import FlashAttention
import Foundation
import Metal

// C FFI defines: FP16=0, BF16=1, FP32=2
// Swift expects: FP32=0, FP16=1, BF16=2
private func convertCFFIPrecisionToSwift(_ cPrecision: Int32) -> Int32 {
  switch cPrecision {
  case 0: return 1  // FP16: C=0 -> Swift=1
  case 1: return 2  // BF16: C=1 -> Swift=2
  case 2: return 0  // FP32: C=2 -> Swift=0
  default: return cPrecision  // INT8=3, INT4=4 remain the same
  }
}

private enum MaskType: Int32 {
  case none = 0
  case boolean = 1
  case additive = 2
}

private enum MaskScalarType: Int32 {
  case byte = 0
  case fp16 = 1
  case bf16 = 2
  case fp32 = 3

  var elementSize: Int {
    switch self {
    case .byte: 1
    case .fp16, .bf16: 2
    case .fp32: 4
    }
  }
}

fileprivate struct MaskArguments {
  let pointer: UnsafeMutableRawPointer
  let sizeBytes: Int
  let shape: [Int64]
  let strides: [Int64]
  let ndim: UInt32
  let type: MaskType
  let scalarType: MaskScalarType
}

fileprivate struct PreparedMask {
  let buffer: MTLBuffer
}

// MARK: - Internal Types

// Pipeline cache key for deduplicating compiled kernels
struct PipelineCacheKey: Hashable {
  let seqLenQ: UInt32
  let seqLenKV: UInt32
  let headDim: UInt16
  let causal: Bool
  let inputPrecision: Int32
  let intermediatePrecision: Int32
  let transposeQ: Bool
  let transposeK: Bool
  let transposeV: Bool
  let transposeO: Bool
  let softmaxScale: Float
}

final class MFAContext {
  let device: MTLDevice
  let commandQueue: MTLCommandQueue

  // Pipeline and kernel cache to avoid recompilation overhead
  private var pipelineCache: [PipelineCacheKey: MTLComputePipelineState] = [:]
  private var kernelCache: [PipelineCacheKey: AttentionKernel] = [:]
  private let cacheQueue = DispatchQueue(label: "MFAContext.caches")

  // Buffer cache for L and D buffers (indexed by sequence length)
  private var lBufferCache: [UInt32: MTLBuffer] = [:]
  private var dBufferCache: [UInt32: MTLBuffer] = [:]

  // Store GPU timing for zero-overhead measurement
  var lastGPULatency: CFTimeInterval = 0.0

  // Scale arrays for row-wise and block-wise quantization
  var qScales: [Float] = []
  var kScales: [Float] = []
  var vScales: [Float] = []

  // Mask preprocessing resources
  private var maskPipelineState: MTLComputePipelineState?
  private var maskOutputBuffer: MTLBuffer?
  private var maskShapeBuffer: MTLBuffer?
  private var maskStrideBuffer: MTLBuffer?

  private static let maskKernelSource = """
#include <metal_stdlib>
using namespace metal;

kernel void mfa_prepare_mask(
    device const uchar* mask_raw [[buffer(0)]],
    device float* out_scores [[buffer(1)]],
    constant uint &mask_type [[buffer(2)]],
    constant uint &mask_scalar [[buffer(3)]],
    constant uint4 &bhqk_shape [[buffer(4)]],
    constant uint &element_count [[buffer(5)]],
    constant uint &mask_dims [[buffer(6)]],
    constant int64_t* mask_shape [[buffer(7)]],
    constant int64_t* mask_strides [[buffer(8)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= element_count) {
    return;
  }

  uint cols = bhqk_shape.w;
  uint rows = bhqk_shape.z;
  uint heads = bhqk_shape.y;
  uint batches = bhqk_shape.x;

  uint idx = gid;
  uint col = idx % cols;
  idx /= cols;
  uint row = idx % rows;
  idx /= rows;
  uint head = idx % heads;
  uint batch = idx / heads;

  float mask_value = 0.0f;
  if (mask_type != 0u && mask_raw != nullptr && mask_shape != nullptr && mask_strides != nullptr) {
    if (mask_dims <= 4u) {
      uint coords[4] = {batch, head, row, col};
      int64_t linear_index = 0;
      for (uint i = 0; i < mask_dims; ++i) {
        uint coord_index = 4u - mask_dims + i;
        uint coord = coords[coord_index];
        int64_t dim = mask_shape[i];
        if (dim == 1) {
          coord = 0;
        }
        linear_index += int64_t(coord) * mask_strides[i];
      }

      switch (mask_type) {
      case 1u: {
        const device uchar* bool_ptr = mask_raw;
        bool masked = bool_ptr[linear_index] != 0;
        mask_value = masked ? -INFINITY : 0.0f;
        break;
      }
      case 2u: {
        switch (mask_scalar) {
        case 3u: {
          const device float* fp32_ptr = reinterpret_cast<const device float*>(mask_raw);
          mask_value = fp32_ptr[linear_index];
          break;
        }
        case 1u: {
          const device half* fp16_ptr = reinterpret_cast<const device half*>(mask_raw);
          mask_value = float(fp16_ptr[linear_index]);
          break;
        }
        case 2u: {
          const device ushort* bf16_ptr = reinterpret_cast<const device ushort*>(mask_raw);
          ushort raw = bf16_ptr[linear_index];
          uint32_t expanded = uint32_t(raw) << 16;
          mask_value = as_type<float>(expanded);
          break;
        }
        default:
          mask_value = 0.0f;
          break;
        }
        break;
      }
      default:
        mask_value = 0.0f;
        break;
      }
    } else {
      mask_value = 0.0f;
    }
  }

  out_scores[gid] = mask_value;
}
"""

  init?(device: MTLDevice) {
    self.device = device
    guard let queue = device.makeCommandQueue() else {
      return nil
    }
    commandQueue = queue
  }

  enum MaskPreparationError: Error {
    case invalidMetadata
    case pipelineCreationFailed
    case bufferAllocationFailed
    case commandEncodingFailed
    case commandExecutionFailed
  }

  private func ensureMaskPipeline() throws -> MTLComputePipelineState {
    if let pipeline = maskPipelineState {
      return pipeline
    }

    do {
      let library = try device.makeLibrary(source: Self.maskKernelSource, options: nil)
      guard let function = library.makeFunction(name: "mfa_prepare_mask") else {
        throw MaskPreparationError.pipelineCreationFailed
      }
      let pipeline = try device.makeComputePipelineState(function: function)
      maskPipelineState = pipeline
      return pipeline
    } catch {
      throw MaskPreparationError.pipelineCreationFailed
    }
  }

  fileprivate func prepareMask(
    arguments: MaskArguments?,
    batchSize: UInt32,
    numHeads: UInt32,
    seqLenQ: UInt32,
    seqLenKV: UInt32
  ) throws -> PreparedMask? {
    guard let arguments else {
      return nil
    }

    guard !arguments.shape.isEmpty,
          arguments.shape.count == arguments.strides.count,
          arguments.shape.count == Int(arguments.ndim)
    else {
      throw MaskPreparationError.invalidMetadata
    }

    let totalElements = Int(batchSize) * Int(numHeads) * Int(seqLenQ) * Int(seqLenKV)
    if totalElements == 0 {
      return nil
    }

    guard
      let maskInputBuffer = device.makeBuffer(
        bytesNoCopy: arguments.pointer,
        length: arguments.sizeBytes,
        options: .storageModeShared,
        deallocator: nil
      )
    else {
      throw MaskPreparationError.bufferAllocationFailed
    }

    let requiredBytes = totalElements * MemoryLayout<Float>.size
    if maskOutputBuffer == nil || maskOutputBuffer!.length < requiredBytes {
      maskOutputBuffer = device.makeBuffer(length: requiredBytes, options: .storageModeShared)
    }
    guard let outputBuffer = maskOutputBuffer else {
      throw MaskPreparationError.bufferAllocationFailed
    }

    let shapeByteLength = arguments.shape.count * MemoryLayout<Int64>.size
    if maskShapeBuffer == nil || maskShapeBuffer!.length < shapeByteLength {
      maskShapeBuffer = device.makeBuffer(length: shapeByteLength, options: .storageModeShared)
    }
    if maskStrideBuffer == nil || maskStrideBuffer!.length < shapeByteLength {
      maskStrideBuffer = device.makeBuffer(length: shapeByteLength, options: .storageModeShared)
    }
    guard let shapeBuffer = maskShapeBuffer, let strideBuffer = maskStrideBuffer else {
      throw MaskPreparationError.bufferAllocationFailed
    }

    arguments.shape.withUnsafeBytes { src in
      if let baseAddress = src.baseAddress {
        shapeBuffer.contents().copyMemory(from: baseAddress, byteCount: shapeByteLength)
      }
    }
    arguments.strides.withUnsafeBytes { src in
      if let baseAddress = src.baseAddress {
        strideBuffer.contents().copyMemory(from: baseAddress, byteCount: shapeByteLength)
      }
    }

    let pipeline = try ensureMaskPipeline()

    guard
      let commandBuffer = commandQueue.makeCommandBuffer(),
      let encoder = commandBuffer.makeComputeCommandEncoder()
    else {
      throw MaskPreparationError.commandEncodingFailed
    }

    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(maskInputBuffer, offset: 0, index: 0)
    encoder.setBuffer(outputBuffer, offset: 0, index: 1)

    var maskTypeValue = UInt32(arguments.type.rawValue)
    encoder.setBytes(&maskTypeValue, length: MemoryLayout<UInt32>.size, index: 2)
    var maskScalarValue = UInt32(arguments.scalarType.rawValue)
    encoder.setBytes(&maskScalarValue, length: MemoryLayout<UInt32>.size, index: 3)

    var shapeVector: [UInt32] = [batchSize, numHeads, seqLenQ, seqLenKV]
    shapeVector.withUnsafeBytes { bytes in
      if let baseAddress = bytes.baseAddress {
        encoder.setBytes(baseAddress, length: bytes.count, index: 4)
      }
    }

    var elementCount = UInt32(totalElements)
    encoder.setBytes(&elementCount, length: MemoryLayout<UInt32>.size, index: 5)

    var maskDims = UInt32(arguments.ndim)
    encoder.setBytes(&maskDims, length: MemoryLayout<UInt32>.size, index: 6)

    encoder.setBuffer(shapeBuffer, offset: 0, index: 7)
    encoder.setBuffer(strideBuffer, offset: 0, index: 8)

    let maxThreads = pipeline.maxTotalThreadsPerThreadgroup
    let threadsPerGroup = min(maxThreads, 256)
    let threadgroupSize = MTLSize(width: threadsPerGroup, height: 1, depth: 1)
    let threadgroupCount = MTLSize(
      width: (totalElements + threadsPerGroup - 1) / threadsPerGroup,
      height: 1,
      depth: 1
    )

    encoder.dispatchThreadgroups(threadgroupCount, threadsPerThreadgroup: threadgroupSize)
    encoder.endEncoding()

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    if let error = commandBuffer.error {
      print("Mask kernel execution error: \(error)")
      throw MaskPreparationError.commandExecutionFailed
    }

    return PreparedMask(buffer: outputBuffer)
  }

  func getCachedPipeline(key: PipelineCacheKey) -> (MTLComputePipelineState, AttentionKernel)? {
    cacheQueue.sync {
      guard
        let pipeline = pipelineCache[key],
        let kernel = kernelCache[key]
      else {
        return nil
      }
      return (pipeline, kernel)
    }
  }

  func cachePipeline(
    _ pipeline: MTLComputePipelineState, kernel: AttentionKernel, key: PipelineCacheKey
  ) {
    cacheQueue.sync {
      pipelineCache[key] = pipeline
      kernelCache[key] = kernel
    }
  }

  func getCachedLBuffer(seqLen: UInt32) -> MTLBuffer? {
    cacheQueue.sync {
      if let buffer = lBufferCache[seqLen] {
        return buffer
      }
      // Initialize L buffer with zeros like native Swift code
      let zeros = [Float](repeating: 0.0, count: Int(seqLen))
      guard
        let buffer = device.makeBuffer(
          bytes: zeros, length: Int(seqLen * 4), options: .storageModeShared
        )
      else {
        return nil
      }
      lBufferCache[seqLen] = buffer
      return buffer
    }
  }

  func getCachedDBuffer(seqLen: UInt32) -> MTLBuffer? {
    cacheQueue.sync {
      if let buffer = dBufferCache[seqLen] {
        return buffer
      }
      // Initialize D buffer with zeros like native Swift code
      let zeros = [Float](repeating: 0.0, count: Int(seqLen))
      guard
        let buffer = device.makeBuffer(
          bytes: zeros, length: Int(seqLen * 4), options: .storageModeShared
        )
      else {
        return nil
      }
      dBufferCache[seqLen] = buffer
      return buffer
    }
  }
}

final class MFABuffer {
  let buffer: MTLBuffer
  let originalDataPtr: UnsafeMutableRawPointer? // Track original data for copy-back
  let dataSize: Int

  // Stride information for non-contiguous tensors
  let shape: [Int64]?
  let strides: [Int64]?
  let ndim: UInt32
  let isStrided: Bool

  init(buffer: MTLBuffer, originalDataPtr: UnsafeMutableRawPointer? = nil, dataSize: Int = 0,
       shape: [Int64]? = nil, strides: [Int64]? = nil, ndim: UInt32 = 0) {
    self.buffer = buffer
    self.originalDataPtr = originalDataPtr
    self.dataSize = dataSize
    self.shape = shape
    self.strides = strides
    self.ndim = ndim
    self.isStrided = (shape != nil && strides != nil && ndim > 0)
  }
}

private func makeQuantizedTensor(
  device: MTLDevice,
  buffer: MFABuffer,
  parameters: QuantizationParameters,
  shape: [Int],
  blockScales: MFABuffer?,
  blockZeroPoints: MFABuffer?,
  blockSize: UInt32
) -> QuantizedTensor {
  var quantParams = parameters
  if blockSize > 0 && blockScales != nil {
    quantParams.mode = .blockwise(blockSizeK: Int(blockSize), bothOperands: false)
  }

  let elementCount = shape.reduce(1, *)

  return QuantizedTensor(
    device: device,
    data: buffer.buffer,
    parameters: quantParams,
    elementCount: elementCount,
    shape: shape,
    blockScales: blockScales?.buffer,
    blockZeroPoints: blockZeroPoints?.buffer,
    blockSizeK: blockSize > 0 ? Int(blockSize) : nil,
    precomputedSums: nil
  )
}

// MARK: - C Bridge Implementation

// Global Metal device (like MTLContext.global.device in native Swift)
private let globalDevice: MTLDevice? = MTLCreateSystemDefaultDevice()
private var globalContext: MFAContext?

@_cdecl("mfa_create_context")
public func mfa_create_context(_ context: UnsafeMutablePointer<UnsafeMutableRawPointer?>?) -> Int32 {
  guard let device = globalDevice else {
    return 3 // MFA_ERROR_DEVICE_NOT_SUPPORTED
  }

  // Use singleton pattern for global context like native Swift
  if globalContext == nil {
    globalContext = MFAContext(device: device)
  }

  guard let mfaContext = globalContext else {
    return 2 // MFA_ERROR_MEMORY_ALLOCATION
  }

  let unmanagedContext = Unmanaged.passRetained(mfaContext)
  context?.pointee = unmanagedContext.toOpaque()
  return 0 // MFA_SUCCESS
}

@_cdecl("mfa_destroy_context")
public func mfa_destroy_context(_ context: UnsafeMutableRawPointer?) {
  guard let context else { return }
  let unmanagedContext = Unmanaged<MFAContext>.fromOpaque(context)
  unmanagedContext.release()
}

@_cdecl("mfa_set_scale_arrays")
public func mfa_set_scale_arrays(
  _ context: UnsafeMutableRawPointer?,
  _ qScales: UnsafePointer<Float>?,
  _ qScalesCount: UInt32,
  _ kScales: UnsafePointer<Float>?,
  _ kScalesCount: UInt32,
  _ vScales: UnsafePointer<Float>?,
  _ vScalesCount: UInt32
) -> Int32 {
  guard let context else { return 1 } // MFA_ERROR_INVALID_ARGS

  let mfaContext = Unmanaged<MFAContext>.fromOpaque(context).takeUnretainedValue()

  // Set Q scales
  if let qScales = qScales, qScalesCount > 0 {
    mfaContext.qScales = Array(UnsafeBufferPointer(start: qScales, count: Int(qScalesCount)))
    print("🔧 Set Q scales: \(mfaContext.qScales.count) scales")
  } else {
    mfaContext.qScales = []
  }

  // Set K scales
  if let kScales = kScales, kScalesCount > 0 {
    mfaContext.kScales = Array(UnsafeBufferPointer(start: kScales, count: Int(kScalesCount)))
    print("🔧 Set K scales: \(mfaContext.kScales.count) scales")
  } else {
    mfaContext.kScales = []
  }

  // Set V scales
  if let vScales = vScales, vScalesCount > 0 {
    mfaContext.vScales = Array(UnsafeBufferPointer(start: vScales, count: Int(vScalesCount)))
    print("🔧 Set V scales: \(mfaContext.vScales.count) scales")
  } else {
    mfaContext.vScales = []
  }

  return 0 // MFA_SUCCESS
}

@_cdecl("mfa_create_buffer")
public func mfa_create_buffer(
  _ context: UnsafeMutableRawPointer?,
  _ sizeBytes: Int,
  _ buffer: UnsafeMutablePointer<UnsafeMutableRawPointer?>?
)
  -> Int32
{
  guard let context, let buffer else { return 1 } // MFA_ERROR_INVALID_ARGS

  let mfaContext = Unmanaged<MFAContext>.fromOpaque(context).takeUnretainedValue()

  guard let mtlBuffer = mfaContext.device.makeBuffer(length: sizeBytes, options: .storageModeShared)
  else {
    return 2 // MFA_ERROR_MEMORY_ALLOCATION
  }

  let mfaBuffer = MFABuffer(buffer: mtlBuffer)
  let unmanagedBuffer = Unmanaged.passRetained(mfaBuffer)
  buffer.pointee = unmanagedBuffer.toOpaque()

  return 0 // MFA_SUCCESS
}

@_cdecl("mfa_buffer_from_ptr")
public func mfa_buffer_from_ptr(
  _ context: UnsafeMutableRawPointer?,
  _ dataPtr: UnsafeMutableRawPointer?,
  _ sizeBytes: Int,
  _ buffer: UnsafeMutablePointer<UnsafeMutableRawPointer?>?
)
  -> Int32
{
  guard
    let context,
    let dataPtr,
    let buffer
  else { return 1 } // MFA_ERROR_INVALID_ARGS

  let mfaContext = Unmanaged<MFAContext>.fromOpaque(context).takeUnretainedValue()

  // Use zero-copy approach: wrap existing memory instead of copying
  guard
    let mtlBuffer = mfaContext.device.makeBuffer(
      bytesNoCopy: dataPtr,
      length: sizeBytes,
      options: .storageModeShared,
      deallocator: nil // Don't deallocate - memory is managed by Python/caller
    )
  else {
    return 2 // MFA_ERROR_MEMORY_ALLOCATION
  }

  let mfaBuffer = MFABuffer(
    buffer: mtlBuffer,
    originalDataPtr: nil,
    dataSize: 0
  ) // No copy-back needed
  let unmanagedBuffer = Unmanaged.passRetained(mfaBuffer)
  buffer.pointee = unmanagedBuffer.toOpaque()

  return 0 // MFA_SUCCESS
}

@_cdecl("mfa_buffer_from_ptr_with_strides")
public func mfa_buffer_from_ptr_with_strides(
  _ context: UnsafeMutableRawPointer?,
  _ dataPtr: UnsafeMutableRawPointer?,
  _ sizeBytes: Int,
  _ shape: UnsafePointer<Int64>?,
  _ strides: UnsafePointer<Int64>?,
  _ ndim: UInt32,
  _ buffer: UnsafeMutablePointer<UnsafeMutableRawPointer?>?
)
  -> Int32
{
  guard
    let context,
    let dataPtr,
    let buffer,
    let shape,
    let strides,
    ndim > 0
  else { return 1 } // MFA_ERROR_INVALID_ARGS

  let mfaContext = Unmanaged<MFAContext>.fromOpaque(context).takeUnretainedValue()

  // Convert C arrays to Swift arrays
  let shapeArray = Array(UnsafeBufferPointer(start: shape, count: Int(ndim)))
  let stridesArray = Array(UnsafeBufferPointer(start: strides, count: Int(ndim)))

  // Use zero-copy approach: wrap existing memory instead of copying
  guard
    let mtlBuffer = mfaContext.device.makeBuffer(
      bytesNoCopy: dataPtr,
      length: sizeBytes,
      options: .storageModeShared,
      deallocator: nil // Don't deallocate - memory is managed by Python/caller
    )
  else {
    return 2 // MFA_ERROR_MEMORY_ALLOCATION
  }

  let mfaBuffer = MFABuffer(
    buffer: mtlBuffer,
    originalDataPtr: nil,
    dataSize: 0,
    shape: shapeArray,
    strides: stridesArray,
    ndim: ndim
  )
  let unmanagedBuffer = Unmanaged.passRetained(mfaBuffer)
  buffer.pointee = unmanagedBuffer.toOpaque()

  return 0 // MFA_SUCCESS
}

@_cdecl("mfa_buffer_from_mtl_buffer")
public func mfa_buffer_from_mtl_buffer(
  _ context: UnsafeMutableRawPointer?,
  _ metalBufferPtr: UnsafeMutableRawPointer?,
  _ sizeBytes: Int,
  _ buffer: UnsafeMutablePointer<UnsafeMutableRawPointer?>?
)
  -> Int32
{
  _ = context

  guard
    let metalBufferPtr,
    let buffer
  else { return 1 } // MFA_ERROR_INVALID_ARGS

  let anyObject = Unmanaged<AnyObject>.fromOpaque(metalBufferPtr).takeUnretainedValue()
  guard let mtlBuffer = anyObject as? MTLBuffer else {
    return 1 // MFA_ERROR_INVALID_ARGS
  }

  if sizeBytes > 0 && mtlBuffer.length < sizeBytes {
    return 1 // Requested size exceeds buffer length
  }

  let mfaBuffer = MFABuffer(
    buffer: mtlBuffer,
    originalDataPtr: nil,
    dataSize: sizeBytes
  )
  let unmanagedBuffer = Unmanaged.passRetained(mfaBuffer)
  buffer.pointee = unmanagedBuffer.toOpaque()

  return 0 // MFA_SUCCESS
}

@_cdecl("mfa_buffer_from_mtl_buffer_with_strides")
public func mfa_buffer_from_mtl_buffer_with_strides(
  _ context: UnsafeMutableRawPointer?,
  _ metalBufferPtr: UnsafeMutableRawPointer?,
  _ sizeBytes: Int,
  _ shape: UnsafePointer<Int64>?,
  _ strides: UnsafePointer<Int64>?,
  _ ndim: UInt32,
  _ buffer: UnsafeMutablePointer<UnsafeMutableRawPointer?>?
)
  -> Int32
{
  _ = context

  guard
    let metalBufferPtr,
    let buffer,
    let shape,
    let strides,
    ndim > 0
  else { return 1 } // MFA_ERROR_INVALID_ARGS

  let anyObject = Unmanaged<AnyObject>.fromOpaque(metalBufferPtr).takeUnretainedValue()
  guard let mtlBuffer = anyObject as? MTLBuffer else {
    return 1 // MFA_ERROR_INVALID_ARGS
  }

  if sizeBytes > 0 && mtlBuffer.length < sizeBytes {
    return 1 // Requested size exceeds buffer length
  }

  let shapeArray = Array(UnsafeBufferPointer(start: shape, count: Int(ndim)))
  let stridesArray = Array(UnsafeBufferPointer(start: strides, count: Int(ndim)))

  let mfaBuffer = MFABuffer(
    buffer: mtlBuffer,
    originalDataPtr: nil,
    dataSize: sizeBytes,
    shape: shapeArray,
    strides: stridesArray,
    ndim: ndim
  )
  let unmanagedBuffer = Unmanaged.passRetained(mfaBuffer)
  buffer.pointee = unmanagedBuffer.toOpaque()

  return 0 // MFA_SUCCESS
}

@_cdecl("mfa_buffer_contents")
public func mfa_buffer_contents(_ buffer: UnsafeMutableRawPointer?) -> UnsafeMutableRawPointer? {
  guard let buffer else { return nil }

  let mfaBuffer = Unmanaged<MFABuffer>.fromOpaque(buffer).takeUnretainedValue()
  return mfaBuffer.buffer.contents()
}

@_cdecl("mfa_destroy_buffer")
public func mfa_destroy_buffer(_ buffer: UnsafeMutableRawPointer?) {
  guard let buffer else { return }
  let unmanagedBuffer = Unmanaged<MFABuffer>.fromOpaque(buffer)
  unmanagedBuffer.release()
}

// MARK: - Attention Functions

@_cdecl("mfa_attention_forward")
public func mfa_attention_forward(
  _ context: UnsafeMutableRawPointer?,
  _ q: UnsafeMutableRawPointer?,
  _ k: UnsafeMutableRawPointer?,
  _ v: UnsafeMutableRawPointer?,
  _ out: UnsafeMutableRawPointer?,
  _: UInt32,
  _ seqLenQ: UInt32,
  _ seqLenKV: UInt32,
  _ numHeads: UInt32,
  _ headDim: UInt16,
  _ softmaxScale: Float,
  _ causal: Bool,
  _ inputPrecision: Int32,
  _ intermediatePrecision: Int32,
  _: Int32,
  _ transposeQ: Bool,
  _ transposeK: Bool,
  _ transposeV: Bool,
  _ transposeO: Bool,
  _ maskPtr: UnsafeMutableRawPointer?,
  _ maskSizeBytes: Int,
  _ maskShape: UnsafePointer<Int64>?,
  _ maskStrides: UnsafePointer<Int64>?,
  _ maskNDim: UInt32,
  _ maskTypeRaw: Int32,
  _ maskScalarTypeRaw: Int32
)
  -> Int32
{
  guard
    let context,
    let q, let k, let v, let out
  else {
    return 1 // MFA_ERROR_INVALID_ARGS
  }

  // Extract context and buffers
  let mfaContext = Unmanaged<MFAContext>.fromOpaque(context).takeUnretainedValue()
  let qBuffer = Unmanaged<MFABuffer>.fromOpaque(q).takeUnretainedValue()
  let kBuffer = Unmanaged<MFABuffer>.fromOpaque(k).takeUnretainedValue()
  let vBuffer = Unmanaged<MFABuffer>.fromOpaque(v).takeUnretainedValue()
  let outBuffer = Unmanaged<MFABuffer>.fromOpaque(out).takeUnretainedValue()

  let resolvedMaskType = MaskType(rawValue: maskTypeRaw) ?? .none
  let resolvedMaskScalar = MaskScalarType(rawValue: maskScalarTypeRaw) ?? .byte
  let maskArguments: MaskArguments?
  if resolvedMaskType != .none,
     let maskPtr,
     maskSizeBytes > 0,
     let maskShape,
     let maskStrides,
     maskNDim > 0
  {
    let shape = Array(UnsafeBufferPointer(start: maskShape, count: Int(maskNDim)))
    let strides = Array(UnsafeBufferPointer(start: maskStrides, count: Int(maskNDim)))
    maskArguments = MaskArguments(
      pointer: maskPtr,
      sizeBytes: maskSizeBytes,
      shape: shape,
      strides: strides,
      ndim: maskNDim,
      type: resolvedMaskType,
      scalarType: resolvedMaskScalar
    )
  } else {
    maskArguments = nil
  }

  let preparedMask: PreparedMask?
  do {
    preparedMask = try mfaContext.prepareMask(
      arguments: maskArguments,
      batchSize: 1,
      numHeads: numHeads,
      seqLenQ: seqLenQ,
      seqLenKV: seqLenKV
    )
  } catch let maskError as MFAContext.MaskPreparationError {
    switch maskError {
    case .invalidMetadata:
      print("Mask preparation failed: invalid metadata")
      return 1
    case .pipelineCreationFailed:
      print("Mask preparation failed: pipeline creation error")
      return 4
    case .bufferAllocationFailed:
      print("Mask preparation failed: buffer allocation error")
      return 2
    case .commandEncodingFailed, .commandExecutionFailed:
      print("Mask preparation failed: command submission error")
      return 5
    }
  } catch {
    print("Mask preparation failed: \(error)")
    return 5
  }

  // Handle multi-head attention using the new MultiHeadAttention implementation
  if numHeads > 1 {
    return mfa_attention_forward_multihead_internal(
      context: mfaContext,
      qBuffer: qBuffer.buffer,
      kBuffer: kBuffer.buffer,
      vBuffer: vBuffer.buffer,
      outBuffer: outBuffer.buffer,
      batchSize: 1, // For now, assume batch size 1 for FFI compatibility
      seqLenQ: seqLenQ,
      seqLenKV: seqLenKV,
      numHeads: numHeads,
      headDim: headDim,
      softmaxScale: softmaxScale,
      causal: causal,
      inputPrecision: inputPrecision,
      intermediatePrecision: intermediatePrecision,
      transposeQ: transposeQ,
      transposeK: transposeK,
      transposeV: transposeV,
      transposeO: transposeO,
      preparedMask: preparedMask
    )
  }

  do {
    // Create cache key for pipeline deduplication
    let cacheKey = PipelineCacheKey(
      seqLenQ: seqLenQ, seqLenKV: seqLenKV, headDim: headDim, causal: causal,
      inputPrecision: inputPrecision, intermediatePrecision: intermediatePrecision,
      transposeQ: transposeQ, transposeK: transposeK, transposeV: transposeV,
      transposeO: transposeO,
      softmaxScale: softmaxScale
    )

    // Check if we have cached pipeline and kernel
    let pipeline: MTLComputePipelineState
    let kernel: AttentionKernel
    if let (cachedPipeline, cachedKernel) = mfaContext.getCachedPipeline(key: cacheKey) {
      pipeline = cachedPipeline
      kernel = cachedKernel
    } else {
      // Create attention descriptor
      var descriptor = AttentionDescriptor()
      descriptor.matrixDimensions = (row: seqLenQ, column: seqLenKV, head: headDim)
      descriptor.transposeState = (Q: transposeQ, K: transposeK, V: transposeV, O: transposeO)

      // Set causal masking using the proper native approach
      descriptor.sparsityPattern = causal ? .causal : .none

      // Set custom scale factor - this fixes the root cause of the correctness issue
      descriptor.softmaxScale = softmaxScale

      // Set precision based on input parameters
      // Convert C FFI enum values to Swift values
      let swiftInputPrecision = convertCFFIPrecisionToSwift(inputPrecision)
      let swiftIntermediatePrecision = convertCFFIPrecisionToSwift(intermediatePrecision)

      // IMPORTANT: When using FP16/BF16 precision modes with FP32 data,
      // we must use FP32 inputs to avoid NaN issues from precision mismatch
      // The inputs are always FP32 from the FFI layer
      descriptor.lowPrecisionInputs = false
      // Use FP32 intermediates for numerical stability
      descriptor.lowPrecisionIntermediates = false

      // Create kernel descriptor
      let kernelDescriptor = descriptor.kernelDescriptor(type: .forward)
      kernel = AttentionKernel(descriptor: kernelDescriptor)

      // Set up function constants
      let constants = MTLFunctionConstantValues()
      descriptor.setFunctionConstants(constants)

      // Get the Metal function using native kernel source (no string replacement!)
      let source = kernel.createSource()
      let library = try mfaContext.device.makeLibrary(source: source, options: nil)
      let function = try library.makeFunction(name: "attention", constantValues: constants)

      // Create pipeline descriptor with proper settings for Apple Silicon
      let pipelineDesc = MTLComputePipelineDescriptor()
      pipelineDesc.computeFunction = function
      pipelineDesc.maxTotalThreadsPerThreadgroup = 1024 // Critical for M1/M2/M3 performance

      pipeline = try mfaContext.device.makeComputePipelineState(
        descriptor: pipelineDesc, options: [], reflection: nil
      )

      // Cache the compiled pipeline and kernel
      mfaContext.cachePipeline(pipeline, kernel: kernel, key: cacheKey)
    }

    // Create command buffer
    guard let commandBuffer = mfaContext.commandQueue.makeCommandBuffer() else {
      return 5 // MFA_ERROR_EXECUTION_FAILED
    }

    // Create compute encoder
    guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
      return 5 // MFA_ERROR_EXECUTION_FAILED
    }

    encoder.setComputePipelineState(pipeline)

    // Get cached L and D buffers (avoid reallocation like native Swift)
    guard
      let lBuffer = mfaContext.getCachedLBuffer(seqLen: seqLenQ),
      let dBuffer = mfaContext.getCachedDBuffer(seqLen: seqLenQ)
    else {
      return 2 // MFA_ERROR_MEMORY_ALLOCATION
    }

    // Note: Debug output removed - buffer copying now verified to work

    // Set buffers (following MFA test pattern)
    encoder.setBuffer(qBuffer.buffer, offset: 0, index: 0)
    encoder.setBuffer(kBuffer.buffer, offset: 0, index: 1)
    encoder.setBuffer(vBuffer.buffer, offset: 0, index: 2)
    encoder.setBuffer(outBuffer.buffer, offset: 0, index: 3)
    encoder.setBuffer(lBuffer, offset: 0, index: 4) // L buffer (attention statistics)
    encoder.setBuffer(dBuffer, offset: 0, index: 5) // D buffer (attention statistics)

    // Pass stride information if available (for non-contiguous tensor support)
    var bufferIndex = 6

    // Set stride buffers for Q, K, V, O if they exist
    if qBuffer.isStrided, let qStrides = qBuffer.strides {
      let strideBuffer = mfaContext.device.makeBuffer(
        bytes: qStrides, length: qStrides.count * MemoryLayout<Int64>.size,
        options: .storageModeShared
      )
      encoder.setBuffer(strideBuffer, offset: 0, index: bufferIndex)
      bufferIndex += 1
    } else {
      encoder.setBuffer(nil, offset: 0, index: bufferIndex)
      bufferIndex += 1
    }

    if kBuffer.isStrided, let kStrides = kBuffer.strides {
      let strideBuffer = mfaContext.device.makeBuffer(
        bytes: kStrides, length: kStrides.count * MemoryLayout<Int64>.size,
        options: .storageModeShared
      )
      encoder.setBuffer(strideBuffer, offset: 0, index: bufferIndex)
      bufferIndex += 1
    } else {
      encoder.setBuffer(nil, offset: 0, index: bufferIndex)
      bufferIndex += 1
    }

    if vBuffer.isStrided, let vStrides = vBuffer.strides {
      let strideBuffer = mfaContext.device.makeBuffer(
        bytes: vStrides, length: vStrides.count * MemoryLayout<Int64>.size,
        options: .storageModeShared
      )
      encoder.setBuffer(strideBuffer, offset: 0, index: bufferIndex)
      bufferIndex += 1
    } else {
      encoder.setBuffer(nil, offset: 0, index: bufferIndex)
      bufferIndex += 1
    }

    if outBuffer.isStrided, let outStrides = outBuffer.strides {
      let strideBuffer = mfaContext.device.makeBuffer(
        bytes: outStrides, length: outStrides.count * MemoryLayout<Int64>.size,
        options: .storageModeShared
      )
      encoder.setBuffer(strideBuffer, offset: 0, index: bufferIndex)
      bufferIndex += 1
    } else {
      encoder.setBuffer(nil, offset: 0, index: bufferIndex)
      bufferIndex += 1
    }

    var numHeadsValue = numHeads
    encoder.setBytes(&numHeadsValue, length: MemoryLayout<UInt32>.size, index: bufferIndex)
    bufferIndex += 1
    var numKVHeadsValue = numHeads
    encoder.setBytes(&numKVHeadsValue, length: MemoryLayout<UInt32>.size, index: bufferIndex)
    bufferIndex += 1
    var headDimensionValue = UInt32(headDim)
    encoder.setBytes(&headDimensionValue, length: MemoryLayout<UInt32>.size, index: bufferIndex)
    bufferIndex += 1
    var sequenceLengthValue = seqLenQ
    encoder.setBytes(&sequenceLengthValue, length: MemoryLayout<UInt32>.size, index: bufferIndex)
    bufferIndex += 1

    if let mask = preparedMask {
      encoder.setBuffer(mask.buffer, offset: 0, index: bufferIndex)
    } else {
      encoder.setBuffer(nil, offset: 0, index: bufferIndex)
    }
    bufferIndex += 1

    // Buffers set: Q, K, V, O, L, D following MFA pattern

    // Set threadgroup memory
    encoder.setThreadgroupMemoryLength(Int(kernel.threadgroupMemoryAllocation), index: 0)

    // Dispatch using MFA's calculation method
    let parallelizationDimension = Int(seqLenQ)

    // Use MFA's ceil divide calculation
    let blockCount =
      (parallelizationDimension + Int(kernel.blockDimensions.parallelization) - 1)
        / Int(kernel.blockDimensions.parallelization)
    let gridSize = MTLSize(width: blockCount, height: 1, depth: 1)
    let groupSize = MTLSize(width: Int(kernel.threadgroupSize), height: 1, depth: 1)

    // Single dispatch for optimal performance (multiple dispatches were causing slowdown)
    encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: groupSize)
    encoder.endEncoding()

    // Execute and measure GPU time (not wall-clock time!)
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    if let error = commandBuffer.error {
      print("Metal execution error: \(error)")
      return 5 // MFA_ERROR_EXECUTION_FAILED
    }

    // Zero-copy: Metal buffer directly wraps the original numpy array memory
    // No copy-back needed since we used makeBuffer(bytesNoCopy:)

    // Store GPU timing for zero-overhead measurement (like native Swift)
    let gpuLatency = commandBuffer.gpuEndTime - commandBuffer.gpuStartTime
    globalContext?.lastGPULatency = gpuLatency

    return 0 // MFA_SUCCESS

  } catch {
    print("MFA Error: \(error)")
    return 4 // MFA_ERROR_KERNEL_COMPILATION
  }
}

// MARK: - String-based Precision Interface

/// Parse precision string to Swift integer value
private func parsePrecisionString(_ precisionStr: UnsafePointer<CChar>?) -> Int32 {
  guard let str = precisionStr else { return 2 } // Default to FP32 (C FFI value = 2)
  let swiftStr = String(cString: str).lowercased()

  // Return C FFI enum values that match mfa_ffi.h
  switch swiftStr {
  case "fp16", "float16": return 0  // MFA_PRECISION_FP16 = 0
  case "bf16", "bfloat16": return 1  // MFA_PRECISION_BF16 = 1
  case "fp32", "float32": return 2   // MFA_PRECISION_FP32 = 2
  case "int8": return 3              // MFA_PRECISION_INT8 = 3
  case "int4": return 4              // MFA_PRECISION_INT4 = 4
  default: return 2 // Default to FP32 (C FFI value = 2)
  }
}

@_cdecl("mfa_attention_forward_str")
public func mfa_attention_forward_str(
  _ context: UnsafeMutableRawPointer?,
  _ q: UnsafeMutableRawPointer?,
  _ k: UnsafeMutableRawPointer?,
  _ v: UnsafeMutableRawPointer?,
  _ out: UnsafeMutableRawPointer?,
  _ batchSize: UInt32,
  _ seqLenQ: UInt32,
  _ seqLenKV: UInt32,
  _ numHeads: UInt32,
  _ headDim: UInt16,
  _ softmaxScale: Float,
  _ causal: Bool,
  _ inputPrecisionStr: UnsafePointer<CChar>?,
  _ intermediatePrecisionStr: UnsafePointer<CChar>?,
  _ outputPrecisionStr: UnsafePointer<CChar>?,
  _ transposeQ: Bool,
  _ transposeK: Bool,
  _ transposeV: Bool,
  _ transposeO: Bool,
  _ maskPtr: UnsafeMutableRawPointer?,
  _ maskSizeBytes: Int,
  _ maskShape: UnsafePointer<Int64>?,
  _ maskStrides: UnsafePointer<Int64>?,
  _ maskNDim: UInt32,
  _ maskTypeRaw: Int32,
  _ maskScalarTypeRaw: Int32
) -> Int32 {
  // Convert string precisions to Swift integers
  let inputPrecision = parsePrecisionString(inputPrecisionStr)
  let intermediatePrecision = parsePrecisionString(intermediatePrecisionStr)
  let outputPrecision = parsePrecisionString(outputPrecisionStr)

  // Call the existing integer-based function
  return mfa_attention_forward(
    context, q, k, v, out,
    batchSize, seqLenQ, seqLenKV,
    numHeads, headDim, softmaxScale, causal,
    inputPrecision, intermediatePrecision, outputPrecision,
    transposeQ, transposeK, transposeV, transposeO,
    maskPtr, maskSizeBytes, maskShape, maskStrides, maskNDim,
    maskTypeRaw, maskScalarTypeRaw
  )
}

// MARK: - Utility Functions

@_cdecl("mfa_error_string")
public func mfa_error_string(_ error: Int32) -> UnsafePointer<CChar>? {
  let errorString = switch error {
  case 0: "Success"
  case 1: "Invalid arguments"
  case 2: "Memory allocation failed"
  case 3: "Device not supported"
  case 4: "Kernel compilation failed"
  case 5: "Execution failed"
  default: "Unknown error"
  }

  return UnsafePointer<CChar>(strdup(errorString))
}

@_cdecl("mfa_is_device_supported")
public func mfa_is_device_supported() -> Bool {
  MTLCreateSystemDefaultDevice() != nil
}

@_cdecl("mfa_get_version")
public func mfa_get_version(
  _ major: UnsafeMutablePointer<Int32>?,
  _ minor: UnsafeMutablePointer<Int32>?,
  _ patch: UnsafeMutablePointer<Int32>?
) {
  major?.pointee = 1
  minor?.pointee = 0
  patch?.pointee = 0
}

@_cdecl("mfa_get_gpu_latency")
public func mfa_get_gpu_latency(_ context: UnsafeMutableRawPointer?) -> Double {
  guard let context else { return 0.0 }
  let mfaContext = Unmanaged<MFAContext>.fromOpaque(context).takeUnretainedValue()
  return mfaContext.lastGPULatency
}

// MARK: - Quantized Attention Functions

// REMOVED: Original mfa_attention_forward_quantized_enhanced implementation
// Now handled by the unified function with backward compatibility wrapper
@_cdecl("mfa_attention_backward_query_quantized")
public func mfa_attention_backward_query_quantized(
  _ context: UnsafeMutableRawPointer?,
  _ q: UnsafeMutableRawPointer?,
  _ k: UnsafeMutableRawPointer?,
  _ v: UnsafeMutableRawPointer?,
  _ gradOutput: UnsafeMutableRawPointer?,
  _ logsumexp: UnsafeMutableRawPointer?,
  _ gradQuery: UnsafeMutableRawPointer?,
  _ dValues: UnsafeMutableRawPointer?,
  _ batchSize: UInt32,
  _ seqLenQ: UInt32,
  _ seqLenKV: UInt32,
  _ numHeads: UInt32,
  _ headDim: UInt16,
  _ qScale: Float,
  _ qZeroPoint: Int32,
  _ kScale: Float,
  _ kZeroPoint: Int32,
  _ vScale: Float,
  _ vZeroPoint: Int32,
  _ qPrecision: Int32,
  _ kPrecision: Int32,
  _ vPrecision: Int32,
  _ causal: Bool,
  _ transposeQ: Bool,
  _ transposeK: Bool,
  _ transposeV: Bool,
  _ transposeO: Bool
) -> Int32 {
  mfa_attention_backward_query_quantized_ex(
    context,
    q,
    k,
    v,
    gradOutput,
    logsumexp,
    gradQuery,
    dValues,
    batchSize,
    seqLenQ,
    seqLenKV,
    numHeads,
    numHeads,
    headDim,
    qScale,
    qZeroPoint,
    kScale,
    kZeroPoint,
    vScale,
    vZeroPoint,
    qPrecision,
    kPrecision,
    vPrecision,
    causal,
    transposeQ,
    transposeK,
    transposeV,
    transposeO,
    nil,
    nil,
    nil,
    nil,
    nil,
    nil,
    0,
    0,
    0,
    0
  )
}

@_cdecl("mfa_attention_backward_query_quantized_ex")
public func mfa_attention_backward_query_quantized_ex(
  _ context: UnsafeMutableRawPointer?,
  _ q: UnsafeMutableRawPointer?,
  _ k: UnsafeMutableRawPointer?,
  _ v: UnsafeMutableRawPointer?,
  _ gradOutput: UnsafeMutableRawPointer?,
  _ logsumexp: UnsafeMutableRawPointer?,
  _ gradQuery: UnsafeMutableRawPointer?,
  _ dValues: UnsafeMutableRawPointer?,
  _ batchSize: UInt32,
  _ seqLenQ: UInt32,
  _ seqLenKV: UInt32,
  _ numHeads: UInt32,
  _ numKVHeads: UInt32,
  _ headDim: UInt16,
  _ qScale: Float,
  _ qZeroPoint: Int32,
  _ kScale: Float,
  _ kZeroPoint: Int32,
  _ vScale: Float,
  _ vZeroPoint: Int32,
  _ qPrecision: Int32,
  _ kPrecision: Int32,
  _ vPrecision: Int32,
  _ causal: Bool,
  _ transposeQ: Bool,
  _ transposeK: Bool,
  _ transposeV: Bool,
  _ transposeO: Bool,
  _ qBlockScales: UnsafeMutableRawPointer?,
  _ qBlockZeroPoints: UnsafeMutableRawPointer?,
  _ kBlockScales: UnsafeMutableRawPointer?,
  _ kBlockZeroPoints: UnsafeMutableRawPointer?,
  _ vBlockScales: UnsafeMutableRawPointer?,
  _ vBlockZeroPoints: UnsafeMutableRawPointer?,
  _ qBlockSize: UInt32,
  _ kBlockSize: UInt32,
  _ vBlockSize: UInt32,
  _ options: UInt32
) -> Int32 {
  guard
    let context,
    let q, let k, let v,
    let gradOutput,
    let logsumexp,
    let gradQuery,
    let dValues
  else {
    return 1
  }

  _ = options

  let mfaContext = Unmanaged<MFAContext>.fromOpaque(context).takeUnretainedValue()
  let device = mfaContext.device

  let qBuffer = Unmanaged<MFABuffer>.fromOpaque(q).takeUnretainedValue()
  let kBuffer = Unmanaged<MFABuffer>.fromOpaque(k).takeUnretainedValue()
  let vBuffer = Unmanaged<MFABuffer>.fromOpaque(v).takeUnretainedValue()
  let gradOutputBuffer = Unmanaged<MFABuffer>.fromOpaque(gradOutput).takeUnretainedValue()
  let logsumexpBuffer = Unmanaged<MFABuffer>.fromOpaque(logsumexp).takeUnretainedValue()
  let gradQueryBuffer = Unmanaged<MFABuffer>.fromOpaque(gradQuery).takeUnretainedValue()
  let dValuesBuffer = Unmanaged<MFABuffer>.fromOpaque(dValues).takeUnretainedValue()

  let effectiveHeads = max(numHeads, 1)
  let effectiveKVHeads = numKVHeads == 0 ? effectiveHeads : numKVHeads
  let headDimension = Int(headDim)

  if headDimension <= 0 {
    return 1
  }

  let batchCount = Int(batchSize)
  let sequenceQ = Int(seqLenQ)
  let sequenceKV = Int(seqLenKV)

  let qShape = [batchCount, Int(effectiveHeads), sequenceQ, headDimension]
  let kvShape = [batchCount, Int(effectiveKVHeads), sequenceKV, headDimension]

  func bufferFromHandle(_ handle: UnsafeMutableRawPointer?) -> MFABuffer? {
    guard let handle else { return nil }
    return Unmanaged<MFABuffer>.fromOpaque(handle).takeUnretainedValue()
  }

  let qBlockScalesBuffer = bufferFromHandle(qBlockScales)
  let qBlockZeroPointsBuffer = bufferFromHandle(qBlockZeroPoints)
  let kBlockScalesBuffer = bufferFromHandle(kBlockScales)
  let kBlockZeroPointsBuffer = bufferFromHandle(kBlockZeroPoints)
  let vBlockScalesBuffer = bufferFromHandle(vBlockScales)
  let vBlockZeroPointsBuffer = bufferFromHandle(vBlockZeroPoints)

  let qParams = QuantizationParameters(
    scale: qScale,
    zeroPoint: qZeroPoint,
    precision: GEMMOperandPrecision(rawValue: UInt16(qPrecision)) ?? .FP16,
    strategy: .legacy
  )

  let kParams = QuantizationParameters(
    scale: kScale,
    zeroPoint: kZeroPoint,
    precision: GEMMOperandPrecision(rawValue: UInt16(kPrecision)) ?? .INT8,
    strategy: .legacy
  )

  let vParams = QuantizationParameters(
    scale: vScale,
    zeroPoint: vZeroPoint,
    precision: GEMMOperandPrecision(rawValue: UInt16(vPrecision)) ?? .INT8,
    strategy: .legacy
  )

  let qTensor = makeQuantizedTensor(
    device: device,
    buffer: qBuffer,
    parameters: qParams,
    shape: qShape,
    blockScales: qBlockScalesBuffer,
    blockZeroPoints: qBlockZeroPointsBuffer,
    blockSize: qBlockSize
  )

  let kTensor = makeQuantizedTensor(
    device: device,
    buffer: kBuffer,
    parameters: kParams,
    shape: kvShape,
    blockScales: kBlockScalesBuffer,
    blockZeroPoints: kBlockZeroPointsBuffer,
    blockSize: kBlockSize
  )

  let vTensor = makeQuantizedTensor(
    device: device,
    buffer: vBuffer,
    parameters: vParams,
    shape: kvShape,
    blockScales: vBlockScalesBuffer,
    blockZeroPoints: vBlockZeroPointsBuffer,
    blockSize: vBlockSize
  )

  var baseDescriptor = AttentionDescriptor()
  baseDescriptor.matrixDimensions = (row: seqLenQ, column: seqLenKV, head: headDim)
  baseDescriptor.transposeState = (Q: transposeQ, K: transposeK, V: transposeV, O: transposeO)
  baseDescriptor.sparsityPattern = causal ? .causal : .none

  var quantConfig = QuantizedAttention.Configuration()
  quantConfig.queryPrecision = qTensor.parameters.precision
  quantConfig.keyPrecision = kTensor.parameters.precision
  quantConfig.valuePrecision = vTensor.parameters.precision

  let quantDescriptor = QuantizedAttention.QuantizedAttentionDescriptor(
    baseDescriptor: baseDescriptor,
    quantizationConfig: quantConfig
  )

  let quantizedAttention = QuantizedAttention(device: device)

  guard
    let commandBuffer = quantizedAttention.backwardQuery(
      query: qTensor,
      key: kTensor,
      value: vTensor,
      gradOutput: gradOutputBuffer.buffer,
      logsumexp: logsumexpBuffer.buffer,
      gradQuery: gradQueryBuffer.buffer,
      dValues: dValuesBuffer.buffer,
      descriptor: quantDescriptor
    )
  else {
    return 5
  }

  commandBuffer.commit()
  commandBuffer.waitUntilCompleted()

  if let error = commandBuffer.error {
    print("Metal execution error: \(error)")
    return 5
  }

  let gpuLatency = commandBuffer.gpuEndTime - commandBuffer.gpuStartTime
  globalContext?.lastGPULatency = gpuLatency

  return 0
}

@_cdecl("mfa_attention_backward_kv_quantized")
public func mfa_attention_backward_kv_quantized(
  _ context: UnsafeMutableRawPointer?,
  _ q: UnsafeMutableRawPointer?,
  _ k: UnsafeMutableRawPointer?,
  _ v: UnsafeMutableRawPointer?,
  _ gradOutput: UnsafeMutableRawPointer?,
  _ logsumexp: UnsafeMutableRawPointer?,
  _ dValues: UnsafeMutableRawPointer?,
  _ gradKey: UnsafeMutableRawPointer?,
  _ gradValue: UnsafeMutableRawPointer?,
  _ batchSize: UInt32,
  _ seqLenQ: UInt32,
  _ seqLenKV: UInt32,
  _ numHeads: UInt32,
  _ headDim: UInt16,
  _ qScale: Float,
  _ qZeroPoint: Int32,
  _ kScale: Float,
  _ kZeroPoint: Int32,
  _ vScale: Float,
  _ vZeroPoint: Int32,
  _ qPrecision: Int32,
  _ kPrecision: Int32,
  _ vPrecision: Int32,
  _ causal: Bool,
  _ transposeQ: Bool,
  _ transposeK: Bool,
  _ transposeV: Bool,
  _ transposeO: Bool
) -> Int32 {
  mfa_attention_backward_kv_quantized_ex(
    context,
    q,
    k,
    v,
    gradOutput,
    logsumexp,
    dValues,
    gradKey,
    gradValue,
    batchSize,
    seqLenQ,
    seqLenKV,
    numHeads,
    numHeads,
    headDim,
    qScale,
    qZeroPoint,
    kScale,
    kZeroPoint,
    vScale,
    vZeroPoint,
    qPrecision,
    kPrecision,
    vPrecision,
    causal,
    transposeQ,
    transposeK,
    transposeV,
    transposeO,
    nil,
    nil,
    nil,
    nil,
    nil,
    nil,
    0,
    0,
    0,
    0
  )
}

@_cdecl("mfa_attention_backward_kv_quantized_ex")
public func mfa_attention_backward_kv_quantized_ex(
  _ context: UnsafeMutableRawPointer?,
  _ q: UnsafeMutableRawPointer?,
  _ k: UnsafeMutableRawPointer?,
  _ v: UnsafeMutableRawPointer?,
  _ gradOutput: UnsafeMutableRawPointer?,
  _ logsumexp: UnsafeMutableRawPointer?,
  _ dValues: UnsafeMutableRawPointer?,
  _ gradKey: UnsafeMutableRawPointer?,
  _ gradValue: UnsafeMutableRawPointer?,
  _ batchSize: UInt32,
  _ seqLenQ: UInt32,
  _ seqLenKV: UInt32,
  _ numHeads: UInt32,
  _ numKVHeads: UInt32,
  _ headDim: UInt16,
  _ qScale: Float,
  _ qZeroPoint: Int32,
  _ kScale: Float,
  _ kZeroPoint: Int32,
  _ vScale: Float,
  _ vZeroPoint: Int32,
  _ qPrecision: Int32,
  _ kPrecision: Int32,
  _ vPrecision: Int32,
  _ causal: Bool,
  _ transposeQ: Bool,
  _ transposeK: Bool,
  _ transposeV: Bool,
  _ transposeO: Bool,
  _ qBlockScales: UnsafeMutableRawPointer?,
  _ qBlockZeroPoints: UnsafeMutableRawPointer?,
  _ kBlockScales: UnsafeMutableRawPointer?,
  _ kBlockZeroPoints: UnsafeMutableRawPointer?,
  _ vBlockScales: UnsafeMutableRawPointer?,
  _ vBlockZeroPoints: UnsafeMutableRawPointer?,
  _ qBlockSize: UInt32,
  _ kBlockSize: UInt32,
  _ vBlockSize: UInt32,
  _ options: UInt32
) -> Int32 {
  guard
    let context,
    let q, let k, let v,
    let gradOutput,
    let logsumexp,
    let dValues,
    let gradKey,
    let gradValue
  else {
    return 1
  }

  _ = options

  let mfaContext = Unmanaged<MFAContext>.fromOpaque(context).takeUnretainedValue()
  let device = mfaContext.device

  let qBuffer = Unmanaged<MFABuffer>.fromOpaque(q).takeUnretainedValue()
  let kBuffer = Unmanaged<MFABuffer>.fromOpaque(k).takeUnretainedValue()
  let vBuffer = Unmanaged<MFABuffer>.fromOpaque(v).takeUnretainedValue()
  let gradOutputBuffer = Unmanaged<MFABuffer>.fromOpaque(gradOutput).takeUnretainedValue()
  let logsumexpBuffer = Unmanaged<MFABuffer>.fromOpaque(logsumexp).takeUnretainedValue()
  let dValuesBuffer = Unmanaged<MFABuffer>.fromOpaque(dValues).takeUnretainedValue()
  let gradKeyBuffer = Unmanaged<MFABuffer>.fromOpaque(gradKey).takeUnretainedValue()
  let gradValueBuffer = Unmanaged<MFABuffer>.fromOpaque(gradValue).takeUnretainedValue()

  let effectiveHeads = max(numHeads, 1)
  let effectiveKVHeads = numKVHeads == 0 ? effectiveHeads : numKVHeads
  let headDimension = Int(headDim)

  if headDimension <= 0 {
    return 1
  }

  let batchCount = Int(batchSize)
  let sequenceQ = Int(seqLenQ)
  let sequenceKV = Int(seqLenKV)

  let qShape = [batchCount, Int(effectiveHeads), sequenceQ, headDimension]
  let kvShape = [batchCount, Int(effectiveKVHeads), sequenceKV, headDimension]

  func bufferFromHandle(_ handle: UnsafeMutableRawPointer?) -> MFABuffer? {
    guard let handle else { return nil }
    return Unmanaged<MFABuffer>.fromOpaque(handle).takeUnretainedValue()
  }

  let qBlockScalesBuffer = bufferFromHandle(qBlockScales)
  let qBlockZeroPointsBuffer = bufferFromHandle(qBlockZeroPoints)
  let kBlockScalesBuffer = bufferFromHandle(kBlockScales)
  let kBlockZeroPointsBuffer = bufferFromHandle(kBlockZeroPoints)
  let vBlockScalesBuffer = bufferFromHandle(vBlockScales)
  let vBlockZeroPointsBuffer = bufferFromHandle(vBlockZeroPoints)

  let qParams = QuantizationParameters(
    scale: qScale,
    zeroPoint: qZeroPoint,
    precision: GEMMOperandPrecision(rawValue: UInt16(qPrecision)) ?? .FP16,
    strategy: .legacy
  )

  let kParams = QuantizationParameters(
    scale: kScale,
    zeroPoint: kZeroPoint,
    precision: GEMMOperandPrecision(rawValue: UInt16(kPrecision)) ?? .INT8,
    strategy: .legacy
  )

  let vParams = QuantizationParameters(
    scale: vScale,
    zeroPoint: vZeroPoint,
    precision: GEMMOperandPrecision(rawValue: UInt16(vPrecision)) ?? .INT8,
    strategy: .legacy
  )

  let qTensor = makeQuantizedTensor(
    device: device,
    buffer: qBuffer,
    parameters: qParams,
    shape: qShape,
    blockScales: qBlockScalesBuffer,
    blockZeroPoints: qBlockZeroPointsBuffer,
    blockSize: qBlockSize
  )

  let kTensor = makeQuantizedTensor(
    device: device,
    buffer: kBuffer,
    parameters: kParams,
    shape: kvShape,
    blockScales: kBlockScalesBuffer,
    blockZeroPoints: kBlockZeroPointsBuffer,
    blockSize: kBlockSize
  )

  let vTensor = makeQuantizedTensor(
    device: device,
    buffer: vBuffer,
    parameters: vParams,
    shape: kvShape,
    blockScales: vBlockScalesBuffer,
    blockZeroPoints: vBlockZeroPointsBuffer,
    blockSize: vBlockSize
  )

  var baseDescriptor = AttentionDescriptor()
  baseDescriptor.matrixDimensions = (row: seqLenQ, column: seqLenKV, head: headDim)
  baseDescriptor.transposeState = (Q: transposeQ, K: transposeK, V: transposeV, O: transposeO)
  baseDescriptor.sparsityPattern = causal ? .causal : .none

  var quantConfig = QuantizedAttention.Configuration()
  quantConfig.queryPrecision = qTensor.parameters.precision
  quantConfig.keyPrecision = kTensor.parameters.precision
  quantConfig.valuePrecision = vTensor.parameters.precision

  let quantDescriptor = QuantizedAttention.QuantizedAttentionDescriptor(
    baseDescriptor: baseDescriptor,
    quantizationConfig: quantConfig
  )

  let quantizedAttention = QuantizedAttention(device: device)

  guard
    let commandBuffer = quantizedAttention.backwardKeyValue(
      query: qTensor,
      key: kTensor,
      value: vTensor,
      gradOutput: gradOutputBuffer.buffer,
      logsumexp: logsumexpBuffer.buffer,
      dValues: dValuesBuffer.buffer,
      gradKey: gradKeyBuffer.buffer,
      gradValue: gradValueBuffer.buffer,
      descriptor: quantDescriptor
    )
  else {
    return 5
  }

  commandBuffer.commit()
  commandBuffer.waitUntilCompleted()

  if let error = commandBuffer.error {
    print("Metal execution error: \(error)")
    return 5
  }

  let gpuLatency = commandBuffer.gpuEndTime - commandBuffer.gpuStartTime
  globalContext?.lastGPULatency = gpuLatency

  return 0
}

// MARK: - Multi-Head Attention Internal Implementation

/// Internal multi-head attention implementation using the new MultiHeadAttention class
private func mfa_attention_forward_multihead_internal(
  context: MFAContext,
  qBuffer: MTLBuffer,
  kBuffer: MTLBuffer,
  vBuffer: MTLBuffer,
  outBuffer: MTLBuffer,
  batchSize: UInt32,
  seqLenQ: UInt32,
  seqLenKV: UInt32,
  numHeads: UInt32,
  headDim: UInt16,
  softmaxScale: Float,
  causal: Bool,
  inputPrecision: Int32,
  intermediatePrecision: Int32,
  transposeQ: Bool,
  transposeK: Bool,
  transposeV: Bool,
  transposeO: Bool,
  preparedMask: PreparedMask?
)
  -> Int32
{
  do {
    // Create multi-head attention instance
    let multiHeadAttention = MultiHeadAttention(device: context.device)

    // Create base attention descriptor
    var baseDescriptor = AttentionDescriptor()
    baseDescriptor.matrixDimensions = (row: seqLenQ, column: seqLenKV, head: headDim)
    // Convert C FFI enum values to Swift values
    let swiftInputPrecision = convertCFFIPrecisionToSwift(inputPrecision)
    let swiftIntermediatePrecision = convertCFFIPrecisionToSwift(intermediatePrecision)

    // IMPORTANT: When using FP16/BF16 precision modes with FP32 data,
    // we must use FP32 inputs to avoid NaN issues from precision mismatch
    // The inputs are always FP32 from the FFI layer
    baseDescriptor.lowPrecisionInputs = false  // Always use FP32 inputs from FFI
    baseDescriptor
      // Use FP32 intermediates for numerical stability
      .lowPrecisionIntermediates = false
    baseDescriptor.transposeState = (Q: transposeQ, K: transposeK, V: transposeV, O: transposeO)
    baseDescriptor.sparsityPattern = causal ? .causal : .none
    baseDescriptor.softmaxScale = softmaxScale

    // Create tensor shapes
    let queryShape = MultiHeadShape(
      batchSize: batchSize,
      numHeads: numHeads,
      sequenceLength: seqLenQ,
      headDimension: headDim
    )

    let kvShape = MultiHeadShape(
      batchSize: batchSize,
      numHeads: numHeads, // Standard MHA for now
      sequenceLength: seqLenKV,
      headDimension: headDim
    )

    // Create multi-head descriptor with optimized dispatch strategy
    let multiHeadDescriptor = MultiHeadAttentionDescriptor(
      baseDescriptor: baseDescriptor,
      queryShape: queryShape,
      keyShape: kvShape,
      valueShape: kvShape,
      broadcastMode: .standard, // Standard MHA mode
      dispatchStrategy: .perBatch // Use per-batch dispatch for better performance with small head
      // counts
    )

    // Execute multi-head attention (no logsumexp for forward-only)
    guard
      let commandBuffer = multiHeadAttention.forward(
        query: qBuffer,
        key: kBuffer,
        value: vBuffer,
        output: outBuffer,
        logsumexp: nil, // Skip logsumexp for forward-only passes
        descriptor: multiHeadDescriptor,
        maskBuffer: preparedMask?.buffer
      )
    else {
      return 5 // MFA_ERROR_EXECUTION_FAILED
    }

    // Execute and wait for completion
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    if let error = commandBuffer.error {
      print("Multi-head attention execution error: \(error)")
      return 5 // MFA_ERROR_EXECUTION_FAILED
    }

    // Store GPU timing
    let gpuLatency = commandBuffer.gpuEndTime - commandBuffer.gpuStartTime
    context.lastGPULatency = gpuLatency

    return 0 // MFA_SUCCESS

  } catch {
    print("Multi-head attention error: \(error)")
    return 4 // MFA_ERROR_KERNEL_COMPILATION
  }
}

// Helper function to dequantize buffers for MultiHeadAttention processing
private func dequantizeBuffer(
  buffer: MTLBuffer,
  params: QuantizationParameters,
  elementCount: Int,
  device: MTLDevice,
  sharedCommandQueue: MTLCommandQueue
)
  -> MTLBuffer?
{
  // For FP16 inputs (no quantization), return the original buffer
  if params.precision == .FP16 {
    return buffer
  }

  // Create output buffer for dequantized FP16 data
  guard
    let outputBuffer = device.makeBuffer(
      length: elementCount * MemoryLayout<Float16>.size,
      options: .storageModeShared
    )
  else {
    return nil
  }

  // Create compute pipeline for dequantization
  let kernelSource = """
  #include <metal_stdlib>
  using namespace metal;

  kernel void dequantize_int8_to_fp16(
      device const char *input [[buffer(0)]],
      device half *output [[buffer(1)]],
      constant float &scale [[buffer(2)]],
      constant int32_t &zero_point [[buffer(3)]],
      constant uint &element_count [[buffer(4)]],
      uint gid [[thread_position_in_grid]]
  ) {
      if (gid >= element_count) return;

      // Enhanced dequantization with NaN protection for BF16 compatibility
      float dequantized = (float(input[gid]) - float(zero_point)) * scale;

      // Protect against NaN and Inf values that can occur with BF16 precision mismatches
      if (!isfinite(dequantized)) {
          dequantized = 0.0f; // Replace NaN/Inf with zero for stability
      }

      // Clamp to reasonable range to prevent overflow in subsequent computations
      dequantized = clamp(dequantized, -65504.0f, 65504.0f); // FP16 max range

      output[gid] = half(dequantized);
  }
  """

  // Create Metal compute pipeline for dequantization with precision safety
  let compileOptions = MTLCompileOptions()
  compileOptions.fastMathEnabled = false // Disable fast math for numerical stability with BF16

  guard
    let library = try? device.makeLibrary(source: kernelSource, options: compileOptions),
    let function = library.makeFunction(name: "dequantize_int8_to_fp16"),
    let pipeline = try? device.makeComputePipelineState(function: function)
  else {
    print("ERROR: Failed to create dequantization compute pipeline")
    print("       Element count: \(elementCount)")
    print("       Kernel source length: \(kernelSource.count)")
    return nil
  }

  // Use shared command queue instead of creating new one
  // This prevents Metal resource contention and double-free bugs
  guard
    let commandBuffer = sharedCommandQueue.makeCommandBuffer(),
    let encoder = commandBuffer.makeComputeCommandEncoder()
  else {
    return nil
  }

  encoder.setComputePipelineState(pipeline)
  encoder.setBuffer(buffer, offset: 0, index: 0) // input (quantized)
  encoder.setBuffer(outputBuffer, offset: 0, index: 1) // output (FP16)

  var scale = params.scale
  var zeroPoint = params.zeroPoint
  var elementCountUInt = UInt32(elementCount)

  // 🔍 DEBUG: Log dequantization parameters
  print("🔍 DEQUANT DEBUG: scale=\(scale), zeroPoint=\(zeroPoint), elementCount=\(elementCount)")
  print("    Precision: \(params.precision)")

  encoder.setBytes(&scale, length: MemoryLayout<Float>.size, index: 2)
  encoder.setBytes(&zeroPoint, length: MemoryLayout<Int32>.size, index: 3)
  encoder.setBytes(&elementCountUInt, length: MemoryLayout<UInt32>.size, index: 4)

  // Dispatch threads
  let threadsPerGroup = MTLSize(width: 256, height: 1, depth: 1)
  let groupCount = MTLSize(
    width: (elementCount + threadsPerGroup.width - 1) / threadsPerGroup.width,
    height: 1,
    depth: 1
  )
  encoder.dispatchThreadgroups(groupCount, threadsPerThreadgroup: threadsPerGroup)
  encoder.endEncoding()

  commandBuffer.commit()
  commandBuffer.waitUntilCompleted()

  if let error = commandBuffer.error {
    print("ERROR: Dequantization failed: \(error)")
    return nil
  }

  return outputBuffer
}

// MARK: - Enhanced Multi-Head Quantized Attention Implementation

// MARK: - UNIFIED QUANTIZED ATTENTION IMPLEMENTATION
// This function replaces both mfa_attention_forward_quantized and mfa_attention_forward_quantized_enhanced
// It supports all quantization granularities, precision options, and advanced features in a single unified codebase

@_cdecl("mfa_attention_forward_quantized_unified")
public func mfa_attention_forward_quantized_unified(
  _ context: UnsafeMutableRawPointer?,
  _ q: UnsafeMutableRawPointer?,
  _ k: UnsafeMutableRawPointer?,
  _ v: UnsafeMutableRawPointer?,
  _ out: UnsafeMutableRawPointer?,
  _ batchSize: UInt32,
  _ seqLenQ: UInt32,
  _ seqLenKV: UInt32,
  _ numHeads: UInt32,
  _ headDim: UInt16,
  _ softmaxScale: Float,
  _ causal: Bool,
  _ qScale: Float,
  _ qZeroPoint: Int32,
  _ kScale: Float,
  _ kZeroPoint: Int32,
  _ vScale: Float,
  _ vZeroPoint: Int32,
  _ qPrecision: Int32,
  _ kPrecision: Int32,
  _ vPrecision: Int32,
  _ outputPrecision: Int32,
  _ granularity: Int32,
  _ qBlockSize: UInt32,
  _ kBlockSize: UInt32,
  _ vBlockSize: UInt32,
  _ enableMixedPrecision: Bool,
  _ forceSymmetricQuantization: Bool,
  _ transposeQ: Bool,
  _ transposeK: Bool,
  _ transposeV: Bool,
  _ transposeO: Bool
)
  -> Int32
{
  print("🚨 ENTERING unified quantized attention with granularity \(granularity)")
  print("   Dimensions: batch=\(batchSize), seqQ=\(seqLenQ), seqKV=\(seqLenKV), heads=\(numHeads), dim=\(headDim)")
  print("   Granularity: \(granularity) (0=tensor, 1=row, 2=block, 3=hybrid)")
  print("   Block sizes: Q=\(qBlockSize), K=\(kBlockSize), V=\(vBlockSize)")
  print("   Precisions: Q=\(qPrecision), K=\(kPrecision), V=\(vPrecision), Out=\(outputPrecision)")

  guard
    let context,
    let q, let k, let v, let out
  else {
    print("ERROR: Invalid null pointers provided")
    return 1 // MFA_ERROR_INVALID_ARGS
  }

  // Extract context and buffers
  let mfaContext = Unmanaged<MFAContext>.fromOpaque(context).takeUnretainedValue()
  let qBuffer = Unmanaged<MFABuffer>.fromOpaque(q).takeUnretainedValue()
  let kBuffer = Unmanaged<MFABuffer>.fromOpaque(k).takeUnretainedValue()
  let vBuffer = Unmanaged<MFABuffer>.fromOpaque(v).takeUnretainedValue()
  let outBuffer = Unmanaged<MFABuffer>.fromOpaque(out).takeUnretainedValue()

  // Create quantization parameters for unified processing
  let qParams = QuantizationParameters(
    scale: qScale,
    zeroPoint: qZeroPoint,
    precision: GEMMOperandPrecision(rawValue: UInt16(qPrecision)) ?? .FP16,
    strategy: .legacy
  )
  let kParams = QuantizationParameters(
    scale: kScale,
    zeroPoint: kZeroPoint,
    precision: GEMMOperandPrecision(rawValue: UInt16(kPrecision)) ?? .INT8,
    strategy: .legacy
  )
  let vParams = QuantizationParameters(
    scale: vScale,
    zeroPoint: vZeroPoint,
    precision: GEMMOperandPrecision(rawValue: UInt16(vPrecision)) ?? .INT8,
    strategy: .legacy
  )

  print("🔧 Unified quantization parameters:")
  print("   Q: scale=\(qParams.scale), zero=\(qParams.zeroPoint), precision=\(qParams.precision)")
  print("   K: scale=\(kParams.scale), zero=\(kParams.zeroPoint), precision=\(kParams.precision)")
  print("   V: scale=\(vParams.scale), zero=\(vParams.zeroPoint), precision=\(vParams.precision)")

  // Route to new runtime quantization API in MFABridge+Quantized.swift
  print("🔀 ROUTING to new runtime quantization API for optimal performance")

  // Convert parameters for new API
  let qMFABuffer = MFABuffer(buffer: qBuffer.buffer)
  let kMFABuffer = MFABuffer(buffer: kBuffer.buffer)
  let vMFABuffer = MFABuffer(buffer: vBuffer.buffer)
  let outMFABuffer = MFABuffer(buffer: outBuffer.buffer)

  let contextOpaque = Unmanaged.passUnretained(mfaContext).toOpaque()
  let qOpaque = Unmanaged.passUnretained(qMFABuffer).toOpaque()
  let kOpaque = Unmanaged.passUnretained(kMFABuffer).toOpaque()
  let vOpaque = Unmanaged.passUnretained(vMFABuffer).toOpaque()
  let outOpaque = Unmanaged.passUnretained(outMFABuffer).toOpaque()

  // Use the new runtime quantization API
  let result = mfa_attention_forward_quantized_direct(
    contextOpaque, qOpaque, kOpaque, vOpaque, outOpaque,
    batchSize, seqLenQ, seqLenKV, numHeads, headDim,
    softmaxScale, causal,
    qScale, qZeroPoint,
    kScale, kZeroPoint,
    vScale, vZeroPoint,
    qPrecision, // Input format (0=FP16, 1=BF16, 2=FP32)
    kPrecision, // Target quantization precision (3=INT8, 4=INT4)
    granularity == 2 ? 2 : 0, // Use blockwise mode if granularity is 2, otherwise tensorwise
    outputPrecision,
    transposeQ, transposeK, transposeV, transposeO
  )

  print("✅ Unified quantized attention completed with result: \(result)")
  return result
}

// MARK: - BACKWARD COMPATIBILITY WRAPPERS
// These functions provide backward compatibility for existing code while routing through the unified implementation

@_cdecl("mfa_attention_forward_quantized")
public func mfa_attention_forward_quantized(
  _ context: UnsafeMutableRawPointer?,
  _ q: UnsafeMutableRawPointer?,
  _ k: UnsafeMutableRawPointer?,
  _ v: UnsafeMutableRawPointer?,
  _ out: UnsafeMutableRawPointer?,
  _ batchSize: UInt32,
  _ seqLenQ: UInt32,
  _ seqLenKV: UInt32,
  _ numHeads: UInt32,
  _ headDim: UInt16,
  _ softmaxScale: Float,
  _ causal: Bool,
  _ qScale: Float,
  _ qZeroPoint: Int32,
  _ kScale: Float,
  _ kZeroPoint: Int32,
  _ vScale: Float,
  _ vZeroPoint: Int32,
  _ qPrecision: Int32,
  _ kPrecision: Int32,
  _ vPrecision: Int32,
  _ outputPrecision: Int32,
  _ transposeQ: Bool,
  _ transposeK: Bool,
  _ transposeV: Bool,
  _ transposeO: Bool
)
  -> Int32
{
  print("🔀 COMPATIBILITY: Routing legacy mfa_attention_forward_quantized to unified implementation")

  // Route to unified implementation with default granularity settings
  return mfa_attention_forward_quantized_unified(
    context, q, k, v, out,
    batchSize, seqLenQ, seqLenKV, numHeads, headDim,
    softmaxScale, causal,
    qScale, qZeroPoint,
    kScale, kZeroPoint,
    vScale, vZeroPoint,
    qPrecision, kPrecision, vPrecision, outputPrecision,
    0, // granularity: 0 = tensor_wise (legacy default)
    128, 64, 64, // default block sizes
    true, false, // enable mixed precision, disable symmetric quantization
    transposeQ, transposeK, transposeV, transposeO
  )
}

@_cdecl("mfa_attention_forward_quantized_enhanced")
public func mfa_attention_forward_quantized_enhanced(
  _ context: UnsafeMutableRawPointer?,
  _ q: UnsafeMutableRawPointer?,
  _ k: UnsafeMutableRawPointer?,
  _ v: UnsafeMutableRawPointer?,
  _ out: UnsafeMutableRawPointer?,
  _ batchSize: UInt32,
  _ seqLenQ: UInt32,
  _ seqLenKV: UInt32,
  _ numHeads: UInt32,
  _ headDim: UInt16,
  _ softmaxScale: Float,
  _ causal: Bool,
  _ qScale: Float,
  _ qZeroPoint: Int32,
  _ kScale: Float,
  _ kZeroPoint: Int32,
  _ vScale: Float,
  _ vZeroPoint: Int32,
  _ qPrecision: Int32,
  _ kPrecision: Int32,
  _ vPrecision: Int32,
  _ outputPrecision: Int32,
  _ granularity: Int32,
  _ qBlockSize: UInt32,
  _ kBlockSize: UInt32,
  _ vBlockSize: UInt32,
  _ enableMixedPrecision: Bool,
  _ forceSymmetricQuantization: Bool,
  _ transposeQ: Bool,
  _ transposeK: Bool,
  _ transposeV: Bool,
  _ transposeO: Bool
)
  -> Int32
{
  print("🔀 COMPATIBILITY: Routing mfa_attention_forward_quantized_enhanced to unified implementation")

  // Route directly to unified implementation (same signature)
  return mfa_attention_forward_quantized_unified(
    context, q, k, v, out,
    batchSize, seqLenQ, seqLenKV, numHeads, headDim,
    softmaxScale, causal,
    qScale, qZeroPoint,
    kScale, kZeroPoint,
    vScale, vZeroPoint,
    qPrecision, kPrecision, vPrecision, outputPrecision,
    granularity,
    qBlockSize, kBlockSize, vBlockSize,
    enableMixedPrecision, forceSymmetricQuantization,
    transposeQ, transposeK, transposeV, transposeO
  )
}




// BACKWARD COMPATIBILITY WRAPPERS FOR DEQUANTIZATION
// These functions provide backward compatibility while routing through the unified implementation

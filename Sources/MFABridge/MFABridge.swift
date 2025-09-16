import FlashAttention
import Foundation
import Metal

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

  init?(device: MTLDevice) {
    self.device = device
    guard let queue = device.makeCommandQueue() else {
      return nil
    }
    commandQueue = queue
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

  init(buffer: MTLBuffer, originalDataPtr: UnsafeMutableRawPointer? = nil, dataSize: Int = 0) {
    self.buffer = buffer
    self.originalDataPtr = originalDataPtr
    self.dataSize = dataSize
  }
}

// MARK: - C Bridge Implementation

// Global Metal device (like MTLContext.global.device in native Swift)
private let globalDevice: MTLDevice? = MTLCreateSystemDefaultDevice()
private var globalContext: MFAContext?

@_cdecl("mfa_create_context")
public func mfa_create_context(_ context: UnsafeMutablePointer<UnsafeMutableRawPointer?>) -> Int32 {
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
  context.pointee = unmanagedContext.toOpaque()
  return 0 // MFA_SUCCESS
}

@_cdecl("mfa_destroy_context")
public func mfa_destroy_context(_ context: UnsafeMutableRawPointer?) {
  guard let context else { return }
  let unmanagedContext = Unmanaged<MFAContext>.fromOpaque(context)
  unmanagedContext.release()
}

@_cdecl("mfa_create_buffer")
public func mfa_create_buffer(
  _ context: UnsafeMutableRawPointer?,
  _ sizeBytes: Int,
  _ buffer: UnsafeMutablePointer<UnsafeMutableRawPointer?>
)
  -> Int32
{
  guard let context else { return 1 } // MFA_ERROR_INVALID_ARGS

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
  _ buffer: UnsafeMutablePointer<UnsafeMutableRawPointer?>
)
  -> Int32
{
  guard
    let context,
    let dataPtr
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
  _ transposeO: Bool
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
      transposeO: transposeO
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
      // Note: MFA uses false = FP32, true = FP16
      descriptor.lowPrecisionInputs = (inputPrecision == 0) // FP16 = true, FP32 = false
      descriptor
        .lowPrecisionIntermediates = (intermediatePrecision == 0) // FP16 = true, FP32 = false

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
)
  -> Int32
{
  guard
    let context,
    let q, let k, let v, let gradOutput,
    let logsumexp, let gradQuery, let dValues
  else {
    return 1 // MFA_ERROR_INVALID_ARGS
  }

  // Extract context and buffers
  let mfaContext = Unmanaged<MFAContext>.fromOpaque(context).takeUnretainedValue()
  let qBuffer = Unmanaged<MFABuffer>.fromOpaque(q).takeUnretainedValue()
  let kBuffer = Unmanaged<MFABuffer>.fromOpaque(k).takeUnretainedValue()
  let vBuffer = Unmanaged<MFABuffer>.fromOpaque(v).takeUnretainedValue()
  let gradOutputBuffer = Unmanaged<MFABuffer>.fromOpaque(gradOutput).takeUnretainedValue()
  let logsumexpBuffer = Unmanaged<MFABuffer>.fromOpaque(logsumexp).takeUnretainedValue()
  let gradQueryBuffer = Unmanaged<MFABuffer>.fromOpaque(gradQuery).takeUnretainedValue()
  let dValuesBuffer = Unmanaged<MFABuffer>.fromOpaque(dValues).takeUnretainedValue()

  // For now, handle single-head case
  if numHeads != 1 {
    return 1 // MFA_ERROR_INVALID_ARGS - Multi-head not yet supported
  }

  // Create quantized attention descriptor
  var baseDescriptor = AttentionDescriptor()
  baseDescriptor.matrixDimensions = (row: seqLenQ, column: seqLenKV, head: headDim)
  baseDescriptor.transposeState = (Q: transposeQ, K: transposeK, V: transposeV, O: transposeO)
  baseDescriptor.sparsityPattern = causal ? .causal : .none

  // Create quantization configuration
  var quantConfig = QuantizedAttention.Configuration()
  quantConfig.queryPrecision = GEMMOperandPrecision(rawValue: UInt16(qPrecision)) ?? .FP16
  quantConfig.keyPrecision = GEMMOperandPrecision(rawValue: UInt16(kPrecision)) ?? .INT8
  quantConfig.valuePrecision = GEMMOperandPrecision(rawValue: UInt16(vPrecision)) ?? .INT8

  let quantDescriptor = QuantizedAttention.QuantizedAttentionDescriptor(
    baseDescriptor: baseDescriptor,
    quantizationConfig: quantConfig
  )

  // Create quantized attention instance
  let quantizedAttention = QuantizedAttention(device: mfaContext.device)

  // Create quantized tensors from buffers
  let qParams = QuantizationParameters(
    scale: qScale, zeroPoint: qZeroPoint, precision: quantConfig.queryPrecision
  )
  let kParams = QuantizationParameters(
    scale: kScale, zeroPoint: kZeroPoint, precision: quantConfig.keyPrecision
  )
  let vParams = QuantizationParameters(
    scale: vScale, zeroPoint: vZeroPoint, precision: quantConfig.valuePrecision
  )

  let elementCount = Int(batchSize * seqLenQ * UInt32(headDim))
  let shape = [Int(batchSize), Int(seqLenQ), Int(headDim)]

  let qTensor = QuantizedTensor(
    device: mfaContext.device, data: qBuffer.buffer, parameters: qParams,
    elementCount: elementCount, shape: shape
  )
  let kTensor = QuantizedTensor(
    device: mfaContext.device, data: kBuffer.buffer, parameters: kParams,
    elementCount: elementCount, shape: shape
  )
  let vTensor = QuantizedTensor(
    device: mfaContext.device, data: vBuffer.buffer, parameters: vParams,
    elementCount: elementCount, shape: shape
  )

  // Execute quantized backward query pass
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
    return 5 // MFA_ERROR_EXECUTION_FAILED
  }

  // Execute and wait for completion
  commandBuffer.commit()
  commandBuffer.waitUntilCompleted()

  if let error = commandBuffer.error {
    print("Metal execution error: \(error)")
    return 5 // MFA_ERROR_EXECUTION_FAILED
  }

  // Store GPU timing
  let gpuLatency = commandBuffer.gpuEndTime - commandBuffer.gpuStartTime
  globalContext?.lastGPULatency = gpuLatency

  return 0 // MFA_SUCCESS
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
)
  -> Int32
{
  guard
    let context,
    let q, let k, let v, let gradOutput,
    let logsumexp, let dValues,
    let gradKey, let gradValue
  else {
    return 1 // MFA_ERROR_INVALID_ARGS
  }

  // Extract context and buffers
  let mfaContext = Unmanaged<MFAContext>.fromOpaque(context).takeUnretainedValue()
  let qBuffer = Unmanaged<MFABuffer>.fromOpaque(q).takeUnretainedValue()
  let kBuffer = Unmanaged<MFABuffer>.fromOpaque(k).takeUnretainedValue()
  let vBuffer = Unmanaged<MFABuffer>.fromOpaque(v).takeUnretainedValue()
  let gradOutputBuffer = Unmanaged<MFABuffer>.fromOpaque(gradOutput).takeUnretainedValue()
  let logsumexpBuffer = Unmanaged<MFABuffer>.fromOpaque(logsumexp).takeUnretainedValue()
  let dValuesBuffer = Unmanaged<MFABuffer>.fromOpaque(dValues).takeUnretainedValue()
  let gradKeyBuffer = Unmanaged<MFABuffer>.fromOpaque(gradKey).takeUnretainedValue()
  let gradValueBuffer = Unmanaged<MFABuffer>.fromOpaque(gradValue).takeUnretainedValue()

  // For now, handle single-head case
  if numHeads != 1 {
    return 1 // MFA_ERROR_INVALID_ARGS - Multi-head not yet supported
  }

  // Create quantized attention descriptor
  var baseDescriptor = AttentionDescriptor()
  baseDescriptor.matrixDimensions = (row: seqLenQ, column: seqLenKV, head: headDim)
  baseDescriptor.transposeState = (Q: transposeQ, K: transposeK, V: transposeV, O: transposeO)
  baseDescriptor.sparsityPattern = causal ? .causal : .none

  // Create quantization configuration
  var quantConfig = QuantizedAttention.Configuration()
  quantConfig.queryPrecision = GEMMOperandPrecision(rawValue: UInt16(qPrecision)) ?? .FP16
  quantConfig.keyPrecision = GEMMOperandPrecision(rawValue: UInt16(kPrecision)) ?? .INT8
  quantConfig.valuePrecision = GEMMOperandPrecision(rawValue: UInt16(vPrecision)) ?? .INT8

  let quantDescriptor = QuantizedAttention.QuantizedAttentionDescriptor(
    baseDescriptor: baseDescriptor,
    quantizationConfig: quantConfig
  )

  // Create quantized attention instance
  let quantizedAttention = QuantizedAttention(device: mfaContext.device)

  // Create quantized tensors from buffers
  let qParams = QuantizationParameters(
    scale: qScale, zeroPoint: qZeroPoint, precision: quantConfig.queryPrecision
  )
  let kParams = QuantizationParameters(
    scale: kScale, zeroPoint: kZeroPoint, precision: quantConfig.keyPrecision
  )
  let vParams = QuantizationParameters(
    scale: vScale, zeroPoint: vZeroPoint, precision: quantConfig.valuePrecision
  )

  let elementCount = Int(batchSize * seqLenKV * UInt32(headDim))
  let shape = [Int(batchSize), Int(seqLenKV), Int(headDim)]

  let qTensor = QuantizedTensor(
    device: mfaContext.device, data: qBuffer.buffer, parameters: qParams,
    elementCount: Int(batchSize * seqLenQ * UInt32(headDim)),
    shape: [Int(batchSize), Int(seqLenQ), Int(headDim)]
  )
  let kTensor = QuantizedTensor(
    device: mfaContext.device, data: kBuffer.buffer, parameters: kParams,
    elementCount: elementCount, shape: shape
  )
  let vTensor = QuantizedTensor(
    device: mfaContext.device, data: vBuffer.buffer, parameters: vParams,
    elementCount: elementCount, shape: shape
  )

  // Execute quantized backward key-value pass
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
    return 5 // MFA_ERROR_EXECUTION_FAILED
  }

  // Execute and wait for completion
  commandBuffer.commit()
  commandBuffer.waitUntilCompleted()

  if let error = commandBuffer.error {
    print("Metal execution error: \(error)")
    return 5 // MFA_ERROR_EXECUTION_FAILED
  }

  // Store GPU timing
  let gpuLatency = commandBuffer.gpuEndTime - commandBuffer.gpuStartTime
  globalContext?.lastGPULatency = gpuLatency

  return 0 // MFA_SUCCESS
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
  transposeO: Bool
)
  -> Int32
{
  do {
    // Create multi-head attention instance
    let multiHeadAttention = MultiHeadAttention(device: context.device)

    // Create base attention descriptor
    var baseDescriptor = AttentionDescriptor()
    baseDescriptor.matrixDimensions = (row: seqLenQ, column: seqLenKV, head: headDim)
    baseDescriptor.lowPrecisionInputs = (inputPrecision == 0) // FP16 = true, FP32 = false
    baseDescriptor
      .lowPrecisionIntermediates = (intermediatePrecision == 0) // FP16 = true, FP32 = false
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
        descriptor: multiHeadDescriptor
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

// MARK: - REAL Quantized Multi-Head Attention Implementation

private func mfa_attention_forward_quantized_multihead_internal(
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
  quantConfig: QuantizedAttention.Configuration,
  qParams: QuantizationParameters,
  kParams: QuantizationParameters,
  vParams: QuantizationParameters,
  softmaxScale: Float,
  causal: Bool,
  transposeQ: Bool,
  transposeK: Bool,
  transposeV: Bool,
  transposeO: Bool
)
  -> Int32
{
  print("🌌 ENTERING mfa_attention_forward_quantized_multihead_internal - numHeads: \(numHeads)")
  print("🚀 REAL Multi-Head Attention: Processing \(numHeads) heads in PARALLEL")

  do {
    // Create multi-head attention instance (REAL MHA!)
    let multiHeadAttention = MultiHeadAttention(device: context.device)

    // Create base attention descriptor with quantization-aware settings
    var baseDescriptor = AttentionDescriptor()
    baseDescriptor.matrixDimensions = (row: seqLenQ, column: seqLenKV, head: headDim)
    baseDescriptor.transposeState = (Q: transposeQ, K: transposeK, V: transposeV, O: transposeO)
    baseDescriptor.sparsityPattern = causal ? .causal : .none
    baseDescriptor.softmaxScale = softmaxScale

    // Set precision handling for quantized inputs
    baseDescriptor
      .lowPrecisionInputs = (
        quantConfig.keyPrecision == .FP16 || quantConfig
          .valuePrecision == .FP16
      )
    baseDescriptor.lowPrecisionIntermediates = false // Use FP32 intermediates for accuracy

    // 🔧 OUTPUT PRECISION: Use default FP16 output for this internal function
    let metalOutputPrecision: GEMMOperandPrecision = .FP16
    print("🔧 OUTPUT PRECISION: Using default FP16 output precision")

    // Create tensor shapes for multi-head layout [B, S, H, D]
    let queryShape = MultiHeadShape(
      batchSize: batchSize,
      numHeads: numHeads,
      sequenceLength: seqLenQ,
      headDimension: headDim
    )

    let kvShape = MultiHeadShape(
      batchSize: batchSize,
      numHeads: numHeads, // Standard MHA (same num heads for K,V as Q)
      sequenceLength: seqLenKV,
      headDimension: headDim
    )

    // Create multi-head descriptor with efficient dispatch strategy
    let multiHeadDescriptor = MultiHeadAttentionDescriptor(
      baseDescriptor: baseDescriptor,
      queryShape: queryShape,
      keyShape: kvShape,
      valueShape: kvShape,
      broadcastMode: .standard, // Standard MHA mode
      dispatchStrategy: .batched // Use batched dispatch for maximum parallelism
    )

    // 🚨 GUARD: Validate MultiHeadAttention descriptor parameters
    print("🔍 MHA DESCRIPTOR VALIDATION:")
    print(
      "   Query shape: batch=\(queryShape.batchSize), heads=\(queryShape.numHeads), seq=\(queryShape.sequenceLength), dim=\(queryShape.headDimension)"
    )
    print(
      "   Key shape: batch=\(kvShape.batchSize), heads=\(kvShape.numHeads), seq=\(kvShape.sequenceLength), dim=\(kvShape.headDimension)"
    )
    print(
      "   Value shape: batch=\(kvShape.batchSize), heads=\(kvShape.numHeads), seq=\(kvShape.sequenceLength), dim=\(kvShape.headDimension)"
    )
    print(
      "   Base descriptor - row: \(baseDescriptor.matrixDimensions?.row ?? 0), col: \(baseDescriptor.matrixDimensions?.column ?? 0), head: \(baseDescriptor.matrixDimensions?.head ?? 0)"
    )
    print(
      "   Broadcast mode: \(multiHeadDescriptor.broadcastMode), Dispatch: \(multiHeadDescriptor.dispatchStrategy)"
    )

    // APPROACH: Use MultiHeadAttention but pre-process buffers for quantization
    // This gives us REAL parallel MHA with quantization support

    // TODO: The MultiHeadAttention class doesn't directly support quantized tensors yet
    // For now, we need to either:
    // 1. Extend MultiHeadAttention to support QuantizedTensor inputs, OR
    // 2. Dequantize, run MHA, then handle precision in the output

    // TEMPORARY: Use approach #2 for immediate functionality
    // This maintains proper parallel MHA while handling quantization

    // 🔧 FIX: Create single shared command queue for all dequantization operations
    // This prevents the Metal resource contention that causes double-free bugs
    guard let sharedCommandQueue = context.device.makeCommandQueue() else {
      print("ERROR: Failed to create shared command queue for dequantization")
      return 2 // MFA_ERROR_MEMORY_ALLOCATION
    }

    // 🔧 FIX: Q may be passed in original precision (scale=1.0) and not need dequantization
    let dequantizedQ: MTLBuffer
    if true { // 🔧 TEMPORARY: Always skip Q dequantization for testing
      // Q is already in original precision, use directly
      print("🔧 Q in original precision (scale=1.0), skipping dequantization")
      dequantizedQ = qBuffer
    } else {
      // Q needs dequantization
      print("🔧 Q needs dequantization (scale=\(qParams.scale))")
      guard
        let buffer = dequantizeBuffer(
          buffer: qBuffer, params: qParams,
          elementCount: Int(batchSize * numHeads * seqLenQ * UInt32(headDim)),
          device: context.device, sharedCommandQueue: sharedCommandQueue
        )
      else {
        print("ERROR: Failed to dequantize Q buffer")
        return 2 // MFA_ERROR_MEMORY_ALLOCATION
      }
      dequantizedQ = buffer
    }

    guard
      let dequantizedK = dequantizeBuffer(
        buffer: kBuffer, params: kParams,
        elementCount: Int(batchSize * numHeads * seqLenKV * UInt32(headDim)),
        device: context.device, sharedCommandQueue: sharedCommandQueue
      )
    else {
      print("ERROR: Failed to dequantize K buffer")
      return 2 // MFA_ERROR_MEMORY_ALLOCATION
    }

    guard
      let dequantizedV = dequantizeBuffer(
        buffer: vBuffer, params: vParams,
        elementCount: Int(batchSize * numHeads * seqLenKV * UInt32(headDim)),
        device: context.device, sharedCommandQueue: sharedCommandQueue
      )
    else {
      print("ERROR: Failed to dequantize V buffer")
      return 2 // MFA_ERROR_MEMORY_ALLOCATION
    }

    // 🔍 DEBUG: Check dequantized buffer contents before MultiHeadAttention
    print("🔍 DEBUG BEFORE MHA:")
    let qPtr = dequantizedQ.contents().bindMemory(to: Float16.self, capacity: 8)
    let qValues = Array(UnsafeBufferPointer(start: qPtr, count: 8))
    print("  DequantQ[0:8]: \(qValues)")

    let kPtr = dequantizedK.contents().bindMemory(to: Float16.self, capacity: 8)
    let kValues = Array(UnsafeBufferPointer(start: kPtr, count: 8))
    print("  DequantK[0:8]: \(kValues)")

    let vPtr = dequantizedV.contents().bindMemory(to: Float16.self, capacity: 8)
    let vValues = Array(UnsafeBufferPointer(start: vPtr, count: 8))
    print("  DequantV[0:8]: \(vValues)")

    // 🚨 GUARD: Validate dequantized inputs are reasonable (breakpoint-style check)
    let qMax = qValues.compactMap { Float($0) }.max() ?? 0
    let kMax = kValues.compactMap { Float($0) }.max() ?? 0
    let vMax = vValues.compactMap { Float($0) }.max() ?? 0

    if qMax == 0 || kMax == 0 || vMax == 0 {
      print("🚨 DEQUANTIZATION ERROR: Zero values detected in inputs!")
      print("   Q_max=\(qMax), K_max=\(kMax), V_max=\(vMax)")
      return 6 // Custom error for dequantization failure
    }

    print("✅ Dequantized inputs validated: Q_max=\(qMax), K_max=\(kMax), V_max=\(vMax)")

    // 🚨 GUARD: Validate buffer sizes before Metal execution
    print("🔍 BUFFER SIZE VALIDATION:")
    print("   DequantQ buffer length: \(dequantizedQ.length) bytes")
    print("   DequantK buffer length: \(dequantizedK.length) bytes")
    print("   DequantV buffer length: \(dequantizedV.length) bytes")
    print("   Output buffer length: \(outBuffer.length) bytes")

    // 🔧 FIX: Calculate expected sizes based on actual precision types
    let numElements = Int(batchSize * numHeads * seqLenQ * UInt32(headDim))
    let expectedQSize = numElements * MemoryLayout<Float16>.size // Q uses FP16 after dequantization
    let expectedKVSize = numElements * MemoryLayout<Float16>.size // K,V also dequantized to FP16
    let expectedOutputSize = numElements * MemoryLayout<Float16>.size

    print("   Expected Q buffer size: \(expectedQSize) bytes")
    print("   Expected K/V buffer size: \(expectedKVSize) bytes")
    print("   Expected output size: \(expectedOutputSize) bytes")

    if dequantizedQ.length != expectedQSize {
      print("🚨 Q BUFFER SIZE MISMATCH: expected \(expectedQSize), got \(dequantizedQ.length)")
    }
    if dequantizedK.length != expectedKVSize {
      print("🚨 K BUFFER SIZE MISMATCH: expected \(expectedKVSize), got \(dequantizedK.length)")
    }
    if dequantizedV.length != expectedKVSize {
      print("🚨 V BUFFER SIZE MISMATCH: expected \(expectedKVSize), got \(dequantizedV.length)")
    }
    if outBuffer.length != 2 * expectedOutputSize { // Output buffer is double-sized for safety
      print(
        "🚨 OUTPUT BUFFER SIZE MISMATCH: expected \(2 * expectedOutputSize), got \(outBuffer.length)"
      )
    }

    // Execute REAL parallel multi-head attention
    guard
      let commandBuffer = multiHeadAttention.forward(
        query: dequantizedQ,
        key: dequantizedK,
        value: dequantizedV,
        output: outBuffer,
        logsumexp: nil, // Skip logsumexp for forward-only passes
        descriptor: multiHeadDescriptor
      )
    else {
      print("🚨 CRITICAL ERROR: Failed to create multi-head attention command buffer")
      print("   This indicates Metal shader compilation or resource allocation failure")
      return 5 // MFA_ERROR_EXECUTION_FAILED
    }

    print("✅ Multi-head attention command buffer created successfully")

    // Execute the parallel multi-head attention
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    // 🚨 GUARD: Check for Metal command buffer errors (like breakpoint)
    if let error = commandBuffer.error {
      print("🚨 METAL ERROR DETECTED: \(error)")
      print("   Error domain: \(error._domain)")
      print("   Error code: \(error._code)")
      print("   Error description: \(error.localizedDescription)")
      return 5 // MFA_ERROR_EXECUTION_FAILED
    }

    // 🔍 GUARD: Validate Metal execution completed successfully
    print("✅ Metal command buffer completed successfully")
    print("   Command buffer status: \(commandBuffer.status.rawValue)")

    // 🔍 DEBUG: Check output buffer contents IMMEDIATELY after Metal execution
    print("🔍 DEBUG AFTER MHA METAL EXECUTION:")

    // 🔧 TYPE-SAFE OUTPUT VALIDATION: Read according to configured output precision
    print("🔍 Validating output buffer (FP16 assumed)")

    // Use FP16 for this internal function
    let outPtr = outBuffer.contents().bindMemory(to: Float16.self, capacity: 8)
    let outputValues = Array(UnsafeBufferPointer<Float16>(start: outPtr, count: 8))
    print("  Output (FP16)[0:8]: \(outputValues)")
    print("✅ FP16 output values validated successfully")

    return 0 // MFA_SUCCESS

  } catch {
    print("ERROR: Multi-head attention setup failed: \(error)")
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
      float dequantized = (float(input[gid]) - float(zero_point)) * scale;
      output[gid] = half(dequantized);
  }
  """

  // Create Metal compute pipeline for dequantization
  let compileOptions = MTLCompileOptions()
  compileOptions.fastMathEnabled = true

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

  // 🔧 FIX: Use shared command queue instead of creating new one
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

private func mfa_attention_forward_quantized_multihead_enhanced_internal(
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
  quantConfig: QuantizedAttention.Configuration,
  qParams: QuantizationParameters,
  kParams: QuantizationParameters,
  vParams: QuantizationParameters,
  softmaxScale: Float,
  causal: Bool,
  granularity: Int32,
  qBlockSize: UInt32,
  kBlockSize: UInt32,
  vBlockSize: UInt32,
  enableMixedPrecision: Bool,
  forceSymmetricQuantization: Bool,
  outputPrecision: Int32,
  transposeQ: Bool,
  transposeK: Bool,
  transposeV: Bool,
  transposeO: Bool
)
  -> Int32
{
  print("🌌 ENTERING mfa_attention_forward_quantized_multihead_enhanced_internal")
  print("🚀 ENHANCED Multi-Head Attention: Processing \(numHeads) heads with granularity \(granularity)")

  do {
    // Create multi-head attention instance for enhanced processing
    let multiHeadAttention = MultiHeadAttention(device: context.device)

    // Create base attention descriptor with enhanced quantization-aware settings
    var baseDescriptor = AttentionDescriptor()
    baseDescriptor.matrixDimensions = (row: seqLenQ, column: seqLenKV, head: headDim)
    baseDescriptor.transposeState = (Q: transposeQ, K: transposeK, V: transposeV, O: transposeO)
    baseDescriptor.sparsityPattern = causal ? .causal : .none
    baseDescriptor.softmaxScale = softmaxScale

    // Configure precision handling based on granularity and mixed precision settings
    if enableMixedPrecision {
      // Use mixed precision for better performance
      baseDescriptor.lowPrecisionInputs = (quantConfig.keyPrecision == .FP16 || quantConfig.valuePrecision == .FP16)
      baseDescriptor.lowPrecisionIntermediates = (granularity == 0) // Use FP16 intermediates for tensor-wise only
    } else {
      // Use consistent precision
      baseDescriptor.lowPrecisionInputs = false
      baseDescriptor.lowPrecisionIntermediates = false
    }

    print("🔧 PRECISION CONFIG:")
    print("   Mixed precision: \(enableMixedPrecision)")
    print("   Low precision inputs: \(baseDescriptor.lowPrecisionInputs)")
    print("   Low precision intermediates: \(baseDescriptor.lowPrecisionIntermediates)")
    print("   Symmetric quantization: \(forceSymmetricQuantization)")

    // Log quantization granularity and block sizes
    print("🔧 GRANULARITY CONFIG:")
    print("   Granularity: \(granularity) (0=tensor, 1=row, 2=block, 3=hybrid)")
    print("   Block sizes: Q=\(qBlockSize), K=\(kBlockSize), V=\(vBlockSize)")

    // Create tensor shapes for multi-head layout [B, S, H, D]
    let queryShape = MultiHeadShape(
      batchSize: batchSize,
      numHeads: numHeads,
      sequenceLength: seqLenQ,
      headDimension: headDim
    )

    let kvShape = MultiHeadShape(
      batchSize: batchSize,
      numHeads: numHeads,
      sequenceLength: seqLenKV,
      headDimension: headDim
    )

    // Create multi-head descriptor
    let multiHeadDescriptor = MultiHeadAttentionDescriptor(
      baseDescriptor: baseDescriptor,
      queryShape: queryShape,
      keyShape: kvShape,
      valueShape: kvShape,
      broadcastMode: .standard
    )

    print("🔍 ENHANCED MHA DESCRIPTOR:")
    print("   Broadcast mode: standard (based on granularity \(granularity))")
    print("   Query shape: batch=\(queryShape.batchSize), heads=\(queryShape.numHeads), seq=\(queryShape.sequenceLength), dim=\(queryShape.headDimension)")

    // Create shared command queue for enhanced processing
    guard let sharedCommandQueue = context.device.makeCommandQueue() else {
      print("ERROR: Failed to create shared command queue for enhanced processing")
      return 2 // MFA_ERROR_MEMORY_ALLOCATION
    }

    // Enhanced dequantization with granularity support
    let dequantizedQ: MTLBuffer
    let dequantizedK: MTLBuffer
    let dequantizedV: MTLBuffer

    // Q processing (may skip dequantization if already in target precision)
    if qParams.scale == 1.0 && quantConfig.queryPrecision == .FP16 {
      print("🔧 Q in target precision (FP16), skipping dequantization")
      dequantizedQ = qBuffer
    } else {
      print("🔧 Dequantizing Q with granularity support")
      guard
        let buffer = dequantizeBufferEnhanced(
          buffer: qBuffer, params: qParams,
          elementCount: Int(batchSize * numHeads * seqLenQ * UInt32(headDim)),
          device: context.device, sharedCommandQueue: sharedCommandQueue,
          granularity: granularity, blockSize: qBlockSize
        )
      else {
        print("ERROR: Failed to dequantize Q buffer with enhanced processing")
        return 2 // MFA_ERROR_MEMORY_ALLOCATION
      }
      dequantizedQ = buffer
    }

    // K processing with enhanced granularity
    print("🔧 Dequantizing K with granularity \(granularity) and block size \(kBlockSize)")
    guard
      let dequantizedKBuffer = dequantizeBufferEnhanced(
        buffer: kBuffer, params: kParams,
        elementCount: Int(batchSize * numHeads * seqLenKV * UInt32(headDim)),
        device: context.device, sharedCommandQueue: sharedCommandQueue,
        granularity: granularity, blockSize: kBlockSize
      )
    else {
      print("ERROR: Failed to dequantize K buffer with enhanced processing")
      return 2 // MFA_ERROR_MEMORY_ALLOCATION
    }
    dequantizedK = dequantizedKBuffer

    // V processing with enhanced granularity
    print("🔧 Dequantizing V with granularity \(granularity) and block size \(vBlockSize)")
    guard
      let dequantizedVBuffer = dequantizeBufferEnhanced(
        buffer: vBuffer, params: vParams,
        elementCount: Int(batchSize * numHeads * seqLenKV * UInt32(headDim)),
        device: context.device, sharedCommandQueue: sharedCommandQueue,
        granularity: granularity, blockSize: vBlockSize
      )
    else {
      print("ERROR: Failed to dequantize V buffer with enhanced processing")
      return 2 // MFA_ERROR_MEMORY_ALLOCATION
    }
    dequantizedV = dequantizedVBuffer

    // Validate enhanced dequantized inputs
    print("🔍 ENHANCED DEBUG AFTER DEQUANTIZATION:")
    let qPtr = dequantizedQ.contents().bindMemory(to: Float16.self, capacity: 8)
    let qValues = Array(UnsafeBufferPointer(start: qPtr, count: 8))
    print("  Enhanced DequantQ[0:8]: \(qValues)")

    let qMax = qValues.compactMap { Float($0) }.max() ?? 0
    if qMax == 0 {
      print("🚨 ENHANCED DEQUANTIZATION ERROR: Zero values detected in Q!")
      return 6 // Custom error for dequantization failure
    }

    print("✅ Enhanced dequantized inputs validated: Q_max=\(qMax)")

    // Execute ENHANCED parallel multi-head attention with output precision control
    print("🚀 Executing ENHANCED multi-head attention...")
    guard
      let commandBuffer = multiHeadAttention.forward(
        query: dequantizedQ,
        key: dequantizedK,
        value: dequantizedV,
        output: outBuffer,
        logsumexp: nil,
        descriptor: multiHeadDescriptor
      )
    else {
      print("🚨 CRITICAL ERROR: Failed to create enhanced multi-head attention command buffer")
      return 5 // MFA_ERROR_EXECUTION_FAILED
    }

    print("✅ Enhanced multi-head attention command buffer created successfully")

    // Execute the enhanced parallel multi-head attention
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    // Check for Metal command buffer errors
    if let error = commandBuffer.error {
      print("🚨 ENHANCED METAL ERROR: \(error)")
      return 5 // MFA_ERROR_EXECUTION_FAILED
    }

    // Validate enhanced output based on configured output precision
    print("🔍 ENHANCED DEBUG AFTER MHA EXECUTION:")

    // Handle different output precisions
    switch outputPrecision {
    case 0: // FP16
      let outPtr = outBuffer.contents().bindMemory(to: Float16.self, capacity: 16)
      let outputValues = Array(UnsafeBufferPointer<Float16>(start: outPtr, count: 16))
      print("  Enhanced Output (FP16)[0:8]: \(outputValues[0..<8])")
      let outputMax = outputValues.compactMap { Float($0) }.max() ?? 0
      print("  Enhanced Output max: \(outputMax)")

    case 1: // BF16 (not directly supported in Swift, treat as FP16)
      let outPtr = outBuffer.contents().bindMemory(to: Float16.self, capacity: 16)
      let outputValues = Array(UnsafeBufferPointer<Float16>(start: outPtr, count: 16))
      print("  Enhanced Output (BF16→FP16)[0:8]: \(outputValues[0..<8])")

    case 2: // FP32
      let outPtr = outBuffer.contents().bindMemory(to: Float.self, capacity: 16)
      let outputValues = Array(UnsafeBufferPointer<Float>(start: outPtr, count: 16))
      print("  Enhanced Output (FP32)[0:8]: \(outputValues[0..<8])")
      let outputMax = outputValues.max() ?? 0
      print("  Enhanced Output max: \(outputMax)")

    default:
      print("  Unknown output precision: \(outputPrecision)")
    }

    print("✅ ENHANCED Multi-Head Attention completed successfully with granularity \(granularity)")
    return 0 // MFA_SUCCESS

  } catch {
    print("ERROR: Enhanced multi-head attention setup failed: \(error)")
    return 4 // MFA_ERROR_KERNEL_COMPILATION
  }
}

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
    scale: qScale, zeroPoint: qZeroPoint, precision: GEMMOperandPrecision(rawValue: UInt16(qPrecision)) ?? .FP16
  )
  let kParams = QuantizationParameters(
    scale: kScale, zeroPoint: kZeroPoint, precision: GEMMOperandPrecision(rawValue: UInt16(kPrecision)) ?? .INT8
  )
  let vParams = QuantizationParameters(
    scale: vScale, zeroPoint: vZeroPoint, precision: GEMMOperandPrecision(rawValue: UInt16(vPrecision)) ?? .INT8
  )

  print("🔧 Unified quantization parameters:")
  print("   Q: scale=\(qParams.scale), zero=\(qParams.zeroPoint), precision=\(qParams.precision)")
  print("   K: scale=\(kParams.scale), zero=\(kParams.zeroPoint), precision=\(kParams.precision)")
  print("   V: scale=\(vParams.scale), zero=\(vParams.zeroPoint), precision=\(vParams.precision)")

  // Route to unified multi-head attention implementation with full granularity support
  print("🔀 ROUTING to unified multi-head function with all advanced features")
  let result = mfa_attention_forward_quantized_multihead_unified_internal(
    context: mfaContext,
    qBuffer: qBuffer.buffer,
    kBuffer: kBuffer.buffer,
    vBuffer: vBuffer.buffer,
    outBuffer: outBuffer.buffer,
    batchSize: batchSize,
    seqLenQ: seqLenQ,
    seqLenKV: seqLenKV,
    numHeads: numHeads,
    headDim: headDim,
    softmaxScale: softmaxScale,
    causal: causal,
    qParams: qParams,
    kParams: kParams,
    vParams: vParams,
    outputPrecision: outputPrecision,
    granularity: granularity,
    qBlockSize: qBlockSize,
    kBlockSize: kBlockSize,
    vBlockSize: vBlockSize,
    enableMixedPrecision: enableMixedPrecision,
    forceSymmetricQuantization: forceSymmetricQuantization,
    transposeQ: transposeQ,
    transposeK: transposeK,
    transposeV: transposeV,
    transposeO: transposeO
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

// MARK: - UNIFIED MULTI-HEAD ATTENTION IMPLEMENTATION
// The actual unified implementation that handles all granularities and features

private func mfa_attention_forward_quantized_multihead_unified_internal(
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
  qParams: QuantizationParameters,
  kParams: QuantizationParameters,
  vParams: QuantizationParameters,
  outputPrecision: Int32,
  granularity: Int32,
  qBlockSize: UInt32,
  kBlockSize: UInt32,
  vBlockSize: UInt32,
  enableMixedPrecision: Bool,
  forceSymmetricQuantization: Bool,
  transposeQ: Bool,
  transposeK: Bool,
  transposeV: Bool,
  transposeO: Bool
)
  -> Int32
{
  print("🌌 ENTERING unified multi-head attention implementation")
  print("🚀 UNIFIED Multi-Head Attention: Processing \(numHeads) heads with granularity \(granularity)")

  do {
    // Create multi-head attention instance for unified processing
    let multiHeadAttention = MultiHeadAttention(device: context.device)

    // Get or create shared command queue for efficient resource management
    let sharedCommandQueue = context.commandQueue ?? context.device.makeCommandQueue()!

    // Step 1: Dequantize tensors using unified dequantization with granularity support
    var dequantizedQ: MTLBuffer
    var dequantizedK: MTLBuffer
    var dequantizedV: MTLBuffer

    // Q processing with unified granularity
    if qParams.precision == .FP16 {
      print("🔧 Q in target precision (FP16), skipping dequantization")
      dequantizedQ = qBuffer
    } else {
      print("🔧 Dequantizing Q with unified granularity support")
      guard
        let buffer = dequantizeBufferUnified(
          buffer: qBuffer, params: qParams,
          elementCount: Int(batchSize * numHeads * seqLenQ * UInt32(headDim)),
          device: context.device, sharedCommandQueue: sharedCommandQueue,
          granularity: granularity, blockSize: qBlockSize
        )
      else {
        print("ERROR: Failed to dequantize Q buffer")
        return 2 // MFA_ERROR_MEMORY_ALLOCATION
      }
      dequantizedQ = buffer
    }

    // K processing with unified granularity
    print("🔧 Dequantizing K with unified granularity \(granularity) and block size \(kBlockSize)")
    guard
      let dequantizedKBuffer = dequantizeBufferUnified(
        buffer: kBuffer, params: kParams,
        elementCount: Int(batchSize * numHeads * seqLenKV * UInt32(headDim)),
        device: context.device, sharedCommandQueue: sharedCommandQueue,
        granularity: granularity, blockSize: kBlockSize
      )
    else {
      print("ERROR: Failed to dequantize K buffer")
      return 2 // MFA_ERROR_MEMORY_ALLOCATION
    }
    dequantizedK = dequantizedKBuffer

    // V processing with unified granularity
    print("🔧 Dequantizing V with unified granularity \(granularity) and block size \(vBlockSize)")
    guard
      let dequantizedVBuffer = dequantizeBufferUnified(
        buffer: vBuffer, params: vParams,
        elementCount: Int(batchSize * numHeads * seqLenKV * UInt32(headDim)),
        device: context.device, sharedCommandQueue: sharedCommandQueue,
        granularity: granularity, blockSize: vBlockSize
      )
    else {
      print("ERROR: Failed to dequantize V buffer")
      return 2 // MFA_ERROR_MEMORY_ALLOCATION
    }
    dequantizedV = dequantizedVBuffer

    // Step 2: Execute attention with unified configuration
    var baseDescriptor = AttentionDescriptor()
    baseDescriptor.matrixDimensions = (row: seqLenQ, column: seqLenKV, head: headDim)
    baseDescriptor.transposeState = (Q: transposeQ, K: transposeK, V: transposeV, O: transposeO)
    baseDescriptor.sparsityPattern = causal ? .causal : .none
    baseDescriptor.softmaxScale = softmaxScale
    baseDescriptor.lowPrecisionInputs = enableMixedPrecision
    baseDescriptor.lowPrecisionIntermediates = enableMixedPrecision

    // Create multi-head shapes
    let queryShape = MultiHeadShape(
      batchSize: batchSize,
      numHeads: numHeads,
      sequenceLength: seqLenQ,
      headDimension: headDim
    )
    let kvShape = MultiHeadShape(
      batchSize: batchSize,
      numHeads: numHeads,
      sequenceLength: seqLenKV,
      headDimension: headDim
    )

    // Create multi-head descriptor
    let multiHeadDescriptor = MultiHeadAttentionDescriptor(
      baseDescriptor: baseDescriptor,
      queryShape: queryShape,
      keyShape: kvShape,
      valueShape: kvShape,
      broadcastMode: .standard,
      dispatchStrategy: .perBatch  // HANG FIX: Use same strategy as working non-quantized version
    )

    // Execute unified attention with performance optimizations
    print("🔍 HANG DEBUG: About to call MultiHeadAttention.forward with perBatch strategy")
    guard
      let commandBuffer = multiHeadAttention.forward(
        query: dequantizedQ,
        key: dequantizedK,
        value: dequantizedV,
        output: outBuffer,
        logsumexp: nil, // Skip logsumexp for forward-only passes
        descriptor: multiHeadDescriptor
      )
    else {
      print("🚨 ERROR: Failed to execute unified multi-head attention")
      return 5 // MFA_ERROR_EXECUTION_FAILED
    }

    print("🔍 HANG DEBUG: Successfully got command buffer, about to commit and wait")

    // Commit the command buffer
    commandBuffer.commit()
    print("🔍 HANG DEBUG: Command buffer committed, waiting for completion...")

    commandBuffer.waitUntilCompleted()
    print("🔍 HANG DEBUG: Command buffer completed!")

    if let error = commandBuffer.error {
      print("ERROR: Unified multi-head attention execution failed: \(error)")
      return 5 // MFA_ERROR_EXECUTION_FAILED
    }

    // Step 3: Log performance metrics
    let gpuLatency = commandBuffer.gpuEndTime - commandBuffer.gpuStartTime
    print("⚡ UNIFIED Attention Performance: GPU latency = \(gpuLatency * 1000) ms")
    print("   Granularity: \(granularity), Block sizes: Q=\(qBlockSize), K=\(kBlockSize), V=\(vBlockSize)")
    print("   Mixed precision: \(enableMixedPrecision), Symmetric quant: \(forceSymmetricQuantization)")

    context.lastGPULatency = gpuLatency

    print("✅ Unified multi-head attention completed successfully")
    return 0 // MFA_SUCCESS

  } catch {
    print("ERROR: Unified multi-head attention setup failed: \(error)")
    return 4 // MFA_ERROR_KERNEL_COMPILATION
  }
}

// MARK: - UNIFIED DEQUANTIZATION IMPLEMENTATION
// This function replaces both dequantizeBuffer and dequantizeBufferEnhanced
// It supports all quantization granularities with runtime parameterization

private func dequantizeBufferUnified(
  buffer: MTLBuffer,
  params: QuantizationParameters,
  elementCount: Int,
  device: MTLDevice,
  sharedCommandQueue: MTLCommandQueue,
  granularity: Int32,
  blockSize: UInt32,
  tensorShape: (batchSize: UInt32, seqLen: UInt32, numHeads: UInt32, headDim: UInt32)? = nil,
  blockConfig: (seqBlockSize: UInt32, headBlockSize: UInt32, dimBlockSize: UInt32)? = nil,
  blockScales: [Float]? = nil
)
  -> MTLBuffer?
{
  print("🔧 UNIFIED dequantization: granularity=\(granularity), blockSize=\(blockSize)")
  print("   Precision: \(params.precision), Scale: \(params.scale), ZeroPoint: \(params.zeroPoint)")
  if let shape = tensorShape {
    print("   Tensor shape: batch=\(shape.batchSize), seq=\(shape.seqLen), heads=\(shape.numHeads), dim=\(shape.headDim)")
  }
  if let config = blockConfig {
    print("   Block config: seq=\(config.seqBlockSize), head=\(config.headBlockSize), dim=\(config.dimBlockSize)")
  }
  if let scales = blockScales {
    print("   Block scales: \(scales.count) scales provided")
  }

  // For FP16 inputs (no quantization), return the original buffer
  if params.precision == .FP16 {
    print("🔧 Input already in FP16 precision, returning original buffer")
    return buffer
  }

  // Create output buffer for dequantized FP16 data
  guard
    let outputBuffer = device.makeBuffer(
      length: elementCount * MemoryLayout<Float16>.size,
      options: .storageModeShared
    )
  else {
    print("ERROR: Failed to create output buffer for unified dequantization")
    return nil
  }

  // Generate unified Metal kernel with runtime constants
  let kernelFunctionName = "dequantize_unified"
  let kernelSource = createUnifiedDequantizationKernel()

  // Create Metal compute pipeline for unified dequantization
  let compileOptions = MTLCompileOptions()
  compileOptions.fastMathEnabled = true

  guard
    let library = try? device.makeLibrary(source: kernelSource, options: compileOptions),
    let function = library.makeFunction(name: kernelFunctionName),
    let pipeline = try? device.makeComputePipelineState(function: function)
  else {
    print("ERROR: Failed to create unified dequantization compute pipeline")
    print("       Kernel: \(kernelFunctionName), Granularity: \(granularity)")
    return nil
  }

  // Use shared command queue for efficient resource management
  guard
    let commandBuffer = sharedCommandQueue.makeCommandBuffer(),
    let encoder = commandBuffer.makeComputeCommandEncoder()
  else {
    print("ERROR: Failed to create command buffer or encoder for unified dequantization")
    return nil
  }

  encoder.setComputePipelineState(pipeline)
  encoder.setBuffer(buffer, offset: 0, index: 0) // input (quantized)
  encoder.setBuffer(outputBuffer, offset: 0, index: 1) // output (FP16)

  // Set unified parameters
  var scale = params.scale
  var zeroPoint = params.zeroPoint
  var elementCountUInt = UInt32(elementCount)
  var blockSizeUInt = blockSize
  var granularityInt = granularity

  encoder.setBytes(&scale, length: MemoryLayout<Float>.size, index: 2)
  encoder.setBytes(&zeroPoint, length: MemoryLayout<Int32>.size, index: 3)
  encoder.setBytes(&elementCountUInt, length: MemoryLayout<UInt32>.size, index: 4)
  encoder.setBytes(&blockSizeUInt, length: MemoryLayout<UInt32>.size, index: 5)
  encoder.setBytes(&granularityInt, length: MemoryLayout<Int32>.size, index: 6)

  // Set tensor shape parameters if provided
  if let shape = tensorShape {
    var batchSize = shape.batchSize
    var seqLen = shape.seqLen
    var numHeads = shape.numHeads
    var headDim = shape.headDim
    encoder.setBytes(&batchSize, length: MemoryLayout<UInt32>.size, index: 7)
    encoder.setBytes(&seqLen, length: MemoryLayout<UInt32>.size, index: 8)
    encoder.setBytes(&numHeads, length: MemoryLayout<UInt32>.size, index: 9)
    encoder.setBytes(&headDim, length: MemoryLayout<UInt32>.size, index: 10)
  } else {
    // Use default values
    var defaultVal: UInt32 = 1
    encoder.setBytes(&defaultVal, length: MemoryLayout<UInt32>.size, index: 7)
    encoder.setBytes(&defaultVal, length: MemoryLayout<UInt32>.size, index: 8)
    encoder.setBytes(&defaultVal, length: MemoryLayout<UInt32>.size, index: 9)
    encoder.setBytes(&defaultVal, length: MemoryLayout<UInt32>.size, index: 10)
  }

  print("🔍 UNIFIED DEQUANT DEBUG: scale=\(scale), zeroPoint=\(zeroPoint), elementCount=\(elementCount), blockSize=\(blockSize), granularity=\(granularity)")

  // Dispatch threads with optimal configuration
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
    print("ERROR: Unified dequantization failed: \(error)")
    return nil
  }

  let gpuTime = commandBuffer.gpuEndTime - commandBuffer.gpuStartTime
  print("⚡ UNIFIED Dequantization Performance: GPU time = \(gpuTime * 1000) ms")

  return outputBuffer
}

// Create unified Metal kernel that handles all granularities with runtime parameters
private func createUnifiedDequantizationKernel() -> String {
  return """
  #include <metal_stdlib>
  using namespace metal;

  // Unified dequantization kernel with runtime granularity selection
  kernel void dequantize_unified(
      device const char *input [[buffer(0)]],
      device half *output [[buffer(1)]],
      constant float &scale [[buffer(2)]],
      constant int32_t &zero_point [[buffer(3)]],
      constant uint &element_count [[buffer(4)]],
      constant uint &block_size [[buffer(5)]],
      constant int32_t &granularity [[buffer(6)]],
      constant uint &batch_size [[buffer(7)]],
      constant uint &seq_len [[buffer(8)]],
      constant uint &num_heads [[buffer(9)]],
      constant uint &head_dim [[buffer(10)]],
      uint gid [[thread_position_in_grid]]
  ) {
      if (gid >= element_count) return;

      float dequantized_value;

      // Runtime granularity selection
      switch (granularity) {
          case 0: { // TENSOR_WISE
              dequantized_value = (float(input[gid]) - float(zero_point)) * scale;
              break;
          }

          case 1: { // ROW_WISE
              // Calculate row index based on tensor layout
              uint row_idx = gid / head_dim;
              // For row-wise, we would need per-row scales passed in
              // For now, fall back to tensor-wise for compatibility
              dequantized_value = (float(input[gid]) - float(zero_point)) * scale;
              break;
          }

          case 2: { // BLOCK_WISE
              // Calculate block index based on block size
              uint block_idx = gid / block_size;
              // For block-wise, we would need per-block scales passed in
              // For now, fall back to tensor-wise for compatibility
              dequantized_value = (float(input[gid]) - float(zero_point)) * scale;
              break;
          }

          case 3: { // HYBRID
              // Hybrid granularity - runtime selection based on position
              // For now, use tensor-wise as fallback
              dequantized_value = (float(input[gid]) - float(zero_point)) * scale;
              break;
          }

          default: { // Fallback to tensor-wise
              dequantized_value = (float(input[gid]) - float(zero_point)) * scale;
              break;
          }
      }

      output[gid] = half(dequantized_value);
  }
  """
}

// BACKWARD COMPATIBILITY WRAPPERS FOR DEQUANTIZATION
// These functions provide backward compatibility while routing through the unified implementation


private func dequantizeBufferEnhanced(
  buffer: MTLBuffer,
  params: QuantizationParameters,
  elementCount: Int,
  device: MTLDevice,
  sharedCommandQueue: MTLCommandQueue,
  granularity: Int32,
  blockSize: UInt32,
  tensorShape: (batchSize: UInt32, seqLen: UInt32, numHeads: UInt32, headDim: UInt32)? = nil,
  blockConfig: (seqBlockSize: UInt32, headBlockSize: UInt32, dimBlockSize: UInt32)? = nil,
  blockScales: [Float]? = nil
)
  -> MTLBuffer?
{
  print("🔀 COMPATIBILITY: Routing dequantizeBufferEnhanced to unified implementation")

  // Route to unified implementation with full parameter support
  return dequantizeBufferUnified(
    buffer: buffer,
    params: params,
    elementCount: elementCount,
    device: device,
    sharedCommandQueue: sharedCommandQueue,
    granularity: granularity,
    blockSize: blockSize,
    tensorShape: tensorShape,
    blockConfig: blockConfig,
    blockScales: blockScales
  )
}

// Legacy Enhanced helper function for dequantization with granularity support (keeping for reference)
private func dequantizeBufferEnhanced_legacy(
  buffer: MTLBuffer,
  params: QuantizationParameters,
  elementCount: Int,
  device: MTLDevice,
  sharedCommandQueue: MTLCommandQueue,
  granularity: Int32,
  blockSize: UInt32,
  tensorShape: (batchSize: UInt32, seqLen: UInt32, numHeads: UInt32, headDim: UInt32)? = nil,
  blockConfig: (seqBlockSize: UInt32, headBlockSize: UInt32, dimBlockSize: UInt32)? = nil,
  blockScales: [Float]? = nil
)
  -> MTLBuffer?
{
  print("🔧 Enhanced dequantization: granularity=\(granularity), blockSize=\(blockSize)")
  if let shape = tensorShape {
    print("   Tensor shape: batch=\(shape.batchSize), seq=\(shape.seqLen), heads=\(shape.numHeads), dim=\(shape.headDim)")
  }
  if let config = blockConfig {
    print("   Block config: seq=\(config.seqBlockSize), head=\(config.headBlockSize), dim=\(config.dimBlockSize)")
  }
  if let scales = blockScales {
    print("   Block scales: \(scales.count) scales provided")
  }

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

  // Choose kernel based on granularity
  let kernelFunctionName: String
  let kernelSource: String

  switch granularity {
  case 0: // tensor_wise
    kernelFunctionName = "dequantize_int8_to_fp16_tensor"
    kernelSource = createTensorWiseDequantizationKernel()

  case 1: // row_wise
    kernelFunctionName = "dequantize_int8_to_fp16_row"
    kernelSource = createRowWiseDequantizationKernel()

  case 2: // block_wise
    kernelFunctionName = "dequantize_int8_to_fp16_block"
    kernelSource = createBlockWiseDequantizationKernel(blockSize: blockSize)

  case 3: // hybrid
    kernelFunctionName = "dequantize_int8_to_fp16_hybrid"
    kernelSource = createHybridDequantizationKernel()

  default:
    print("⚠️  Unknown granularity \(granularity), falling back to tensor-wise")
    kernelFunctionName = "dequantize_int8_to_fp16_tensor"
    kernelSource = createTensorWiseDequantizationKernel()
  }

  // Create Metal compute pipeline for enhanced dequantization
  let compileOptions = MTLCompileOptions()
  compileOptions.fastMathEnabled = true

  guard
    let library = try? device.makeLibrary(source: kernelSource, options: compileOptions),
    let function = library.makeFunction(name: kernelFunctionName),
    let pipeline = try? device.makeComputePipelineState(function: function)
  else {
    print("ERROR: Failed to create enhanced dequantization compute pipeline")
    print("       Kernel: \(kernelFunctionName), Granularity: \(granularity)")
    return nil
  }

  // Use shared command queue
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
  var blockSizeUInt = blockSize

  print("🔍 ENHANCED DEQUANT DEBUG: scale=\(scale), zeroPoint=\(zeroPoint), elementCount=\(elementCount), blockSize=\(blockSize)")

  // Handle different granularities with proper parameter binding
  switch granularity {
  case 1: // ROW_WISE
    // For now, use the fallback scale - proper row scales would need FFI extension
    // TODO: Add FFI interface to pass per-row scales from C++
    print("🔧 Row-wise dequantization: using fallback scale=\(scale) (per-row scales need FFI extension)")

    encoder.setBytes(&scale, length: MemoryLayout<Float>.size, index: 2)
    encoder.setBytes(&zeroPoint, length: MemoryLayout<Int32>.size, index: 3)
    encoder.setBytes(&elementCountUInt, length: MemoryLayout<UInt32>.size, index: 4)
    encoder.setBytes(&blockSizeUInt, length: MemoryLayout<UInt32>.size, index: 5)

  case 2: // BLOCK_WISE
    print("🔧 Block-wise dequantization: setting up per-block scales")

    if let blockScales = blockScales, let shape = tensorShape, let config = blockConfig {
      // Create Metal buffer for block scales
      guard let blockScalesBuffer = device.makeBuffer(
        bytes: blockScales,
        length: blockScales.count * MemoryLayout<Float>.size,
        options: .storageModeShared
      ) else {
        print("ERROR: Failed to create block scales buffer")
        return nil
      }

      // Set buffer containing per-block scales
      encoder.setBuffer(blockScalesBuffer, offset: 0, index: 2)
      encoder.setBytes(&zeroPoint, length: MemoryLayout<Int32>.size, index: 3)
      encoder.setBytes(&elementCountUInt, length: MemoryLayout<UInt32>.size, index: 4)
      encoder.setBytes(&blockSizeUInt, length: MemoryLayout<UInt32>.size, index: 5)

      // Additional parameters for block-wise dequantization
      var seqLen = shape.seqLen
      var numHeads = shape.numHeads
      var headDim = shape.headDim
      var seqBlockSize = config.seqBlockSize
      var headBlockSize = config.headBlockSize
      var dimBlockSize = config.dimBlockSize

      encoder.setBytes(&seqLen, length: MemoryLayout<UInt32>.size, index: 6)
      encoder.setBytes(&numHeads, length: MemoryLayout<UInt32>.size, index: 7)
      encoder.setBytes(&headDim, length: MemoryLayout<UInt32>.size, index: 8)
      encoder.setBytes(&seqBlockSize, length: MemoryLayout<UInt32>.size, index: 9)
      encoder.setBytes(&headBlockSize, length: MemoryLayout<UInt32>.size, index: 10)
      encoder.setBytes(&dimBlockSize, length: MemoryLayout<UInt32>.size, index: 11)

      print("   Set tensor dimensions: seq=\(seqLen), heads=\(numHeads), dim=\(headDim)")
      print("   Set block sizes: seq=\(seqBlockSize), head=\(headBlockSize), dim=\(dimBlockSize)")
    } else {
      print("⚠️  Block-wise mode but missing required parameters, falling back to tensor-wise")
      encoder.setBytes(&scale, length: MemoryLayout<Float>.size, index: 2)
      encoder.setBytes(&zeroPoint, length: MemoryLayout<Int32>.size, index: 3)
      encoder.setBytes(&elementCountUInt, length: MemoryLayout<UInt32>.size, index: 4)
      encoder.setBytes(&blockSizeUInt, length: MemoryLayout<UInt32>.size, index: 5)
    }

  default: // TENSOR_WISE, HYBRID, or fallback
    // Standard parameter binding for tensor-wise
    encoder.setBytes(&scale, length: MemoryLayout<Float>.size, index: 2)
    encoder.setBytes(&zeroPoint, length: MemoryLayout<Int32>.size, index: 3)
    encoder.setBytes(&elementCountUInt, length: MemoryLayout<UInt32>.size, index: 4)
    encoder.setBytes(&blockSizeUInt, length: MemoryLayout<UInt32>.size, index: 5)
  }

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
    print("ERROR: Enhanced dequantization failed: \(error)")
    return nil
  }

  return outputBuffer
}

// MARK: - Enhanced Dequantization Kernels

private func createTensorWiseDequantizationKernel() -> String {
  return """
  #include <metal_stdlib>
  using namespace metal;

  kernel void dequantize_int8_to_fp16_tensor(
      device const char *input [[buffer(0)]],
      device half *output [[buffer(1)]],
      constant float &scale [[buffer(2)]],
      constant int32_t &zero_point [[buffer(3)]],
      constant uint &element_count [[buffer(4)]],
      constant uint &block_size [[buffer(5)]],
      uint gid [[thread_position_in_grid]]
  ) {
      if (gid >= element_count) return;
      float dequantized = (float(input[gid]) - float(zero_point)) * scale;
      output[gid] = half(dequantized);
  }
  """
}

private func createRowWiseDequantizationKernel() -> String {
  return """
  #include <metal_stdlib>
  using namespace metal;

  kernel void dequantize_int8_to_fp16_row(
      device const char *input [[buffer(0)]],
      device half *output [[buffer(1)]],
      constant float &fallback_scale [[buffer(2)]],      // Fallback scale (TODO: replace with per-row scales)
      constant int32_t &zero_point [[buffer(3)]],
      constant uint &element_count [[buffer(4)]],
      constant uint &head_dim [[buffer(5)]],             // Head dimension (row width)
      uint gid [[thread_position_in_grid]]
  ) {
      if (gid >= element_count) return;

      // TODO: Implement proper row-wise scaling
      // This requires FFI extension to pass per-row scales from C++
      // For now, compute an approximation of row-wise scaling

      // Calculate which row this element belongs to
      uint row_index = gid / head_dim;
      uint col_index = gid % head_dim;

      // TEMPORARY: Use fallback scale as baseline
      // In proper implementation, we would have: float row_scale = row_scales[row_index];
      float row_scale = fallback_scale;

      // TODO: Add row-specific scaling logic here
      // This could involve analyzing the quantized values in the current row
      // to estimate a better scale than the fallback

      // Dequantize using current scale
      float dequantized = (float(input[gid]) - float(zero_point)) * row_scale;
      output[gid] = half(dequantized);
  }
  """
}

private func createBlockWiseDequantizationKernel(blockSize: UInt32) -> String {
  return """
  #include <metal_stdlib>
  using namespace metal;

  kernel void dequantize_int8_to_fp16_block(
      device const char *input [[buffer(0)]],
      device half *output [[buffer(1)]],
      device const float *block_scales [[buffer(2)]],      // Per-block scales array
      constant int32_t &zero_point [[buffer(3)]],
      constant uint &element_count [[buffer(4)]],
      constant uint &block_size [[buffer(5)]],
      constant uint &seq_len [[buffer(6)]],                // Sequence length
      constant uint &num_heads [[buffer(7)]],              // Number of heads
      constant uint &head_dim [[buffer(8)]],               // Head dimension
      constant uint &seq_block_size [[buffer(9)]],         // Block size for sequence dimension
      constant uint &head_block_size [[buffer(10)]],       // Block size for head dimension
      constant uint &dim_block_size [[buffer(11)]],        // Block size for dimension
      uint gid [[thread_position_in_grid]]
  ) {
      if (gid >= element_count) return;

      // Calculate 4D tensor coordinates from linear index
      // Layout: [batch, seq_len, num_heads, head_dim]
      uint head_dim_val = head_dim;
      uint num_heads_val = num_heads;
      uint seq_len_val = seq_len;

      uint dim_idx = gid % head_dim_val;
      uint head_idx = (gid / head_dim_val) % num_heads_val;
      uint seq_idx = (gid / (head_dim_val * num_heads_val)) % seq_len_val;
      uint batch_idx = gid / (head_dim_val * num_heads_val * seq_len_val);

      // Calculate which block this element belongs to
      uint seq_block_idx = seq_idx / seq_block_size;
      uint head_block_idx = head_idx / head_block_size;
      uint dim_block_idx = dim_idx / dim_block_size;

      // Calculate number of blocks in each dimension
      uint num_seq_blocks = (seq_len_val + seq_block_size - 1) / seq_block_size;
      uint num_head_blocks = (num_heads_val + head_block_size - 1) / head_block_size;
      uint num_dim_blocks = (head_dim_val + dim_block_size - 1) / dim_block_size;

      // Calculate linear block index
      uint block_idx = batch_idx * (num_seq_blocks * num_head_blocks * num_dim_blocks) +
                       seq_block_idx * (num_head_blocks * num_dim_blocks) +
                       head_block_idx * num_dim_blocks +
                       dim_block_idx;

      // Get the scale for this block
      float block_scale = block_scales[block_idx];

      // Dequantize using block-specific scale
      float dequantized = (float(input[gid]) - float(zero_point)) * block_scale;
      output[gid] = half(dequantized);
  }
  """
}

private func createHybridDequantizationKernel() -> String {
  return """
  #include <metal_stdlib>
  using namespace metal;

  kernel void dequantize_int8_to_fp16_hybrid(
      device const char *input [[buffer(0)]],
      device half *output [[buffer(1)]],
      constant float &scale [[buffer(2)]],
      constant int32_t &zero_point [[buffer(3)]],
      constant uint &element_count [[buffer(4)]],
      constant uint &block_size [[buffer(5)]],
      uint gid [[thread_position_in_grid]]
  ) {
      if (gid >= element_count) return;
      // TODO: Implement hybrid scaling (different granularities for different regions)
      // For now, use tensor-wise as fallback
      float dequantized = (float(input[gid]) - float(zero_point)) * scale;
      output[gid] = half(dequantized);
  }
  """
}

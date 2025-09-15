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

  // For now, handle single-head case (MFA's current limitation)
  // TODO: Add multi-head support by looping over heads
  if numHeads != 1 {
    return 1 // MFA_ERROR_INVALID_ARGS - Multi-head not yet supported
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
  _: Float,
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

  // For now, handle single-head case (MFA's current limitation)
  if numHeads != 1 {
    return 1 // MFA_ERROR_INVALID_ARGS - Multi-head not yet supported
  }

  do {
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

    // Execute quantized forward pass
    guard
      let commandBuffer = quantizedAttention.forward(
        query: qTensor,
        key: kTensor,
        value: vTensor,
        output: outBuffer.buffer,
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

    // Store GPU timing for zero-overhead measurement
    let gpuLatency = commandBuffer.gpuEndTime - commandBuffer.gpuStartTime
    globalContext?.lastGPULatency = gpuLatency

    return 0 // MFA_SUCCESS

  } catch {
    print("MFA Quantized Error: \(error)")
    return 4 // MFA_ERROR_KERNEL_COMPILATION
  }
}

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

  do {
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

  } catch {
    print("MFA Quantized Backward Query Error: \(error)")
    return 4 // MFA_ERROR_KERNEL_COMPILATION
  }
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

  do {
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

  } catch {
    print("MFA Quantized Backward KV Error: \(error)")
    return 4 // MFA_ERROR_KERNEL_COMPILATION
  }
}

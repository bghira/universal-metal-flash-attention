import FlashAttention
import Foundation
import Metal

// MARK: - Direct Quantized Attention (No Dequantization)

// This replaces the overcomplicated dequantization approach with direct INT8/INT4 compute

/// Execute quantized attention using runtime quantization
/// This uses the new forwardWithRuntimeQuantization API that takes FP16/BF16/FP32 inputs
/// and performs quantization internally for optimal performance
@_cdecl("mfa_attention_forward_quantized_direct")
public func mfa_attention_forward_quantized_direct(
  _ context: UnsafeMutableRawPointer?,
  _ q: UnsafeMutableRawPointer?, // FP16/BF16/FP32 buffer (not pre-quantized)
  _ k: UnsafeMutableRawPointer?, // FP16/BF16/FP32 buffer (not pre-quantized)
  _ v: UnsafeMutableRawPointer?, // FP16/BF16/FP32 buffer (not pre-quantized)
  _ out: UnsafeMutableRawPointer?,
  _ batchSize: UInt32,
  _ seqLenQ: UInt32,
  _ seqLenKV: UInt32,
  _ numHeads: UInt32,
  _ headDim: UInt16,
  _ softmaxScale: Float,
  _ causal: Bool,
  _: Float, // Not used in new API
  _: Int32, // Not used in new API
  _: Float, // Not used in new API
  _: Int32, // Not used in new API
  _: Float, // Not used in new API
  _: Int32, // Not used in new API
  _: Int32, // Input precision: 0=FP16, 1=BF16, 2=FP32
  _: Int32, // Target quantization precision: 3=INT8, 4=INT4
  _: Int32, // Quantization mode: 0=tensorWise, 2=blockwise
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
  let qBuffer = Unmanaged<MFABuffer>.fromOpaque(q).takeUnretainedValue().buffer
  let kBuffer = Unmanaged<MFABuffer>.fromOpaque(k).takeUnretainedValue().buffer
  let vBuffer = Unmanaged<MFABuffer>.fromOpaque(v).takeUnretainedValue().buffer
  let outBuffer = Unmanaged<MFABuffer>.fromOpaque(out).takeUnretainedValue().buffer

  // Convert precision values to GEMMOperandPrecision
  func toGEMMPrecision(_ precision: Int32) -> GEMMOperandPrecision {
    switch precision {
    case 0: .FP16
    case 1: .BF16
    case 2: .FP32
    case 3: .INT8
    case 4: .INT4
    default: .FP16 // Default to FP16 for input
    }
  }

  // Convert quantization mode
  func toQuantizationMode(_ mode: Int32) -> QuantizationMode {
    switch mode {
    case 0: .tensorWise
    case 2: .blockwise(blockSizeK: 64) // Use default block size for blockwise quantization
    default: .tensorWise // Default to tensor-wise
    }
  }

  // Note: Quantization parameters are preserved for API compatibility but not used in
  // MultiHeadAttention
  // The MultiHeadAttention infrastructure handles precision internally

  // Validate parameters to prevent underflow
  guard batchSize > 0, numHeads > 0, seqLenQ > 0, seqLenKV > 0, headDim > 0 else {
    print(
      "❌ Invalid parameters: batch=\(batchSize), heads=\(numHeads), seqQ=\(seqLenQ), seqKV=\(seqLenKV), dim=\(headDim)"
    )
    return 2 // MFA_ERROR_INVALID_ARGUMENT
  }

  // Create 4D tensor shape to preserve head dimension for parallel processing
  // This maintains [batch, heads, sequence, headDim] structure instead of flattening heads into
  // batch
  let shape = [Int(batchSize), Int(numHeads), Int(seqLenQ), Int(headDim)]

  // Validate shape doesn't overflow
  guard shape.allSatisfy({ $0 > 0 }) else {
    print("❌ Invalid shape after calculation: \(shape)")
    return 2 // MFA_ERROR_INVALID_ARGUMENT
  }

  // Use cached multi-head attention instance (reuses compiled pipelines)
  let multiHeadAttention = mfaContext.multiHeadAttention

  // Create proper MultiHeadAttentionDescriptor with 4D shape support
  var baseDescriptor = AttentionDescriptor()
  baseDescriptor.matrixDimensions = (
    row: seqLenQ,
    column: seqLenKV,
    head: headDim
  )
  baseDescriptor.transposeState = (Q: transposeQ, K: transposeK, V: transposeV, O: transposeO)
  baseDescriptor.softmaxScale = softmaxScale
  if causal {
    baseDescriptor.sparsityPattern = .causal
  }

  // Create multi-head shapes preserving 4D structure
  let queryShape = MultiHeadShape(
    batchSize: batchSize,
    numHeads: numHeads,
    sequenceLength: seqLenQ,
    headDimension: headDim
  )
  let keyShape = MultiHeadShape(
    batchSize: batchSize,
    numHeads: numHeads,
    sequenceLength: seqLenKV,
    headDimension: headDim
  )
  let valueShape = MultiHeadShape(
    batchSize: batchSize,
    numHeads: numHeads,
    sequenceLength: seqLenKV,
    headDimension: headDim
  )

  let multiHeadDescriptor = MultiHeadAttentionDescriptor(
    baseDescriptor: baseDescriptor,
    queryShape: queryShape,
    keyShape: keyShape,
    valueShape: valueShape,
    broadcastMode: .standard,
    dispatchStrategy: .perBatchHead // Enable parallel head processing
  )

  // Execute multi-head attention with proper 4D tensor handling
  guard
    let commandBuffer = multiHeadAttention.forward(
      query: qBuffer,
      key: kBuffer,
      value: vBuffer,
      output: outBuffer,
      descriptor: multiHeadDescriptor
    )
  else {
    print("❌ Failed to create multi-head attention command buffer")
    return 5 // MFA_ERROR_EXECUTION_FAILED
  }

  // Execute and wait
  commandBuffer.commit()
  commandBuffer.waitUntilCompleted()

  if let error = commandBuffer.error {
    print("❌ Quantized attention execution error: \(error)")
    return 5 // MFA_ERROR_EXECUTION_FAILED
  }

  print(
    "✅ Multi-head quantized attention completed successfully - parallel head processing enabled"
  )
  return 0 // MFA_SUCCESS
}

// MARK: - Simplified Quantized Multi-Head Attention

/// Multi-head quantized attention using parallel head processing
@_cdecl("mfa_multihead_attention_quantized_direct")
public func mfa_multihead_attention_quantized_direct(
  _ context: UnsafeMutableRawPointer?,
  _ q: UnsafeMutableRawPointer?, // FP16/BF16/FP32 buffer (not pre-quantized)
  _ k: UnsafeMutableRawPointer?, // FP16/BF16/FP32 buffer (not pre-quantized)
  _ v: UnsafeMutableRawPointer?, // FP16/BF16/FP32 buffer (not pre-quantized)
  _ out: UnsafeMutableRawPointer?,
  _ batchSize: UInt32,
  _ seqLenQ: UInt32,
  _ seqLenKV: UInt32,
  _ numHeads: UInt32,
  _ headDim: UInt16,
  _ softmaxScale: Float,
  _ causal: Bool,
  _: Float, // Not used in new API
  _: Int32, // Not used in new API
  _: Float, // Not used in new API
  _: Int32, // Not used in new API
  _: Float, // Not used in new API
  _: Int32, // Not used in new API
  _ qPrecision: Int32, // Input precision: 0=FP16, 1=BF16, 2=FP32
  _ kPrecision: Int32, // Target quantization precision: 3=INT8, 4=INT4
  _ vPrecision: Int32 // Quantization mode: 0=tensorWise, 2=blockwise
)
  -> Int32
{
  // Now delegates to the improved multi-head implementation with parallel head processing
  // This ensures proper 4D tensor handling and eliminates the head flattening bottleneck

  mfa_attention_forward_quantized_direct(
    context, q, k, v, out,
    batchSize, seqLenQ, seqLenKV, numHeads, headDim,
    softmaxScale, causal,
    0, 0, // qScale, qZeroPoint - not used
    0, 0, // kScale, kZeroPoint - not used
    0, 0, // vScale, vZeroPoint - not used
    qPrecision, kPrecision, vPrecision,
    2, // outputPrecision = FP32
    false, false, false, false // no transpose
  )
}

// MARK: - Multi-Head Quantized Attention with Autograd Support

/// Forward pass with runtime INT8 quantization + LSE output for autograd.
///
/// Quantizes Q/K/V to INT8 (per-tensor), then dispatches the quantized flash
/// attention kernel per head. Writes both the attention output and the
/// logsumexp buffer needed by the backward pass.
@_cdecl("mfa_quantized_forward_with_lse")
public func mfa_quantized_forward_with_lse(
  _ context: UnsafeMutableRawPointer?,
  _ q: UnsafeMutableRawPointer?,
  _ k: UnsafeMutableRawPointer?,
  _ v: UnsafeMutableRawPointer?,
  _ out: UnsafeMutableRawPointer?,
  _ lse: UnsafeMutableRawPointer?,
  _ mask: UnsafeMutableRawPointer?,
  _ batchSize: UInt32,
  _ seqLenQ: UInt32,
  _ seqLenKV: UInt32,
  _ numHeads: UInt32,
  _ headDim: UInt16,
  _ softmaxScale: Float,
  _ causal: Bool,
  _ qTargetPrecision: Int32,
  _ kvTargetPrecision: Int32,
  _ quantMode: Int32,
  _ inputPrecision: Int32,
  _ externalCommandBuffer: UnsafeMutableRawPointer?
)
  -> Int32
{
  guard
    let context,
    let q, let k, let v, let out, let lse
  else {
    return 1
  }

  let mfaContext = Unmanaged<MFAContext>.fromOpaque(context).takeUnretainedValue()

  let qBuffer = Unmanaged<MFABuffer>.fromOpaque(q).takeUnretainedValue()
  let kBuffer = Unmanaged<MFABuffer>.fromOpaque(k).takeUnretainedValue()
  let vBuffer = Unmanaged<MFABuffer>.fromOpaque(v).takeUnretainedValue()
  let outBuffer = Unmanaged<MFABuffer>.fromOpaque(out).takeUnretainedValue()
  let lseBuffer = Unmanaged<MFABuffer>.fromOpaque(lse).takeUnretainedValue()
  let maskBuffer: MFABuffer? = mask.map {
    Unmanaged<MFABuffer>.fromOpaque($0).takeUnretainedValue()
  }

  let qPrec = GEMMOperandPrecision(rawValue: UInt16(qTargetPrecision)) ?? .INT8
  let kvPrec = GEMMOperandPrecision(rawValue: UInt16(kvTargetPrecision)) ?? .INT8
  let mode: QuantizationMode = switch quantMode {
  case 0: .tensorWise
  case 2: .blockwise(blockSizeK: 64)
  default: .tensorWise
  }
  let inputPrec: GEMMOperandPrecision = switch inputPrecision {
  case 0: .FP16
  case 1: .BF16
  case 2: .FP32
  default: .FP32
  }

  // Use the cached singleton — recreating QuantizedAttention per call would
  // destroy the pipeline cache and force kernel recompilation every call.
  let quantAttention = mfaContext.quantizedAttention

  let fullQShape = [Int(batchSize), Int(numHeads), Int(seqLenQ), Int(headDim)]
  let fullKVShape = [Int(batchSize), Int(numHeads), Int(seqLenKV), Int(headDim)]

  guard
    let qTensor = quantAttention.createQuantizedTensorFromBufferPublic(
      buffer: qBuffer.buffer, shape: fullQShape,
      inputPrecision: inputPrec, targetPrecision: qPrec,
      quantizationMode: mode, targetStrategy: .symmetric
    ),
    let kTensor = quantAttention.createQuantizedTensorFromBufferPublic(
      buffer: kBuffer.buffer, shape: fullKVShape,
      inputPrecision: inputPrec, targetPrecision: kvPrec,
      quantizationMode: mode, targetStrategy: .symmetric
    ),
    let vTensor = quantAttention.createQuantizedTensorFromBufferPublic(
      buffer: vBuffer.buffer, shape: fullKVShape,
      inputPrecision: inputPrec, targetPrecision: kvPrec,
      quantizationMode: mode, targetStrategy: .symmetric
    )
  else {
    return 5
  }

  var baseDescriptor = AttentionDescriptor()
  baseDescriptor.matrixDimensions = (
    row: seqLenQ, column: seqLenKV, head: headDim
  )
  baseDescriptor.transposeState = (Q: false, K: false, V: false, O: false)
  baseDescriptor.softmaxScale = softmaxScale
  baseDescriptor.sparsityPattern = causal ? .causal : .none

  var quantConfig = QuantizedAttention.Configuration()
  quantConfig.queryPrecision = qTensor.parameters.precision
  quantConfig.keyPrecision = kTensor.parameters.precision
  quantConfig.valuePrecision = vTensor.parameters.precision

  let quantDescriptor = QuantizedAttention.QuantizedAttentionDescriptor(
    baseDescriptor: baseDescriptor, quantizationConfig: quantConfig
  )

  // If an external command buffer is provided (from PyTorch's MPS stream),
  // encode the attention dispatch into it WITHOUT committing — the caller
  // owns submission. Otherwise create our own and block.
  if
    let cbPtr = externalCommandBuffer,
    let extCB = Unmanaged<AnyObject>.fromOpaque(cbPtr).takeUnretainedValue() as? MTLCommandBuffer
  {
    guard
      quantAttention.forwardMultiHead(
        query: qTensor, key: kTensor, value: vTensor,
        output: outBuffer.buffer,
        descriptor: quantDescriptor,
        batchSize: batchSize,
        numHeads: numHeads,
        numKVHeads: numHeads,
        seqLenQ: seqLenQ,
        headDim: headDim,
        logsumexp: lseBuffer.buffer,
        mask: maskBuffer?.buffer,
        into: extCB
      ) != nil
    else { return 5 }
    return 0 // caller commits
  }

  // Blocking fallback: own command buffer, commit + wait.
  guard let commandBuffer = quantAttention.makeCommandBuffer() else {
    return 5
  }

  guard
    quantAttention.forwardMultiHead(
      query: qTensor, key: kTensor, value: vTensor,
      output: outBuffer.buffer,
      descriptor: quantDescriptor,
      batchSize: batchSize,
      numHeads: numHeads,
      numKVHeads: numHeads,
      seqLenQ: seqLenQ,
      headDim: headDim,
      logsumexp: lseBuffer.buffer,
      mask: maskBuffer?.buffer,
      into: commandBuffer
    ) != nil
  else {
    return 5
  }

  commandBuffer.commit()
  commandBuffer.waitUntilCompleted()

  if let error = commandBuffer.error {
    print("Quantized forward error: \(error)")
    return 5
  }

  return 0
}

/// Backward pass with runtime INT8 quantization.
///
/// Re-quantizes Q/K/V to INT8 (deterministic — same inputs produce same
/// quantized values as the forward), then dispatches the quantized flash
/// backward kernels per head. Computes dQ, dK, dV.
@_cdecl("mfa_quantized_backward")
public func mfa_quantized_backward(
  _ context: UnsafeMutableRawPointer?,
  _ q: UnsafeMutableRawPointer?,
  _ k: UnsafeMutableRawPointer?,
  _ v: UnsafeMutableRawPointer?,
  _ out: UnsafeMutableRawPointer?,
  _ gradOut: UnsafeMutableRawPointer?,
  _ lse: UnsafeMutableRawPointer?,
  _ gradQ: UnsafeMutableRawPointer?,
  _ gradK: UnsafeMutableRawPointer?,
  _ gradV: UnsafeMutableRawPointer?,
  _ mask: UnsafeMutableRawPointer?,
  _ batchSize: UInt32,
  _ seqLenQ: UInt32,
  _ seqLenKV: UInt32,
  _ numHeads: UInt32,
  _ headDim: UInt16,
  _ softmaxScale: Float,
  _ causal: Bool,
  _ qTargetPrecision: Int32,
  _ kvTargetPrecision: Int32,
  _ quantMode: Int32,
  _ inputPrecision: Int32
)
  -> Int32
{
  guard
    let context,
    let q, let k, let v, let out,
    let gradOut, let lse,
    let gradQ, let gradK, let gradV
  else {
    return 1
  }

  let mfaContext = Unmanaged<MFAContext>.fromOpaque(context).takeUnretainedValue()
  let device = mfaContext.device

  let qBuffer = Unmanaged<MFABuffer>.fromOpaque(q).takeUnretainedValue()
  let kBuffer = Unmanaged<MFABuffer>.fromOpaque(k).takeUnretainedValue()
  let vBuffer = Unmanaged<MFABuffer>.fromOpaque(v).takeUnretainedValue()
  let outBuffer = Unmanaged<MFABuffer>.fromOpaque(out).takeUnretainedValue()
  let gradOutBuffer = Unmanaged<MFABuffer>.fromOpaque(gradOut).takeUnretainedValue()
  let lseBuffer = Unmanaged<MFABuffer>.fromOpaque(lse).takeUnretainedValue()
  let gradQBuffer = Unmanaged<MFABuffer>.fromOpaque(gradQ).takeUnretainedValue()
  let gradKBuffer = Unmanaged<MFABuffer>.fromOpaque(gradK).takeUnretainedValue()
  let gradVBuffer = Unmanaged<MFABuffer>.fromOpaque(gradV).takeUnretainedValue()
  let maskBuffer: MFABuffer? = mask.map {
    Unmanaged<MFABuffer>.fromOpaque($0).takeUnretainedValue()
  }

  let qPrec = GEMMOperandPrecision(rawValue: UInt16(qTargetPrecision)) ?? .INT8
  let kvPrec = GEMMOperandPrecision(rawValue: UInt16(kvTargetPrecision)) ?? .INT8
  let mode: QuantizationMode = switch quantMode {
  case 0: .tensorWise
  case 2: .blockwise(blockSizeK: 64)
  default: .tensorWise
  }
  let inputPrec: GEMMOperandPrecision = switch inputPrecision {
  case 0: .FP16
  case 1: .BF16
  case 2: .FP32
  default: .FP32
  }

  // Use the cached singleton — see mfa_quantized_forward_with_lse.
  let quantAttention = mfaContext.quantizedAttention

  let fullQShape = [Int(batchSize), Int(numHeads), Int(seqLenQ), Int(headDim)]
  let fullKVShape = [Int(batchSize), Int(numHeads), Int(seqLenKV), Int(headDim)]

  guard
    let qTensor = quantAttention.createQuantizedTensorFromBufferPublic(
      buffer: qBuffer.buffer, shape: fullQShape,
      inputPrecision: inputPrec, targetPrecision: qPrec,
      quantizationMode: mode, targetStrategy: .symmetric
    ),
    let kTensor = quantAttention.createQuantizedTensorFromBufferPublic(
      buffer: kBuffer.buffer, shape: fullKVShape,
      inputPrecision: inputPrec, targetPrecision: kvPrec,
      quantizationMode: mode, targetStrategy: .symmetric
    ),
    let vTensor = quantAttention.createQuantizedTensorFromBufferPublic(
      buffer: vBuffer.buffer, shape: fullKVShape,
      inputPrecision: inputPrec, targetPrecision: kvPrec,
      quantizationMode: mode, targetStrategy: .symmetric
    )
  else {
    return 5
  }

  var baseDescriptor = AttentionDescriptor()
  baseDescriptor.matrixDimensions = (
    row: seqLenQ, column: seqLenKV, head: headDim
  )
  baseDescriptor.transposeState = (Q: false, K: false, V: false, O: false)
  baseDescriptor.softmaxScale = softmaxScale
  baseDescriptor.sparsityPattern = causal ? .causal : .none

  var quantConfig = QuantizedAttention.Configuration()
  quantConfig.queryPrecision = qTensor.parameters.precision
  quantConfig.keyPrecision = kTensor.parameters.precision
  quantConfig.valuePrecision = vTensor.parameters.precision

  let quantDescriptor = QuantizedAttention.QuantizedAttentionDescriptor(
    baseDescriptor: baseDescriptor, quantizationConfig: quantConfig
  )

  let fp32Size = MemoryLayout<Float>.stride

  // D buffer sized for ALL heads (each head writes to its own slice).
  let dBuf = device.makeBuffer(
    length: Int(seqLenQ) * Int(numHeads) * Int(batchSize) * fp32Size,
    options: .storageModeShared
  )!

  // Two 3D-grid dispatches (backwardQuery + backwardKeyValue) covering all
  // heads in parallel, replacing the previous B×H×2 per-head loop.
  guard let commandBuffer = quantAttention.makeCommandBuffer() else {
    return 5
  }

  guard
    quantAttention.backwardQueryMultiHead(
      query: qTensor, key: kTensor, value: vTensor,
      output: outBuffer.buffer,
      gradOutput: gradOutBuffer.buffer,
      logsumexp: lseBuffer.buffer,
      gradQuery: gradQBuffer.buffer,
      dValues: dBuf,
      descriptor: quantDescriptor,
      batchSize: batchSize,
      numHeads: numHeads,
      numKVHeads: numHeads,
      seqLenQ: seqLenQ,
      headDim: headDim,
      mask: maskBuffer?.buffer,
      into: commandBuffer
    ) != nil
  else { return 5 }

  guard
    quantAttention.backwardKeyValueMultiHead(
      query: qTensor, key: kTensor, value: vTensor,
      gradOutput: gradOutBuffer.buffer,
      logsumexp: lseBuffer.buffer,
      dValues: dBuf,
      gradKey: gradKBuffer.buffer,
      gradValue: gradVBuffer.buffer,
      descriptor: quantDescriptor,
      batchSize: batchSize,
      numHeads: numHeads,
      numKVHeads: numHeads,
      seqLenKV: seqLenKV,
      headDim: headDim,
      mask: maskBuffer?.buffer,
      into: commandBuffer
    ) != nil
  else { return 5 }

  commandBuffer.commit()
  commandBuffer.waitUntilCompleted()

  if let error = commandBuffer.error {
    print("Quantized backward error: \(error)")
    return 5
  }

  return 0
}

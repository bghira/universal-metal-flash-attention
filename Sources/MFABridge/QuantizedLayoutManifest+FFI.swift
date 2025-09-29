import FlashAttention

@frozen
public struct mfa_quantized_layout_t {
  public var qData: Int32 = -1
  public var kData: Int32 = -1
  public var vData: Int32 = -1
  public var output: Int32 = -1
  public var gradOutput: Int32 = -1
  public var logsumexp: Int32 = -1
  public var gradQuery: Int32 = -1
  public var dValues: Int32 = -1
  public var gradKey: Int32 = -1
  public var gradValue: Int32 = -1
  public var qScale: Int32 = -1
  public var qZeroPoint: Int32 = -1
  public var kScale: Int32 = -1
  public var kZeroPoint: Int32 = -1
  public var vScale: Int32 = -1
  public var vZeroPoint: Int32 = -1
  public var dims: Int32 = -1
  public var steClipRange: Int32 = -1
  public var qBlockScales: Int32 = -1
  public var qBlockZeroPoints: Int32 = -1
  public var kBlockScales: Int32 = -1
  public var kBlockZeroPoints: Int32 = -1
  public var vBlockScales: Int32 = -1
  public var vBlockZeroPoints: Int32 = -1
  public var qPrecomputedSums: Int32 = -1
  public var kPrecomputedSums: Int32 = -1
  public var vPrecomputedSums: Int32 = -1
  public var qStrides: Int32 = -1
  public var kStrides: Int32 = -1
  public var vStrides: Int32 = -1
  public var oStrides: Int32 = -1
  public var maskBuffer: Int32 = -1
  public var numHeads: Int32 = -1
  public var numKeyValueHeads: Int32 = -1
  public var headDimension: Int32 = -1
  public var sequenceLength: Int32 = -1
  public var scratch0: Int32 = -1
  public var scratch1: Int32 = -1

  public init() {}
}

private func defaultQuantizedLayout() -> mfa_quantized_layout_t {
  mfa_quantized_layout_t()
}

private func populateLayout(_ layout: QuantizedKernelLayoutManifest.Layout) -> mfa_quantized_layout_t {
  var result = defaultQuantizedLayout()
  result.qData = Int32(layout.qData)
  result.kData = Int32(layout.kData)
  result.vData = Int32(layout.vData)
  result.output = Int32(layout.output)
  result.gradOutput = Int32(layout.gradOutput)
  result.logsumexp = Int32(layout.logsumexp)
  result.gradQuery = Int32(layout.gradQuery)
  result.dValues = Int32(layout.dValues)
  result.gradKey = Int32(layout.gradKey)
  result.gradValue = Int32(layout.gradValue)
  result.qScale = Int32(layout.qScale)
  result.qZeroPoint = Int32(layout.qZeroPoint)
  result.kScale = Int32(layout.kScale)
  result.kZeroPoint = Int32(layout.kZeroPoint)
  result.vScale = Int32(layout.vScale)
  result.vZeroPoint = Int32(layout.vZeroPoint)
  result.dims = Int32(layout.dims)
  result.steClipRange = Int32(layout.steClipRange)
  result.qBlockScales = Int32(layout.qBlockScales)
  result.qBlockZeroPoints = Int32(layout.qBlockZeroPoints)
  result.kBlockScales = Int32(layout.kBlockScales)
  result.kBlockZeroPoints = Int32(layout.kBlockZeroPoints)
  result.vBlockScales = Int32(layout.vBlockScales)
  result.vBlockZeroPoints = Int32(layout.vBlockZeroPoints)
  result.qPrecomputedSums = Int32(layout.qPrecomputedSums)
  result.kPrecomputedSums = Int32(layout.kPrecomputedSums)
  result.vPrecomputedSums = Int32(layout.vPrecomputedSums)
  result.qStrides = Int32(layout.qStrides)
  result.kStrides = Int32(layout.kStrides)
  result.vStrides = Int32(layout.vStrides)
  result.oStrides = Int32(layout.oStrides)
  result.maskBuffer = Int32(layout.maskBuffer)
  result.numHeads = Int32(layout.numHeads)
  result.numKeyValueHeads = Int32(layout.numKeyValueHeads)
  result.headDimension = Int32(layout.headDimension)
  result.sequenceLength = Int32(layout.sequenceLength)
  result.scratch0 = Int32(layout.scratch0)
  result.scratch1 = Int32(layout.scratch1)
  return result
}

@frozen
public struct mfa_quantized_capabilities_t {
  public var supports_multi_head_backward: UInt8
  public var supports_blockwise_backward: UInt8
  public var max_heads: UInt32
  public var max_block_size: UInt32

  public init(
    supports_multi_head_backward: UInt8 = 0,
    supports_blockwise_backward: UInt8 = 0,
    max_heads: UInt32 = 1,
    max_block_size: UInt32 = 0
  ) {
    self.supports_multi_head_backward = supports_multi_head_backward
    self.supports_blockwise_backward = supports_blockwise_backward
    self.max_heads = max_heads
    self.max_block_size = max_block_size
  }
}

@_cdecl("mfa_get_quantized_capabilities")
public func mfa_get_quantized_capabilities(_ outPtr: UnsafeMutableRawPointer?) {
  guard let outPtr else {
    return
  }

  let capabilities = mfa_quantized_capabilities_t(
    supports_multi_head_backward: 1,
    supports_blockwise_backward: 1,
    max_heads: 128,
    max_block_size: 256
  )

  var localCopy = capabilities
  outPtr.copyMemory(from: &localCopy, byteCount: MemoryLayout<mfa_quantized_capabilities_t>.stride)
}

@_cdecl("mfa_get_quantized_layout")
public func mfa_get_quantized_layout(
  _ kernelRaw: Int32,
  _ outPtr: UnsafeMutableRawPointer?
) {
  guard let outPtr else {
    return
  }

  guard let kernel = QuantizedKernelLayoutManifest.Kernel(rawValue: kernelRaw) else {
    var layout = defaultQuantizedLayout()
    outPtr.copyMemory(from: &layout, byteCount: MemoryLayout<mfa_quantized_layout_t>.stride)
    return
  }

  let layout = QuantizedKernelLayoutManifest.layout(for: kernel)
  var populated = populateLayout(layout)
  outPtr.copyMemory(from: &populated, byteCount: MemoryLayout<mfa_quantized_layout_t>.stride)
}

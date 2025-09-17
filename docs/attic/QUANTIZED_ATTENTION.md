# Quantised Flash Attention (Architecture Overview)

This document explains how the current Universal Metal Flash Attention stack implements
INT8/INT4 support, how the pieces interact (Swift, Metal, and C FFI), and what trade-offs
we observe in practice.

## Feature Highlights – 2025

- **Per-operand strategies**: `QuantizationStrategy` (`legacy`, `asymmetric`, `symmetric`)
  travels with every tensor, along with a `strategyVersion` for forward compatibility.
- **Blockwise bias restoration**: Symmetric INT8 uses a fused runtime quantiser that
  subtracts block means on GPU (`GEMMRuntimeQuantization`), stores per-block scales/biases,
  and adds them back after GEMM.
- **Unified kernel bindings**: Each quantised operand now exposes four buffers in the
  generated Metal source: scale, zero point, strategy, and strategy version. The bindings
  are present regardless of whether GLUON heuristics are enabled.
- **Multi-head aware**: The Swift `MultiHeadAttention` path carries quantisation state for
  each operand. FFI forward calls (`mfa_attention_forward_quantized_unified`) route through
  this code path automatically when `numHeads > 1`.
- **Backward support**: Single-head backward passes are available via
  `mfa_attention_backward_query_quantized` and `mfa_attention_backward_kv_quantized`.
  Multi-head backward remains on the roadmap.

## Data Flow

```text
Host (Swift/Python/Rust)  →  QuantizationParameters (scale/zero/strategy)
                         →  GEMMRuntimeQuantization (optional fused blockwise)
                         →  QuantizedTensor (MTLBuffer + metadata)
                         →  QuantizedAttention.forward/backward
                         →  Metal kernels (load_quantized_int8/int4 + bias restore)
```

### QuantizedTensor Structure

```swift
let tensor = QuantizedTensor.from(
  device: device,
  floatData: values,
  shape: [batch, heads, seq, headDim],
  precision: .INT8,
  mode: .blockwise(blockSizeK: 64),
  strategy: .symmetric
)

// tensor.parameters.scale / zeroPoint / strategy / strategyVersion available here
```

`QuantizedTensor.from` calls into `GEMMRuntimeQuantization` when a symmetric blockwise
INT8 request is made, so you obtain both the quantised buffer and the per-block metadata
needed by the kernels.

## Swift API Primer

```swift
import FlashAttention

var config = QuantizedAttention.Configuration()
config.queryPrecision = .FP16
config.keyPrecision   = .INT8
config.valuePrecision = .INT8
config.queryStrategy  = .legacy
config.keyStrategy    = .symmetric
config.valueStrategy  = .symmetric
config.quantizationParameters[.K] = QuantizationParameters(
  scale: 0.12,
  zeroPoint: 0,
  precision: .INT8,
  mode: .blockwise(blockSizeK: 64),
  strategy: .symmetric
)
config.quantizationParameters[.V] = QuantizationParameters(
  scale: 0.08,
  zeroPoint: 0,
  precision: .INT8,
  strategy: .symmetric
)

let tensors = quantizedAttention.createQuantizedTensors(
  queryData: qFloats,
  keyData:   kFloats,
  valueData: vFloats,
  queryShape: [batch, seq, headDim],
  keyShape:   [batch, seq, headDim],
  valueShape: [batch, seq, headDim],
  config: config
)

var baseDescriptor = AttentionDescriptor()
baseDescriptor.matrixDimensions = (
  row: UInt32(seq),
  column: UInt32(seq),
  head: UInt16(headDim)
)
baseDescriptor.transposeState = (Q: false, K: false, V: false, O: false)

let descriptor = QuantizedAttention.QuantizedAttentionDescriptor(
  baseDescriptor: baseDescriptor,
  quantizationConfig: config
)

let commandBuffer = quantizedAttention.forward(
  query: tensors.query,
  key: tensors.key,
  value: tensors.value,
  output: outputBuffer,
  descriptor: descriptor
)
commandBuffer?.commit()
commandBuffer?.waitUntilCompleted()
```

### FFI Entry Points

- `mfa_attention_forward_quantized_unified` – main path (supports mask metadata, strategy
  flags, block sizes, and multi-head forward).
- `mfa_attention_forward_quantized` / `_enhanced` – legacy wrappers that forward into the
  unified function.
- `mfa_attention_backward_query_quantized`, `mfa_attention_backward_kv_quantized` – single-
  head backward routines.
- `mfa_set_scale_arrays` – optional helper to pass precomputed per-block scales when using
  the FFI directly.

See [API.md](../API.md) for complete signatures.

## Performance at a Glance

| Scenario | INT8 | INT4 | Notes |
| --- | --- | --- | --- |
| Flux 1024×1024 image generation | +15 % vs FP16 | +37 % vs FP16 | Memory footprint reduced by ×2 / ×4 respectively; accuracy maintained on tested prompts. |
| `QuantizedAttention.benchmark` (1×1024×1024, head=128) | 33 ms | 30.5 ms | Both faster than FP16 (38.5 ms) once the workload is bandwidth bound. |

Quantisation error remains ≈ 0.1 % (INT8) / 2 % (INT4) relative to FP32 reference across the
synthetic workloads used in `QuantizedAttentionTest`.

## Testing Checklist

```bash
swift test --filter QuantizedAttentionTest      # Kernel signature + quantisation accuracy
swift test --filter MultiHeadAttentionTest      # Multi-head + quantised bindings
swift test --filter QuantizedBackwardTest       # Single-head backward FFI
```

These suites ensure the strategy metadata, bias buffers, and fused runtime quantisation
stay in sync with code generation.

## Roadmap

- Multi-head backward FFI surface.
- Additional heuristics for GLUON enablement tuned to newer GPUs (M4 family).
- Expanded symmetric INT4 runtime quantiser (currently relies on CPU packing).

Contributions with new measurements or features are welcome—raise an issue or submit a
PR with updates to this document and the associated benchmarks.

# Quantised Metal SIMD Implementation (2025)

This document explains how the current Metal kernels handle INT8/INT4 tensors, how the
kernel generator binds quantisation metadata, and where the runtime blockwise pipeline
plugs in.

## Building Blocks

| Component | Location | Purpose |
| --- | --- | --- |
| Metal kernels | `metal-flash-attention/Sources/FlashAttention/GEMM/GEMMHeaders.swift` & friends | Provide `load_quantized_int8/int4` helpers that dequantise directly into register tiles. |
| Kernel generator | `AttentionKernel+Source.swift`, `AttentionKernel+GluonOptimizations.swift` | Emits kernel source with the appropriate bindings for quantised operands. |
| Runtime quantiser | `GEMMRuntimeQuantization.swift` + `.metal` | Fused GPU pre-processing for symmetric blockwise INT8 (centering + scale computation). |
| Swift API | `QuantizedAttention.swift` | Configures precision/strategy per operand, manages `QuantizedTensor` lifetimes, and calls into kernels. |

## Metadata Bound Per Operand

For every operand marked as `.INT8` or `.INT4`, the generated Metal function receives four
buffers:

```metal
constant float  &q_scale             [[buffer(i)]];
constant int32_t&q_zero_point        [[buffer(i+1)]];
constant uint   &q_strategy          [[buffer(i+2)]];   // QuantizationStrategy raw value
constant uint   &q_strategy_version  [[buffer(i+3)]];   // Increment on breaking changes
```

This matches the Swift-side structure:

```swift
struct QuantizationParameters: Codable {
  var scale: Float
  var zeroPoint: Int32
  var precision: GEMMOperandPrecision
  var mode: QuantizationMode
  var strategy: QuantizationStrategy   // legacy / asymmetric / symmetric
  var strategyVersion: UInt8
  var additionalScales: [Float]?       // optional block tables
  var additionalZeroPoints: [Int32]?
}
```

The helper `quantizationBindings(for:)` in `MultiHeadAttention.swift` iterates Q/K/V/O and
binds only the operands that actually have quantisation metadata.

## Load/Store Helpers

`GEMMHeaders.swift` provides overloads that the generator calls automatically:

```metal
METAL_FUNC void load_quantized_int8(
    const device char *src,
    uint elements_per_row,
    ushort2 matrix_origin,
    float scale,
    int32_t zero_point,
    bool transpose_matrix = false)
```

These helpers dequantise directly into `simdgroup_matrix_storage<float>` registers so the
main GEMM loop can continue to use the standard `multiply` instruction. INT4 uses a packed
representation (two 4-bit values per byte) with a similar helper.

## Blockwise Symmetric Pipeline

For symmetric INT8 you can request `.blockwise(blockSize: 64, granularity: .perHead)` while
setting `strategy = .symmetric`. The Swift runtime then:

1. Calls `GEMMRuntimeQuantization.quantizeBlockwiseCenteredTensor`, which
   - computes block means on GPU,
   - subtracts them before quantising,
   - writes per-block scale / bias tables, and
   - returns a `QuantizedTensor` with those buffers attached.
2. Feeds the tensor into `QuantizedAttention.forward`, which
   - binds the per-block scale arrays (via `additionalScales`),
   - restores the bias after GEMM using the stored block means,
   - routes through multi-head if needed.

If the fused runtime quantiser is unavailable (e.g. for non-symmetric strategies), the
code falls back to CPU quantisation but still binds the same metadata.

## Interaction With GLUON Heuristics

GLUON (`optimizedSoftmax`) and the baseline path are both aware of the additional buffers;
no special-case code is required. The subtiled softmax runs on the dequantised FP32 tiles,
so accuracy is governed by the quantiser rather than the optimisation itself.

## Practical Tips

- Always run `swift build -c release` before benchmarking; Metal shaders are emitted in
  the same step.
- Verify kernel signatures with `swift test --filter QuantizedAttentionTest` after touching
  binding logic; the test suite inspects the emitted source and catches missing buffers.
- When adding new strategies or block layouts, bump `QuantizationParameters.currentStrategyVersion`
  to avoid cross-version deserialisation surprises.

This design keeps the Metal kernels agnostic to the higher-level strategy logic while
exposing enough metadata to evolve quantisation behaviour without rewriting the core
GEMM/softmax loops.

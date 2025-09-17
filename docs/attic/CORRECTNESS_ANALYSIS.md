# GLUON Optimizations – Current Correctness Snapshot

This note captures the present status of the GLUON-inspired optimizations that live in
`Sources/FlashAttention/Attention/AttentionKernel/AttentionKernel+GluonOptimizations.swift`.
It explains what we have high confidence in, what still needs validation, and how the new
quantization/multi-head architecture interacts with the optimizations.

## 🔍 Summary

- GLUON optimizations remain **opt-in heuristics** layered on top of the current
  `AttentionKernel` generator. They are enabled automatically for large tiles via
  `shouldEnableGluonOptimizations()` (sequence ≥ 512, head ≥ 64) and fall back to the
  baseline path for smaller workloads.
- The broader test suite has expanded since the original write‑up. In addition to the
  classic attention tests, we now ship:
  - `QuantizedAttentionTest` – verifies kernel signatures (strategy/zero-point buffers,
    stride bindings) and round-trip quantization accuracy.
  - `MultiHeadAttentionTest` – covers the Swift multi-head path and its quantization
    bindings.
  - `QuantizedBackwardTest` – exercises the single-head backward FFI.
  These suites run cleanly with GLUON heuristics enabled.
- Flux end-to-end runs (INT8 / INT4) demonstrate the optimizations do not introduce
  instability: 1024×1024 image generation remains numerically stable while benefitting
  from the 15–37 % speed-ups observed in practice.

## ✅ What We Have Confidence In

### Subtiled Softmax

- The decomposition produced by `subtiledSoftmaxDecomposition(derivative:)` still
  reduces to the standard softmax formula. Intermediate max/sum accumulators are stored
  in FP32, preserving accuracy even with the extra loop structure.
- Quantized tensor tests confirm the softmax stage does not produce NaNs/Infs across the
  ranges we exercise (INT8 symmetric/asymmetric, INT4 packed, FP16 inputs).

### Multi-Stage Pipelining

- The pipeline emitted by `multiStagePipelinedAttention()` continues to fence each
  stage with `simdgroup_event` and `threadgroup_barrier`. No race conditions have surfaced
  when toggling GLUON on/off.
- Kernel signature tests ensure the extra stride and strategy buffers introduced by the
  quantization rework are always bound, regardless of the optimization path chosen.

### Numerical Bounds

- Quantization error monitoring (`QuantizedAttentionTest.testQuantizeAndDequantize`) shows
  INT8 relative error ≈ 0.1 %, INT4 ≈ 2 %. GLUON softmax operates on the same FP32
  intermediates, so these envelopes carry over.
- The fused runtime quantizer (`GEMMRuntimeQuantization`) injects block-bias subtraction
  prior to GLUON softmax; unit tests verify the bias add-back keeps attention weights in
  the [0, 1] range.

## ⚠️ Areas to Keep an Eye On

| Area | Status | Notes |
| --- | --- | --- |
| Blockwise symmetric quantization + GLUON | 🔄 In progress | GPU centering kernels are new; keep validating that block-bias buffers are plumbed before/after GLUON softmax. |
| Multi-head INT8/INT4 path | 🔄 Partial | Swift multi-head support is active, but FFI backward routines are currently capped at one head. Re-run tests as multi-head backward lands. |
| Benchmark parity | ✅ Manual | `benchmarks/GluonOptimizationBenchmark.swift` runs but is not part of `swift test`; run it manually when tweaking heuristics. |

## Test Checklist (2025)

```bash
swift test --filter QuantizedAttentionTest
swift test --filter MultiHeadAttentionTest
swift test --filter QuantizedBackwardTest
# Optional performance spot-check
swift run --package-path benchmarks GluonOptimizationBenchmark
```

Running the three test suites above covers both GLUON code paths and the new quantization
plumbing. The benchmark target remains useful for spotting performance regressions.

## Takeaways

- GLUON optimizations remain mathematically equivalent to the baseline implementation
  and coexist with the new quantization strategies.
- Modern validation focuses less on raw test counts and more on the scenarios that matter
  (kernel signature validation, multi-head quantization, fused runtime quantization).
- When introducing new heuristics or touching GLUON code, re-run the checklist above to
  keep confidence high.

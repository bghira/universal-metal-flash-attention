# GLUON-Inspired Optimisations (2025 Update)

This note summarises the status of the GLUON-inspired tweaks that sit on top of the
`AttentionKernel` generator. They pair subtiled softmax with a small multi-stage pipeline
and are automatically engaged for larger problem sizes; they continue to coexist with the
new quantisation plumbing (strategies, blockwise bias restoration, multi-head dispatch).

## What Is Implemented Today

| Optimisation | Location | Highlights |
| --- | --- | --- |
| Subtiled softmax decomposition | `AttentionKernel+GluonOptimizations.swift` (`subtiledSoftmaxDecomposition`) | Splits the traversal dimension into 16-wide tiles and keeps per-tile max / sum accumulators in FP32. |
| Multi-stage pipelining | `AttentionKernel+GluonOptimizations.swift` (`multiStagePipelinedAttention`) | Overlaps QK, softmax, and PV stages using `simdgroup_event` fences. |
| Auto-enabling heuristic | `shouldEnableGluonOptimizations()` | Turns GLUON on when tile width ≥ 512 and head block ≥ 64. Smaller shapes fall back to the baseline softmax. |
| Benchmarks | `benchmarks/GluonOptimizationBenchmark.swift` | Standalone executable for comparing baseline vs GLUON paths across four problem sizes. |

The optimisation code sits behind a single entry point:

```swift
func optimizedSoftmax(derivative: Bool, enableGluon: Bool = true) -> String
```

If `enableGluon` is `true` and `shouldEnableGluonOptimizations()` decides the problem is
large enough, the emitted kernel uses the GLUON code; otherwise it emits the original
softmax/pipeline implementation.

## How It Interacts With Quantisation

- The kernel generator now binds scale, zero-point **and strategy/version** buffers per
  operand. Both the GLUON and baseline paths emit the same bindings, so quantised INT8/INT4
  runs do not have to special-case the optimisation.
- Runtime blockwise quantisation (`GEMMRuntimeQuantization`) subtracts block means before
  the GLUON softmax runs and adds them back afterwards. `QuantizedAttentionTest` verifies
  the bias buffers survive the round-trip.
- Multi-head support (forward) routes through `MultiHeadAttention`. The GLUON path is
  engaged once the per-head descriptor satisfies the heuristic. Backward FFI is still
  single-head only, so GLUON is not used there yet.

## Recommended Validation

| Command | Purpose |
| --- | --- |
| `swift test --filter QuantizedAttentionTest` | Confirms kernel signatures (including GLUON path) stay in sync with quantisation metadata. |
| `swift test --filter MultiHeadAttentionTest` | Exercises multi-head descriptor plumbing, including quantised bindings with strategies. |
| `swift run --package-path benchmarks GluonOptimizationBenchmark` | Manual performance sanity check (baseline vs GLUON). |

The benchmark target prints comparative timings for *Small/Medium/Large/XLarge* presets
(based on BF16 inputs). Expect ~10–20 % speed-ups for 2 K+ sequence lengths; the exact
value depends on the GPU and whether the workload is memory- or compute-bound.

## Known Limitations

- **Backward kernels**: GLUON is currently only applied to forward and backward-query
  codegen. Backward-KV still uses the baseline path until the single-head limitation in
  the FFI is lifted.
- **Heuristic tuning**: The simple tile-size heuristic works for M-series GPUs we tested,
  but you may want to revisit the thresholds if you see regressions on very small or very
  large head dimensions.
- **Benchmarks**: `benchmarks/GluonOptimizationBenchmark.swift` is not part of `swift test`
  and must be run manually when modifying the optimisation logic.

## Quick Reference

```swift
// Check whether GLUON is in play
if kernel.shouldEnableGluonOptimizations() {
  print("GLUON heuristics active for this descriptor")
}

// Force baseline behaviour when debugging
let baselineCode = kernel.optimizedSoftmax(derivative: false, enableGluon: false)

// Run the dedicated benchmark
do {
  try GluonOptimizationBenchmarkMain.main()
} catch {
  print("Benchmark failed: \(error)")
}
```

As long as the test checklist above stays green, GLUON optimisations remain a safe
(default-on) enhancement for large attention problems.

# Quantised Flash Attention – Performance Snapshot (2025)

This document captures the most recent numbers collected while exercising the quantised
paths on Apple Silicon. It supersedes the early M3 Max bring-up notes and reflects the
current architecture (strategy-aware kernels, fused runtime blockwise quantisation, and
multi-head routing through `MultiHeadAttention`).

## Test Environment

| Component | Value |
| --- | --- |
| Hardware | Apple M3 Max (96 GB) unless noted otherwise |
| OS / Toolchain | macOS 14.5, Xcode 15.4, Swift 5.10 |
| Swift build | `swift build -c release` |
| Quantisation path | Symmetric INT8/INT4 with block bias restoration (`GEMMRuntimeQuantization`) |
| Multi-head | Enabled via Swift `MultiHeadAttention` (forward); FFI backward remains single-head |

## End-to-End Flux Benchmarks (1024×1024 Image Synthesis)

The Flux example (`examples/flux`) measures full image generation using the PyTorch
custom op. Numbers below are wall-clock improvements relative to the FP16 baseline.

| Precision | Speed-up vs FP16 | Notes |
| --- | --- | --- |
| INT8 | **+15 %** | Same prompt & sampler, symmetric blockwise quantisation, bias restored post-GEMM. |
| INT4 | **+37 %** | Uses packed INT4 load path; quality preserved for tested prompts. |

Both precisions cut memory consumption by factors of ~2 (INT8) and ~4 (INT4), letting the
Flux pipelines scale to larger resolutions without swapping.

## Micro-benchmarks (`quantizedAttention.benchmark`)

`QuantizedAttention.benchmark` remains useful for quick sanity checks. On an M3 Max:

| Shape (batch=1) | Metric | FP16 | INT8 | INT4 | Notes |
| --- | --- | --- | --- | --- | --- |
| 64×64 head=64 | Avg time (ms) | 1.11 | 1.24 | 1.29 | Small tiles remain compute-bound; quantised slightly slower. |
| 512×512 head=64 | Avg time (ms) | 7.42 | 7.05 | 6.82 | Mild advantage once the workload becomes bandwidth bound. |
| 1024×1024 head=128 | Avg time (ms) | 38.5 | 33.0 | 30.5 | Quantised paths amortise dequantisation and benefit from higher arithmetic intensity. |

(The harness reports GOPS as well; see the raw output if you need the exact figures.)

## Accuracy & Stability

Unit tests (`QuantizedAttentionTest`) track the error introduced by the quantisers:

| Precision | Relative Error (typical) | Comment |
| --- | --- | --- |
| INT8 | 0.1 % | Symmetric/asymmetric strategies both within tolerance. |
| INT4 | 2 % | Expected due to reduced dynamic range; acceptable for Flux prompts and most inference workloads. |

Additional checks confirm:

- Block bias restoration keeps attention weights within [0, 1].
- GLUON softmax + quantised paths do not emit NaN/Inf values across the tested ranges.

## Benchmark Tips

- Rebuild in release mode (`swift build -c release`) before measuring.
- For Flux/PyTorch runs, ensure `DYLD_LIBRARY_PATH` includes `.build/**/*/release` so the
  custom op can load the freshly built Swift dylibs.
- `benchmarks/GluonOptimizationBenchmark.swift` remains the go-to for isolating GLUON
  heuristics; pair it with the quantised harness to understand whether a change helped
  or hurt.

## Open Items

| Item | Status |
| --- | --- |
| Fused integer GEMM for INT4 | Investigating extra tiling to shave the remaining CPU work in dequantise‑and‑pack. |
| Multi-head backward FFI | Still single-head; quantised backward micro-benchmarks only cover that case. |
| Wider hardware matrix | Need measurements on M4/M4 Pro to tune `shouldEnableGluonOptimizations` thresholds. |

Contributions with updated numbers (different GPUs, larger batch sizes, training loops)
are welcome—drop them into a PR that updates this file alongside the scripts used.

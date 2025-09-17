# Quantised Flash Attention – Cross‑Language Bindings (2025)

This document summarises the C FFI surface that Universal Metal Flash Attention exports
and how the shipped examples (Objective‑C, Python, Rust, Flux) consume it. It replaces the
old “planned bindings” document and reflects the current code in `Sources/MFABridge` and
`examples/*`.

## Architecture Recap

```
Metal kernels  ← Swift (FlashAttention, QuantizedAttention) ← C FFI (@_cdecl) ← Language bindings
```

- Swift remains the reference implementation: `QuantizedAttention` handles tensor
  creation, runtime blockwise quantisation, and kernel dispatch. `MultiHeadAttention`
  provides multi-head routing.
- `MFABridge.swift` exposes a small C ABI; it is the only interface other languages link
  against.
- Bindings (Rust via bindgen, Python via ctypes, Objective‑C via a thin wrapper) simply
  forward through these symbols.

## Exported C Functions

| Category | Functions | Notes |
| --- | --- | --- |
| Context | `mfa_create_context`, `mfa_destroy_context`, `mfa_is_device_supported`, `mfa_get_version`, `mfa_get_gpu_latency` | Single shared context per process; `mfa_get_gpu_latency` reports last GPU time. |
| Buffers | `mfa_create_buffer`, `mfa_buffer_from_ptr`, `mfa_buffer_from_ptr_with_strides`, `mfa_buffer_from_mtl_buffer`, `mfa_buffer_from_mtl_buffer_with_strides`, `mfa_buffer_contents`, `mfa_destroy_buffer` | Support both copied and zero-copy buffers. |
| Masks | `mfa_attention_forward` / `_str` accept optional mask pointer + metadata; `MaskType`/`MaskScalarType` enums set the interpretation. | |
| Forward (dense) | `mfa_attention_forward`, `mfa_attention_forward_str` | Accept transpose flags, precision enums, and optional mask metadata. Multi-head flows through Swift `MultiHeadAttention`. |
| Forward (quantised) | `mfa_attention_forward_quantized_unified` (+ legacy wrappers `_quantized`, `_quantized_enhanced`, `_quantized_direct`) | Parameters include per-tensor scales/zero points, block sizes, granularity flag, and strategy toggles (`force_symmetric_quantization`). Call `mfa_set_scale_arrays` when supplying per-block scale tables. |
| Backward (quantised) | `mfa_attention_backward_query_quantized`, `mfa_attention_backward_kv_quantized` | Currently **single-head only**. Each expects pre-quantised Q/K/V buffers plus FP32 gradients and log-sum-exp scratch tensors. |
| Utilities | `mfa_error_string` | Returns a heap-allocated C string for a given error code. |

See [API.md](../API.md) for full signatures and sample calls.

## Language Bindings Today

| Language | Entry point | Status / Notes |
| --- | --- | --- |
| Python (ctypes) | `examples/python-ffi/src/umfa` | Wraps the full FFI, including `mfa_attention_forward_quantized_unified`. `examples/flux` builds on this to integrate with PyTorch. |
| PyTorch custom op | `examples/pytorch-custom-op-ffi` | C++ extension calling into the FFI. Supports INT8/INT4 forward (multi-head) and the new mask parameters. |
| Rust | `examples/rust-ffi` | Uses `bindgen` to generate the FFI; exposes safe RAII wrappers for context/buffers. Currently limited to forward paths. |
| Objective‑C | `examples/objc` | Simple bridge class that forwards to `mfa_attention_forward` and demonstrates Metal buffer interop. |

## Gaps & Future Work

| Area | Status |
| --- | --- |
| Multi-head backward via FFI | ❌ Not yet supported; QuantizedAttention handles it in Swift, but the C API still asserts `numHeads == 1` for backward calls. |
| Dedicated quantisation helpers | ❌ Functions such as `mfa_quantize_tensor_batch` or `mfa_create_quantized_buffer` are not implemented; callers quantise data in their native language (or use the Swift runtime quantiser). |
| Error-handling helpers | ⚠️ `mfa_error_string` exists, but higher-level helpers (e.g. to classify recoverable vs fatal errors) are left to each binding. |

## Recommended Validation Pipeline

1. Follow [INSTALL.md](../INSTALL.md) to build the Swift package.
2. Run the core tests:

   ```bash
   swift test --filter QuantizedAttentionTest
   swift test --filter MultiHeadAttentionTest
   swift test --filter QuantizedBackwardTest
   ```

3. Exercise the FFI from your language of choice (see the README in each `examples/*`
   directory). Flux/PyTorch benchmarks provide the most demanding real-world workload.

Keeping bindings thin and delegating quantisation details to Swift means the C surface
remains stable even as strategies and blockwise logic evolve. When new functionality is
added (e.g. multi-head backward), update this document alongside the examples so every
integration stays in sync.

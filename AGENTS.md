# AGENTS.md

## Repository: universal-metal-flash-attention

Universal Metal Flash Attention (UMFA) — a Swift/Metal flash-attention library
with a PyTorch custom-op FFI for Apple Silicon.

## Architecture

```
Python (metal_sdpa_extension)
  → C++ (metal_sdpa_backend.cpp, mla_wrappers.cpp)
    → Swift FFI (MFABridge.swift — @_cdecl functions)
      → FlashAttention module (submodule: metal-flash-attention)
        → AttentionKernel (code-generated Metal source)
        → MultiHeadAttention (multi-head dispatch)
        → QuantizedAttention (INT8/INT4 dequantize-on-load)
```

The `metal-flash-attention` directory is a git submodule pointing at
`github.com/bghira/metal-flash-attention-plus`.

## Key files

| File | Purpose |
|------|---------|
| `Sources/MFABridge/MFABridge.swift` | All `@_cdecl` FFI functions: forward, backward, MLA, Hadamard rotation |
| `Sources/MFAFFI/include/mfa_ffi.h` | C header declaring the FFI ABI |
| `examples/pytorch-custom-op-ffi/src/metal_sdpa_backend.cpp` | C++ backend: tensor binding, layout conversion, autograd (`MetalFlashAttentionFn`) |
| `examples/pytorch-custom-op-ffi/src/python_bindings.cpp` | pybind11 module definition |
| `metal-flash-attention/.../QuantizedAttention.swift` | Quantized attention: forward, backward, runtime quantization |
| `metal-flash-attention/.../MultiHeadAttention.swift` | Multi-head dispatch: forward, backward, pipeline cache |
| `metal-flash-attention/.../AttentionKernel+Source.swift` | Code-generated Metal kernel source (all kernel types) |
| `metal-flash-attention/.../AttentionKernel+Accumulate.swift` | Outer-product accumulate loop with blockwise dequantize-on-load |
| `metal-flash-attention/.../GEMMHeaders.swift` | Metal header: simdgroup storage, load_quantized_int8/int4, buffer bindings |
| `metal-flash-attention/.../HadamardRotation.swift` | ConvRot-style FWHT kernel for outlier smoothing |

## Build

```bash
# Swift package (libMFAFFI.dylib)
swift build -c release

# Python extension (.so)
cd examples/pytorch-custom-op-ffi
python setup.py build_ext --inplace

# Run tests
swift test -c release --filter MLAMFAUsageExample
python -m pytest tests/ -m "not gpu and not slow"
```

Requires: macOS 15+, Xcode 16+, torch >= 2.12 (unified memory).

## Debug

Set `MFA_DEBUG=1` to enable verbose diagnostic prints in `QuantizedAttention`
(buffer sizes, conversion samples, quantization fallback warnings).

```bash
MFA_DEBUG=1 swift test
```

## Known issues

### MPS cross-queue synchronization

`torch.mps.synchronize()` is required before any UMFA kernel dispatch when
tensors have pending producer ops on PyTorch's MPS command queue. This is
handled in `call_swift_flash_attention_impl` (entry-point sync) and after
`.contiguous()` calls. The SimpleTuner wrapper also does a pre-dispatch sync.

### MSL 3.2 requirement for bfloat

`torch.mps` lowers the Metal device's default language version, which
disables `__HAVE_BFLOAT__`. All `device.makeLibrary` calls must use
`languageVersion = .version3_2`. The `mfaCompileOptions()` helper in
`MFABridge.swift` and the inline options in `MultiHeadAttention.swift`
enforce this.

### Multi-head buffer binding contract

The generated kernel (`AttentionKernel.createBufferBindings`) expects:

```
Q@0 K@1 V@2 O@3 L@4  [nil strides]@5-7  num_heads@8-11  mask@12
```

The dispatch (`dispatchBatched` in `MultiHeadAttention.swift`) must bind
exactly this layout. Nil strides @5-7 force the contiguous-BHSD fallback
offset math.

## Quantized attention subsystem

### What works

- **INT8/INT4 forward** (per-tensor + blockwise) — dequantize-on-load via
  `load_quantized_int8/int4` in `GEMMHeaders.swift`.
- **INT8 backward** (per-tensor + blockwise) — uses the same flash-attention
  backward kernel with dequantize-on-load. `getOrCreateCorePipeline` sets
  `HAS_BLOCKWISE_*` function constants from actual operand properties.
- **Blockwise GEMM compensation** — `createRowSumComputation()` and
  `createBlockwiseCompensation()` in `AttentionKernel+Accumulate.swift`
  track per-block quantized sums and apply zero-point correction. The
  compensation formula is validated in `BlockwiseCompensationTest.swift`.
- **Fused GPU runtime quantization** — `GEMMBlockwiseQuantization.metal` computes
  block stats + quantizes in one pass via simdgroup reductions.
- **STE for quantization rounding** — handled at the C++ autograd level
  (`MetalFlashAttentionFn` in `metal_sdpa_backend.cpp`): forward fake-quants
  Q/K/V, backward passes gradients straight through the rounding step.

### What's incomplete

- **Blockwise backward with non-aligned blocks** — `getOrCreateCorePipeline`
  now accepts `hasBlockwiseQ/K/V` + `blockSizeK` and sets the function
  constants correctly. The dequantize-on-load path handles per-block
  scales when `BLOCK_SIZE_K` is a multiple of 8 (the simdgroup tile width).
  Non-aligned block sizes are not yet compensated in backward.
- **Strategy field in backward FFI** — always hardcoded `.legacy`;
  asymmetric/symmetric distinction is lost at the Swift↔FFI boundary.

### Quantized attention data flow

```
FP32/FP16/BF16 input → [runtime quantize or CPU fallback]
  → QuantizedTensor (INT8/INT4 MTLBuffer + scales/zero-points)
    → AttentionKernel.createSource() (code-generated MSL with function constants)
      → load_quantized_int8/int4 (dequantize-on-load into FP32 simdgroup tiles)
        → FP32 flash-attention math
          → FP32 output buffer
```

### Buffer layout manifest

`QuantizedKernelLayoutManifest` defines static buffer slots (0–30, Metal limit).
The core flash kernel path uses `AttentionKernel.createBufferBindings()` which
has its own independent scheme. These two systems must be kept in sync when
adding new buffer slots.

### ConvRot (Hadamard rotation)

`HadamardRotation.swift` implements the Fast Walsh-Hadamard Transform on Metal
for outlier smoothing before quantization. Usage:

```python
ext.hadamard_rotate(tensor, block_size=256)  # in-place, tensor on MPS
```

Double application = identity (normalized by 1/sqrt(N)).

## Git workflow

- Submodule changes go to `metal-flash-attention` (mfa+ repo) first, then
  the universal repo records the new submodule pointer.
- Always push the submodule before the superproject.
- CI runs on macos-15 runners (virtualized M1, 7 GiB). Mark memory-heavy
  tests `@pytest.mark.slow` and GPU-only tests `@pytest.mark.gpu`.

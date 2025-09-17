# Examples Directory

> For environment setup and package build instructions, read [INSTALL.md](/docs/INSTALL.md). The sections below summarise each supported example and link to the detailed README in that example's folder.

| Example | Summary | Location |
| --- | --- | --- |
| Flux + Quantized Attention | End-to-end image generation benchmarks (INT8/INT4) using the Flux diffusion model and the PyTorch custom op. | [`examples/flux/README.md`](/examples/flux/README.md) |
| Objective‑C Bridge | Minimal host app demonstrating how to call the C FFI directly from Objective‑C. | [`examples/objc/README.md`](/examples/objc/README.md) |
| Python FFI | `ctypes` bindings, unit tests, and CLI utilities for invoking the Metal kernels from Python. | [`examples/python-ffi/README.md`](/examples/python-ffi/README.md) |
| PyTorch Custom Op | Drop-in `scaled_dot_product_attention` replacement for PyTorch with quantized + sparse support. | [`examples/pytorch-custom-op-ffi/README.md`](/examples/pytorch-custom-op-ffi/README.md) |
| Rust FFI | Rust crate that links against the C API and exercises forward attention with benchmarks. | [`examples/rust-ffi/README.md`](/examples/rust-ffi/README.md) |
| Swift Reference Snippets | Standalone Swift programs illustrating pure-Swift quantized attention and bridge benchmarks. | [`examples/swift/README.md`](/examples/swift/README.md) |

## Usage Notes

- Examples rely on the Swift package being built (`swift build`) and, where applicable, the quantized Metal shaders compiled. Always run through the steps in [INSTALL.md](/docs/INSTALL.md) first.
- Some examples (Flux, PyTorch) build a native extension that expects `DYLD_LIBRARY_PATH` to include `.build/…/release`. The README for each example calls this out explicitly.
- When experimenting with quantized attention, pay attention to the expected scale/zero-point metadata. The Python and Flux examples show how to set `mfa_set_scale_arrays` or pass per-tensor parameters.
- Additional examples or language ports should add a `README.md` and extend this table so users can discover them.

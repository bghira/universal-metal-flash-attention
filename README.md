# Universal Metal Flash Attention

A universal C Foreign Function Interface (FFI) for the Metal Flash Attention library, providing Flash Attention 3-style API for seamless integration with Rust, Python, Objective-C, and any language supporting C FFI.

## Overview

This library bridges the high-performance Metal Flash Attention implementation to other programming languages through a clean C API. It maintains zero-copy semantics by working directly with Metal buffers and provides the same interface patterns as Flash Attention 3 or PyTorch SDPA.

## Features

### Language support

- **Rust FFI**: 1135 GINSTRS/s (matches native Swift performance)
- **Objective-C FFI**: 1148 GINSTRS/s peak performance
- **Python FFI**: Zero-copy generic Python integration, compatible with PyTorch and others
- **PyTorch Custom Op**: Experimental deep integration with PyTorch via PrivateUse1 backend
- **Zero-copy tensor operations** supported by MFABridge layer for low-latency integration
- **Language agnostic**: C interface works with Rust, Python, Julia, etc.

### Advanced features

- **Multiple precision support**: FP16, BF16, FP32 with automatic conversion
- **Experimental quantised attention**: Leveraging SageAttention2's lessons to reduce memory overhead for attention matmuls via `int8` and `int4`
  - Provided by custom Metal kernels with support for vectorised multi-head attention
  - Tensor-wise, Row-wise, and Block-wise quantisation strategies are supported for varying levels of performance and accuracy
- **Optimized for Apple Silicon**: Leverages unified memory architecture to avoid unnecessary memory copying
- **Sparse Attention Patterns**: FlexAttention-style sparsity with superior performance
- **GLUON-inspired improvements**
  - Subtiled softmax calculations take advantage of Metal GPU's preference for smaller operations
  - Multi-stage pipelining for reduced call overhead and synchronisation
  - Vectorised operations where possible, using Swift's fast exp2

## Installation

### Prerequisites

- macOS 15+ / iOS 17+ / tvOS 17+ / visionOS 1+
  - Not validated for iOS, tvOS, visionOS due to lack of hardware
- Xcode 15+ with Swift 5.10+
- Metal-capable device
  - Tested only so far on M3 Max 128G

See the [INSTALL](/docs/INSTALL.md) document for specific help with installation.

## Performance & Quality

- ✅ **Full performance**: 1148 GINSTRS/s peak performance for common workloads for **all** adapter languages
- ✅ **Drop-in PyTorch SDPA replacement**: Up to 1.3x faster than PyTorch SDPA when quantising attention computations on memory-bound workloads (eg. FLUX or video diffusion models)
- ✅ **FlexAttention-compatible API** with superior performance and higher quality than PyTorch MPS SDPA efficient backend
  - ✅ **Sliding Window Attention**: 33% faster than standard attention
  - ✅ **Causal Masking**: Full autoregressive model support
  - ✅ **Arbitrary binary or bias masks**: High-performance masking for eg. Chroma, PixArt, and other diffusion models

## Current Limitations

- Only accelerates attention calc
- No native variable sequence length batching support yet
- No native fused QKV+MLP interface
- Mixed-precision BF16 flash-attention is currently 10% slower than equivalent operation in PyTorch 2.8 SDPA, but we have **more accurate results**
  - PyTorch's MPS backend has a historically high number of correctness and performance issues, so this is not a particularly surprising result
- Intermediary activations must be kept in fp32 for reduction & accumulation precision guarantees
  - BF16: Supported, has low occurrence of NaN (but less stable than fp32)
  - FP16: Experimental, has high occurrence of NaN due to lack of auto-scaler

**Note:** The underlying Metal Flash Attention library supports full forward + backward passes with gradients, even through its native quantised GEMM kernels.

## Language Examples

See the [EXAMPLES](/docs/EXAMPLES.md) for language-specific adapter examples, how to integrate Universal Metal Flash Attention into downstream projects.

### Quantized Training Support

**2025 September:** Added full quantized backpropagation support with performance-optimized gradient computation.

**Training Performance Results:**

- **1.14-1.48x faster** than PyTorch backward pass
- **25-40% memory savings** during training
- **FP32 gradient precision** maintained for stability
- **Straight-through estimator** for quantization-aware training

Quantised training semantics were inspired by the GLUON project provided by Triton.

## Real-world Performance

### FLUX.1 Schnell

| Resolution      | Configuration     | Time (s) | Speedup | Notes                                                              |
| --------------- | ----------------- | -------- | ------- | ------------------------------------------------------------------ |
| **256x256**     | PyTorch Vanilla   | 6.42     | baseline| Baseline for comparison.                                           |
|                 | Metal UMFA BF16   | 6.91     | 0.93x   | Slower; overhead not justified at this resolution.                 |
|                 | Metal UMFA INT8   | 6.65     | 0.97x   | Slower; overhead not justified at this resolution.                 |
|                 | Metal UMFA INT4   | 6.45     | 1.00x   | Matches baseline; minimal benefit.                                 |
| **512x512**     | PyTorch Vanilla   | 11.23    | baseline| Baseline for comparison.                                           |
|                 | Metal UMFA BF16   | 13.10    | 0.86x   | Slower; BF16 is less efficient but more accurate.                  |
|                 | Metal UMFA INT8   | 21.28    | 0.53x   | Significantly slower due to quantization dominating time spent.    |
|                 | Metal UMFA INT4   | 25.37    | 0.44x   | Significantly slower due to quantization dominating time spent.    |
| **1024x1024**   | PyTorch Vanilla   | 77.29    | baseline| Baseline for comparison.                                           |
|                 | Metal UMFA BF16   | 87.77    | 0.88x   | Slower, but offers higher precision than quantized options.        |
|                 | Metal UMFA INT8   | 67.40    | **1.15x**   | **Faster.** Good balance of speed and quality for large inputs.    |
|                 | Metal UMFA INT4   | 56.23    | **1.37x**   | **Fastest.** Best for memory-bound workloads where max speed is key. |

Higher resolutions and longer sequence lengths benefit **the most**.

More work can be done to identify performance loss in the baseline non-quantised results.

## Integration with Other Languages

See [EXAMPLES](/docs/EXAMPLES.md) document for integration guides.

## Contributing

1. Fork the repository
2. Create your feature branch
3. Ensure tests pass: `swift test`
4. Submit a pull request

## License

MIT.

Same license as the parent Metal Flash Attention project.

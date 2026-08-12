# Universal Metal Flash Attention

A universal C Foreign Function Interface (FFI) for the Metal Flash Attention library, providing Flash Attention 3-style API for seamless integration with Rust, Python, Objective-C, and any language supporting C FFI.

## Overview

This library bridges the high-performance Metal Flash Attention implementation to other programming languages through a clean C API. It maintains zero-copy semantics by working directly with Metal buffers and provides the same interface patterns as Flash Attention 3 or PyTorch SDPA.

🍎 **NOTE**: This project is mostly a proof-of-concept to answer the age-old question, _"Can **I** accomplish that?"_

### When do I use your library?

- If you are stuck with PyTorch
  - You know who you are 🙈
- If you run models with **long** sequence lengths and would like to experiment with potential performance improvement
  - Long-context text models and high-resolution image models
  - In theory, video models also benefit, but this hasn't been verified
- If you require **more accuracy** than PyTorch SDPA can provide
  - Some image and video models with eg. Diffusers do not produce correct outputs on PyTorch, but do with UMFA
- If you **don't** require accuracy but instead require **memory efficiency** similar to what SageAttention provides NVIDIA users
  - A lower memory system might be able to run a larger model than usual, for example

### When do I NOT use your library?

- If you can make use of [MLX](https://github.com/ml-explore/mlx) instead, you should do this
- If there's a way to use [NVIDIA](https://nvidia.com) hardware
  - However, Apple GPU is **more efficient** than NVIDIA GPU for the same workload, they just don't make a super large one yet. **YET**.

## Features

### Language support

- **Rust FFI**: 1135 GINSTRS/s (matches native Swift performance)
- **Objective-C FFI**: 1148 GINSTRS/s peak performance
- **Python FFI**: Zero-copy generic Python integration, compatible with PyTorch and matches Rust & Objective-C performance
- **PyTorch Custom Op**: Experimental deep integration with PyTorch via PrivateUse1 backend
- **Zero-copy tensor operations** supported by MFABridge layer for low-latency integration
- **Language agnostic**: C interface works with Rust, Python, Julia, etc.

See the [EXAMPLES](/docs/EXAMPLES.md) document for more details on integrations.

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

## Usage

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
| **256x256**     | PyTorch Vanilla   | 5.53     | baseline| Baseline for comparison.                                           |
|                 | Metal UMFA BF16   | 4.88     | **1.13x**   | **Faster** — lower overhead at small resolutions.              |
|                 | Metal UMFA INT8   | 5.15     | **1.08x**   | **Faster** — quantization overhead amortised.                  |
|                 | Metal UMFA INT4   | 4.92     | **1.13x**   | **Faster** — best memory efficiency.                          |
| **512x512**     | PyTorch Vanilla   | 9.59     | baseline| Baseline for comparison.                                           |
|                 | Metal UMFA BF16   | 9.43     | **1.02x**   | Marginal improvement.                                          |
|                 | Metal UMFA INT8   | 9.76     | 0.98x   | Slight overhead from per-layer runtime quantization.               |
|                 | Metal UMFA INT4   | 10.17    | 0.94x   | Slight overhead from per-layer runtime quantization.               |
| **1024x1024**   | PyTorch Vanilla   | 32.88    | baseline| Baseline for comparison.                                           |
|                 | Metal UMFA BF16   | 34.61    | 0.95x   | Comparable; offers higher precision.                                |
|                 | Metal UMFA INT8   | 43.75    | 0.75x   | Quantization dispatch overhead dominates at this resolution.        |
|                 | Metal UMFA INT4   | 50.75    | 0.65x   | Quantization dispatch overhead dominates at this resolution.        |

BF16 UMFA is competitive with or faster than PyTorch SDPA at all resolutions.

Quantised modes (INT8/INT4) currently incur per-attention-layer runtime quantization
sync overhead (3 GPU command-buffer commits per forward call). This overhead is
amortised at short sequence lengths (256x256) but dominates at longer ones.
Chaining quantization dispatches into the attention command buffer (eliminating
the sync points) is on the roadmap and will close this gap.

Tested on M3 Max 128 GB, 4 inference steps, FLUX.1-schnell (Apache 2.0).

## Roadmap

- Make it simpler to install this package, is probably step number one eg. providing precompiled wheels
- Better abstraction for downstream use, eg. helpers for quantised buffers instead of having to reimplement in each adapter language
- Per-channel asymmetric quantisation to provide more options for granularity over the tensorWise, blockWise and rowWise impl we've got currently
- Chain quantization dispatches into the attention command buffer (eliminate per-tensor sync points) for further latency reduction
- Testing the gains from GLUON and other heuristics that don't exist in the original MFA repo on newer hardware, maybe someone buys me an M4 🤪
- Attention dropout, low priority, as I have no personal use-case for it
- Experimentation with newer attention strategies as they become available (_open a feature request!_)

## Contributing

1. Fork the repository
2. Create your feature branch
3. Ensure tests pass: `swift test`
4. Submit a pull request

## Citation

If you use this library in your research, please cite both this repository and the original Metal Flash Attention codebase:

**This Repository**

```
@misc{universal-metal-flash-attention,
  author = {bghira},
  title = {Universal Metal Flash Attention},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/bghira/universal-metal-flash-attention}},
}
```

**Metal Flash Attention**

```
@misc{metal-flash-attention,
  author = {Philip Turner},
    title = {Metal Flash Attention},
    year = {2024},  
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/philipturner/metal-flash-attention}},
}
```

## Acknowledgements

Thanks to [Philip Turner](https://github.com/philipturner) for creating and so generously open-sourcing the original Metal Flash Attention library under the MIT license.

His work has inspired this project and it would not have been possible without this foundation.

All of our interfaces rely on and are derived from his original work.

Thanks to [Mario Lezcano Casado](https://github.com/lezcano) for publishing the work on [GLUON](https://github.com/triton-lang/triton/blob/main/python/examples/gluon/01-attention-forward.py).

This work has inspired our quantised attention implementation; I have adapted some of his project's ideas to fit our needs.

The initial project framework was coded with [Claude Code](https://www.anthropic.com/claude), as I'd never worked on Swift before.

Further debugging of the multi-head quantised attention kernel and backward implementation were assisted by ChatGPT-5 Codex.

## License

MIT.

Same license as the parent Metal Flash Attention project.

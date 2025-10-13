# deepseek-v3.2-exp reference implementation

reference implementation of deepseek-v3.2 mla architecture for metal. the 671B model needs 1.3TB ram so you can't actually run it, but the architecture patterns work for smaller mla models.

based on [deepseek-v3.2-exp](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp)

## what this is

- reference for multi-latent attention (mla) architecture
- sparse attention indexer using metal/mps
- mixture of experts (moe) routing
- works with pytorch mps backend on apple silicon

## what works

- config loading (671B spec)
- mla forward pass structure (kv compression 32x: 16384→512 dims)
- sparse indexer via mfa bridge (complete ffi stack: python → c++ → swift → metal)
- moe routing (256 experts, 8 active per token)
- demo with 26M param model (random weights)
- graceful fallback to pytorch if extension not built

## what doesn't

- can't load 671B weights (need 1.3TB ram)
- mla kv decompression weights not loaded (ffi exists, needs weight integration)
- no tokenizer
- no pre-trained weights

## build

```bash
# build mfa bridge + python extension (enables optimized indexer)
cd ../../
swift build -c release
cd examples/pytorch-custom-op-ffi
pip install -e .

# extension provides sparse_indexer_scores via mfa bridge
# falls back to pytorch mps if not built
```

## run

```bash
cd ../deepseek-v3.2-exp

# test components
python config.py           # config loader
python kernel.py           # sparse indexer (uses mfa if built)
python generate.py --demo  # 26M model generation

# unit tests
python tests/test_mla_indexer.py
python tests/test_moe.py
```

## files

```
config.py              # 671B config loader
model.py               # mla + moe + indexer + decoder
kernel.py              # metal/mps sparse indexer
generate.py            # text generation (demo)
tests/                 # unit tests
```

## how it works

**mla:** compresses kv cache 32x (16384→512 dims) via low-rank projection. decompression uses learned weights W_k, W_v (ffi stack exists, weights not loaded yet).

**sparse indexer:**

- complete ffi stack: `kernel.py` → `metal_sdpa_ffi` → c++ extension → swift bridge → metal
- computes Q@K^T with scale, applies relu, returns scores [batch, heads, seq_q, seq_k]
- uses mfa bridge if extension built, falls back to pytorch mps if not
- currently both paths use mps gemm (can swap for mfa gemm later)

**moe:** 256 experts, 8 active per token. sigmoid routing (scale 2.5). first 3 layers dense, remaining 58 use moe.

## why this exists

mla (multi-latent attention) is useful for long-context models because it compresses kv cache 32x. smaller mla models (7B-16B range) will likely emerge since sequence length is the main bottleneck. this implementation shows how mla works on metal and can be adapted when those models arrive.

## realistic use cases

- reference for when smaller mla models are released
- test mla decompression kernels at small scale
- understand mla architecture before implementing at scale
- adapt for custom mla models (1-7B range)

## what you'd need for actual inference

- smaller model (7B-16B, not 671B)
- quantization (4-8 bit weights)
- mla ffi integration (swift decompression kernels)
- kv cache management
- tokenizer

metal/mps on m3 max: ~5-10 TFLOPS vs 580 TFLOPS on h100. good enough for 7B models, not 671B.

## ffi integration

complete sparse indexer stack implemented:

```
python (deepseek-v3.2-exp/kernel.py)
  ↓ imports sparse_indexer_scores
pytorch-custom-op-ffi/python/metal_sdpa_ffi.py
  ↓ pybind11 binding
pytorch-custom-op-ffi/src/python_bindings.cpp
  ↓ calls MetalSDPABackend::sparse_indexer_scores
pytorch-custom-op-ffi/src/metal_sdpa_backend.cpp
  ↓ wraps tensors, calls C FFI
Sources/MFAFFI/include/mfa_ffi.h
  ↓ declares mfa_sparse_indexer_scores
Sources/MFABridge/MFABridge.swift
  ↓ implements @_cdecl("mfa_sparse_indexer_scores")
metal (MPSMatrixMultiplication)
  ↓ Q@K^T with scale
```

graceful degradation: if extension not built, `kernel.py` falls back to `torch.matmul` on mps device.

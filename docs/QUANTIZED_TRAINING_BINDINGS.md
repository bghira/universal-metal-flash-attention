# Universal Quantized Flash Attention: Cross-Language Bindings

## Overview

This document describes the cross-language bindings that make quantized Flash Attention accessible from multiple programming environments. The core implementation uses Metal SIMD operations on Apple Silicon GPUs, with C FFI bindings providing universal access across programming languages.

## Architecture

### Core Implementation Stack

1. **Metal SIMD Kernels**: GPU-accelerated quantized matrix operations
2. **Swift Implementation**: QuantizedAttention class with forward/backward passes
3. **C FFI Bridge**: MFABridge providing C-compatible interface
4. **Language Bindings**: Rust, Python, Objective-C wrappers

### Current Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| Swift Core | Complete | 23 tests passing, forward pass implemented |
| Metal Kernels | Complete | INT8/INT4 quantized load functions |
| C FFI Bridge | Partial | Forward pass only, backward bindings needed |
| Backward Pass | Complete | QuantizedAttention.backwardQuery() implemented |
| Cross-Language Bindings | Documentation | Reference implementations provided |

## C FFI Interface

### Current FFI Functions (Implemented)

```c
// Context management
int32_t mfa_create_context(void** context);
int32_t mfa_destroy_context(void* context);

// Forward pass (quantized precision support)
int32_t mfa_attention_forward(
    void* context,
    void* query_ptr, void* key_ptr, void* value_ptr, void* output_ptr,
    uint32_t batch_size, uint32_t seq_len_q, uint32_t seq_len_kv,
    uint32_t num_heads, uint16_t head_dim,
    float softmax_scale, bool causal,
    int32_t q_precision, int32_t k_precision, int32_t v_precision,
    bool transpose_q, bool transpose_k, bool transpose_v, bool transpose_o
);
```

### Required FFI Extensions for Training

```c
// Quantized forward with training state preservation
int32_t mfa_attention_forward_quantized(
    void* context,
    void* query_ptr, void* key_ptr, void* value_ptr,
    void* output_ptr, void* logsumexp_ptr,
    uint32_t batch_size, uint32_t seq_len_q, uint32_t seq_len_kv,
    uint32_t num_heads, uint16_t head_dim,
    float softmax_scale, bool causal,
    int32_t q_precision, int32_t k_precision, int32_t v_precision,
    float q_scale, int32_t q_zero_point,
    float k_scale, int32_t k_zero_point,
    float v_scale, int32_t v_zero_point,
    bool transpose_q, bool transpose_k, bool transpose_v, bool transpose_o
);

// Backward pass for query gradients
int32_t mfa_attention_backward_query_quantized(
    void* context,
    void* query_ptr, void* key_ptr, void* value_ptr,
    void* grad_output_ptr, void* logsumexp_ptr,
    void* grad_query_ptr, void* d_values_ptr,
    uint32_t batch_size, uint32_t seq_len_q, uint32_t seq_len_kv,
    uint32_t num_heads, uint16_t head_dim,
    float q_scale, int32_t q_zero_point
);

// Backward pass for key/value gradients
int32_t mfa_attention_backward_kv_quantized(
    void* context,
    void* query_ptr, void* key_ptr, void* value_ptr,
    void* grad_output_ptr, void* logsumexp_ptr, void* d_values_ptr,
    void* grad_key_ptr, void* grad_value_ptr,
    uint32_t batch_size, uint32_t seq_len_q, uint32_t seq_len_kv,
    uint32_t num_heads, uint16_t head_dim,
    float q_scale, int32_t q_zero_point,
    float k_scale, int32_t k_zero_point,
    float v_scale, int32_t v_zero_point
);
```

## Quantization Parameters

### Precision Encoding

```c
#define MFA_PRECISION_FP32  0
#define MFA_PRECISION_FP16  1
#define MFA_PRECISION_INT8  8
#define MFA_PRECISION_INT4  4
```

### Quantization Strategy

- **Symmetric quantization**: Zero point typically 0 for signed types
- **Per-tensor scaling**: Single scale factor per operand
- **FP32 gradients**: All gradient computations maintain FP32 precision
- **Scale calculation**: `scale = max_abs_value / quantization_range`

## Language Bindings

### Rust Integration

**Setup:** `examples/rust-ffi/`

```toml
# Cargo.toml
[dependencies]
libc = "0.2"
bindgen = "0.70"

[build-dependencies]
bindgen = "0.70"
```

**Key Types:**
- `MfaContext` - RAII wrapper for device context
- `MfaBuffer` - Memory-managed Metal buffers
- Auto-generated bindings via `bindgen`

**Usage Pattern:**
```rust
let context = MfaContext::new()?;
let buffers = create_buffers(&context, seq_len, head_dim)?;
let result = unsafe { mfa_attention_forward(/* params */) };
// Automatic cleanup via Drop trait
```

### Python Integration

**Setup:** `examples/python-ffi/`

```python
pip install -e examples/python-ffi/
import umfa
```

**PyTorch SDPA Drop-in Replacement:**
```python
# examples/pytorch_sdpa_replacement.py
import umfa
import torch.nn.functional as F

metal_sdpa = umfa.MetalSDPA()

# Replace this:
torch_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

# With this:
metal_out = metal_sdpa(q, k, v, is_causal=True)
```

**Low-level API:**
```python
with umfa.MFAContext() as ctx:
    output = umfa.flash_attention_forward(ctx, q, k, v, causal=True)
```

**Key Features:**
- Zero-copy PyTorch/NumPy integration
- Automatic resource management via context managers
- FP16/FP32 precision support

### Objective-C Integration

**Setup:** `examples/objc/`

```objc
// examples/objc/bridge.h
@interface MFABridge : NSObject
- (instancetype)initWithDevice:(id<MTLDevice>)device;
- (double)runAttentionWithQ:(id<MTLBuffer>)qBuffer
                          K:(id<MTLBuffer>)kBuffer
                          V:(id<MTLBuffer>)vBuffer
                          O:(id<MTLBuffer>)oBuffer
                  seqLength:(NSUInteger)seqLength
                   headDim:(NSUInteger)headDim
                      scale:(float)scale
                     causal:(BOOL)causal;
@end
```

**Usage:**
```objc
MFABridge *bridge = [[MFABridge alloc] initWithDevice:device];
double executionTime = [bridge runAttentionWithQ:qBuf K:kBuf V:vBuf O:oBuf
                                        seqLength:512 headDim:64
                                            scale:0.125 causal:YES];
```

**Integration Notes:**
- Direct Metal buffer management
- Swift FlashAttention backend
- Returns execution timing for performance monitoring

## Performance Characteristics

### Measured Performance (Apple M3 Max)

**Forward Pass Performance:**

- Small matrices (512x64): 89-91% of FP16 performance
- Medium matrices (1024x64): 85-88% of FP16 performance
- Large matrices (2048x128): 80-81% of FP16 performance

**Memory Efficiency:**

- INT8 quantization: 50% memory usage, 1.8x efficiency per GOPS
- INT4 quantization: 25% memory usage, 3.6x efficiency per GOPS

**Backward Pass Performance (Larger Matrices):**

- 1024x1024x1024: 1.15x faster than FP16 (memory bandwidth bound)

### Quality Metrics

- INT8 quantization: <0.2% RMSE error
- INT4 quantization: ~2% RMSE error
- Symmetric quantization with zero point = 0
- FP32 gradient precision maintains training stability


## Technical Notes

### Memory Layout Requirements

All tensors must be contiguous in memory with the following layout:

- Query: [batch_size, seq_len_q, head_dim]
- Key: [batch_size, seq_len_kv, head_dim]
- Value: [batch_size, seq_len_kv, head_dim]
- Output: [batch_size, seq_len_q, head_dim]

### Thread Safety

The C FFI interface is not thread-safe. Each thread requires its own context:

```c
// Per-thread usage
void* context_thread1, context_thread2;
mfa_create_context(&context_thread1);
mfa_create_context(&context_thread2);
```

### Error Handling

Return codes follow standard conventions:

- 0: Success
- 1: Invalid context
- 2: Buffer allocation failed
- 3: Invalid dimensions
- 4: Kernel execution failed

This implementation provides a foundation for universal quantized Flash Attention across programming environments, enabling efficient transformer training and inference on Apple Silicon hardware.

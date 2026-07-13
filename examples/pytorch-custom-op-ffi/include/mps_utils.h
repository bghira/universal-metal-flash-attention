#pragma once

#include <ATen/Tensor.h>

namespace metal_sdpa {
namespace mps_utils {

bool is_mps_tensor(const at::Tensor& tensor);

// Returns opaque pointer to id<MTLBuffer> (or nullptr on failure)
void* get_mtl_buffer_handle(const at::Tensor& tensor);

// Encode UMFA attention into PyTorch's current MPS-stream command buffer and
// commit (without waiting). Q/K/V/output must be contiguous MPS tensors in
// BHSD layout with matching dtypes. Optional masks must be MPS tensors with
// <=4 dims; they are expanded on the same command buffer before attention.
// Ordering against surrounding torch ops is automatic because the kernels
// ride the same stream — no host synchronization happens anywhere.
// Returns 0 on success, an mfa_error_t code otherwise.
// Encode one RoPE rotation (interleaved-pair, fp32 math, pair-duplicated
// fp32 tables) into the current MPS-stream command buffer WITHOUT
// committing. src may be a strided BHSD view (last dim contiguous); dst is
// dense BHSD, same dtype. negate_sin selects the inverse rotation (dQ/dK
// backward). The caller encodes attention afterwards and commits once.
int encode_rope_rotate_on_torch_stream(
    void* mfa_context,
    const at::Tensor& src,
    const at::Tensor& dst,
    const at::Tensor& cos_table,
    const at::Tensor& sin_table,
    int64_t table_batch_stride,
    bool negate_sin,
    uint32_t batch_size,
    uint32_t num_heads,
    uint32_t seq_len,
    uint32_t head_dim,
    const char* precision);

int encode_attention_on_torch_stream(
    void* mfa_context,
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& out,
    const at::Tensor* mask,
    int mask_type,
    int mask_scalar_type,
    uint32_t batch_size,
    uint32_t seq_len_q,
    uint32_t seq_len_kv,
    uint32_t num_heads,
    uint16_t head_dim,
    float softmax_scale,
    bool is_causal,
    const char* input_precision,
    const char* intermediate_precision);

} // namespace mps_utils
} // namespace metal_sdpa

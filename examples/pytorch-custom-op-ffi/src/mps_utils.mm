#include "mps_utils.h"

#if defined(__APPLE__)
#import <Metal/Metal.h>
#include <ATen/native/mps/OperationUtils.h>
#include <torch/mps.h>
#include <vector>

extern "C" {
int mfa_rope_rotate_encode_mtl(
    void* context,
    void* command_buffer,
    void* src_buffer, int64_t src_offset,
    int64_t src_batch_stride, int64_t src_head_stride, int64_t src_seq_stride,
    void* dst_buffer, int64_t dst_offset,
    void* cos_buffer, int64_t cos_offset,
    void* sin_buffer, int64_t sin_offset,
    int64_t table_batch_stride,
    bool negate_sin,
    uint32_t batch_size, uint32_t num_heads, uint32_t seq_len, uint32_t head_dim,
    const char* precision);

int mfa_attention_encode_mtl(
    void* context,
    void* command_buffer,
    void* q_buffer, int64_t q_offset, const int64_t* q_strides,
    void* k_buffer, int64_t k_offset, const int64_t* k_strides,
    void* v_buffer, int64_t v_offset, const int64_t* v_strides,
    void* out_buffer, int64_t out_offset,
    void* mask_buffer, int64_t mask_offset,
    const int64_t* mask_shape,
    const int64_t* mask_strides,
    uint32_t mask_ndim,
    int mask_type,
    int mask_scalar_type,
    uint32_t batch_size, uint32_t seq_len_q, uint32_t seq_len_kv,
    uint32_t num_heads, uint16_t head_dim, float softmax_scale,
    bool causal,
    const char* input_precision, const char* intermediate_precision);
}

namespace metal_sdpa {
namespace mps_utils {

bool is_mps_tensor(const at::Tensor& tensor) {
    return tensor.device().type() == c10::DeviceType::MPS;
}

void* get_mtl_buffer_handle(const at::Tensor& tensor) {
    if (!is_mps_tensor(tensor)) {
        return nullptr;
    }

    id<MTLBuffer> buffer = at::native::mps::getMTLBufferStorage(tensor);
    if (buffer == nil) {
        return nullptr;
    }
    return (__bridge void*)buffer;
}

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
    const char* precision) {
    id<MTLBuffer> src_buf = at::native::mps::getMTLBufferStorage(src);
    id<MTLBuffer> dst_buf = at::native::mps::getMTLBufferStorage(dst);
    id<MTLBuffer> cos_buf = at::native::mps::getMTLBufferStorage(cos_table);
    id<MTLBuffer> sin_buf = at::native::mps::getMTLBufferStorage(sin_table);
    if (src_buf == nil || dst_buf == nil || cos_buf == nil || sin_buf == nil) {
        return 1; // MFA_ERROR_INVALID_ARGS
    }

    auto st = src.strides();
    __block int rc = 0;
    @autoreleasepool {
        at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
        if (!stream) {
            return 5; // MFA_ERROR_EXECUTION_FAILED
        }
        dispatch_sync(stream->queue(), ^{
            stream->endKernelCoalescing();
            id<MTLCommandBuffer> command_buffer = stream->commandBuffer();
            if (command_buffer == nil) {
                rc = 5; // MFA_ERROR_EXECUTION_FAILED
                return;
            }
            // No commit here: the caller encodes attention onto the same
            // command buffer right after, then commits once.
            rc = mfa_rope_rotate_encode_mtl(
                mfa_context,
                (__bridge void*)command_buffer,
                (__bridge void*)src_buf, src.storage_offset() * src.element_size(),
                st[0], st[1], st[2],
                (__bridge void*)dst_buf, dst.storage_offset() * dst.element_size(),
                (__bridge void*)cos_buf, cos_table.storage_offset() * cos_table.element_size(),
                (__bridge void*)sin_buf, sin_table.storage_offset() * sin_table.element_size(),
                table_batch_stride,
                negate_sin,
                batch_size, num_heads, seq_len, head_dim,
                precision);
        });
    }
    return rc;
}

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
    const char* intermediate_precision) {
    id<MTLBuffer> q_buf = at::native::mps::getMTLBufferStorage(q);
    id<MTLBuffer> k_buf = at::native::mps::getMTLBufferStorage(k);
    id<MTLBuffer> v_buf = at::native::mps::getMTLBufferStorage(v);
    id<MTLBuffer> out_buf = at::native::mps::getMTLBufferStorage(out);
    if (q_buf == nil || k_buf == nil || v_buf == nil || out_buf == nil) {
        return 1; // MFA_ERROR_INVALID_ARGS
    }

    const int64_t q_off = q.storage_offset() * q.element_size();
    const int64_t k_off = k.storage_offset() * k.element_size();
    const int64_t v_off = v.storage_offset() * v.element_size();
    const int64_t out_off = out.storage_offset() * out.element_size();

    // Element strides in BHSD order for non-contiguous views (the kernel
    // requires a contiguous last dim; the caller guarantees that). Null for
    // contiguous tensors selects the kernel's dense addressing.
    int64_t q_strides[4], k_strides[4], v_strides[4];
    auto fill_strides = [](const at::Tensor& t, int64_t* dst) -> const int64_t* {
        if (t.is_contiguous()) {
            return nullptr;
        }
        auto st = t.strides();
        for (int i = 0; i < 4; ++i) {
            dst[i] = st[i];
        }
        return dst;
    };
    const int64_t* q_strides_ptr = fill_strides(q, q_strides);
    const int64_t* k_strides_ptr = fill_strides(k, k_strides);
    const int64_t* v_strides_ptr = fill_strides(v, v_strides);

    id<MTLBuffer> mask_buf = nil;
    int64_t mask_off = 0;
    std::vector<int64_t> mask_shape_vec;
    std::vector<int64_t> mask_stride_vec;
    uint32_t mask_ndim = 0;
    if (mask && mask->defined() && mask_type != 0) {
        mask_buf = at::native::mps::getMTLBufferStorage(*mask);
        if (mask_buf == nil) {
            return 1; // MFA_ERROR_INVALID_ARGS
        }
        mask_off = mask->storage_offset() * mask->element_size();
        auto sizes = mask->sizes();
        auto strides = mask->strides();
        mask_shape_vec.assign(sizes.begin(), sizes.end());
        mask_stride_vec.assign(strides.begin(), strides.end());
        mask_ndim = static_cast<uint32_t>(mask_shape_vec.size());
    } else {
        mask_type = 0;
        mask_scalar_type = 0;
    }

    __block int rc = 0;
    @autoreleasepool {
        at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
        if (!stream) {
            return 5; // MFA_ERROR_EXECUTION_FAILED
        }
        // Debug aid: MFA_INSTREAM_ENTRY_SYNC=1 drains the stream before
        // encoding to distinguish producer-ordering races from encode bugs.
        static const bool debug_entry_sync = []() {
            const char* env = std::getenv("MFA_INSTREAM_ENTRY_SYNC");
            return env && env[0] == '1';
        }();
        if (debug_entry_sync) {
            torch::mps::synchronize();
        }
        // Encoding must be serialized against the MPS backend's own encodes.
        dispatch_sync(stream->queue(), ^{
            // Close the stream's coalesced eager-kernel encoder (if open) so
            // our encoder gets its own pass, properly ordered after pending
            // eager ops. Torch's MPS buffers are hazard-untracked; encoder
            // boundaries on the stream's command buffer provide the ordering.
            stream->endKernelCoalescing();
            id<MTLCommandBuffer> command_buffer = stream->commandBuffer();
            if (command_buffer == nil) {
                rc = 5; // MFA_ERROR_EXECUTION_FAILED
                return;
            }
            rc = mfa_attention_encode_mtl(
                mfa_context,
                (__bridge void*)command_buffer,
                (__bridge void*)q_buf, q_off, q_strides_ptr,
                (__bridge void*)k_buf, k_off, k_strides_ptr,
                (__bridge void*)v_buf, v_off, v_strides_ptr,
                (__bridge void*)out_buf, out_off,
                mask_buf ? (__bridge void*)mask_buf : nullptr,
                mask_off,
                mask_shape_vec.empty() ? nullptr : mask_shape_vec.data(),
                mask_stride_vec.empty() ? nullptr : mask_stride_vec.data(),
                mask_ndim,
                mask_type,
                mask_scalar_type,
                batch_size, seq_len_q, seq_len_kv,
                num_heads, head_dim, softmax_scale,
                is_causal,
                input_precision, intermediate_precision);
        });
        if (rc == 0) {
            // Submit without waiting; later torch ops on the stream stay
            // ordered after this kernel.
            torch::mps::commit();
            // Debug aid: MFA_INSTREAM_SYNC=1 fully synchronizes after every
            // encode to distinguish overlap hazards from encode bugs.
            static const bool debug_sync = []() {
                const char* env = std::getenv("MFA_INSTREAM_SYNC");
                return env && env[0] == '1';
            }();
            if (debug_sync) {
                torch::mps::synchronize();
            }
        }
    }
    return rc;
}

} // namespace mps_utils
} // namespace metal_sdpa

#else

namespace metal_sdpa {
namespace mps_utils {

bool is_mps_tensor(const at::Tensor&) {
    return false;
}

void* get_mtl_buffer_handle(const at::Tensor&) {
    return nullptr;
}

int encode_rope_rotate_on_torch_stream(
    void*, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, int64_t, bool, uint32_t, uint32_t, uint32_t, uint32_t,
    const char*) {
    return 3; // MFA_ERROR_DEVICE_NOT_SUPPORTED
}

int encode_attention_on_torch_stream(
    void*, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const at::Tensor*, int, int,
    uint32_t, uint32_t, uint32_t, uint32_t, uint16_t, float, bool,
    const char*, const char*) {
    return 3; // MFA_ERROR_DEVICE_NOT_SUPPORTED
}

} // namespace mps_utils
} // namespace metal_sdpa

#endif

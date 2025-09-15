#pragma once

#include <torch/extension.h>
#include <torch/library.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/core/DeviceType.h>
#include <c10/core/ScalarType.h>

namespace metal_sdpa {

// Forward declarations for Swift FFI (from mfa_ffi.h)
extern "C" {
    // MFA types
    typedef void* mfa_context_t;
    typedef void* mfa_buffer_t;
    typedef enum {
        MFA_SUCCESS = 0,
        MFA_ERROR_INVALID_ARGS = 1,
        MFA_ERROR_MEMORY_ALLOCATION = 2,
        MFA_ERROR_DEVICE_NOT_SUPPORTED = 3,
        MFA_ERROR_KERNEL_COMPILATION = 4,
        MFA_ERROR_EXECUTION_FAILED = 5
    } mfa_error_t;
    typedef enum {
        MFA_PRECISION_FP16 = 0,
        MFA_PRECISION_BF16 = 1,
        MFA_PRECISION_FP32 = 2,
        MFA_PRECISION_INT8 = 3,
        MFA_PRECISION_INT4 = 4
    } mfa_precision_t;

    // MFA functions
    mfa_error_t mfa_create_context(mfa_context_t* context);
    void mfa_destroy_context(mfa_context_t context);

    mfa_error_t mfa_buffer_from_ptr(mfa_context_t context, void* data_ptr, size_t size_bytes, mfa_buffer_t* buffer);
    void* mfa_buffer_contents(mfa_buffer_t buffer);
    void mfa_destroy_buffer(mfa_buffer_t buffer);

    mfa_error_t mfa_attention_forward(
        mfa_context_t context,
        mfa_buffer_t q, mfa_buffer_t k, mfa_buffer_t v, mfa_buffer_t out,
        uint32_t batch_size, uint32_t seq_len_q, uint32_t seq_len_kv,
        uint32_t num_heads, uint16_t head_dim, float softmax_scale,
        bool causal,
        mfa_precision_t input_precision,
        mfa_precision_t intermediate_precision,
        mfa_precision_t output_precision,
        bool transpose_q, bool transpose_k, bool transpose_v, bool transpose_o
    );

    mfa_error_t mfa_attention_forward_quantized(
        mfa_context_t context,
        mfa_buffer_t q, mfa_buffer_t k, mfa_buffer_t v, mfa_buffer_t out,
        uint32_t batch_size, uint32_t seq_len_q, uint32_t seq_len_kv,
        uint32_t num_heads, uint16_t head_dim, float softmax_scale,
        bool causal,
        float q_scale, int32_t q_zero_point,
        float k_scale, int32_t k_zero_point,
        float v_scale, int32_t v_zero_point,
        mfa_precision_t q_precision, mfa_precision_t k_precision,
        mfa_precision_t v_precision, mfa_precision_t output_precision,
        bool transpose_q, bool transpose_k, bool transpose_v, bool transpose_o
    );

    bool mfa_is_device_supported(void);
    void mfa_get_version(int* major, int* minor, int* patch);
}

// PyTorch SDPA backend registration
class MetalSDPABackend {
public:
    static void register_backend();
    static void unregister_backend();

    static torch::Tensor scaled_dot_product_attention(
        const torch::Tensor& query,
        const torch::Tensor& key,
        const torch::Tensor& value,
        const c10::optional<torch::Tensor>& attn_mask = c10::nullopt,
        double dropout_p = 0.0,
        bool is_causal = false,
        c10::optional<double> scale = c10::nullopt,
        bool enable_gqa = false
    );

    static torch::Tensor quantized_scaled_dot_product_attention(
        const torch::Tensor& query,
        const torch::Tensor& key,
        const torch::Tensor& value,
        const std::string& precision = "int8",
        bool is_causal = false,
        c10::optional<double> scale = c10::nullopt
    );

private:
    static mfa_context_t swift_context_;
    static bool is_initialized_;
    static std::mutex init_mutex_;

    static void ensure_initialized();
    static void cleanup();

    // Helper methods
    static torch::Tensor call_swift_flash_attention(
        const torch::Tensor& q,
        const torch::Tensor& k,
        const torch::Tensor& v,
        bool is_causal,
        float softmax_scale
    );

    static mfa_precision_t torch_dtype_to_mfa_dtype(torch::ScalarType dtype);
    static torch::Tensor ensure_contiguous_cpu(const torch::Tensor& tensor);
};

// Custom dispatcher key for Metal SDPA backend
// This will be registered as a PrivateUse1 backend
constexpr c10::DispatchKey kMetalSDPADispatchKey = c10::DispatchKey::PrivateUse1;

} // namespace metal_sdpa

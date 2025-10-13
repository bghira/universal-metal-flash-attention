#include "metal_sdpa_backend.h"
#include "mps_utils.h"
#include <torch/torch.h>
#include <stdexcept>
#include <iostream>

// Forward declarations from C FFI
extern "C" {
    typedef void* mfa_context_t;
    typedef void* mfa_mla_context_t;
    typedef void* mfa_buffer_t;
    typedef int32_t mfa_error_t;

    mfa_error_t mfa_create_context(mfa_context_t* context);
    void mfa_destroy_context(mfa_context_t context);

    mfa_error_t mfa_mla_create_context(mfa_mla_context_t* context);
    void mfa_mla_destroy_context(mfa_mla_context_t context);

    mfa_error_t mfa_mla_init_weights(
        mfa_mla_context_t context,
        uint32_t num_heads,
        uint32_t head_dim,
        uint32_t kv_latent_dim
    );

    mfa_error_t mfa_mla_load_weights(
        mfa_mla_context_t context,
        mfa_buffer_t wk,
        mfa_buffer_t wv
    );

    mfa_error_t mfa_mla_forward(
        mfa_mla_context_t context,
        mfa_context_t mfa_context,
        mfa_buffer_t kv_latent,
        mfa_buffer_t* decompressed_k,
        mfa_buffer_t* decompressed_v,
        uint32_t batch_size,
        uint32_t num_heads,
        uint32_t sequence_length,
        uint32_t head_dim,
        uint32_t kv_latent_dim
    );
}

namespace metal_sdpa {

// =============================================================================
// MlaContextWrapper Implementation
// =============================================================================

MlaContextWrapper::MlaContextWrapper() {
    // Create MFA context (needed for command queue)
    mfa_error_t result = mfa_create_context(&mfa_context_);
    if (result != 0) {
        throw std::runtime_error("Failed to create MFA context");
    }

    // Create MLA context
    result = mfa_mla_create_context(&mla_context_);
    if (result != 0) {
        if (mfa_context_) {
            mfa_destroy_context(mfa_context_);
        }
        throw std::runtime_error("Failed to create MLA context");
    }
}

MlaContextWrapper::~MlaContextWrapper() {
    if (mla_context_) {
        mfa_mla_destroy_context(mla_context_);
    }
    if (mfa_context_) {
        mfa_destroy_context(mfa_context_);
    }
}

void MlaContextWrapper::init_random_weights(uint32_t num_heads, uint32_t head_dim, uint32_t kv_latent_dim) {
    if (!mla_context_) {
        throw std::runtime_error("MLA context not initialized");
    }

    mfa_error_t result = mfa_mla_init_weights(mla_context_, num_heads, head_dim, kv_latent_dim);
    if (result != 0) {
        throw std::runtime_error("Failed to initialize MLA weights");
    }
}

void MlaContextWrapper::load_weights(const torch::Tensor& wk, const torch::Tensor& wv) {
    if (!mla_context_) {
        throw std::runtime_error("MLA context not initialized");
    }

    // Get Metal buffers from tensors
    auto wk_buffer = get_mtl_buffer_storage(wk);
    auto wv_buffer = get_mtl_buffer_storage(wv);

    mfa_error_t result = mfa_mla_load_weights(mla_context_, wk_buffer, wv_buffer);
    if (result != 0) {
        throw std::runtime_error("Failed to load MLA weights");
    }
}

std::tuple<torch::Tensor, torch::Tensor> MlaContextWrapper::forward(
    const torch::Tensor& kv_latent,
    uint32_t batch_size,
    uint32_t num_heads,
    uint32_t sequence_length,
    uint32_t head_dim,
    uint32_t kv_latent_dim
) {
    if (!mla_context_ || !mfa_context_) {
        throw std::runtime_error("MLA/MFA context not initialized");
    }

    // Get Metal buffer from kv_latent tensor
    auto kv_latent_buffer = get_mtl_buffer_storage(kv_latent);

    // Call MLA forward
    mfa_buffer_t decompressed_k_buffer = nullptr;
    mfa_buffer_t decompressed_v_buffer = nullptr;

    mfa_error_t result = mfa_mla_forward(
        mla_context_,
        mfa_context_,
        kv_latent_buffer,
        &decompressed_k_buffer,
        &decompressed_v_buffer,
        batch_size,
        num_heads,
        sequence_length,
        head_dim,
        kv_latent_dim
    );

    if (result != 0) {
        throw std::runtime_error("MLA forward pass failed");
    }

    // Create output tensors from Metal buffers
    auto output_shape = std::vector<int64_t>{
        static_cast<int64_t>(batch_size * sequence_length),
        static_cast<int64_t>(num_heads * head_dim)
    };

    auto decompressed_k = create_tensor_from_mtl_buffer(
        decompressed_k_buffer,
        output_shape,
        torch::kFloat16,
        kv_latent.device()
    );

    auto decompressed_v = create_tensor_from_mtl_buffer(
        decompressed_v_buffer,
        output_shape,
        torch::kFloat16,
        kv_latent.device()
    );

    return std::make_tuple(decompressed_k, decompressed_v);
}

// =============================================================================
// Standalone wrapper functions
// =============================================================================

void* mla_create_context_wrapper() {
    auto wrapper = new MlaContextWrapper();
    return static_cast<void*>(wrapper);
}

void mla_destroy_context_wrapper(void* context) {
    if (context) {
        delete static_cast<MlaContextWrapper*>(context);
    }
}

void mla_init_weights_wrapper(void* context, uint32_t num_heads, uint32_t head_dim, uint32_t kv_latent_dim) {
    if (!context) {
        throw std::runtime_error("Invalid MLA context");
    }
    static_cast<MlaContextWrapper*>(context)->init_random_weights(num_heads, head_dim, kv_latent_dim);
}

void mla_load_weights_wrapper(void* context, const torch::Tensor& wk, const torch::Tensor& wv) {
    if (!context) {
        throw std::runtime_error("Invalid MLA context");
    }
    static_cast<MlaContextWrapper*>(context)->load_weights(wk, wv);
}

std::tuple<torch::Tensor, torch::Tensor> mla_forward_wrapper(
    void* context,
    const torch::Tensor& kv_latent,
    uint32_t batch_size,
    uint32_t num_heads,
    uint32_t sequence_length,
    uint32_t head_dim,
    uint32_t kv_latent_dim
) {
    if (!context) {
        throw std::runtime_error("Invalid MLA context");
    }
    return static_cast<MlaContextWrapper*>(context)->forward(
        kv_latent, batch_size, num_heads, sequence_length, head_dim, kv_latent_dim
    );
}

} // namespace metal_sdpa

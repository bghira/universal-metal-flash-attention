#include "metal_sdpa_backend.h"
#include "mps_utils.h"
#include <torch/torch.h>
#include <stdexcept>
#include <iostream>

// The FFI declarations are already in metal_sdpa_backend.h
// No need to re-declare them here

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
    auto wk_buffer = mps_utils::get_mtl_buffer_handle(wk);
    auto wv_buffer = mps_utils::get_mtl_buffer_handle(wv);

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

    // Create output tensors on the same device as input
    auto output_shape = std::vector<int64_t>{
        static_cast<int64_t>(batch_size * sequence_length),
        static_cast<int64_t>(num_heads * head_dim)
    };

    auto decompressed_k = torch::empty(output_shape,
        torch::TensorOptions()
            .dtype(torch::kFloat16)
            .device(kv_latent.device()));

    auto decompressed_v = torch::empty(output_shape,
        torch::TensorOptions()
            .dtype(torch::kFloat16)
            .device(kv_latent.device()));

    // Get Metal buffer handles
    auto kv_latent_buffer = mps_utils::get_mtl_buffer_handle(kv_latent);
    auto k_buffer = mps_utils::get_mtl_buffer_handle(decompressed_k);
    auto v_buffer = mps_utils::get_mtl_buffer_handle(decompressed_v);

    // Wrap Metal buffers in mfa_buffer_t for FFI
    mfa_buffer_t kv_latent_mfa_buffer = nullptr;
    mfa_buffer_t k_mfa_buffer = nullptr;
    mfa_buffer_t v_mfa_buffer = nullptr;

    mfa_error_t result;

    // Create MFA buffers from Metal buffers
    result = mfa_buffer_from_mtl_buffer(mfa_context_, kv_latent_buffer,
                                        kv_latent.nbytes(), &kv_latent_mfa_buffer);
    if (result != MFA_SUCCESS) {
        throw std::runtime_error("Failed to create MFA buffer for kv_latent");
    }

    result = mfa_buffer_from_mtl_buffer(mfa_context_, k_buffer,
                                        decompressed_k.nbytes(), &k_mfa_buffer);
    if (result != MFA_SUCCESS) {
        mfa_destroy_buffer(kv_latent_mfa_buffer);
        throw std::runtime_error("Failed to create MFA buffer for decompressed_k");
    }

    result = mfa_buffer_from_mtl_buffer(mfa_context_, v_buffer,
                                        decompressed_v.nbytes(), &v_mfa_buffer);
    if (result != MFA_SUCCESS) {
        mfa_destroy_buffer(kv_latent_mfa_buffer);
        mfa_destroy_buffer(k_mfa_buffer);
        throw std::runtime_error("Failed to create MFA buffer for decompressed_v");
    }

    // Call MLA forward - note: it now takes mfa_buffer_t* as outputs to fill in
    result = mfa_mla_forward(
        mla_context_,
        mfa_context_,
        kv_latent_mfa_buffer,
        &k_mfa_buffer,
        &v_mfa_buffer,
        batch_size,
        num_heads,
        sequence_length,
        head_dim,
        kv_latent_dim
    );

    // Clean up MFA buffers (they're just wrappers, not the underlying Metal buffers)
    mfa_destroy_buffer(kv_latent_mfa_buffer);
    mfa_destroy_buffer(k_mfa_buffer);
    mfa_destroy_buffer(v_mfa_buffer);

    if (result != MFA_SUCCESS) {
        throw std::runtime_error("MLA forward pass failed");
    }

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

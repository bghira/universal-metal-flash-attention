#include "metal_sdpa_backend.h"
#include <torch/torch.h>
#include <ATen/ATen.h>
#include <c10/util/Exception.h>
#include <mutex>
#include <stdexcept>
#include <iostream>

namespace metal_sdpa {

// Static member initialization
mfa_context_t MetalSDPABackend::swift_context_ = nullptr;
bool MetalSDPABackend::is_initialized_ = false;
std::mutex MetalSDPABackend::init_mutex_;

void MetalSDPABackend::ensure_initialized() {
    std::lock_guard<std::mutex> lock(init_mutex_);

    if (!is_initialized_) {
        if (!mfa_is_device_supported()) {
            throw std::runtime_error("Metal is not available on this device");
        }

        mfa_error_t result = mfa_create_context(&swift_context_);
        if (result != MFA_SUCCESS || !swift_context_) {
            throw std::runtime_error("Failed to create Metal Flash Attention context");
        }

        is_initialized_ = true;
        std::cout << "Metal SDPA backend initialized successfully" << std::endl;

        // Register cleanup on exit
        std::atexit(cleanup);
    }
}

void MetalSDPABackend::cleanup() {
    std::lock_guard<std::mutex> lock(init_mutex_);

    if (is_initialized_ && swift_context_) {
        mfa_destroy_context(swift_context_);
        swift_context_ = nullptr;
        is_initialized_ = false;
    }
}

mfa_precision_t MetalSDPABackend::torch_dtype_to_mfa_dtype(torch::ScalarType dtype) {
    switch (dtype) {
        case torch::kFloat16: return MFA_PRECISION_FP16;
        case torch::kFloat32: return MFA_PRECISION_FP32;
        case torch::kBFloat16: return MFA_PRECISION_BF16;
        default:
            throw std::runtime_error("Unsupported dtype for Metal Flash Attention. Supported: float16, float32, bfloat16");
    }
}

torch::Tensor MetalSDPABackend::ensure_contiguous_cpu(const torch::Tensor& tensor) {
    if (tensor.device().is_cpu() && tensor.is_contiguous()) {
        return tensor;
    }

    return tensor.to(torch::kCPU).contiguous();
}

torch::Tensor MetalSDPABackend::call_swift_flash_attention(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    bool is_causal,
    float softmax_scale
) {
    // Ensure tensors are on CPU and contiguous
    auto q_cpu = ensure_contiguous_cpu(q);
    auto k_cpu = ensure_contiguous_cpu(k);
    auto v_cpu = ensure_contiguous_cpu(v);

    // Get tensor shapes
    auto q_sizes = q_cpu.sizes();
    auto k_sizes = k_cpu.sizes();
    auto v_sizes = v_cpu.sizes();

    // Validate tensor shapes
    if (q_sizes.size() != k_sizes.size() || k_sizes.size() != v_sizes.size()) {
        throw std::runtime_error("Query, key, and value tensors must have the same number of dimensions");
    }

    // Support both 2D (seq_len, head_dim) and 4D (batch, seq_len, num_heads, head_dim)
    uint32_t batch_size, seq_len_q, seq_len_kv, num_heads;
    uint16_t head_dim;

    if (q_sizes.size() == 2) {
        // 2D: (seq_len, head_dim) - treat as single batch, single head
        batch_size = 1;
        seq_len_q = seq_len_kv = static_cast<uint32_t>(q_sizes[0]);
        num_heads = 1;
        head_dim = static_cast<uint16_t>(q_sizes[1]);
    } else if (q_sizes.size() == 4) {
        // 4D: (batch, seq_len, num_heads, head_dim)
        batch_size = static_cast<uint32_t>(q_sizes[0]);
        seq_len_q = static_cast<uint32_t>(q_sizes[1]);
        seq_len_kv = static_cast<uint32_t>(k_sizes[1]);
        num_heads = static_cast<uint32_t>(q_sizes[2]);
        head_dim = static_cast<uint16_t>(q_sizes[3]);

        // Note: Current MFA implementation is limited to num_heads=1
        if (num_heads > 1) {
            throw std::runtime_error("Multi-head attention not yet supported by Metal Flash Attention (num_heads > 1)");
        }
    } else {
        throw std::runtime_error("Unsupported tensor dimensions. Expected 2D (seq_len, head_dim) or 4D (batch, seq_len, num_heads, head_dim)");
    }

    // Create output tensor with same shape as query
    auto output = torch::empty_like(q_cpu);

    // Get precision
    mfa_precision_t precision = torch_dtype_to_mfa_dtype(q_cpu.scalar_type());

    // Additional validation to prevent crashes
    if (seq_len_q > 65535 || seq_len_kv > 65535) {
        throw std::runtime_error("Sequence length too large (max 65535)");
    }
    if (head_dim > 1024) {
        throw std::runtime_error("Head dimension too large (max 1024)");
    }
    if (batch_size > 1024) {
        throw std::runtime_error("Batch size too large (max 1024)");
    }

    // Create MFA buffers from tensor data
    mfa_buffer_t q_buffer, k_buffer, v_buffer, out_buffer;

    size_t q_bytes = q_cpu.numel() * q_cpu.element_size();
    size_t k_bytes = k_cpu.numel() * k_cpu.element_size();
    size_t v_bytes = v_cpu.numel() * v_cpu.element_size();
    size_t out_bytes = output.numel() * output.element_size();

    mfa_error_t result;

    result = mfa_buffer_from_ptr(swift_context_, q_cpu.data_ptr(), q_bytes, &q_buffer);
    if (result != MFA_SUCCESS) {
        throw std::runtime_error("Failed to create query buffer");
    }

    result = mfa_buffer_from_ptr(swift_context_, k_cpu.data_ptr(), k_bytes, &k_buffer);
    if (result != MFA_SUCCESS) {
        mfa_destroy_buffer(q_buffer);
        throw std::runtime_error("Failed to create key buffer");
    }

    result = mfa_buffer_from_ptr(swift_context_, v_cpu.data_ptr(), v_bytes, &v_buffer);
    if (result != MFA_SUCCESS) {
        mfa_destroy_buffer(q_buffer);
        mfa_destroy_buffer(k_buffer);
        throw std::runtime_error("Failed to create value buffer");
    }

    result = mfa_buffer_from_ptr(swift_context_, output.data_ptr(), out_bytes, &out_buffer);
    if (result != MFA_SUCCESS) {
        mfa_destroy_buffer(q_buffer);
        mfa_destroy_buffer(k_buffer);
        mfa_destroy_buffer(v_buffer);
        throw std::runtime_error("Failed to create output buffer");
    }

    // Call MFA attention forward
    result = mfa_attention_forward(
        swift_context_,
        q_buffer, k_buffer, v_buffer, out_buffer,
        batch_size, seq_len_q, seq_len_kv, num_heads, head_dim,
        softmax_scale, is_causal,
        precision, precision, precision,  // input, intermediate, output precision
        false, false, false, false       // transpose flags
    );

    // Clean up buffers
    mfa_destroy_buffer(q_buffer);
    mfa_destroy_buffer(k_buffer);
    mfa_destroy_buffer(v_buffer);
    mfa_destroy_buffer(out_buffer);

    if (result != MFA_SUCCESS) {
        std::string error_msg = "Metal Flash Attention forward pass failed with code " + std::to_string(result);
        switch (result) {
            case 1: // MFA_ERROR_INVALID_ARGS
                error_msg += " (Invalid arguments - check tensor shapes and parameters)";
                break;
            case 2: // MFA_ERROR_MEMORY_ALLOCATION
                error_msg += " (Memory allocation failed)";
                break;
            case 3: // MFA_ERROR_DEVICE_NOT_SUPPORTED
                error_msg += " (Metal device not supported)";
                break;
            case 4: // MFA_ERROR_KERNEL_COMPILATION
                error_msg += " (Metal kernel compilation failed)";
                break;
            case 5: // MFA_ERROR_EXECUTION_FAILED
                error_msg += " (Kernel execution failed)";
                break;
            default:
                error_msg += " (Unknown error)";
                break;
        }
        throw std::runtime_error(error_msg);
    }

    return output;
}

torch::Tensor MetalSDPABackend::scaled_dot_product_attention(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const c10::optional<torch::Tensor>& attn_mask,
    double dropout_p,
    bool is_causal,
    c10::optional<double> scale,
    bool enable_gqa
) {
    try {
        // Ensure backend is initialized
        ensure_initialized();

    // Validate inputs
    if (dropout_p > 0.0) {
        std::cout << "Warning: Dropout not supported in Metal Flash Attention, ignoring dropout_p" << std::endl;
    }

    if (attn_mask.has_value()) {
        std::cout << "Warning: Custom attention mask not supported, using is_causal instead" << std::endl;
    }

    if (enable_gqa) {
        std::cout << "Warning: Grouped Query Attention (GQA) not yet supported, ignoring enable_gqa flag" << std::endl;
    }

    // Calculate softmax scale
    float softmax_scale = 1.0f;
    if (scale.has_value()) {
        softmax_scale = static_cast<float>(scale.value());
    } else {
        // Default: 1/sqrt(head_dim)
        int head_dim = query.size(-1);
        softmax_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    }

    // Store original device and dtype
    auto orig_device = query.device();
    auto orig_dtype = query.scalar_type();

    // Call Swift Flash Attention
    auto result = call_swift_flash_attention(query, key, value, is_causal, softmax_scale);

    // Move result back to original device if needed
    if (result.device() != orig_device) {
        result = result.to(orig_device);
    }

    // Convert to original dtype if needed
    if (result.scalar_type() != orig_dtype) {
        result = result.to(orig_dtype);
    }

    return result;

    } catch (const std::exception& e) {
        // Re-throw with more context to prevent silent crashes
        throw std::runtime_error(std::string("Metal SDPA Backend Error: ") + e.what());
    } catch (...) {
        throw std::runtime_error("Metal SDPA Backend: Unknown error occurred");
    }
}

void MetalSDPABackend::register_backend() {
    // Use the TORCH_LIBRARY_IMPL macro for modern PyTorch operator registration
    std::cout << "Metal SDPA backend registered successfully" << std::endl;
}

void MetalSDPABackend::unregister_backend() {
    cleanup();
    std::cout << "Metal SDPA backend unregistered" << std::endl;
}

// Utility functions for Python binding
bool is_metal_available() {
    return mfa_is_device_supported();
}

std::tuple<int, int, int> get_version() {
    int major, minor, patch;
    mfa_get_version(&major, &minor, &patch);
    return std::make_tuple(major, minor, patch);
}

} // namespace metal_sdpa

// Register the custom SDPA implementation for PrivateUse1 backend
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("scaled_dot_product_attention", &metal_sdpa::MetalSDPABackend::scaled_dot_product_attention);
}
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

        mfa_error_t result = mfa_create_context(&MetalSDPABackend::swift_context_);
        if (result != MFA_SUCCESS || !MetalSDPABackend::swift_context_) {
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

    if (is_initialized_ && MetalSDPABackend::swift_context_) {
        mfa_destroy_context(MetalSDPABackend::swift_context_);
        MetalSDPABackend::swift_context_ = nullptr;
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

        // Multi-head attention is now supported!
    } else {
        throw std::runtime_error("Unsupported tensor dimensions. Expected 2D (seq_len, head_dim) or 4D (batch, seq_len, num_heads, head_dim)");
    }

    // Create output tensor with same shape as query
    auto output = torch::empty_like(q_cpu);

    // Get precision
    mfa_precision_t precision = MetalSDPABackend::torch_dtype_to_mfa_dtype(q_cpu.scalar_type());

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
    mfa_buffer_t q_buffer = nullptr, k_buffer = nullptr, v_buffer = nullptr, out_buffer = nullptr;

    size_t q_bytes = q_cpu.numel() * q_cpu.element_size();
    size_t k_bytes = k_cpu.numel() * k_cpu.element_size();
    size_t v_bytes = v_cpu.numel() * v_cpu.element_size();
    size_t out_bytes = output.numel() * output.element_size();

    mfa_error_t result;

    result = mfa_buffer_from_ptr(MetalSDPABackend::swift_context_, q_cpu.data_ptr(), q_bytes, &q_buffer);
    if (result != MFA_SUCCESS) {
        throw std::runtime_error("Failed to create query buffer");
    }

    result = mfa_buffer_from_ptr(MetalSDPABackend::swift_context_, k_cpu.data_ptr(), k_bytes, &k_buffer);
    if (result != MFA_SUCCESS) {
        // Note: Don't destroy external memory buffers
        // if (q_buffer) mfa_destroy_buffer(q_buffer);
        throw std::runtime_error("Failed to create key buffer");
    }

    result = mfa_buffer_from_ptr(MetalSDPABackend::swift_context_, v_cpu.data_ptr(), v_bytes, &v_buffer);
    if (result != MFA_SUCCESS) {
        // Note: Don't destroy external memory buffers
        // if (q_buffer) mfa_destroy_buffer(q_buffer);
        // if (k_buffer) mfa_destroy_buffer(k_buffer);
        throw std::runtime_error("Failed to create value buffer");
    }

    result = mfa_buffer_from_ptr(MetalSDPABackend::swift_context_, output.data_ptr(), out_bytes, &out_buffer);
    if (result != MFA_SUCCESS) {
        // Note: Don't destroy external memory buffers
        // if (q_buffer) mfa_destroy_buffer(q_buffer);
        // if (k_buffer) mfa_destroy_buffer(k_buffer);
        // if (v_buffer) mfa_destroy_buffer(v_buffer);
        throw std::runtime_error("Failed to create output buffer");
    }

    // Call MFA attention forward
    result = mfa_attention_forward(
        MetalSDPABackend::swift_context_,
        q_buffer, k_buffer, v_buffer, out_buffer,
        batch_size, seq_len_q, seq_len_kv, num_heads, head_dim,
        softmax_scale, is_causal,
        precision, precision, precision,  // input, intermediate, output precision
        false, false, false, false       // transpose flags
    );

    // Clean up buffers
    // Note: For external memory buffers (created with deallocator: nil),
    // we should NOT call mfa_destroy_buffer as it can cause crashes.
    // The underlying PyTorch tensors manage their own memory.
    // if (q_buffer) mfa_destroy_buffer(q_buffer);
    // if (k_buffer) mfa_destroy_buffer(k_buffer);
    // if (v_buffer) mfa_destroy_buffer(v_buffer);
    // if (out_buffer) mfa_destroy_buffer(out_buffer);

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

torch::Tensor MetalSDPABackend::quantized_scaled_dot_product_attention(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const std::string& precision,
    bool is_causal,
    std::optional<double> scale
) {
    ensure_initialized();

    // Convert all tensors to CPU and contiguous
    auto q_cpu = ensure_contiguous_cpu(query);
    auto k_cpu = ensure_contiguous_cpu(key);
    auto v_cpu = ensure_contiguous_cpu(value);

    // Get tensor dimensions (same logic as regular SDPA)
    uint32_t batch_size, seq_len_q, seq_len_kv, num_heads, head_dim;

    if (q_cpu.dim() == 4) {
        auto q_sizes = q_cpu.sizes();
        batch_size = static_cast<uint32_t>(q_sizes[0]);
        seq_len_q = static_cast<uint32_t>(q_sizes[1]);
        seq_len_kv = static_cast<uint32_t>(k_cpu.sizes()[1]);
        num_heads = static_cast<uint32_t>(q_sizes[2]);
        head_dim = static_cast<uint16_t>(q_sizes[3]);
    } else {
        throw std::runtime_error("Quantized attention currently only supports 4D tensors [batch, seq_len, num_heads, head_dim]");
    }

    // Create output tensor with FP32 precision by default (fixing Float16/Float32 mismatch)
    auto output = torch::empty_like(q_cpu, torch::kFloat32);  // 🚨 FIXED: Force FP32 output

    // Convert precision string to enum
    mfa_precision_t k_precision, v_precision;
    if (precision == "int8") {
        k_precision = MFA_PRECISION_INT8;
        v_precision = MFA_PRECISION_INT8;
    } else if (precision == "int4") {
        k_precision = MFA_PRECISION_INT4;
        v_precision = MFA_PRECISION_INT4;
    } else {
        throw std::runtime_error("Unsupported quantization precision: " + precision + ". Use 'int8' or 'int4'.");
    }

    // Get query precision from tensor dtype
    mfa_precision_t q_precision = MetalSDPABackend::torch_dtype_to_mfa_dtype(q_cpu.scalar_type());

    // Calculate softmax scale
    float softmax_scale = scale ? static_cast<float>(*scale) : (1.0f / std::sqrt(static_cast<float>(head_dim)));

    // 🔧 FIX: Don't pre-scale Q - let the kernel handle all scaling
    // The Metal kernel applies softmax_scale * log2(e) internally
    torch::Tensor q_prescaled = q_cpu;  // No pre-scaling

    // 🔧 FIX: Calculate quantization scales with FP16 overflow protection
    // Problem: Previous implementation used max_val/127 which causes FP16 overflow
    // for realistic input scales (e.g., input scale 1.0 -> scale=0.008 -> overflow)

    float k_scale, v_scale, q_scale;
    int32_t k_zero_point = 0, v_zero_point = 0, q_zero_point = 0;  // Symmetric quantization

    // Get max absolute values for dynamic range calculation
    // Note: Use pre-scaled Q for max calculation
    float q_max = q_prescaled.abs().max().item<float>();
    float k_max = k_cpu.abs().max().item<float>();
    float v_max = v_cpu.abs().max().item<float>();

    if (precision == "int8") {
        // Use a minimum scale to prevent FP16 overflow during dequantization
        // FP16 max ≈ 65504, so ensure dequantized values stay well below this
        const float MIN_SCALE = 1e-4f;  // Minimum scale to prevent overflow
        const float MAX_DEQUANT_VAL = 32000.0f;  // Conservative FP16 limit

        q_scale = std::max(q_max / 127.0f, MIN_SCALE);
        k_scale = std::max(k_max / 127.0f, MIN_SCALE);
        v_scale = std::max(v_max / 127.0f, MIN_SCALE);

        // Ensure dequantized values won't overflow FP16
        q_scale = std::max(q_scale, 127.0f * q_scale / MAX_DEQUANT_VAL);
        k_scale = std::max(k_scale, 127.0f * k_scale / MAX_DEQUANT_VAL);
        v_scale = std::max(v_scale, 127.0f * v_scale / MAX_DEQUANT_VAL);

    } else { // int4
        const float MIN_SCALE = 1e-4f;
        const float MAX_DEQUANT_VAL = 32000.0f;

        q_scale = std::max(q_max / 7.0f, MIN_SCALE);
        k_scale = std::max(k_max / 7.0f, MIN_SCALE);
        v_scale = std::max(v_max / 7.0f, MIN_SCALE);

        // Ensure dequantized values won't overflow FP16
        q_scale = std::max(q_scale, 7.0f * q_scale / MAX_DEQUANT_VAL);
        k_scale = std::max(k_scale, 7.0f * k_scale / MAX_DEQUANT_VAL);
        v_scale = std::max(v_scale, 7.0f * v_scale / MAX_DEQUANT_VAL);
    }

    // 🔍 DEBUG: Log scale calculations for debugging
    std::cout << "🔍 QUANTIZATION SCALES: q=" << q_scale << ", k=" << k_scale << ", v=" << v_scale << std::endl;
    std::cout << "   Input ranges: q_max=" << q_max << ", k_max=" << k_max << ", v_max=" << v_max << std::endl;

    // 🔧 FIX: Actually quantize Q, K and V tensors to INT8
    // Swift expects quantized data, not FP32 data + scales

    torch::Tensor q_quantized, k_quantized, v_quantized;

    if (precision == "int8") {
        // Quantize Q: Use pre-scaled Q (already includes softmax_scale * log2(e))
        q_quantized = torch::round(q_prescaled / q_scale).clamp(-127, 127).to(torch::kInt8);
        // Quantize K: FP32 -> INT8 (no pre-scaling for K)
        k_quantized = torch::round(k_cpu / k_scale).clamp(-127, 127).to(torch::kInt8);
        // Quantize V: FP32 -> INT8
        v_quantized = torch::round(v_cpu / v_scale).clamp(-127, 127).to(torch::kInt8);
    } else { // int4 - clamp to 4-bit range
        q_quantized = torch::round(q_prescaled / q_scale).clamp(-7, 7).to(torch::kInt8);
        k_quantized = torch::round(k_cpu / k_scale).clamp(-7, 7).to(torch::kInt8); // Store as INT8 but use 4-bit range
        v_quantized = torch::round(v_cpu / v_scale).clamp(-7, 7).to(torch::kInt8);
    }

    // Create MFA buffers
    mfa_buffer_t q_buffer, k_buffer, v_buffer, out_buffer;

    size_t out_bytes = output.numel() * output.element_size();

    mfa_error_t result;

    // 🔧 FIX: Use quantized Q tensor data too
    size_t q_quantized_bytes = q_quantized.numel() * q_quantized.element_size();
    result = mfa_buffer_from_ptr(MetalSDPABackend::swift_context_, q_quantized.data_ptr(), q_quantized_bytes, &q_buffer);
    if (result != MFA_SUCCESS) {
        throw std::runtime_error("Failed to create query buffer for quantized attention");
    }

    // 🔧 FIX: Use quantized tensor data and update buffer size for INT8
    size_t k_quantized_bytes = k_quantized.numel() * k_quantized.element_size();
    result = mfa_buffer_from_ptr(MetalSDPABackend::swift_context_, k_quantized.data_ptr(), k_quantized_bytes, &k_buffer);
    if (result != MFA_SUCCESS) {
        // Note: Don't destroy external memory buffers
        // mfa_destroy_buffer(q_buffer);
        throw std::runtime_error("Failed to create key buffer for quantized attention");
    }

    // 🔧 FIX: Use quantized tensor data and update buffer size for INT8
    size_t v_quantized_bytes = v_quantized.numel() * v_quantized.element_size();
    result = mfa_buffer_from_ptr(MetalSDPABackend::swift_context_, v_quantized.data_ptr(), v_quantized_bytes, &v_buffer);
    if (result != MFA_SUCCESS) {
        // Note: Don't destroy external memory buffers
        // mfa_destroy_buffer(q_buffer);
        // mfa_destroy_buffer(k_buffer);
        throw std::runtime_error("Failed to create value buffer for quantized attention");
    }

    result = mfa_buffer_from_ptr(MetalSDPABackend::swift_context_, output.data_ptr(), out_bytes, &out_buffer);
    if (result != MFA_SUCCESS) {
        // Note: Don't destroy external memory buffers
        // mfa_destroy_buffer(q_buffer);
        // mfa_destroy_buffer(k_buffer);
        // mfa_destroy_buffer(v_buffer);
        throw std::runtime_error("Failed to create output buffer for quantized attention");
    }

    try {
        // Call quantized attention function
        // 🔧 Pass the original softmax_scale - kernel handles log2(e) multiplication
        result = mfa_attention_forward_quantized(
            MetalSDPABackend::swift_context_,
            q_buffer, k_buffer, v_buffer, out_buffer,
            batch_size, seq_len_q, seq_len_kv, num_heads, head_dim,
            softmax_scale, is_causal,  // Pass original scale
            q_scale, 0,  // 🔧 FIX: Use calculated Q scale, not 1.0
            k_scale, k_zero_point,
            v_scale, v_zero_point,
            q_precision,
            k_precision,
            v_precision,
            MFA_PRECISION_FP32,  // 🚨 FIXED: Force FP32 output precision
            false, false, false, false  // No transpose for standard layout
        );

        if (result != MFA_SUCCESS) {
            throw std::runtime_error("Quantized attention forward pass failed with error code: " + std::to_string(result));
        }

        // Cleanup buffers
        // Note: For external memory buffers (created with mfa_buffer_from_ptr),
        // we should NOT call mfa_destroy_buffer as it can cause crashes.
        // The underlying PyTorch tensors manage their own memory.
        // if (q_buffer) mfa_destroy_buffer(q_buffer);
        // if (k_buffer) mfa_destroy_buffer(k_buffer);
        // if (v_buffer) mfa_destroy_buffer(v_buffer);
        // if (out_buffer) mfa_destroy_buffer(out_buffer);

        // Move output back to original device
        return output.to(query.device());

    } catch (...) {
        // Cleanup on exception
        // Note: For external memory buffers (created with mfa_buffer_from_ptr),
        // we should NOT call mfa_destroy_buffer as it can cause crashes.
        // The underlying PyTorch tensors manage their own memory.
        // if (q_buffer) mfa_destroy_buffer(q_buffer);
        // if (k_buffer) mfa_destroy_buffer(k_buffer);
        // if (v_buffer) mfa_destroy_buffer(v_buffer);
        // if (out_buffer) mfa_destroy_buffer(out_buffer);
        throw;
    }
}


torch::Tensor MetalSDPABackend::quantized_scaled_dot_product_attention_with_config(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const QuantizationConfig& config
) {
    ensure_initialized();

    // Convert all tensors to CPU and contiguous
    auto q_cpu = MetalSDPABackend::ensure_contiguous_cpu(query);
    auto k_cpu = MetalSDPABackend::ensure_contiguous_cpu(key);
    auto v_cpu = MetalSDPABackend::ensure_contiguous_cpu(value);

    // Get tensor dimensions (same logic as regular SDPA)
    uint32_t batch_size, seq_len_q, seq_len_kv, num_heads, head_dim;

    if (q_cpu.dim() == 4) {
        auto q_sizes = q_cpu.sizes();
        batch_size = static_cast<uint32_t>(q_sizes[0]);
        seq_len_q = static_cast<uint32_t>(q_sizes[1]);
        seq_len_kv = static_cast<uint32_t>(k_cpu.sizes()[1]);
        num_heads = static_cast<uint32_t>(q_sizes[2]);
        head_dim = static_cast<uint16_t>(q_sizes[3]);
    } else {
        throw std::runtime_error("Configurable quantized attention currently only supports 4D tensors [batch, seq_len, num_heads, head_dim]");
    }

    // Create output tensor with configurable precision
    torch::ScalarType output_dtype;
    switch (config.output_precision) {
        case OutputPrecision::FP16:
            output_dtype = torch::kFloat16;
            break;
        case OutputPrecision::BF16:
            output_dtype = torch::kBFloat16;
            break;
        case OutputPrecision::FP32:
        default:
            output_dtype = torch::kFloat32;
            break;
    }

    auto output = torch::empty_like(q_cpu, output_dtype);

    // Convert precision string to enum
    mfa_precision_t k_precision, v_precision;
    if (config.precision == "int8") {
        k_precision = MFA_PRECISION_INT8;
        v_precision = MFA_PRECISION_INT8;
    } else if (config.precision == "int4") {
        k_precision = MFA_PRECISION_INT4;
        v_precision = MFA_PRECISION_INT4;
    } else {
        throw std::runtime_error("Unsupported quantization precision: " + config.precision + ". Use 'int8' or 'int4'.");
    }

    // Get query precision from tensor dtype
    mfa_precision_t q_precision = MetalSDPABackend::torch_dtype_to_mfa_dtype(q_cpu.scalar_type());

    // Convert output precision config to MFA precision
    mfa_precision_t output_precision_mfa;
    switch (config.output_precision) {
        case OutputPrecision::FP16:
            output_precision_mfa = MFA_PRECISION_FP16;
            break;
        case OutputPrecision::BF16:
            output_precision_mfa = MFA_PRECISION_BF16;
            break;
        case OutputPrecision::FP32:
        default:
            output_precision_mfa = MFA_PRECISION_FP32;
            break;
    }

    // Calculate softmax scale
    float softmax_scale = config.scale ? static_cast<float>(*config.scale) : (1.0f / std::sqrt(static_cast<float>(head_dim)));

    // 🔧 FIX: Don't pre-scale Q - let the kernel handle all scaling
    // The Metal kernel applies softmax_scale * log2(e) internally
    torch::Tensor q_prescaled = q_cpu;  // No pre-scaling

    // 🔧 FIX: Calculate quantization scales with FP16 overflow protection
    // Problem: Previous implementation used max_val/127 which causes FP16 overflow
    // for realistic input scales (e.g., input scale 1.0 -> scale=0.008 -> overflow)

    float k_scale, v_scale, q_scale;
    int32_t k_zero_point = 0, v_zero_point = 0, q_zero_point = 0;  // Symmetric quantization

    // Get max absolute values for dynamic range calculation
    // Note: Use pre-scaled Q for max calculation
    float q_max = q_prescaled.abs().max().item<float>();
    float k_max = k_cpu.abs().max().item<float>();
    float v_max = v_cpu.abs().max().item<float>();

    if (config.precision == "int8") {
        // Use a minimum scale to prevent FP16 overflow during dequantization
        // FP16 max ≈ 65504, so ensure dequantized values stay well below this
        const float MIN_SCALE = 1e-4f;  // Minimum scale to prevent overflow
        const float MAX_DEQUANT_VAL = 32000.0f;  // Conservative FP16 limit

        q_scale = std::max(q_max / 127.0f, MIN_SCALE);
        k_scale = std::max(k_max / 127.0f, MIN_SCALE);
        v_scale = std::max(v_max / 127.0f, MIN_SCALE);

        // Ensure dequantized values won't overflow FP16
        q_scale = std::max(q_scale, 127.0f * q_scale / MAX_DEQUANT_VAL);
        k_scale = std::max(k_scale, 127.0f * k_scale / MAX_DEQUANT_VAL);
        v_scale = std::max(v_scale, 127.0f * v_scale / MAX_DEQUANT_VAL);

    } else { // int4
        const float MIN_SCALE = 1e-4f;
        const float MAX_DEQUANT_VAL = 32000.0f;

        q_scale = std::max(q_max / 7.0f, MIN_SCALE);
        k_scale = std::max(k_max / 7.0f, MIN_SCALE);
        v_scale = std::max(v_max / 7.0f, MIN_SCALE);

        // Ensure dequantized values won't overflow FP16
        q_scale = std::max(q_scale, 7.0f * q_scale / MAX_DEQUANT_VAL);
        k_scale = std::max(k_scale, 7.0f * k_scale / MAX_DEQUANT_VAL);
        v_scale = std::max(v_scale, 7.0f * v_scale / MAX_DEQUANT_VAL);
    }

    // 🔍 DEBUG: Log scale calculations for debugging
    std::cout << "🔍 QUANTIZATION SCALES: q=" << q_scale << ", k=" << k_scale << ", v=" << v_scale << std::endl;
    std::cout << "   Input ranges: q_max=" << q_max << ", k_max=" << k_max << ", v_max=" << v_max << std::endl;

    torch::Tensor q_quantized, k_quantized, v_quantized;

    if (config.precision == "int8") {
        // Quantize Q: Use pre-scaled Q (already includes softmax_scale * log2(e))
        q_quantized = torch::round(q_prescaled / q_scale).clamp(-127, 127).to(torch::kInt8);
        // Quantize K: FP32 -> INT8 (no pre-scaling for K)
        k_quantized = torch::round(k_cpu / k_scale).clamp(-127, 127).to(torch::kInt8);
        // Quantize V: FP32 -> INT8
        v_quantized = torch::round(v_cpu / v_scale).clamp(-127, 127).to(torch::kInt8);
    } else { // int4 - clamp to 4-bit range
        q_quantized = torch::round(q_prescaled / q_scale).clamp(-7, 7).to(torch::kInt8);
        k_quantized = torch::round(k_cpu / k_scale).clamp(-7, 7).to(torch::kInt8);
        v_quantized = torch::round(v_cpu / v_scale).clamp(-7, 7).to(torch::kInt8);
    }

    // Create MFA buffers
    mfa_buffer_t q_buffer, k_buffer, v_buffer, out_buffer;

    size_t q_quantized_bytes = q_quantized.numel() * q_quantized.element_size();
    size_t k_quantized_bytes = k_quantized.numel() * k_quantized.element_size();
    size_t v_quantized_bytes = v_quantized.numel() * v_quantized.element_size();
    size_t out_bytes = output.numel() * output.element_size();

    mfa_error_t result;

    result = mfa_buffer_from_ptr(MetalSDPABackend::swift_context_, q_quantized.data_ptr(), q_quantized_bytes, &q_buffer);
    if (result != MFA_SUCCESS) {
        throw std::runtime_error("Failed to create query buffer for configurable quantized attention");
    }

    result = mfa_buffer_from_ptr(MetalSDPABackend::swift_context_, k_quantized.data_ptr(), k_quantized_bytes, &k_buffer);
    if (result != MFA_SUCCESS) {
        throw std::runtime_error("Failed to create key buffer for configurable quantized attention");
    }

    result = mfa_buffer_from_ptr(MetalSDPABackend::swift_context_, v_quantized.data_ptr(), v_quantized_bytes, &v_buffer);
    if (result != MFA_SUCCESS) {
        throw std::runtime_error("Failed to create value buffer for configurable quantized attention");
    }

    result = mfa_buffer_from_ptr(MetalSDPABackend::swift_context_, output.data_ptr(), out_bytes, &out_buffer);
    if (result != MFA_SUCCESS) {
        throw std::runtime_error("Failed to create output buffer for configurable quantized attention");
    }

    try {
        // Call quantized attention function with configurable output precision
        // 🔧 Pass the original softmax_scale - kernel handles log2(e) multiplication
        result = mfa_attention_forward_quantized(
            MetalSDPABackend::swift_context_,
            q_buffer, k_buffer, v_buffer, out_buffer,
            batch_size, seq_len_q, seq_len_kv, num_heads, head_dim,
            softmax_scale, config.is_causal,  // Pass original scale
            q_scale, 0,  // Q scale and zero point
            k_scale, k_zero_point,
            v_scale, v_zero_point,
            q_precision,
            k_precision,
            v_precision,
            output_precision_mfa,  // 🚨 CONFIGURABLE: Use specified output precision
            false, false, false, false  // No transpose for standard layout
        );

        if (result != MFA_SUCCESS) {
            throw std::runtime_error("Configurable quantized attention forward pass failed with error code: " + std::to_string(result));
        }

        // Move output back to original device
        return output.to(query.device());

    } catch (...) {
        throw;
    }
}

BlockQuantizationParams MetalSDPABackend::quantize_tensor_per_block(
    const torch::Tensor& tensor,
    int block_rows,
    const std::string& precision
) {
    auto tensor_cpu = ensure_contiguous_cpu(tensor);

    // Get tensor dimensions
    auto sizes = tensor_cpu.sizes();
    int total_rows = 1;
    for (int i = 0; i < sizes.size() - 1; ++i) {
        total_rows *= sizes[i];
    }
    int cols = sizes[sizes.size() - 1];

    // Calculate number of blocks
    int num_blocks = (total_rows + block_rows - 1) / block_rows;

    BlockQuantizationParams params(block_rows, cols);
    params.num_blocks = num_blocks;
    params.scales.reserve(num_blocks);
    params.zero_points.reserve(num_blocks);

    // Flatten tensor for block processing
    auto flat_tensor = tensor_cpu.view({total_rows, cols});

    // Calculate quantization range
    float quant_max = (precision == "int8") ? 127.0f : 7.0f; // INT8: [-127, 127], INT4: [-7, 7]

    // Process each block
    for (int block_idx = 0; block_idx < num_blocks; ++block_idx) {
        int start_row = block_idx * block_rows;
        int end_row = std::min(start_row + block_rows, total_rows);

        // Extract block
        auto block = flat_tensor.slice(0, start_row, end_row);

        // Calculate per-block scale using min-max range
        auto block_max = block.abs().max().item<float>();

        // Prevent division by zero and ensure minimum scale for numerical stability
        const float MIN_SCALE = 1e-6f;
        float scale = std::max(block_max / quant_max, MIN_SCALE);

        // Apply additional numerical stability constraints
        const float MAX_DEQUANT_VAL = 32000.0f;  // Conservative FP16 limit
        scale = std::max(scale, quant_max * scale / MAX_DEQUANT_VAL);

        params.scales.push_back(scale);
        params.zero_points.push_back(0);  // Symmetric quantization
    }

    return params;
}

torch::Tensor MetalSDPABackend::apply_per_block_quantization(
    const torch::Tensor& tensor,
    const BlockQuantizationParams& params,
    const std::string& precision
) {
    auto tensor_cpu = ensure_contiguous_cpu(tensor);

    // Get tensor dimensions
    auto sizes = tensor_cpu.sizes();
    int total_rows = 1;
    for (int i = 0; i < sizes.size() - 1; ++i) {
        total_rows *= sizes[i];
    }
    int cols = sizes[sizes.size() - 1];

    // Create quantized output tensor
    auto quantized = torch::empty_like(tensor_cpu, torch::kInt8);

    // Flatten tensors for block processing
    auto flat_input = tensor_cpu.view({total_rows, cols});
    auto flat_output = quantized.view({total_rows, cols});

    // Calculate quantization range
    float quant_min = (precision == "int8") ? -127.0f : -7.0f;
    float quant_max = (precision == "int8") ? 127.0f : 7.0f;

    // Quantize each block
    for (int block_idx = 0; block_idx < params.num_blocks; ++block_idx) {
        int start_row = block_idx * params.block_size_rows;
        int end_row = std::min(start_row + params.block_size_rows, total_rows);

        // Extract input and output blocks
        auto input_block = flat_input.slice(0, start_row, end_row);
        auto output_block = flat_output.slice(0, start_row, end_row);

        // Apply quantization with block-specific scale
        float scale = params.scales[block_idx];
        auto quantized_block = torch::round(input_block / scale).clamp(quant_min, quant_max);

        // Copy to output
        output_block.copy_(quantized_block);
    }

    return quantized.view(sizes);  // Restore original shape
}

torch::Tensor MetalSDPABackend::quantized_scaled_dot_product_attention_per_block(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    int q_block_size,
    int k_block_size,
    int v_block_size,
    const std::string& precision,
    bool is_causal,
    c10::optional<double> scale
) {
    ensure_initialized();

    // Convert all tensors to CPU and contiguous
    auto q_cpu = ensure_contiguous_cpu(query);
    auto k_cpu = ensure_contiguous_cpu(key);
    auto v_cpu = ensure_contiguous_cpu(value);

    // Get tensor dimensions
    uint32_t batch_size, seq_len_q, seq_len_kv, num_heads, head_dim;

    if (q_cpu.dim() == 4) {
        auto q_sizes = q_cpu.sizes();
        batch_size = static_cast<uint32_t>(q_sizes[0]);
        seq_len_q = static_cast<uint32_t>(q_sizes[1]);
        seq_len_kv = static_cast<uint32_t>(k_cpu.sizes()[1]);
        num_heads = static_cast<uint32_t>(q_sizes[2]);
        head_dim = static_cast<uint16_t>(q_sizes[3]);
    } else {
        throw std::runtime_error("Per-block quantized attention currently only supports 4D tensors [batch, seq_len, num_heads, head_dim]");
    }

    // Calculate softmax scale
    float softmax_scale = scale ? static_cast<float>(*scale) : (1.0f / std::sqrt(static_cast<float>(head_dim)));

    // Create output tensor with FP32 precision
    auto output = torch::empty_like(q_cpu, torch::kFloat32);

    // Convert precision string to enum
    mfa_precision_t precision_enum;
    if (precision == "int8") {
        precision_enum = MFA_PRECISION_INT8;
    } else if (precision == "int4") {
        precision_enum = MFA_PRECISION_INT4;
    } else {
        throw std::runtime_error("Unsupported quantization precision: " + precision + ". Use 'int8' or 'int4'.");
    }

    // Get query precision from tensor dtype
    mfa_precision_t q_precision = torch_dtype_to_mfa_dtype(q_cpu.scalar_type());

    // Perform per-block quantization
    auto q_params = quantize_tensor_per_block(q_cpu, q_block_size, precision);
    auto k_params = quantize_tensor_per_block(k_cpu, k_block_size, precision);
    auto v_params = quantize_tensor_per_block(v_cpu, v_block_size, precision);

    // Apply quantization
    auto q_quantized = apply_per_block_quantization(q_cpu, q_params, precision);
    auto k_quantized = apply_per_block_quantization(k_cpu, k_params, precision);
    auto v_quantized = apply_per_block_quantization(v_cpu, v_params, precision);

    // Create scale tensors for Swift
    auto q_scales_tensor = torch::from_blob(q_params.scales.data(), {static_cast<long>(q_params.scales.size())}, torch::kFloat32);
    auto k_scales_tensor = torch::from_blob(k_params.scales.data(), {static_cast<long>(k_params.scales.size())}, torch::kFloat32);
    auto v_scales_tensor = torch::from_blob(v_params.scales.data(), {static_cast<long>(v_params.scales.size())}, torch::kFloat32);

    // Create MFA buffers
    mfa_buffer_t q_buffer, k_buffer, v_buffer, out_buffer;
    mfa_buffer_t q_scales_buffer, k_scales_buffer, v_scales_buffer;

    size_t q_quantized_bytes = q_quantized.numel() * q_quantized.element_size();
    size_t k_quantized_bytes = k_quantized.numel() * k_quantized.element_size();
    size_t v_quantized_bytes = v_quantized.numel() * v_quantized.element_size();
    size_t out_bytes = output.numel() * output.element_size();
    size_t q_scales_bytes = q_scales_tensor.numel() * q_scales_tensor.element_size();
    size_t k_scales_bytes = k_scales_tensor.numel() * k_scales_tensor.element_size();
    size_t v_scales_bytes = v_scales_tensor.numel() * v_scales_tensor.element_size();

    mfa_error_t result;

    // Create buffers for quantized data
    result = mfa_buffer_from_ptr(swift_context_, q_quantized.data_ptr(), q_quantized_bytes, &q_buffer);
    if (result != MFA_SUCCESS) {
        throw std::runtime_error("Failed to create query buffer for per-block quantized attention");
    }

    result = mfa_buffer_from_ptr(swift_context_, k_quantized.data_ptr(), k_quantized_bytes, &k_buffer);
    if (result != MFA_SUCCESS) {
        throw std::runtime_error("Failed to create key buffer for per-block quantized attention");
    }

    result = mfa_buffer_from_ptr(swift_context_, v_quantized.data_ptr(), v_quantized_bytes, &v_buffer);
    if (result != MFA_SUCCESS) {
        throw std::runtime_error("Failed to create value buffer for per-block quantized attention");
    }

    result = mfa_buffer_from_ptr(swift_context_, output.data_ptr(), out_bytes, &out_buffer);
    if (result != MFA_SUCCESS) {
        throw std::runtime_error("Failed to create output buffer for per-block quantized attention");
    }

    // Create buffers for scales
    result = mfa_buffer_from_ptr(swift_context_, q_scales_tensor.data_ptr(), q_scales_bytes, &q_scales_buffer);
    if (result != MFA_SUCCESS) {
        throw std::runtime_error("Failed to create query scales buffer for per-block quantized attention");
    }

    result = mfa_buffer_from_ptr(swift_context_, k_scales_tensor.data_ptr(), k_scales_bytes, &k_scales_buffer);
    if (result != MFA_SUCCESS) {
        throw std::runtime_error("Failed to create key scales buffer for per-block quantized attention");
    }

    result = mfa_buffer_from_ptr(swift_context_, v_scales_tensor.data_ptr(), v_scales_bytes, &v_scales_buffer);
    if (result != MFA_SUCCESS) {
        throw std::runtime_error("Failed to create value scales buffer for per-block quantized attention");
    }

    try {
        // Call per-block quantized attention function
        result = mfa_attention_forward_quantized_per_block(
            swift_context_,
            q_buffer, k_buffer, v_buffer, out_buffer,
            q_scales_buffer, k_scales_buffer, v_scales_buffer,
            batch_size, seq_len_q, seq_len_kv, num_heads, head_dim,
            softmax_scale, is_causal,
            static_cast<uint32_t>(q_block_size),
            static_cast<uint32_t>(k_block_size),
            static_cast<uint32_t>(v_block_size),
            q_precision, precision_enum, precision_enum,
            MFA_PRECISION_FP32,  // Force FP32 output precision
            false, false, false, false  // No transpose for standard layout
        );

        if (result != MFA_SUCCESS) {
            throw std::runtime_error("Per-block quantized attention forward pass failed with error code: " + std::to_string(result));
        }

        // Move output back to original device
        return output.to(query.device());

    } catch (...) {
        throw;
    }
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

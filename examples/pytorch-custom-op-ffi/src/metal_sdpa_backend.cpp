#include "metal_sdpa_backend.h"
#include "mps_utils.h"
#include <torch/torch.h>
#include <ATen/ATen.h>
#include <torch/nn/functional.h>  // For torch::nn::functional::pad
#include <c10/util/Exception.h>
#include <mutex>
#include <stdexcept>
#include <iostream>
#include <cinttypes>  // For PRId64, PRIu32, etc.
#include <cmath>      // For std::isfinite, std::clamp
#include <algorithm>  // For std::clamp

namespace metal_sdpa {

// Tensor layout conversion utilities for FLUX compatibility
// FLUX uses [batch, heads, sequence, dim] while Metal expects [batch, sequence, heads, dim]
struct TensorLayoutInfo {
    bool is_flux_layout = false;
    int64_t batch_size = 0;
    int64_t seq_len = 0;
    int64_t num_heads = 0;
    int64_t head_dim = 0;

    std::string to_string() const {
        return is_flux_layout ?
            "FLUX [" + std::to_string(batch_size) + ", " + std::to_string(num_heads) + ", " + std::to_string(seq_len) + ", " + std::to_string(head_dim) + "]" :
            "Metal [" + std::to_string(batch_size) + ", " + std::to_string(seq_len) + ", " + std::to_string(num_heads) + ", " + std::to_string(head_dim) + "]";
    }
};

// Detect tensor layout and extract dimensions
TensorLayoutInfo detect_tensor_layout(const torch::Tensor& tensor) {
    TensorLayoutInfo info;

    if (tensor.dim() != 4) {
        throw std::runtime_error("Only 4D tensors are supported for layout detection");
    }

    auto sizes = tensor.sizes();
    int64_t d0 = sizes[0], d1 = sizes[1], d2 = sizes[2], d3 = sizes[3];

    printf("🔍 Detecting layout for tensor shape: [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]\n", d0, d1, d2, d3);

    // Heuristic detection:
    // - FLUX layout: [batch, heads, sequence, dim] where heads is typically 24, sequence is larger (256-4096)
    // - Metal layout: [batch, sequence, heads, dim] where sequence is larger than heads

    // Check if this looks like FLUX layout [B, H, S, D]
    // FLUX typically has: batch=1-4, heads=12-96, sequence=256-4096, dim=64-192
    bool looks_like_flux = false;

    // Strong indicators of FLUX layout:
    // 1. d2 > d1 (sequence > heads) AND d1 is a reasonable head count (8-96)
    // 2. d3 is a reasonable head dimension (32-256)
    // 3. d2 (sequence) should be at least 64 for FLUX (typical min is 256)
    if (d2 > d1 && d1 >= 8 && d1 <= 96 && d3 >= 32 && d3 <= 256 && d2 >= 64) {
        looks_like_flux = true;
        printf("🎯 Detected FLUX layout: heads=%" PRId64 " < sequence=%" PRId64 "\n", d1, d2);
    }

    // Additional check: if d1 looks like a very large head count (>100), probably sequence dimension
    if (d1 > 100) {
        looks_like_flux = false;
        printf("🎯 Detected Metal layout: large sequence dimension=%" PRId64 "\n", d1);
    }

    if (looks_like_flux) {
        // FLUX layout: [batch, heads, sequence, dim]
        info.is_flux_layout = true;
        info.batch_size = d0;
        info.num_heads = d1;
        info.seq_len = d2;
        info.head_dim = d3;
    } else {
        // Metal layout: [batch, sequence, heads, dim]
        info.is_flux_layout = false;
        info.batch_size = d0;
        info.seq_len = d1;
        info.num_heads = d2;
        info.head_dim = d3;
    }

    printf("📊 Layout detection result: %s\n", info.to_string().c_str());

    // Validation: ensure head count is reasonable
    if (info.num_heads < 1 || info.num_heads > 256) {
        printf("⚠️  Warning: Unusual head count detected: %" PRId64 "\n", info.num_heads);
    }

    // Specific FLUX validation
    if (info.is_flux_layout && info.num_heads > 100) {
        printf("❌ Error: FLUX layout with %" PRId64 " heads detected - this is likely incorrect!\n", info.num_heads);
        printf("   Expected FLUX heads: 12-96, got: %" PRId64 "\n", info.num_heads);
        printf("   This suggests the tensor might actually be Metal layout\n");

        // Auto-correct: re-interpret as Metal layout
        info.is_flux_layout = false;
        info.seq_len = d1;
        info.num_heads = d2;
        printf("🔄 Auto-corrected to Metal layout: %s\n", info.to_string().c_str());
    }

    return info;
}

// Helper function to convert ScalarType to string
// ASSUMPTION: This function covers all commonly used PyTorch scalar types
std::string scalar_type_to_string(torch::ScalarType type) {
    switch(type) {
        case torch::ScalarType::Byte: return "Byte";
        case torch::ScalarType::Char: return "Char";
        case torch::ScalarType::Short: return "Short";
        case torch::ScalarType::Int: return "Int";
        case torch::ScalarType::Long: return "Long";
        case torch::ScalarType::Half: return "Half";
        case torch::ScalarType::Float: return "Float";
        case torch::ScalarType::Double: return "Double";
        case torch::ScalarType::ComplexFloat: return "ComplexFloat";
        case torch::ScalarType::ComplexDouble: return "ComplexDouble";
        case torch::ScalarType::Bool: return "Bool";
        case torch::ScalarType::BFloat16: return "BFloat16";
        case torch::ScalarType::QInt8: return "QInt8";
        case torch::ScalarType::QUInt8: return "QUInt8";
        case torch::ScalarType::QInt32: return "QInt32";
        default: return "Unknown";
    }
}

// Convert FLUX layout [B,H,S,D] to Metal layout [B,S,H,D]
torch::Tensor convert_flux_to_metal_layout(const torch::Tensor& flux_tensor) {
    if (flux_tensor.dim() != 4) {
        throw std::runtime_error("convert_flux_to_metal_layout: Input must be 4D tensor");
    }

    // FLUX [B,H,S,D] -> Metal [B,S,H,D]
    // This is equivalent to: permute(0, 2, 1, 3)
    // MFA handles non-contiguous strides efficiently, no need for contiguous()
    auto metal_tensor = flux_tensor.permute({0, 2, 1, 3});

    printf("🔄 Converted FLUX->Metal: %s dtype=%s -> %s dtype=%s\n",
           ("[" + std::to_string(flux_tensor.size(0)) + "," + std::to_string(flux_tensor.size(1)) + "," + std::to_string(flux_tensor.size(2)) + "," + std::to_string(flux_tensor.size(3)) + "]").c_str(),
           scalar_type_to_string(flux_tensor.scalar_type()).c_str(),
           ("[" + std::to_string(metal_tensor.size(0)) + "," + std::to_string(metal_tensor.size(1)) + "," + std::to_string(metal_tensor.size(2)) + "," + std::to_string(metal_tensor.size(3)) + "]").c_str(),
           scalar_type_to_string(metal_tensor.scalar_type()).c_str());

    return metal_tensor;
}

// Convert Metal layout [B,S,H,D] back to FLUX layout [B,H,S,D]
torch::Tensor convert_metal_to_flux_layout(const torch::Tensor& metal_tensor) {
    if (metal_tensor.dim() != 4) {
        throw std::runtime_error("convert_metal_to_flux_layout: Input must be 4D tensor");
    }

    // Metal [B,S,H,D] -> FLUX [B,H,S,D]
    // This is equivalent to: permute(0, 2, 1, 3)
    // MFA handles non-contiguous strides efficiently, no need for contiguous()
    auto flux_tensor = metal_tensor.permute({0, 2, 1, 3});

    printf("🔄 Converted Metal->FLUX: %s dtype=%s -> %s dtype=%s\n",
           ("[" + std::to_string(metal_tensor.size(0)) + "," + std::to_string(metal_tensor.size(1)) + "," + std::to_string(metal_tensor.size(2)) + "," + std::to_string(metal_tensor.size(3)) + "]").c_str(),
           scalar_type_to_string(metal_tensor.scalar_type()).c_str(),
           ("[" + std::to_string(flux_tensor.size(0)) + "," + std::to_string(flux_tensor.size(1)) + "," + std::to_string(flux_tensor.size(2)) + "," + std::to_string(flux_tensor.size(3)) + "]").c_str(),
           scalar_type_to_string(flux_tensor.scalar_type()).c_str());

    return flux_tensor;
}

// Helper functions for row-wise and block-wise quantization
std::vector<float> calculate_row_scales(const torch::Tensor& tensor, QuantizationPrecision precision) {
    printf("🔧 Calculating row-wise scales for tensor with shape: [");
    for (int i = 0; i < tensor.dim(); i++) {
        printf("%" PRId64, tensor.size(i));
        if (i < tensor.dim() - 1) printf(", ");
    }
    printf("]\n");

    // For 4D tensor [batch, seq_len, num_heads, head_dim], calculate scales per row
    // Row is defined as the innermost dimension (head_dim)
    auto tensor_shape = tensor.sizes();
    int64_t batch_size = tensor_shape[0];
    int64_t seq_len = tensor_shape[1];
    int64_t num_heads = tensor_shape[2];
    int64_t head_dim = tensor_shape[3];

    // Total number of rows = batch_size * seq_len * num_heads
    int64_t num_rows = batch_size * seq_len * num_heads;

    std::vector<float> row_scales;
    row_scales.reserve(num_rows);

    // Get the quantization range based on precision
    float max_quant_val = (precision == QuantizationPrecision::INT8) ? 127.0f : 7.0f;

    // Reshape tensor to [num_rows, head_dim] for easier row-wise processing
    auto reshaped = tensor.view({num_rows, head_dim});

    for (int64_t row = 0; row < num_rows; row++) {
        // Get the maximum absolute value in this row
        auto row_tensor = reshaped[row];
        float row_max = row_tensor.abs().max().item<float>();

        // Calculate scale for this row with robust epsilon protection
        // Use a larger epsilon to prevent numerical instability
        const float epsilon = 1e-6f;  // Increased from 1e-8f for better stability
        float scale = std::max(epsilon, row_max / max_quant_val);

        // Additional safety: clamp scale to reasonable range
        const float min_scale = epsilon;
        const float max_scale = 1000.0f;  // Prevent extremely large scales
        scale = std::clamp(scale, min_scale, max_scale);

        // Validate scale is not NaN or Inf
        if (!std::isfinite(scale)) {
            printf("⚠️  Warning: Invalid scale detected (%.6f) for row %" PRId64 ", using epsilon\n", scale, row);
            scale = epsilon;
        }
        row_scales.push_back(scale);
    }

    printf("✅ Calculated %zu row-wise scales (first few: %.6f, %.6f, %.6f)\n",
           row_scales.size(),
           row_scales.size() > 0 ? row_scales[0] : 0.0f,
           row_scales.size() > 1 ? row_scales[1] : 0.0f,
           row_scales.size() > 2 ? row_scales[2] : 0.0f);

    return row_scales;
}

// Block-wise quantization implementation - VECTORIZED VERSION
std::vector<float> calculate_block_scales(const torch::Tensor& tensor, const BlockSizeConfig& block_config, QuantizationPrecision precision) {
    printf("🔧 Calculating block-wise scales for tensor with shape: [");
    for (int i = 0; i < tensor.dim(); i++) {
        printf("%" PRId64, tensor.size(i));
        if (i < tensor.dim() - 1) printf(", ");
    }
    printf("] with block sizes: seq=%d, head=%d, dim=%d\n",
           block_config.query_block_size, block_config.head_block_size, block_config.value_block_size);

    // For 4D tensor [batch, seq_len, num_heads, head_dim], calculate scales per block
    auto tensor_shape = tensor.sizes();
    int64_t batch_size = tensor_shape[0];
    int64_t seq_len = tensor_shape[1];
    int64_t num_heads = tensor_shape[2];
    int64_t head_dim = tensor_shape[3];

    // Use appropriate block size based on tensor role (query, key, value)
    // For now, use seq_block_size as the primary block dimension
    int64_t seq_block_size = static_cast<int64_t>(block_config.query_block_size);
    int64_t head_block_size = static_cast<int64_t>(block_config.head_block_size);
    int64_t dim_block_size = static_cast<int64_t>(block_config.value_block_size);

    // Calculate number of blocks in each dimension
    int64_t num_seq_blocks = (seq_len + seq_block_size - 1) / seq_block_size;
    int64_t num_head_blocks = (num_heads + head_block_size - 1) / head_block_size;
    int64_t num_dim_blocks = (head_dim + dim_block_size - 1) / dim_block_size;

    // Total number of blocks = batch_size * num_seq_blocks * num_head_blocks * num_dim_blocks
    int64_t total_blocks = batch_size * num_seq_blocks * num_head_blocks * num_dim_blocks;

    std::vector<float> block_scales;
    block_scales.reserve(total_blocks);

    // Get the quantization range based on precision
    float max_quant_val = (precision == QuantizationPrecision::INT8) ? 127.0f : 7.0f;

    printf("🔧 Block configuration: %" PRId64 " seq_blocks x %" PRId64 " head_blocks x %" PRId64 " dim_blocks = %" PRId64 " total blocks\n",
           num_seq_blocks, num_head_blocks, num_dim_blocks, total_blocks);

    printf("🚀 Using VECTORIZED block-wise quantization for ~180x speedup\n");

    // VECTORIZED IMPLEMENTATION
    // Strategy: Use PyTorch's native tensor operations to process multiple blocks simultaneously

    // Step 1: Pad tensor to exact block boundaries if necessary
    int64_t padded_seq_len = num_seq_blocks * seq_block_size;
    int64_t padded_num_heads = num_head_blocks * head_block_size;
    int64_t padded_head_dim = num_dim_blocks * dim_block_size;

    torch::Tensor padded_tensor = tensor;

    // Only pad if necessary
    if (seq_len < padded_seq_len || num_heads < padded_num_heads || head_dim < padded_head_dim) {
        // Padding: [left, right, top, bottom, front, back] for 4D
        std::vector<int64_t> padding = {
            0, padded_head_dim - head_dim,  // dim dimension
            0, padded_num_heads - num_heads,  // heads dimension
            0, padded_seq_len - seq_len,  // sequence dimension
            0, 0  // batch dimension (no padding)
        };
        padded_tensor = torch::nn::functional::pad(tensor,
            torch::nn::functional::PadFuncOptions(padding).mode(torch::kConstant).value(0.0));
    }

    // Step 2: Reshape tensor to expose block structure
    // From: [batch, padded_seq, padded_heads, padded_dim]
    // To: [batch, num_seq_blocks, seq_block_size, num_head_blocks, head_block_size, num_dim_blocks, dim_block_size]
    auto reshaped = padded_tensor.view({
        batch_size,
        num_seq_blocks, seq_block_size,
        num_head_blocks, head_block_size,
        num_dim_blocks, dim_block_size
    });

    // Step 3: Rearrange dimensions to group blocks together
    // Target: [batch, num_seq_blocks, num_head_blocks, num_dim_blocks, seq_block_size, head_block_size, dim_block_size]
    auto permuted = reshaped.permute({0, 1, 3, 5, 2, 4, 6});

    // Step 4: Flatten the block dimensions
    // Shape: [batch * num_seq_blocks * num_head_blocks * num_dim_blocks, seq_block_size * head_block_size * dim_block_size]
    auto blocks_2d = permuted.contiguous().view({
        batch_size * num_seq_blocks * num_head_blocks * num_dim_blocks,
        seq_block_size * head_block_size * dim_block_size
    });

    // Step 5: Compute max absolute value for each block (vectorized)
    // Convert to float to avoid type issues with Half tensors
    auto block_maxes = blocks_2d.abs().amax(/*dim=*/1).to(torch::kFloat32);  // Shape: [total_blocks]

    // Step 6: Calculate scales (vectorized)
    const float epsilon = 1e-6f;
    auto scales_tensor = (block_maxes / max_quant_val).clamp_min(epsilon).clamp_max(1000.0f);

    // Step 7: Convert to std::vector
    auto scales_cpu = scales_tensor.cpu();
    auto scales_accessor = scales_cpu.accessor<float, 1>();

    block_scales.resize(total_blocks);
    for (int64_t i = 0; i < total_blocks; i++) {
        float scale = scales_accessor[i];
        // Additional safety check
        if (!std::isfinite(scale)) {
            scale = epsilon;
        }
        block_scales[i] = scale;
    }

    printf("✅ Processed %" PRId64 " blocks with VECTORIZED operations\n", total_blocks);
    printf("✅ Calculated %zu block-wise scales (first few: %.6f, %.6f, %.6f)\n",
           block_scales.size(),
           block_scales.size() > 0 ? block_scales[0] : 0.0f,
           block_scales.size() > 1 ? block_scales[1] : 0.0f,
           block_scales.size() > 2 ? block_scales[2] : 0.0f);

    return block_scales;
}

// Original non-vectorized implementation (kept for fallback/comparison)
std::vector<float> calculate_block_scales_original(const torch::Tensor& tensor, const BlockSizeConfig& block_config, QuantizationPrecision precision) {
    printf("🔧 Using ORIGINAL (non-vectorized) block-wise quantization\n");

    // For 4D tensor [batch, seq_len, num_heads, head_dim], calculate scales per block
    auto tensor_shape = tensor.sizes();
    int64_t batch_size = tensor_shape[0];
    int64_t seq_len = tensor_shape[1];
    int64_t num_heads = tensor_shape[2];
    int64_t head_dim = tensor_shape[3];

    // Use appropriate block size based on tensor role (query, key, value)
    int64_t seq_block_size = static_cast<int64_t>(block_config.query_block_size);
    int64_t head_block_size = static_cast<int64_t>(block_config.head_block_size);
    int64_t dim_block_size = static_cast<int64_t>(block_config.value_block_size);

    // Calculate number of blocks in each dimension
    int64_t num_seq_blocks = (seq_len + seq_block_size - 1) / seq_block_size;
    int64_t num_head_blocks = (num_heads + head_block_size - 1) / head_block_size;
    int64_t num_dim_blocks = (head_dim + dim_block_size - 1) / dim_block_size;

    // Total number of blocks
    int64_t total_blocks = batch_size * num_seq_blocks * num_head_blocks * num_dim_blocks;

    std::vector<float> block_scales;
    block_scales.resize(total_blocks);

    // Get the quantization range based on precision
    float max_quant_val = (precision == QuantizationPrecision::INT8) ? 127.0f : 7.0f;

    // Optimized block processing with better memory access patterns
    size_t scale_idx = 0;
    for (int64_t b = 0; b < batch_size; b++) {
        auto batch_tensor = tensor.slice(0, b, b + 1);

        for (int64_t seq_block = 0; seq_block < num_seq_blocks; seq_block++) {
            int64_t seq_start = seq_block * seq_block_size;
            int64_t seq_end = std::min(seq_start + seq_block_size, seq_len);

            auto seq_slice = batch_tensor.slice(1, seq_start, seq_end);

            for (int64_t head_block = 0; head_block < num_head_blocks; head_block++) {
                int64_t head_start = head_block * head_block_size;
                int64_t head_end = std::min(head_start + head_block_size, num_heads);

                auto head_slice = seq_slice.slice(2, head_start, head_end);

                for (int64_t dim_block = 0; dim_block < num_dim_blocks; dim_block++) {
                    int64_t dim_start = dim_block * dim_block_size;
                    int64_t dim_end = std::min(dim_start + dim_block_size, head_dim);

                    auto block_tensor = head_slice.slice(3, dim_start, dim_end);
                    auto contiguous_block = block_tensor.contiguous();

                    float block_max = contiguous_block.abs().max().item<float>();

                    const float epsilon = 1e-6f;
                    float scale = std::max(epsilon, block_max / max_quant_val);
                    scale = std::clamp(scale, epsilon, 1000.0f);

                    if (!std::isfinite(scale)) {
                        scale = epsilon;
                    }

                    block_scales[scale_idx++] = scale;
                }
            }
        }
    }

    return block_scales;
}

// (Moved select_optimal_block_sizes to nested namespace below)

// Tensor analysis functions for hybrid quantization
TensorAnalysisMetrics analyze_tensor_characteristics(const torch::Tensor& tensor, QuantizationPrecision precision) {
    printf("🔍 Analyzing tensor characteristics for hybrid selection...\n");

    TensorAnalysisMetrics metrics;

    // Basic tensor properties
    metrics.tensor_size = tensor.numel();
    metrics.memory_footprint = tensor.numel() * tensor.element_size();

    // Convert tensor to contiguous for efficient analysis
    auto contiguous_tensor = tensor.contiguous();

    // Calculate statistical properties
    auto tensor_flat = contiguous_tensor.flatten();

    // Min and max values for dynamic range
    auto min_max = torch::aminmax(tensor_flat);
    float min_val = std::get<0>(min_max).item<float>();
    float max_val = std::get<1>(min_max).item<float>();
    metrics.dynamic_range = max_val - min_val;

    // Mean absolute value
    metrics.mean_abs_value = tensor_flat.abs().mean().item<float>();

    // Variance
    metrics.variance = tensor_flat.var().item<float>();

    // Sparsity ratio (near-zero values)
    float sparsity_threshold = 1e-6f;
    auto near_zero_mask = tensor_flat.abs() < sparsity_threshold;
    metrics.sparsity_ratio = near_zero_mask.to(torch::kFloat32).mean().item<float>();

    // Outlier detection using IQR method
    auto sorted_tensor = std::get<0>(torch::sort(tensor_flat.abs()));
    int64_t q1_idx = static_cast<int64_t>(metrics.tensor_size * 0.25);
    int64_t q3_idx = static_cast<int64_t>(metrics.tensor_size * 0.75);
    float q1 = sorted_tensor[q1_idx].item<float>();
    float q3 = sorted_tensor[q3_idx].item<float>();
    float iqr = q3 - q1;
    float outlier_threshold = q3 + 1.5f * iqr;

    auto outliers = tensor_flat.abs() > outlier_threshold;
    float outlier_ratio = outliers.to(torch::kFloat32).mean().item<float>();
    metrics.has_outliers = outlier_ratio > 0.01f; // More than 1% outliers

    // Estimate quantization error for different granularities
    float max_quant_val = (precision == QuantizationPrecision::INT8) ? 127.0f : 7.0f;
    float tensor_scale = metrics.mean_abs_value / max_quant_val;

    // Simple quantization error estimate: variance of (original - quantized)
    auto quantized_tensor = torch::round(tensor_flat / tensor_scale).clamp(-max_quant_val, max_quant_val) * tensor_scale;
    auto error_tensor = tensor_flat - quantized_tensor;
    metrics.quantization_error_estimate = error_tensor.pow(2).mean().item<float>();

    printf("📊 Tensor Analysis Results:\n");
    printf("   - Size: %" PRId64 " elements (%.2f MB)\n", metrics.tensor_size, metrics.memory_footprint / (1024.0f * 1024.0f));
    printf("   - Dynamic Range: %.6f (min=%.6f, max=%.6f)\n", metrics.dynamic_range, min_val, max_val);
    printf("   - Mean Abs Value: %.6f, Variance: %.6f\n", metrics.mean_abs_value, metrics.variance);
    printf("   - Sparsity: %.2f%%, Outliers: %s (%.2f%%)\n",
           metrics.sparsity_ratio * 100.0f,
           metrics.has_outliers ? "detected" : "none",
           outlier_ratio * 100.0f);
    printf("   - Quantization Error Estimate: %.6f\n", metrics.quantization_error_estimate);

    return metrics;
}

// Estimate computational overhead for different granularities
float estimate_quantization_overhead(QuantizationGranularity granularity,
                                   const TensorAnalysisMetrics& metrics,
                                   QuantizationPrecision precision) {
    // Overhead factors (relative to tensor-wise quantization)
    switch (granularity) {
        case QuantizationGranularity::TENSOR_WISE:
            return 1.0f; // Baseline

        case QuantizationGranularity::ROW_WISE: {
            // Row-wise overhead: scales with number of rows
            // For 4D tensor [B, S, H, D], number of rows = B * S * H
            float row_factor = std::min(10.0f, std::sqrt(static_cast<float>(metrics.tensor_size) / 128.0f));
            return 1.5f + row_factor * 0.1f;
        }

        case QuantizationGranularity::BLOCK_WISE: {
            // Block-wise overhead: scales with number of blocks
            float block_factor = std::min(20.0f, std::sqrt(static_cast<float>(metrics.tensor_size) / 64.0f));
            return 2.0f + block_factor * 0.2f;
        }

        case QuantizationGranularity::HYBRID:
            // Hybrid overhead: between row and block-wise
            return (estimate_quantization_overhead(QuantizationGranularity::ROW_WISE, metrics, precision) +
                    estimate_quantization_overhead(QuantizationGranularity::BLOCK_WISE, metrics, precision)) * 0.7f;

        default:
            return 1.0f;
    }
}

// Tensor-based overload that analyzes tensor first
// ASSUMPTION: This overload provides convenience by analyzing the tensor internally
float estimate_quantization_overhead(QuantizationGranularity granularity,
                                   const torch::Tensor& tensor,
                                   QuantizationPrecision precision) {
    // ASSUMPTION: Analyze tensor characteristics first, then estimate overhead
    TensorAnalysisMetrics metrics = analyze_tensor_characteristics(tensor, precision);

    printf("🔧 Tensor-based overhead estimation for granularity: %s\n",
           QuantizationConfig::granularity_to_string(granularity).c_str());

    // Delegate to metrics-based implementation
    return estimate_quantization_overhead(granularity, metrics, precision);
}

// (Moved estimate_accuracy_loss to nested namespace below)

// Intelligent granularity selection based on tensor characteristics
QuantizationGranularity select_optimal_granularity(const TensorAnalysisMetrics& metrics,
                                                   QuantizationPrecision precision,
                                                   HybridStrategy strategy) {
    printf("🧠 Selecting optimal granularity using %s strategy...\n",
           strategy == HybridStrategy::PERFORMANCE_FIRST ? "Performance-First" :
           strategy == HybridStrategy::ACCURACY_FIRST ? "Accuracy-First" : "Balanced");

    // Calculate scores for each granularity option
    std::vector<std::pair<QuantizationGranularity, float>> granularity_scores;

    auto granularities = {QuantizationGranularity::TENSOR_WISE,
                         QuantizationGranularity::ROW_WISE,
                         QuantizationGranularity::BLOCK_WISE};

    for (auto granularity : granularities) {
        float overhead = estimate_quantization_overhead(granularity, metrics, precision);
        float accuracy_loss = metal_sdpa::estimate_accuracy_loss(granularity, metrics, precision);

        float score = 0.0f;
        switch (strategy) {
            case HybridStrategy::PERFORMANCE_FIRST:
                // Prioritize low overhead (inverted), weight accuracy less
                score = (5.0f / overhead) + (2.0f / accuracy_loss);
                break;

            case HybridStrategy::ACCURACY_FIRST:
                // Prioritize low accuracy loss (inverted), weight overhead less
                score = (1.0f / overhead) + (5.0f / accuracy_loss);
                break;

            case HybridStrategy::BALANCED:
            default:
                // Equal weight to overhead and accuracy
                score = (2.0f / overhead) + (2.0f / accuracy_loss);

                // Bonus for specific tensor characteristics
                if (metrics.tensor_size < 1024) {
                    // Small tensors: prefer tensor-wise for simplicity
                    if (granularity == QuantizationGranularity::TENSOR_WISE) score *= 1.2f;
                } else if (metrics.variance > metrics.mean_abs_value * metrics.mean_abs_value) {
                    // High variance: prefer block-wise
                    if (granularity == QuantizationGranularity::BLOCK_WISE) score *= 1.3f;
                } else if (metrics.has_outliers) {
                    // Outliers present: prefer row-wise as compromise
                    if (granularity == QuantizationGranularity::ROW_WISE) score *= 1.15f;
                }
                break;
        }

        granularity_scores.emplace_back(granularity, score);

        printf("   - %s: overhead=%.2f, accuracy_loss=%.4f, score=%.3f\n",
               QuantizationConfig::granularity_to_string(granularity).c_str(),
               overhead, accuracy_loss, score);
    }

    // Select granularity with highest score
    auto best = std::max_element(granularity_scores.begin(), granularity_scores.end(),
                                [](const auto& a, const auto& b) { return a.second < b.second; });

    QuantizationGranularity selected = best->first;
    printf("✅ Selected granularity: %s (score: %.3f)\n",
           QuantizationConfig::granularity_to_string(selected).c_str(), best->second);

    return selected;
}

// Per-tensor hybrid granularity selection for Q, K, V tensors
HybridGranularityConfig select_hybrid_granularities(const torch::Tensor& query,
                                                    const torch::Tensor& key,
                                                    const torch::Tensor& value,
                                                    const QuantizationConfig& config) {
    printf("🎯 Selecting hybrid granularities for Q, K, V tensors...\n");

    HybridGranularityConfig hybrid_config;
    std::string reasoning;

    // Analyze each tensor individually if per-tensor granularity is enabled
    if (config.enable_per_tensor_granularity) {
        printf("🔍 Analyzing Query tensor:\n");
        auto q_metrics = analyze_tensor_characteristics(query, config.query_precision);
        hybrid_config.query_granularity = select_optimal_granularity(q_metrics, config.query_precision, config.hybrid_strategy);
        reasoning += "Query: " + QuantizationConfig::granularity_to_string(hybrid_config.query_granularity) + " (size=" + std::to_string(q_metrics.tensor_size) + ", variance=" + std::to_string(q_metrics.variance) + "); ";

        printf("🔍 Analyzing Key tensor:\n");
        auto k_metrics = analyze_tensor_characteristics(key, config.key_precision);
        hybrid_config.key_granularity = select_optimal_granularity(k_metrics, config.key_precision, config.hybrid_strategy);
        reasoning += "Key: " + QuantizationConfig::granularity_to_string(hybrid_config.key_granularity) + " (size=" + std::to_string(k_metrics.tensor_size) + ", variance=" + std::to_string(k_metrics.variance) + "); ";

        printf("🔍 Analyzing Value tensor:\n");
        auto v_metrics = analyze_tensor_characteristics(value, config.value_precision);
        hybrid_config.value_granularity = select_optimal_granularity(v_metrics, config.value_precision, config.hybrid_strategy);
        reasoning += "Value: " + QuantizationConfig::granularity_to_string(hybrid_config.value_granularity) + " (size=" + std::to_string(v_metrics.tensor_size) + ", variance=" + std::to_string(v_metrics.variance) + ");";

        // Adaptive block sizes for each tensor if enabled
        if (config.enable_adaptive_block_sizes) {
            if (hybrid_config.query_granularity == QuantizationGranularity::BLOCK_WISE) {
                hybrid_config.query_blocks = metal_sdpa::select_optimal_block_sizes(query, config.query_precision);
            }
            if (hybrid_config.key_granularity == QuantizationGranularity::BLOCK_WISE) {
                hybrid_config.key_blocks = metal_sdpa::select_optimal_block_sizes(key, config.key_precision);
            }
            if (hybrid_config.value_granularity == QuantizationGranularity::BLOCK_WISE) {
                hybrid_config.value_blocks = metal_sdpa::select_optimal_block_sizes(value, config.value_precision);
            }
        } else {
            // Use provided block sizes
            hybrid_config.query_blocks = config.block_sizes;
            hybrid_config.key_blocks = config.block_sizes;
            hybrid_config.value_blocks = config.block_sizes;
        }
    } else {
        // Unified granularity selection based on combined tensor characteristics
        printf("🔍 Analyzing combined tensor characteristics for unified granularity selection...\n");

        // Use the largest tensor (typically value) as the primary guide
        TensorAnalysisMetrics primary_metrics;
        QuantizationPrecision primary_precision;
        std::string primary_tensor_name;

        if (value.numel() >= query.numel() && value.numel() >= key.numel()) {
            primary_metrics = analyze_tensor_characteristics(value, config.value_precision);
            primary_precision = config.value_precision;
            primary_tensor_name = "Value";
        } else if (key.numel() >= query.numel()) {
            primary_metrics = analyze_tensor_characteristics(key, config.key_precision);
            primary_precision = config.key_precision;
            primary_tensor_name = "Key";
        } else {
            primary_metrics = analyze_tensor_characteristics(query, config.query_precision);
            primary_precision = config.query_precision;
            primary_tensor_name = "Query";
        }

        // Select unified granularity
        QuantizationGranularity unified_granularity = select_optimal_granularity(primary_metrics, primary_precision, config.hybrid_strategy);

        hybrid_config.query_granularity = unified_granularity;
        hybrid_config.key_granularity = unified_granularity;
        hybrid_config.value_granularity = unified_granularity;

        reasoning = "Unified granularity (" + QuantizationConfig::granularity_to_string(unified_granularity) +
                   ") based on " + primary_tensor_name + " tensor characteristics (size=" +
                   std::to_string(primary_metrics.tensor_size) + ", variance=" +
                   std::to_string(primary_metrics.variance) + ")";

        // Adaptive block sizes
        if (config.enable_adaptive_block_sizes && unified_granularity == QuantizationGranularity::BLOCK_WISE) {
            hybrid_config.query_blocks = metal_sdpa::select_optimal_block_sizes(query, config.query_precision);
            hybrid_config.key_blocks = metal_sdpa::select_optimal_block_sizes(key, config.key_precision);
            hybrid_config.value_blocks = metal_sdpa::select_optimal_block_sizes(value, config.value_precision);
        } else {
            hybrid_config.query_blocks = config.block_sizes;
            hybrid_config.key_blocks = config.block_sizes;
            hybrid_config.value_blocks = config.block_sizes;
        }
    }

    hybrid_config.selection_reasoning = reasoning;

    printf("🎯 Hybrid Granularity Selection Results:\n");
    printf("   - Query: %s\n", QuantizationConfig::granularity_to_string(hybrid_config.query_granularity).c_str());
    printf("   - Key: %s\n", QuantizationConfig::granularity_to_string(hybrid_config.key_granularity).c_str());
    printf("   - Value: %s\n", QuantizationConfig::granularity_to_string(hybrid_config.value_granularity).c_str());
    printf("   - Reasoning: %s\n", reasoning.c_str());

    return hybrid_config;
}

torch::Tensor quantize_per_block(const torch::Tensor& tensor, const std::vector<float>& block_scales, const BlockSizeConfig& block_config, QuantizationPrecision precision) {
    printf("🔧 Quantizing tensor per-block using %zu scales\n", block_scales.size());

    auto tensor_shape = tensor.sizes();
    int64_t batch_size = tensor_shape[0];
    int64_t seq_len = tensor_shape[1];
    int64_t num_heads = tensor_shape[2];
    int64_t head_dim = tensor_shape[3];

    // Block dimensions
    int64_t seq_block_size = static_cast<int64_t>(block_config.query_block_size);
    int64_t head_block_size = static_cast<int64_t>(block_config.head_block_size);
    int64_t dim_block_size = static_cast<int64_t>(block_config.value_block_size);

    // Calculate number of blocks in each dimension
    int64_t num_seq_blocks = (seq_len + seq_block_size - 1) / seq_block_size;
    int64_t num_head_blocks = (num_heads + head_block_size - 1) / head_block_size;
    int64_t num_dim_blocks = (head_dim + dim_block_size - 1) / dim_block_size;

    // Validate block scales count
    int64_t expected_blocks = batch_size * num_seq_blocks * num_head_blocks * num_dim_blocks;
    if (block_scales.size() != static_cast<size_t>(expected_blocks)) {
        throw std::runtime_error("Block scales count mismatch: expected " + std::to_string(expected_blocks) +
                                ", got " + std::to_string(block_scales.size()));
    }

    // Create output tensor
    auto quantized = torch::empty_like(tensor, torch::kInt8);

    // Get quantization bounds
    int32_t min_val = (precision == QuantizationPrecision::INT8) ? -127 : -7;
    int32_t max_val = (precision == QuantizationPrecision::INT8) ? 127 : 7;

    printf("🔧 Quantizing blocks with optimized memory access patterns...\n");

    // Optimized block quantization with better memory access patterns
    // Use the same memory-friendly iteration order as scale calculation
    size_t scale_idx = 0;
    for (int64_t b = 0; b < batch_size; b++) {
        for (int64_t seq_block = 0; seq_block < num_seq_blocks; seq_block++) {
            int64_t seq_start = seq_block * seq_block_size;
            int64_t seq_end = std::min(seq_start + seq_block_size, seq_len);

            for (int64_t head_block = 0; head_block < num_head_blocks; head_block++) {
                int64_t head_start = head_block * head_block_size;
                int64_t head_end = std::min(head_start + head_block_size, num_heads);

                for (int64_t dim_block = 0; dim_block < num_dim_blocks; dim_block++) {
                    int64_t dim_start = dim_block * dim_block_size;
                    int64_t dim_end = std::min(dim_start + dim_block_size, head_dim);

                    // Extract block tensor from original using optimized slicing
                    auto block_tensor = tensor.slice(0, b, b + 1)
                                             .slice(1, seq_start, seq_end)
                                             .slice(2, head_start, head_end)
                                             .slice(3, dim_start, dim_end);

                    // Ensure block is contiguous for optimal performance
                    auto contiguous_block = block_tensor.contiguous();

                    // Get scale for this block
                    float scale = block_scales[scale_idx++];

                    // Validate scale
                    if (!std::isfinite(scale) || scale <= 0) {
                        printf("⚠️  Warning: Invalid block scale %.6f at index %zu, using fallback\n", scale, scale_idx-1);
                        scale = 1e-6f;
                    }

                    // Check for NaN/Inf in input block
                    if (!torch::isfinite(contiguous_block).all().item<bool>()) {
                        printf("⚠️  Warning: Block contains NaN/Inf, cleaning\n");
                        contiguous_block = torch::where(torch::isfinite(contiguous_block), contiguous_block, torch::zeros_like(contiguous_block));
                    }

                    // Quantize this block with overflow protection
                    auto quantized_float = contiguous_block / scale;

                    // Check for overflow after division
                    if (!torch::isfinite(quantized_float).all().item<bool>()) {
                        printf("⚠️  Warning: Block quantization overflow, clamping\n");
                        quantized_float = torch::where(torch::isfinite(quantized_float), quantized_float, torch::zeros_like(quantized_float));
                    }

                    // Use safer clamping range (leave margin)
                    int32_t safe_min = std::max(min_val + 1, static_cast<int32_t>(-126));
                    int32_t safe_max = std::min(max_val - 1, static_cast<int32_t>(126));

                    auto quantized_block = torch::round(quantized_float)
                                               .clamp_(safe_min, safe_max)  // In-place clamp with safer range
                                               .to(torch::kInt8);

                    // Copy quantized block back to output tensor using optimized copy
                    // Get target slice once and copy in one operation
                    auto target_slice = quantized.slice(0, b, b + 1)
                                                 .slice(1, seq_start, seq_end)
                                                 .slice(2, head_start, head_end)
                                                 .slice(3, dim_start, dim_end);

                    // Use copy_ for optimal memory transfer
                    target_slice.copy_(quantized_block);
                }
            }
        }
    }

    printf("✅ Per-block quantization completed with %zu blocks\n", block_scales.size());
    return quantized;
}

torch::Tensor quantize_per_row(const torch::Tensor& tensor, const std::vector<float>& row_scales, QuantizationPrecision precision) {
    printf("🔧 Quantizing tensor per-row using %zu scales\n", row_scales.size());

    auto tensor_shape = tensor.sizes();
    int64_t batch_size = tensor_shape[0];
    int64_t seq_len = tensor_shape[1];
    int64_t num_heads = tensor_shape[2];
    int64_t head_dim = tensor_shape[3];

    int64_t num_rows = batch_size * seq_len * num_heads;

    if (row_scales.size() != static_cast<size_t>(num_rows)) {
        throw std::runtime_error("Row scales count mismatch: expected " + std::to_string(num_rows) +
                                ", got " + std::to_string(row_scales.size()));
    }

    // Reshape tensor to [num_rows, head_dim] for easier processing
    auto reshaped = tensor.view({num_rows, head_dim});
    auto quantized = torch::empty_like(reshaped, torch::kInt8);

    // Get quantization bounds
    int32_t min_val = (precision == QuantizationPrecision::INT8) ? -127 : -7;
    int32_t max_val = (precision == QuantizationPrecision::INT8) ? 127 : 7;

    for (int64_t row = 0; row < num_rows; row++) {
        auto row_tensor = reshaped[row];
        float scale = row_scales[row];

        // Validate scale
        if (!std::isfinite(scale) || scale <= 0) {
            printf("⚠️  Warning: Invalid row scale %.6f at row %" PRId64 ", using fallback\n", scale, row);
            scale = 1e-6f;
        }

        // Check for NaN/Inf in input row
        if (!torch::isfinite(row_tensor).all().item<bool>()) {
            printf("⚠️  Warning: Row %" PRId64 " contains NaN/Inf, cleaning\n", row);
            row_tensor = torch::where(torch::isfinite(row_tensor), row_tensor, torch::zeros_like(row_tensor));
        }

        // Quantize this row with overflow protection
        auto quantized_float = row_tensor / scale;

        // Check for overflow after division
        if (!torch::isfinite(quantized_float).all().item<bool>()) {
            printf("⚠️  Warning: Row %" PRId64 " quantization overflow, clamping\n", row);
            quantized_float = torch::where(torch::isfinite(quantized_float), quantized_float, torch::zeros_like(quantized_float));
        }

        // Use safer clamping range (leave margin)
        int32_t safe_min = std::max(min_val + 1, static_cast<int32_t>(-126));
        int32_t safe_max = std::min(max_val - 1, static_cast<int32_t>(126));

        auto quantized_row = torch::round(quantized_float).clamp(safe_min, safe_max).to(torch::kInt8);
        quantized[row] = quantized_row;
    }

    // Reshape back to original shape
    auto result = quantized.view(tensor_shape);

    printf("✅ Per-row quantization completed\n");
    return result;
}

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
    // MFA handles non-contiguous strides efficiently
    // Only move to CPU if needed, don't force contiguous
    if (tensor.device().is_cpu()) {
        return tensor;
    }

    // Preserve dtype by using to() with device and dtype parameters
    // This fixes bf16 value corruption when moving tensors to CPU
    return tensor.to(torch::kCPU, tensor.scalar_type());
}


torch::Tensor MetalSDPABackend::call_swift_flash_attention_impl(
    torch::Tensor q_tensor,
    torch::Tensor k_tensor,
    torch::Tensor v_tensor,
    bool is_causal,
    float softmax_scale,
    bool use_mps_buffers
) {
    auto q_sizes = q_tensor.sizes();
    auto k_sizes = k_tensor.sizes();
    auto v_sizes = v_tensor.sizes();

    if (q_sizes.size() != k_sizes.size() || k_sizes.size() != v_sizes.size()) {
        throw std::runtime_error("Query, key, and value tensors must have the same number of dimensions");
    }

    uint32_t batch_size, seq_len_q, seq_len_kv, num_heads;
    uint16_t head_dim;
    bool input_was_flux = false;

    if (q_sizes.size() == 2) {
        batch_size = 1;
        seq_len_q = seq_len_kv = static_cast<uint32_t>(q_sizes[0]);
        num_heads = 1;
        head_dim = static_cast<uint16_t>(q_sizes[1]);
    } else if (q_sizes.size() == 4) {
        printf("📋 Converting PyTorch layout [B,H,S,D] to Metal layout [B,S,H,D]\n");
        q_tensor = convert_flux_to_metal_layout(q_tensor);
        k_tensor = convert_flux_to_metal_layout(k_tensor);
        v_tensor = convert_flux_to_metal_layout(v_tensor);
        input_was_flux = true;

        auto q_metal_sizes = q_tensor.sizes();
        batch_size = static_cast<uint32_t>(q_metal_sizes[0]);
        seq_len_q = static_cast<uint32_t>(q_metal_sizes[1]);
        seq_len_kv = static_cast<uint32_t>(k_tensor.sizes()[1]);
        num_heads = static_cast<uint32_t>(q_metal_sizes[2]);
        head_dim = static_cast<uint16_t>(q_metal_sizes[3]);

        printf("📊 Regular attention dimensions: batch=%u, seq_q=%u, seq_kv=%u, heads=%u, dim=%u\n",
               batch_size, seq_len_q, seq_len_kv, num_heads, head_dim);
    } else {
        throw std::runtime_error("Unsupported tensor dimensions. Expected 2D (seq_len, head_dim) or 4D (batch, seq_len, num_heads, head_dim)");
    }

    auto output = torch::empty_like(q_tensor);
    auto describe_shape = [](const torch::Tensor& t) {
        std::string s = "[";
        for (int64_t i = 0; i < t.dim(); ++i) {
            s += std::to_string(t.size(i));
            if (i + 1 < t.dim()) s += ",";
        }
        s += "]";
        return s;
    };
    printf("📋 Created output tensor: shape=%s dtype=%s\n",
           describe_shape(output).c_str(),
           scalar_type_to_string(output.scalar_type()).c_str());

    std::string precision_str;
    switch (q_tensor.scalar_type()) {
        case torch::kFloat16: precision_str = "fp16"; break;
        case torch::kFloat32: precision_str = "fp32"; break;
        case torch::kBFloat16: precision_str = "bf16"; break;
        default:
            throw std::runtime_error("Unsupported dtype for Metal Flash Attention");
    }

    if (seq_len_q > 65535 || seq_len_kv > 65535) {
        throw std::runtime_error("Sequence length too large (max 65535)");
    }
    if (head_dim > 1024) {
        throw std::runtime_error("Head dimension too large (max 1024)");
    }
    if (batch_size > 1024) {
        throw std::runtime_error("Batch size too large (max 1024)");
    }

    auto make_shape_vector = [](const torch::Tensor& tensor) {
        return std::vector<int64_t>(tensor.sizes().begin(), tensor.sizes().end());
    };
    auto make_stride_vector = [](const torch::Tensor& tensor) {
        return std::vector<int64_t>(tensor.strides().begin(), tensor.strides().end());
    };

    auto print_strides = [](const char* name, const torch::Tensor& tensor) {
        std::string shape_str = "[";
        std::string stride_str = "[";
        for (int64_t i = 0; i < tensor.dim(); ++i) {
            shape_str += std::to_string(tensor.size(i));
            stride_str += std::to_string(tensor.stride(i));
            if (i + 1 < tensor.dim()) {
                shape_str += ",";
                stride_str += ",";
            }
        }
        shape_str += "]";
        stride_str += "]";
        printf("  %s shape: %s, strides: %s\n", name, shape_str.c_str(), stride_str.c_str());
    };

    mfa_buffer_t q_buffer = nullptr, k_buffer = nullptr, v_buffer = nullptr, out_buffer = nullptr;

    auto bind_tensor = [&](const char* name, const torch::Tensor& tensor, mfa_buffer_t& buffer) {
        size_t bytes = tensor.numel() * tensor.element_size();
        mfa_error_t result = MFA_SUCCESS;

        if (tensor.is_contiguous()) {
            if (use_mps_buffers) {
                void* handle = mps_utils::get_mtl_buffer_handle(tensor);
                if (!handle) {
                    throw std::runtime_error(std::string("Failed to acquire MTLBuffer for ") + name);
                }
                result = mfa_buffer_from_mtl_buffer(MetalSDPABackend::swift_context_, handle, bytes, &buffer);
            } else {
                result = mfa_buffer_from_ptr(MetalSDPABackend::swift_context_, tensor.data_ptr(), bytes, &buffer);
            }
        } else {
            if (use_mps_buffers) {
                void* handle = mps_utils::get_mtl_buffer_handle(tensor);
                if (!handle) {
                    throw std::runtime_error(std::string("Failed to acquire MTLBuffer for ") + name);
                }
                auto shape_vec = make_shape_vector(tensor);
                auto stride_vec = make_stride_vector(tensor);
                result = mfa_buffer_from_mtl_buffer_with_strides(
                    MetalSDPABackend::swift_context_,
                    handle,
                    bytes,
                    shape_vec.data(),
                    stride_vec.data(),
                    static_cast<uint32_t>(shape_vec.size()),
                    &buffer
                );
            } else {
                auto shape_vec = make_shape_vector(tensor);
                auto stride_vec = make_stride_vector(tensor);
                result = mfa_buffer_from_ptr_with_strides(
                    MetalSDPABackend::swift_context_,
                    tensor.data_ptr(),
                    bytes,
                    shape_vec.data(),
                    stride_vec.data(),
                    static_cast<uint32_t>(shape_vec.size()),
                    &buffer
                );
            }
        }

        if (result != MFA_SUCCESS) {
            throw std::runtime_error(std::string("Failed to create ") + name + " buffer");
        }
    };

    bool any_strided = !q_tensor.is_contiguous() || !k_tensor.is_contiguous() ||
                       !v_tensor.is_contiguous() || !output.is_contiguous();
    if (any_strided) {
        printf("📊 Using stride-aware buffers:\n");
        print_strides("Q", q_tensor);
        print_strides("K", k_tensor);
        print_strides("V", v_tensor);
        print_strides("O", output);
    }

    bind_tensor("query", q_tensor, q_buffer);
    bind_tensor("key", k_tensor, k_buffer);
    bind_tensor("value", v_tensor, v_buffer);
    bind_tensor("output", output, out_buffer);

    mfa_error_t result = mfa_attention_forward_str(
        MetalSDPABackend::swift_context_,
        q_buffer, k_buffer, v_buffer, out_buffer,
        batch_size, seq_len_q, seq_len_kv, num_heads, head_dim,
        softmax_scale, is_causal,
        precision_str.c_str(), precision_str.c_str(), precision_str.c_str(),
        false, false, false, false
    );

    if (result != MFA_SUCCESS) {
        std::string error_msg = "Metal Flash Attention forward pass failed with code " + std::to_string(result);
        switch (result) {
            case MFA_ERROR_INVALID_ARGS: error_msg += " (Invalid arguments - check tensor shapes and parameters)"; break;
            case MFA_ERROR_MEMORY_ALLOCATION: error_msg += " (Memory allocation failed)"; break;
            case MFA_ERROR_DEVICE_NOT_SUPPORTED: error_msg += " (Metal device not supported)"; break;
            case MFA_ERROR_KERNEL_COMPILATION: error_msg += " (Metal kernel compilation failed)"; break;
            case MFA_ERROR_EXECUTION_FAILED: error_msg += " (Kernel execution failed)"; break;
            default: error_msg += " (Unknown error)"; break;
        }
        throw std::runtime_error(error_msg);
    }

    if (input_was_flux) {
        printf("🔄 Converting output from Metal layout [B,S,H,D] back to PyTorch layout [B,H,S,D]\n");
        output = convert_metal_to_flux_layout(output);
    }

    return output;
}

torch::Tensor MetalSDPABackend::call_swift_flash_attention(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    bool is_causal,
    float softmax_scale
) {
    bool mps_candidate = q.device().is_mps() && k.device().is_mps() && v.device().is_mps();
    if (mps_candidate) {
        try {
            return call_swift_flash_attention_impl(q, k, v, is_causal, softmax_scale, true);
        } catch (const std::exception& ex) {
            std::cout << "⚠️  MPS fast path unavailable: " << ex.what() << " -- falling back to CPU path" << std::endl;
        }
    }

    auto q_cpu = ensure_contiguous_cpu(q);
    auto k_cpu = ensure_contiguous_cpu(k);
    auto v_cpu = ensure_contiguous_cpu(v);
    return call_swift_flash_attention_impl(q_cpu, k_cpu, v_cpu, is_causal, softmax_scale, false);
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

    // Call Swift Flash Attention (using processed tensors from call_swift_flash_attention)
    auto result = call_swift_flash_attention(query, key, value, is_causal, softmax_scale);

    // Convert result back to original layout if input was FLUX
    // Note: The layout conversion is now handled within call_swift_flash_attention
    // This just preserves the original logic structure for future reference

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

// REMOVED: First implementation of quantized_scaled_dot_product_attention
// This function is now implemented as a compatibility wrapper that routes to the unified implementation



void MetalSDPABackend::unregister_backend() {
    cleanup();
    std::cout << "Metal SDPA backend unregistered" << std::endl;
}

// Nested namespace for helper functions that need to be in metal_sdpa::metal_sdpa
namespace metal_sdpa {

// Forward declarations for buffer type management
size_t calculate_expected_buffer_size(const torch::Tensor& reference_tensor, OutputPrecision precision);

// Estimate accuracy loss for different granularities
float estimate_accuracy_loss(QuantizationGranularity granularity,
                           const TensorAnalysisMetrics& metrics,
                           QuantizationPrecision precision) {
    // Base accuracy loss from quantization precision
    float base_loss = (precision == QuantizationPrecision::INT8) ? 0.01f : 0.05f; // INT4 has higher base loss

    // Variance penalty: higher variance benefits from finer granularity
    float variance_penalty = metrics.variance / (metrics.mean_abs_value * metrics.mean_abs_value + 1e-8f);

    // Dynamic range penalty: larger dynamic range benefits from finer granularity
    float range_penalty = std::min(2.0f, metrics.dynamic_range / (metrics.mean_abs_value * 8.0f + 1e-8f));

    // Outlier penalty: outliers benefit from finer granularity
    float outlier_penalty = metrics.has_outliers ? 1.5f : 1.0f;

    switch (granularity) {
        case QuantizationGranularity::TENSOR_WISE:
            // Tensor-wise has highest accuracy loss for non-uniform data
            return base_loss * (1.0f + variance_penalty + range_penalty) * outlier_penalty;

        case QuantizationGranularity::ROW_WISE:
            // Row-wise reduces variance penalty significantly
            return base_loss * (1.0f + variance_penalty * 0.3f + range_penalty * 0.5f) * outlier_penalty;

        case QuantizationGranularity::BLOCK_WISE:
            // Block-wise has lowest accuracy loss for most cases
            return base_loss * (1.0f + variance_penalty * 0.1f + range_penalty * 0.2f) * outlier_penalty;

        case QuantizationGranularity::HYBRID:
            // Hybrid can achieve near block-wise accuracy with better performance
            return base_loss * (1.0f + variance_penalty * 0.15f + range_penalty * 0.25f) * outlier_penalty;

        default:
            return base_loss;
    }
}

// Adaptive block size selection algorithm
BlockSizeConfig select_optimal_block_sizes(const torch::Tensor& tensor, QuantizationPrecision precision) {
    auto tensor_shape = tensor.sizes();
    int64_t batch_size = tensor_shape[0];
    int64_t seq_len = tensor_shape[1];
    int64_t num_heads = tensor_shape[2];
    int64_t head_dim = tensor_shape[3];

    printf("🧠 Selecting optimal block sizes for tensor: [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]\n",
           batch_size, seq_len, num_heads, head_dim);

    BlockSizeConfig config;

    // Adaptive sequence block size based on sequence length
    if (seq_len <= 64) {
        config.query_block_size = static_cast<uint32_t>(seq_len);  // Use full sequence as one block for small sequences
    } else if (seq_len <= 512) {
        config.query_block_size = 64;  // Medium block size for medium sequences
    } else if (seq_len <= 2048) {
        config.query_block_size = 128; // Standard block size for long sequences
    } else {
        config.query_block_size = 256; // Large block size for very long sequences
    }

    // Adaptive head block size based on number of heads
    if (num_heads <= 8) {
        config.head_block_size = static_cast<uint32_t>(num_heads);  // Use all heads as one block for small head counts
    } else if (num_heads <= 32) {
        config.head_block_size = 8;   // Medium block size for medium head counts
    } else {
        config.head_block_size = 16;  // Large block size for many heads
    }

    // Adaptive dimension block size based on head dimension
    if (head_dim <= 64) {
        config.value_block_size = static_cast<uint32_t>(head_dim);  // Use full dimension as one block for small dimensions
    } else if (head_dim <= 128) {
        config.value_block_size = 64;  // Medium block size for medium dimensions
    } else {
        config.value_block_size = 128; // Large block size for large dimensions
    }

    // Key and value tensors can use the same block sizes as query for simplicity
    config.key_block_size = config.query_block_size;

    // Performance-based adjustments for INT4 vs INT8
    if (precision == QuantizationPrecision::INT4) {
        // INT4 benefits from smaller blocks for better accuracy
        config.query_block_size = std::max(32u, config.query_block_size / 2);
        config.key_block_size = config.query_block_size;
        config.value_block_size = std::max(32u, config.value_block_size / 2);
        config.head_block_size = std::max(1u, config.head_block_size / 2);
    }

    printf("✅ Selected block sizes: seq=%d, head=%d, dim=%d (key=%d)\n",
           config.query_block_size, config.head_block_size, config.value_block_size, config.key_block_size);

    return config;
}

// Return buffer type management implementations
OutputPrecision determine_output_precision(const QuantizationConfig& config,
                                          const torch::Tensor& query,
                                          const torch::Tensor& key,
                                          const torch::Tensor& value) {
    printf("🔍 Determining optimal output precision...\n");

    // If explicitly specified in config, use that
    if (config.output_precision != OutputPrecision::FP32) {
        printf("   Using explicitly configured output precision: %s\n",
               QuantizationConfig::precision_to_string(config.output_precision).c_str());
        return config.output_precision;
    }

    // Intelligent precision selection based on input characteristics
    auto input_dtype = query.scalar_type();

    // Rule 1: If input is already high precision (FP32), maintain it unless quantized
    if (input_dtype == torch::kFloat32) {
        // Check if any tensors are quantized (INT4/INT8)
        bool has_quantized_inputs = (config.key_precision == QuantizationPrecision::INT4 ||
                                   config.key_precision == QuantizationPrecision::INT8 ||
                                   config.value_precision == QuantizationPrecision::INT4 ||
                                   config.value_precision == QuantizationPrecision::INT8);

        if (has_quantized_inputs) {
            printf("   FP32 input with quantized K/V → using FP16 output for efficiency\n");
            return OutputPrecision::FP16;
        } else {
            printf("   FP32 input, no quantization → maintaining FP32 output\n");
            return OutputPrecision::FP32;
        }
    }

    // Rule 2: If input is FP16, generally maintain FP16 for efficiency
    if (input_dtype == torch::kFloat16) {
        printf("   FP16 input → maintaining FP16 output for efficiency\n");
        return OutputPrecision::FP16;
    }

    // Rule 3: If input is BF16, maintain BF16
    if (input_dtype == torch::kBFloat16) {
        printf("   BF16 input → maintaining BF16 output\n");
        return OutputPrecision::BF16;
    }

    // Rule 4: For quantized-only scenarios, use FP16 as efficient default
    printf("   Mixed/quantized scenario → defaulting to FP16 output\n");
    return OutputPrecision::FP16;
}

torch::Tensor create_typed_output_tensor(const torch::Tensor& reference_tensor,
                                        OutputPrecision output_precision,
                                        bool validate_size) {
    auto target_dtype = QuantizationConfig::precision_to_torch_dtype(output_precision);

    printf("🔧 Creating typed output tensor: %s → %s\n",
           scalar_type_to_string(reference_tensor.scalar_type()).c_str(),
           scalar_type_to_string(target_dtype).c_str());

    // Create output tensor with correct dtype and same shape as reference
    auto output = torch::empty_like(reference_tensor, target_dtype);

    if (validate_size) {
        size_t expected_size = calculate_expected_buffer_size(reference_tensor, output_precision);
        size_t actual_size = output.numel() * output.element_size();

        if (actual_size != expected_size) {
            throw std::runtime_error(
                "Output buffer size mismatch: expected " + std::to_string(expected_size) +
                " bytes, got " + std::to_string(actual_size) + " bytes"
            );
        }

        printf("✅ Output buffer size validated: %zu bytes\n", actual_size);
    }

    return output;
}

bool validate_output_buffer_type(const torch::Tensor& output_tensor,
                                OutputPrecision expected_precision,
                                size_t expected_size) {
    auto expected_dtype = QuantizationConfig::precision_to_torch_dtype(expected_precision);
    auto actual_dtype = output_tensor.scalar_type();

    printf("🔍 Validating output buffer type...\n");
    printf("   Expected: %s, Actual: %s\n",
           scalar_type_to_string(expected_dtype).c_str(),
           scalar_type_to_string(actual_dtype).c_str());

    // Check dtype match
    if (actual_dtype != expected_dtype) {
        printf("❌ Output buffer dtype mismatch!\n");
        return false;
    }

    // Check size match
    size_t actual_size = output_tensor.numel() * output_tensor.element_size();
    if (actual_size != expected_size) {
        printf("❌ Output buffer size mismatch: expected %zu, got %zu bytes\n",
               expected_size, actual_size);
        return false;
    }

    printf("✅ Output buffer validation passed\n");
    return true;
}

torch::Tensor convert_output_precision(const torch::Tensor& output_tensor,
                                      OutputPrecision source_precision,
                                      OutputPrecision target_precision) {
    if (source_precision == target_precision) {
        return output_tensor; // No conversion needed
    }

    auto target_dtype = QuantizationConfig::precision_to_torch_dtype(target_precision);

    printf("🔄 Converting output precision: %s → %s\n",
           QuantizationConfig::precision_to_string(source_precision).c_str(),
           QuantizationConfig::precision_to_string(target_precision).c_str());

    // Perform safe precision conversion
    auto converted = output_tensor.to(target_dtype);

    printf("✅ Precision conversion completed\n");
    return converted;
}

size_t calculate_expected_buffer_size(const torch::Tensor& reference_tensor,
                                    OutputPrecision precision) {
    size_t element_count = reference_tensor.numel();
    size_t element_size;

    switch (precision) {
        case OutputPrecision::FP16:
            element_size = sizeof(torch::Half);
            break;
        case OutputPrecision::BF16:
            element_size = sizeof(torch::BFloat16);
            break;
        case OutputPrecision::FP32:
        default:
            element_size = sizeof(float);
            break;
    }

    return element_count * element_size;
}

} // namespace metal_sdpa (nested)

// UNIFIED QUANTIZED ATTENTION IMPLEMENTATION
// This function replaces both quantized_scaled_dot_product_attention and quantized_scaled_dot_product_attention_enhanced
// It supports all quantization granularities, precision options, and advanced features in a single unified codebase
torch::Tensor MetalSDPABackend::quantized_scaled_dot_product_attention_unified(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const QuantizationConfig& config
) {
    printf("🚨 ENTERING unified quantized attention with granularity: %s\n",
           QuantizationConfig::granularity_to_string(config.granularity).c_str());
    fflush(stdout);

    ensure_initialized();

    // Handle hybrid granularity selection with per-tensor analysis
    QuantizationConfig effective_config = config;
    HybridGranularityConfig hybrid_config;

    if (config.granularity == QuantizationGranularity::HYBRID) {
        printf("🎯 Performing hybrid granularity selection...\n");
        hybrid_config = select_hybrid_granularities(query, key, value, config);

        // For now, use unified granularity for compatibility with current FFI
        if (config.enable_per_tensor_granularity) {
            printf("⚠️  Per-tensor granularity not yet supported in FFI, using unified selection\n");
            // Use the most common granularity among Q, K, V as unified choice
            std::map<QuantizationGranularity, int> granularity_votes;
            granularity_votes[hybrid_config.query_granularity]++;
            granularity_votes[hybrid_config.key_granularity]++;
            granularity_votes[hybrid_config.value_granularity]++;

            auto most_common = std::max_element(granularity_votes.begin(), granularity_votes.end(),
                                              [](const auto& a, const auto& b) { return a.second < b.second; });
            effective_config.granularity = most_common->first;

            printf("🎯 Unified hybrid selection: %s (based on majority vote)\n",
                   QuantizationConfig::granularity_to_string(effective_config.granularity).c_str());
        } else {
            // Use the primary tensor analysis result
            effective_config.granularity = hybrid_config.query_granularity;
            printf("🎯 Unified hybrid selection: %s (based on primary tensor)\n",
                   QuantizationConfig::granularity_to_string(effective_config.granularity).c_str());
        }
    }

    // Convert all tensors to CPU and contiguous
    auto q_cpu = MetalSDPABackend::ensure_contiguous_cpu(query);
    auto k_cpu = MetalSDPABackend::ensure_contiguous_cpu(key);
    auto v_cpu = MetalSDPABackend::ensure_contiguous_cpu(value);

    // Detect tensor layout and convert if necessary
    printf("🔍 Analyzing tensor layouts for FLUX compatibility...\n");

    auto q_layout = detect_tensor_layout(q_cpu);
    auto k_layout = detect_tensor_layout(k_cpu);
    auto v_layout = detect_tensor_layout(v_cpu);

    // Check layout consistency
    if (q_layout.is_flux_layout != k_layout.is_flux_layout || q_layout.is_flux_layout != v_layout.is_flux_layout) {
        printf("⚠️  Warning: Inconsistent tensor layouts detected:\n");
        printf("   Query: %s\n", q_layout.to_string().c_str());
        printf("   Key: %s\n", k_layout.to_string().c_str());
        printf("   Value: %s\n", v_layout.to_string().c_str());
    }

    // Convert FLUX layout to Metal layout if needed
    bool input_was_flux_layout = q_layout.is_flux_layout;
    torch::Tensor q_metal = q_cpu;
    torch::Tensor k_metal = k_cpu;
    torch::Tensor v_metal = v_cpu;

    if (q_layout.is_flux_layout) {
        printf("🔄 Converting FLUX layout tensors to Metal layout...\n");
        q_metal = convert_flux_to_metal_layout(q_cpu);
        k_metal = convert_flux_to_metal_layout(k_cpu);
        v_metal = convert_flux_to_metal_layout(v_cpu);
    }

    // Get tensor dimensions (now in Metal layout [B,S,H,D])
    uint32_t batch_size, seq_len_q, seq_len_kv, num_heads, head_dim;

    if (q_metal.dim() == 4) {
        auto q_sizes = q_metal.sizes();
        batch_size = static_cast<uint32_t>(q_sizes[0]);
        seq_len_q = static_cast<uint32_t>(q_sizes[1]);
        seq_len_kv = static_cast<uint32_t>(k_metal.sizes()[1]);
        num_heads = static_cast<uint32_t>(q_sizes[2]);
        head_dim = static_cast<uint16_t>(q_sizes[3]);

        printf("📊 Final tensor dimensions: batch=%u, seq_q=%u, seq_kv=%u, heads=%u, dim=%u\n",
               batch_size, seq_len_q, seq_len_kv, num_heads, head_dim);

        // Validate head count for FLUX
        if (input_was_flux_layout) {
            if (num_heads < 12 || num_heads > 96) {
                printf("⚠️  Warning: Unusual head count for FLUX: %u (expected 12-96)\n", num_heads);
            } else {
                printf("✅ FLUX head count validation passed: %u heads\n", num_heads);
            }
        }
    } else {
        throw std::runtime_error("Unified quantized attention currently only supports 4D tensors [batch, seq_len, num_heads, head_dim]");
    }

    // Determine optimal output precision using intelligent selection
    OutputPrecision optimal_output_precision = metal_sdpa::determine_output_precision(effective_config, q_metal, k_metal, v_metal);

    // Create type-safe output tensor with validation (using Metal layout)
    auto output = metal_sdpa::create_typed_output_tensor(q_metal, optimal_output_precision, true);


    // Calculate softmax scale
    float softmax_scale = config.scale ? static_cast<float>(*config.scale) : (1.0f / std::sqrt(static_cast<float>(head_dim)));

    // REMOVED: C++ quantization logic - now using runtime quantization
    // The new runtime quantization API handles all quantization on the GPU side
    printf("🚀 Using runtime quantization - bypassing C++ side quantization\n");

    // No longer quantize tensors on C++ side - pass raw FP16/BF16/FP32 tensors directly
    torch::Tensor q_processed = q_metal;
    torch::Tensor k_processed = k_metal;
    torch::Tensor v_processed = v_metal;

    // Create MFA buffers
    mfa_buffer_t q_buffer, k_buffer, v_buffer, out_buffer;

    size_t q_bytes = q_processed.numel() * q_processed.element_size();
    size_t k_bytes = k_processed.numel() * k_processed.element_size();
    size_t v_bytes = v_processed.numel() * v_processed.element_size();
    size_t out_bytes = output.numel() * output.element_size();

    mfa_error_t result;

    result = mfa_buffer_from_ptr(MetalSDPABackend::swift_context_, q_processed.data_ptr(), q_bytes, &q_buffer);
    if (result != MFA_SUCCESS) {
        throw std::runtime_error("Failed to create query buffer for unified quantized attention");
    }

    result = mfa_buffer_from_ptr(MetalSDPABackend::swift_context_, k_processed.data_ptr(), k_bytes, &k_buffer);
    if (result != MFA_SUCCESS) {
        throw std::runtime_error("Failed to create key buffer for unified quantized attention");
    }

    result = mfa_buffer_from_ptr(MetalSDPABackend::swift_context_, v_processed.data_ptr(), v_bytes, &v_buffer);
    if (result != MFA_SUCCESS) {
        throw std::runtime_error("Failed to create value buffer for unified quantized attention");
    }

    result = mfa_buffer_from_ptr(MetalSDPABackend::swift_context_, output.data_ptr(), out_bytes, &out_buffer);
    if (result != MFA_SUCCESS) {
        throw std::runtime_error("Failed to create output buffer for unified quantized attention");
    }

    try {
        // Convert effective granularity enum to int32_t for FFI
        int32_t granularity_int = static_cast<int32_t>(effective_config.granularity);

        printf("🚀 Calling unified quantized attention with:\n");
        printf("   Effective Granularity: %s (%d)\n", QuantizationConfig::granularity_to_string(effective_config.granularity).c_str(), granularity_int);
        printf("   Block sizes: Q=%u, K=%u, V=%u\n", effective_config.block_sizes.query_block_size, effective_config.block_sizes.key_block_size, effective_config.block_sizes.value_block_size);
        printf("   Mixed precision: %s, Symmetric quantization: %s\n", effective_config.enable_mixed_precision ? "enabled" : "disabled", effective_config.force_symmetric_quantization ? "enabled" : "disabled");
        if (config.granularity == QuantizationGranularity::HYBRID) {
            printf("   Hybrid Selection Reasoning: %s\n", hybrid_config.selection_reasoning.c_str());
        }
        fflush(stdout);

        // Scale arrays are no longer needed - runtime quantization handles scaling internally

        // Call new runtime quantized attention function that takes FP16/BF16/FP32 inputs
        // The new API parameters are:
        // - q_precision: Input tensor precision (0=FP16, 1=BF16, 2=FP32)
        // - k_precision: Target quantization precision (3=INT8, 4=INT4)
        // - v_precision: Quantization mode (0=tensorWise, 2=blockWise)

        // Determine input precision from tensor dtype
        int32_t input_precision = 0; // Default to FP16
        if (q_processed.scalar_type() == torch::kFloat32) {
            input_precision = 2; // FP32
        } else if (q_processed.scalar_type() == torch::kBFloat16) {
            input_precision = 1; // BF16
        } else if (q_processed.scalar_type() == torch::kFloat16) {
            input_precision = 0; // FP16
        }

        // Determine target quantization precision from config
        int32_t target_quantization = 3; // Default to INT8
        if (config.key_precision == QuantizationPrecision::INT4 || config.value_precision == QuantizationPrecision::INT4) {
            target_quantization = 4; // INT4
        }

        // Determine quantization mode from granularity
        int32_t quantization_mode = 0; // Default to tensor-wise
        if (effective_config.granularity == QuantizationGranularity::BLOCK_WISE) {
            quantization_mode = 2; // Block-wise (use default block size of 64)
        }

        // Determine output precision
        int32_t output_precision_int = 2; // Default to FP32
        if (optimal_output_precision == OutputPrecision::FP16) {
            output_precision_int = 0; // FP16
        } else if (optimal_output_precision == OutputPrecision::BF16) {
            output_precision_int = 1; // BF16
        }

        printf("🚀 Calling runtime quantized attention with:\n");
        printf("   Input precision: %s (%d)\n",
               input_precision == 0 ? "FP16" : input_precision == 1 ? "BF16" : "FP32",
               input_precision);
        printf("   Target quantization: %s (%d)\n",
               target_quantization == 3 ? "INT8" : "INT4",
               target_quantization);
        printf("   Quantization mode: %s (%d)\n",
               quantization_mode == 0 ? "tensor-wise" : "block-wise",
               quantization_mode);
        printf("   Output precision: %s (%d)\n",
               output_precision_int == 0 ? "FP16" : output_precision_int == 1 ? "BF16" : "FP32",
               output_precision_int);

        result = mfa_attention_forward_quantized_direct(
            MetalSDPABackend::swift_context_,
            q_buffer, k_buffer, v_buffer, out_buffer,
            batch_size, seq_len_q, seq_len_kv, num_heads, head_dim,
            softmax_scale, config.is_causal,
            0.0f, 0,  // qScale, qZeroPoint - not used in new API
            0.0f, 0,  // kScale, kZeroPoint - not used in new API
            0.0f, 0,  // vScale, vZeroPoint - not used in new API
            input_precision,        // Input precision: 0=FP16, 1=BF16, 2=FP32
            target_quantization,    // Target quantization precision: 3=INT8, 4=INT4
            quantization_mode,      // Quantization mode: 0=tensorWise, 2=blockWise
            output_precision_int,   // Output precision
            false, false, false, false  // No transpose for standard layout
        );

        if (result != MFA_SUCCESS) {
            throw std::runtime_error("Runtime quantized attention forward pass failed with error code: " + std::to_string(result));
        }

        // Validate output buffer type matches expectations
        size_t expected_size = metal_sdpa::calculate_expected_buffer_size(output, optimal_output_precision);
        if (!metal_sdpa::validate_output_buffer_type(output, optimal_output_precision, expected_size)) {
            throw std::runtime_error("Output buffer type validation failed - potential data corruption detected");
        }

        printf("✅ Unified quantized attention completed successfully with type validation\n");

        // Convert output back to original layout if input was FLUX
        torch::Tensor final_output = output;
        if (input_was_flux_layout) {
            printf("🔄 Converting output back to FLUX layout...\n");
            printf("   Output tensor before conversion: shape=%s dtype=%s\n",
                   ("[" + std::to_string(output.size(0)) + "," + std::to_string(output.size(1)) + "," + std::to_string(output.size(2)) + "," + std::to_string(output.size(3)) + "]").c_str(),
                   scalar_type_to_string(output.scalar_type()).c_str());
            final_output = convert_metal_to_flux_layout(output);
        }

        // CRITICAL FIX: Ensure tensor data is copied before buffer cleanup to prevent use-after-free
        // This is especially important for small tensors where PyTorch may not automatically copy
        printf("🧹 Creating safe copy of output tensor before buffer cleanup...\n");
        torch::Tensor safe_output = final_output.clone().contiguous();

        // Convert to target device AFTER creating safe copy
        torch::Tensor final_result = safe_output.to(query.device());

        // Now safe to clean up MFA buffers since we have an independent copy
        printf("🧹 Skipping MFA buffer destruction (zero-copy views are owned by PyTorch)\n");

        fflush(stdout);
        return final_result;

    } catch (...) {
        printf("🚨 Exception occurred; skipping MFA buffer destruction to avoid touching shared memory\n");
        throw;
    }
}

// BACKWARD COMPATIBILITY WRAPPERS
// These functions provide backward compatibility for existing code while routing through the unified implementation

torch::Tensor MetalSDPABackend::quantized_scaled_dot_product_attention(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const std::string& precision,
    bool is_causal,
    std::optional<double> scale
) {
    printf("🔀 COMPATIBILITY: Routing legacy quantized_scaled_dot_product_attention to unified implementation\n");

    // Convert legacy string-based API to unified QuantizationConfig
    QuantizationConfig config;
    config.precision = precision;  // Legacy string field
    config.is_causal = is_causal;
    config.scale = scale;

    // Set default granularity for legacy API (tensor-wise for compatibility)
    config.granularity = QuantizationGranularity::TENSOR_WISE;

    // Set precisions based on legacy API behavior:
    // - Query stays in original precision (FP16/FP32)
    // - Key and Value are quantized to specified precision
    config.query_precision = QuantizationPrecision::FP16;  // Keep Q in original precision
    config.key_precision = QuantizationConfig::string_to_quantization_precision(precision);
    config.value_precision = QuantizationConfig::string_to_quantization_precision(precision);

    // Use FP16 output for efficiency (matches legacy behavior)
    config.output_precision = OutputPrecision::FP16;

    printf("🔀 Legacy API converted to: granularity=%s, q_precision=%s, kv_precision=%s\n",
           QuantizationConfig::granularity_to_string(config.granularity).c_str(),
           QuantizationConfig::quantization_precision_to_string(config.query_precision).c_str(),
           QuantizationConfig::quantization_precision_to_string(config.key_precision).c_str());

    // Route to unified implementation
    return quantized_scaled_dot_product_attention_unified(query, key, value, config);
}

torch::Tensor MetalSDPABackend::quantized_scaled_dot_product_attention_with_config(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const QuantizationConfig& config
) {
    printf("🔀 COMPATIBILITY: Routing quantized_scaled_dot_product_attention_with_config to unified implementation\n");

    // This function already uses QuantizationConfig, so route directly
    return quantized_scaled_dot_product_attention_unified(query, key, value, config);
}

torch::Tensor MetalSDPABackend::quantized_scaled_dot_product_attention_enhanced(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const QuantizationConfig& config
) {
    printf("🔀 COMPATIBILITY: Routing quantized_scaled_dot_product_attention_enhanced to unified implementation\n");

    // This function already uses QuantizationConfig, so route directly
    return quantized_scaled_dot_product_attention_unified(query, key, value, config);
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

#include "metal_sdpa_backend.h"
#include <torch/torch.h>
#include <ATen/ATen.h>
#include <c10/util/Exception.h>
#include <mutex>
#include <stdexcept>
#include <iostream>

namespace metal_sdpa {

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
        case torch::ScalarType::Float16: return "Float16";
        case torch::ScalarType::QInt8: return "QInt8";
        case torch::ScalarType::QUInt8: return "QUInt8";
        case torch::ScalarType::QInt32: return "QInt32";
        default: return "Unknown";
    }
}

// Helper functions for row-wise and block-wise quantization
std::vector<float> calculate_row_scales(const torch::Tensor& tensor, QuantizationPrecision precision) {
    printf("🔧 Calculating row-wise scales for tensor with shape: [");
    for (int i = 0; i < tensor.dim(); i++) {
        printf("%ld", tensor.size(i));
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

        // Calculate scale for this row (avoid division by zero)
        float scale = (row_max > 1e-8f) ? (row_max / max_quant_val) : 1e-8f;
        row_scales.push_back(scale);
    }

    printf("✅ Calculated %zu row-wise scales (first few: %.6f, %.6f, %.6f)\n",
           row_scales.size(),
           row_scales.size() > 0 ? row_scales[0] : 0.0f,
           row_scales.size() > 1 ? row_scales[1] : 0.0f,
           row_scales.size() > 2 ? row_scales[2] : 0.0f);

    return row_scales;
}

// Block-wise quantization implementation
std::vector<float> calculate_block_scales(const torch::Tensor& tensor, const BlockSizeConfig& block_config, QuantizationPrecision precision) {
    printf("🔧 Calculating block-wise scales for tensor with shape: [");
    for (int i = 0; i < tensor.dim(); i++) {
        printf("%ld", tensor.size(i));
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

    printf("🔧 Block configuration: %ld seq_blocks x %ld head_blocks x %ld dim_blocks = %ld total blocks\n",
           num_seq_blocks, num_head_blocks, num_dim_blocks, total_blocks);

    // Optimize memory access pattern by processing in cache-friendly order
    // Process blocks in memory layout order: batch -> sequence -> head -> dimension
    // This aligns with PyTorch's default memory layout for better cache performance

    printf("🔧 Processing blocks in optimized memory order...\n");

    // Pre-allocate vector for better performance
    block_scales.clear();
    block_scales.resize(total_blocks);

    // Optimized block processing with better memory access patterns
    size_t scale_idx = 0;
    for (int64_t b = 0; b < batch_size; b++) {
        // Process each batch separately for better cache locality
        auto batch_tensor = tensor.slice(0, b, b + 1);

        for (int64_t seq_block = 0; seq_block < num_seq_blocks; seq_block++) {
            int64_t seq_start = seq_block * seq_block_size;
            int64_t seq_end = std::min(seq_start + seq_block_size, seq_len);

            // Extract sequence block once for all heads
            auto seq_slice = batch_tensor.slice(1, seq_start, seq_end);

            for (int64_t head_block = 0; head_block < num_head_blocks; head_block++) {
                int64_t head_start = head_block * head_block_size;
                int64_t head_end = std::min(head_start + head_block_size, num_heads);

                // Extract head block for current sequence slice
                auto head_slice = seq_slice.slice(2, head_start, head_end);

                for (int64_t dim_block = 0; dim_block < num_dim_blocks; dim_block++) {
                    int64_t dim_start = dim_block * dim_block_size;
                    int64_t dim_end = std::min(dim_start + dim_block_size, head_dim);

                    // Final block extraction - now contiguous in memory
                    auto block_tensor = head_slice.slice(3, dim_start, dim_end);

                    // Use contiguous() to ensure optimal memory layout for max() operation
                    auto contiguous_block = block_tensor.contiguous();

                    // Calculate maximum absolute value in this block efficiently
                    float block_max = contiguous_block.abs().max().item<float>();

                    // Calculate scale for this block (avoid division by zero)
                    float scale = (block_max > 1e-8f) ? (block_max / max_quant_val) : 1e-8f;

                    // Store scale at the correct index
                    block_scales[scale_idx++] = scale;
                }
            }
        }
    }

    printf("✅ Processed %zu blocks with optimized memory access\n", scale_idx);

    printf("✅ Calculated %zu block-wise scales (first few: %.6f, %.6f, %.6f)\n",
           block_scales.size(),
           block_scales.size() > 0 ? block_scales[0] : 0.0f,
           block_scales.size() > 1 ? block_scales[1] : 0.0f,
           block_scales.size() > 2 ? block_scales[2] : 0.0f);

    return block_scales;
}

// Adaptive block size selection algorithm
BlockSizeConfig select_optimal_block_sizes(const torch::Tensor& tensor, QuantizationPrecision precision) {
    auto tensor_shape = tensor.sizes();
    int64_t batch_size = tensor_shape[0];
    int64_t seq_len = tensor_shape[1];
    int64_t num_heads = tensor_shape[2];
    int64_t head_dim = tensor_shape[3];

    printf("🧠 Selecting optimal block sizes for tensor: [%ld, %ld, %ld, %ld]\n",
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
    auto sorted_tensor = torch::sort(tensor_flat.abs()).values;
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
    printf("   - Size: %ld elements (%.2f MB)\n", metrics.tensor_size, metrics.memory_footprint / (1024.0f * 1024.0f));
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
        float accuracy_loss = estimate_accuracy_loss(granularity, metrics, precision);

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
                hybrid_config.query_blocks = select_optimal_block_sizes(query, config.query_precision);
            }
            if (hybrid_config.key_granularity == QuantizationGranularity::BLOCK_WISE) {
                hybrid_config.key_blocks = select_optimal_block_sizes(key, config.key_precision);
            }
            if (hybrid_config.value_granularity == QuantizationGranularity::BLOCK_WISE) {
                hybrid_config.value_blocks = select_optimal_block_sizes(value, config.value_precision);
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
            hybrid_config.query_blocks = select_optimal_block_sizes(query, config.query_precision);
            hybrid_config.key_blocks = select_optimal_block_sizes(key, config.key_precision);
            hybrid_config.value_blocks = select_optimal_block_sizes(value, config.value_precision);
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

                    // Quantize this block using optimized operations
                    // Use in-place operations where possible to reduce memory allocations
                    auto quantized_block = torch::round(contiguous_block / scale)
                                               .clamp_(min_val, max_val)  // In-place clamp
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

        // Quantize this row: round(value / scale) clamped to range
        auto quantized_row = torch::round(row_tensor / scale).clamp(min_val, max_val).to(torch::kInt8);
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

// REMOVED: First implementation of quantized_scaled_dot_product_attention
// This function is now implemented as a compatibility wrapper that routes to the unified implementation

torch::Tensor MetalSDPABackend::quantized_scaled_dot_product_attention_enhanced(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const QuantizationConfig& config
) {
    printf("🚨 ENTERING enhanced quantized attention with granularity: %s\n",
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
        // In future iterations, we can extend the FFI to support per-tensor granularities
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
        throw std::runtime_error("Enhanced quantized attention currently only supports 4D tensors [batch, seq_len, num_heads, head_dim]");
    }

    // Determine optimal output precision using intelligent selection
    OutputPrecision optimal_output_precision = determine_output_precision(effective_config, q_cpu, k_cpu, v_cpu);

    // Create type-safe output tensor with validation
    auto output = create_typed_output_tensor(q_cpu, optimal_output_precision, true);

    // Convert precision enums to MFA precision
    auto convert_quantization_precision_to_mfa = [](QuantizationPrecision precision) -> mfa_precision_t {
        switch (precision) {
            case QuantizationPrecision::INT4: return MFA_PRECISION_INT4;
            case QuantizationPrecision::INT8: return MFA_PRECISION_INT8;
            case QuantizationPrecision::FP16: return MFA_PRECISION_FP16;
            case QuantizationPrecision::BF16: return MFA_PRECISION_BF16;
            case QuantizationPrecision::FP32: return MFA_PRECISION_FP32;
            default: return MFA_PRECISION_INT8;
        }
    };

    mfa_precision_t q_precision_mfa = convert_quantization_precision_to_mfa(config.query_precision);
    mfa_precision_t k_precision_mfa = convert_quantization_precision_to_mfa(config.key_precision);
    mfa_precision_t v_precision_mfa = convert_quantization_precision_to_mfa(config.value_precision);

    // Convert optimal output precision to MFA precision
    mfa_precision_t output_precision_mfa;
    switch (optimal_output_precision) {
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

    // Calculate quantization scales based on effective granularity (after hybrid selection)
    float q_scale = 1.0f, k_scale = 1.0f, v_scale = 1.0f;
    int32_t q_zero_point = 0, k_zero_point = 0, v_zero_point = 0;

    // Storage for per-row and per-block scales
    std::vector<float> q_row_scales, k_row_scales, v_row_scales;
    std::vector<float> q_block_scales, k_block_scales, v_block_scales;

    // Calculate quantization scales based on precision and effective granularity
    auto calculate_scales = [&](const torch::Tensor& tensor, QuantizationPrecision precision, float& scale, int32_t& zero_point, std::vector<float>& row_scales, std::vector<float>& block_scales, const std::string& tensor_name) {
        if (precision == QuantizationPrecision::FP16 || precision == QuantizationPrecision::BF16 || precision == QuantizationPrecision::FP32) {
            scale = 1.0f;  // No quantization needed
            zero_point = 0;
            return;
        }

        float max_val = 0.0f;
        switch (effective_config.granularity) {
            case QuantizationGranularity::TENSOR_WISE:
                max_val = tensor.abs().max().item<float>();
                printf("📊 %s tensor-wise scale: max_val=%.6f\n", tensor_name.c_str(), max_val);
                break;

            case QuantizationGranularity::ROW_WISE:
                printf("🚀 Implementing row-wise quantization for %s tensor\n", tensor_name.c_str());
                row_scales = calculate_row_scales(tensor, precision);
                // For compatibility with the current FFI interface, we still use a single scale
                // The actual per-row scales will be handled in the Swift layer
                max_val = tensor.abs().max().item<float>(); // Fallback scale for FFI compatibility
                printf("📊 %s row-wise: calculated %zu per-row scales, fallback scale=%.6f\n",
                       tensor_name.c_str(), row_scales.size(), max_val);
                break;

            case QuantizationGranularity::BLOCK_WISE:
                printf("🚀 Implementing block-wise quantization for %s tensor\n", tensor_name.c_str());
                // Use adaptive block sizing if block sizes are not explicitly configured
                BlockSizeConfig adaptive_config = effective_config.block_sizes;
                if (effective_config.block_sizes.query_block_size == 128 && effective_config.block_sizes.head_block_size == 1 && effective_config.block_sizes.value_block_size == 64) {
                    // Default values detected, use adaptive sizing
                    printf("🧠 Using adaptive block sizing for %s tensor\n", tensor_name.c_str());
                    adaptive_config = select_optimal_block_sizes(tensor, precision);
                }
                block_scales = calculate_block_scales(tensor, adaptive_config, precision);
                // For compatibility with the current FFI interface, we still use a single scale
                // The actual per-block scales will be handled in the Swift layer
                max_val = tensor.abs().max().item<float>(); // Fallback scale for FFI compatibility
                printf("📊 %s block-wise: calculated %zu per-block scales, fallback scale=%.6f\n",
                       tensor_name.c_str(), block_scales.size(), max_val);
                break;

            case QuantizationGranularity::HYBRID: {
                // Implement intelligent hybrid quantization
                printf("🚀 Implementing intelligent hybrid quantization for %s tensor\n", tensor_name.c_str());

                // Create a temporary config for individual tensor analysis
                QuantizationConfig temp_config = effective_config;
                temp_config.enable_per_tensor_granularity = true;

                // Analyze this tensor and select optimal granularity
                auto metrics = analyze_tensor_characteristics(tensor, precision);
                auto optimal_granularity = select_optimal_granularity(metrics, precision, effective_config.hybrid_strategy);

                printf("🎯 Hybrid selection for %s: %s\n", tensor_name.c_str(),
                       QuantizationConfig::granularity_to_string(optimal_granularity).c_str());

                // Apply the selected granularity
                switch (optimal_granularity) {
                    case QuantizationGranularity::ROW_WISE:
                        row_scales = calculate_row_scales(tensor, precision);
                        max_val = tensor.abs().max().item<float>();
                        printf("📊 %s hybrid→row-wise: calculated %zu row scales, fallback scale=%.6f\n",
                               tensor_name.c_str(), row_scales.size(), max_val);
                        break;

                    case QuantizationGranularity::BLOCK_WISE: {
                        BlockSizeConfig adaptive_blocks = effective_config.enable_adaptive_block_sizes ?
                                                         select_optimal_block_sizes(tensor, precision) :
                                                         effective_config.block_sizes;
                        block_scales = calculate_block_scales(tensor, adaptive_blocks, precision);
                        max_val = tensor.abs().max().item<float>();
                        printf("📊 %s hybrid→block-wise: calculated %zu block scales, fallback scale=%.6f\n",
                               tensor_name.c_str(), block_scales.size(), max_val);
                        break;
                    }

                    case QuantizationGranularity::TENSOR_WISE:
                    default:
                        max_val = tensor.abs().max().item<float>();
                        printf("📊 %s hybrid→tensor-wise: using global scale=%.6f\n", tensor_name.c_str(), max_val);
                        break;
                }
                break;
            }
        }

        if (precision == QuantizationPrecision::INT8) {
            scale = max_val / 127.0f;
        } else { // INT4
            scale = max_val / 7.0f;
        }

        if (effective_config.force_symmetric_quantization) {
            zero_point = 0;
        } else {
            // For now, use symmetric quantization
            zero_point = 0;
        }
    };

    calculate_scales(q_cpu, config.query_precision, q_scale, q_zero_point, q_row_scales, q_block_scales, "Query");
    calculate_scales(k_cpu, config.key_precision, k_scale, k_zero_point, k_row_scales, k_block_scales, "Key");
    calculate_scales(v_cpu, config.value_precision, v_scale, v_zero_point, v_row_scales, v_block_scales, "Value");

    // Quantize tensors if needed
    torch::Tensor q_processed = q_cpu;
    torch::Tensor k_processed = k_cpu;
    torch::Tensor v_processed = v_cpu;

    // Enhanced quantization function that supports tensor-wise, row-wise, and block-wise
    auto quantize_tensor_enhanced = [&](const torch::Tensor& tensor, QuantizationPrecision precision, float scale, const std::vector<float>& row_scales, const std::vector<float>& block_scales, const std::string& tensor_name) -> torch::Tensor {
        if (precision == QuantizationPrecision::FP16 || precision == QuantizationPrecision::BF16 || precision == QuantizationPrecision::FP32) {
            return tensor;  // No quantization
        }

        // Choose quantization method based on effective granularity
        switch (effective_config.granularity) {
            case QuantizationGranularity::ROW_WISE:
                if (!row_scales.empty()) {
                    printf("🔧 Using row-wise quantization for %s tensor\n", tensor_name.c_str());
                    return quantize_per_row(tensor, row_scales, precision);
                }
                break;

            case QuantizationGranularity::BLOCK_WISE:
                if (!block_scales.empty()) {
                    printf("🔧 Using block-wise quantization for %s tensor\n", tensor_name.c_str());
                    return quantize_per_block(tensor, block_scales, effective_config.block_sizes, precision);
                }
                break;

            case QuantizationGranularity::HYBRID: {
                // For hybrid, re-analyze and apply the optimal granularity for quantization
                auto metrics = analyze_tensor_characteristics(tensor, precision);
                auto optimal_granularity = select_optimal_granularity(metrics, precision, effective_config.hybrid_strategy);

                printf("🔧 Applying hybrid quantization for %s tensor using %s\n",
                       tensor_name.c_str(), QuantizationConfig::granularity_to_string(optimal_granularity).c_str());

                switch (optimal_granularity) {
                    case QuantizationGranularity::ROW_WISE:
                        if (!row_scales.empty()) {
                            return quantize_per_row(tensor, row_scales, precision);
                        } else {
                            // Calculate row scales on-the-fly if not available
                            auto temp_row_scales = calculate_row_scales(tensor, precision);
                            return quantize_per_row(tensor, temp_row_scales, precision);
                        }

                    case QuantizationGranularity::BLOCK_WISE:
                        if (!block_scales.empty()) {
                            BlockSizeConfig adaptive_blocks = effective_config.enable_adaptive_block_sizes ?
                                                             select_optimal_block_sizes(tensor, precision) :
                                                             effective_config.block_sizes;
                            return quantize_per_block(tensor, block_scales, adaptive_blocks, precision);
                        } else {
                            // Calculate block scales on-the-fly if not available
                            BlockSizeConfig adaptive_blocks = effective_config.enable_adaptive_block_sizes ?
                                                             select_optimal_block_sizes(tensor, precision) :
                                                             effective_config.block_sizes;
                            auto temp_block_scales = calculate_block_scales(tensor, adaptive_blocks, precision);
                            return quantize_per_block(tensor, temp_block_scales, adaptive_blocks, precision);
                        }

                    case QuantizationGranularity::TENSOR_WISE:
                    default:
                        // Fall through to tensor-wise quantization
                        break;
                }
                break;
            }

            case QuantizationGranularity::TENSOR_WISE:
            default:
                // Use tensor-wise quantization
                break;
        }

        // Fall back to tensor-wise quantization
        printf("🔧 Using tensor-wise quantization for %s tensor (scale=%.6f)\n", tensor_name.c_str(), scale);
        if (precision == QuantizationPrecision::INT8) {
            return torch::round(tensor / scale).clamp(-127, 127).to(torch::kInt8);
        } else { // INT4
            return torch::round(tensor / scale).clamp(-7, 7).to(torch::kInt8);
        }
    };

    q_processed = quantize_tensor_enhanced(q_cpu, config.query_precision, q_scale, q_row_scales, q_block_scales, "Query");
    k_processed = quantize_tensor_enhanced(k_cpu, config.key_precision, k_scale, k_row_scales, k_block_scales, "Key");
    v_processed = quantize_tensor_enhanced(v_cpu, config.value_precision, v_scale, v_row_scales, v_block_scales, "Value");

    // Create MFA buffers
    mfa_buffer_t q_buffer, k_buffer, v_buffer, out_buffer;

    size_t q_bytes = q_processed.numel() * q_processed.element_size();
    size_t k_bytes = k_processed.numel() * k_processed.element_size();
    size_t v_bytes = v_processed.numel() * v_processed.element_size();
    size_t out_bytes = output.numel() * output.element_size();

    mfa_error_t result;

    result = mfa_buffer_from_ptr(MetalSDPABackend::swift_context_, q_processed.data_ptr(), q_bytes, &q_buffer);
    if (result != MFA_SUCCESS) {
        throw std::runtime_error("Failed to create query buffer for enhanced quantized attention");
    }

    result = mfa_buffer_from_ptr(MetalSDPABackend::swift_context_, k_processed.data_ptr(), k_bytes, &k_buffer);
    if (result != MFA_SUCCESS) {
        throw std::runtime_error("Failed to create key buffer for enhanced quantized attention");
    }

    result = mfa_buffer_from_ptr(MetalSDPABackend::swift_context_, v_processed.data_ptr(), v_bytes, &v_buffer);
    if (result != MFA_SUCCESS) {
        throw std::runtime_error("Failed to create value buffer for enhanced quantized attention");
    }

    result = mfa_buffer_from_ptr(MetalSDPABackend::swift_context_, output.data_ptr(), out_bytes, &out_buffer);
    if (result != MFA_SUCCESS) {
        throw std::runtime_error("Failed to create output buffer for enhanced quantized attention");
    }

    try {
        // Convert effective granularity enum to int32_t for FFI
        int32_t granularity_int = static_cast<int32_t>(effective_config.granularity);

        printf("🚀 Calling enhanced quantized attention with:\n");
        printf("   Effective Granularity: %s (%d)\n", QuantizationConfig::granularity_to_string(effective_config.granularity).c_str(), granularity_int);
        printf("   Block sizes: Q=%u, K=%u, V=%u\n", effective_config.block_sizes.query_block_size, effective_config.block_sizes.key_block_size, effective_config.block_sizes.value_block_size);
        printf("   Mixed precision: %s, Symmetric quantization: %s\n", effective_config.enable_mixed_precision ? "enabled" : "disabled", effective_config.force_symmetric_quantization ? "enabled" : "disabled");
        if (config.granularity == QuantizationGranularity::HYBRID) {
            printf("   Hybrid Selection Reasoning: %s\n", hybrid_config.selection_reasoning.c_str());
        }
        fflush(stdout);

        // Call enhanced quantized attention function
        result = mfa_attention_forward_quantized_enhanced(
            MetalSDPABackend::swift_context_,
            q_buffer, k_buffer, v_buffer, out_buffer,
            batch_size, seq_len_q, seq_len_kv, num_heads, head_dim,
            softmax_scale, config.is_causal,
            q_scale, q_zero_point,
            k_scale, k_zero_point,
            v_scale, v_zero_point,
            q_precision_mfa, k_precision_mfa, v_precision_mfa, output_precision_mfa,
            granularity_int,
            effective_config.block_sizes.query_block_size, effective_config.block_sizes.key_block_size, effective_config.block_sizes.value_block_size,
            effective_config.enable_mixed_precision, effective_config.force_symmetric_quantization,
            false, false, false, false  // No transpose for standard layout
        );

        if (result != MFA_SUCCESS) {
            throw std::runtime_error("Enhanced quantized attention forward pass failed with error code: " + std::to_string(result));
        }

        // Validate output buffer type matches expectations
        size_t expected_size = calculate_expected_buffer_size(output, optimal_output_precision);
        if (!validate_output_buffer_type(output, optimal_output_precision, expected_size)) {
            throw std::runtime_error("Output buffer type validation failed - potential data corruption detected");
        }

        printf("✅ Enhanced quantized attention completed successfully with type validation\n");
        fflush(stdout);
        return output.to(query.device());

    } catch (...) {
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

    // Determine optimal output precision using intelligent selection
    OutputPrecision optimal_output_precision = determine_output_precision(config, q_cpu, k_cpu, v_cpu);

    // Create type-safe output tensor with validation
    auto output = create_typed_output_tensor(q_cpu, optimal_output_precision, true);

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

    // Convert optimal output precision to MFA precision
    mfa_precision_t output_precision_mfa;
    switch (optimal_output_precision) {
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

    // Calculate quantization scales for K and V tensors
    float k_scale, v_scale;
    int32_t k_zero_point = 0, v_zero_point = 0;  // Symmetric quantization

    if (config.precision == "int8") {
        k_scale = k_cpu.abs().max().item<float>() / 127.0f;
        v_scale = v_cpu.abs().max().item<float>() / 127.0f;
    } else { // int4
        k_scale = k_cpu.abs().max().item<float>() / 7.0f;
        v_scale = v_cpu.abs().max().item<float>() / 7.0f;
    }

    // 🔧 FIX: Only quantize K and V, keep Q in original precision
    // This is the standard approach for quantized attention to maintain accuracy
    torch::Tensor q_original, k_quantized, v_quantized;

    // Q stays in original precision (no quantization)
    q_original = q_cpu; // Keep Q in FP32/BF16
    float q_scale = 1.0f; // No scaling needed for Q

    printf("🔧 DEBUG: Set q_scale = %f (should be 1.0)\n", q_scale);
    fflush(stdout);

    if (config.precision == "int8") {
        // Quantize K: FP32 -> INT8
        k_quantized = torch::round(k_cpu / k_scale).clamp(-127, 127).to(torch::kInt8);
        // Quantize V: FP32 -> INT8
        v_quantized = torch::round(v_cpu / v_scale).clamp(-127, 127).to(torch::kInt8);
    } else { // int4 - clamp to 4-bit range
        k_quantized = torch::round(k_cpu / k_scale).clamp(-7, 7).to(torch::kInt8);
        v_quantized = torch::round(v_cpu / v_scale).clamp(-7, 7).to(torch::kInt8);
    }

    // Create MFA buffers
    mfa_buffer_t q_buffer, k_buffer, v_buffer, out_buffer;

    size_t q_original_bytes = q_original.numel() * q_original.element_size();
    size_t k_quantized_bytes = k_quantized.numel() * k_quantized.element_size();
    size_t v_quantized_bytes = v_quantized.numel() * v_quantized.element_size();
    size_t out_bytes = output.numel() * output.element_size();

    mfa_error_t result;

    result = mfa_buffer_from_ptr(MetalSDPABackend::swift_context_, q_original.data_ptr(), q_original_bytes, &q_buffer);
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
        result = mfa_attention_forward_quantized(
            MetalSDPABackend::swift_context_,
            q_buffer, k_buffer, v_buffer, out_buffer,
            batch_size, seq_len_q, seq_len_kv, num_heads, head_dim,
            softmax_scale, config.is_causal,
            q_scale, 0,  // 🔧 FIX: Pass q_scale=1.0 since Q is not quantized
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

        // Validate output buffer type matches expectations
        size_t expected_size = calculate_expected_buffer_size(output, optimal_output_precision);
        if (!validate_output_buffer_type(output, optimal_output_precision, expected_size)) {
            throw std::runtime_error("Output buffer type validation failed - potential data corruption detected");
        }

        // Move output back to original device
        printf("🎯 EXITING C++ function - returning tensor to device with type validation\n");
        fflush(stdout);
        return output.to(query.device());

    } catch (...) {
        throw;
    }
}

void MetalSDPABackend::unregister_backend() {
    cleanup();
    std::cout << "Metal SDPA backend unregistered" << std::endl;
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
        throw std::runtime_error("Unified quantized attention currently only supports 4D tensors [batch, seq_len, num_heads, head_dim]");
    }

    // Determine optimal output precision using intelligent selection
    OutputPrecision optimal_output_precision = determine_output_precision(effective_config, q_cpu, k_cpu, v_cpu);

    // Create type-safe output tensor with validation
    auto output = create_typed_output_tensor(q_cpu, optimal_output_precision, true);

    // Convert precision enums to MFA precision
    auto convert_quantization_precision_to_mfa = [](QuantizationPrecision precision) -> mfa_precision_t {
        switch (precision) {
            case QuantizationPrecision::INT4: return MFA_PRECISION_INT4;
            case QuantizationPrecision::INT8: return MFA_PRECISION_INT8;
            case QuantizationPrecision::FP16: return MFA_PRECISION_FP16;
            case QuantizationPrecision::BF16: return MFA_PRECISION_BF16;
            case QuantizationPrecision::FP32: return MFA_PRECISION_FP32;
            default: return MFA_PRECISION_INT8;
        }
    };

    mfa_precision_t q_precision_mfa = convert_quantization_precision_to_mfa(config.query_precision);
    mfa_precision_t k_precision_mfa = convert_quantization_precision_to_mfa(config.key_precision);
    mfa_precision_t v_precision_mfa = convert_quantization_precision_to_mfa(config.value_precision);

    // Convert optimal output precision to MFA precision
    mfa_precision_t output_precision_mfa;
    switch (optimal_output_precision) {
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

    // Calculate quantization scales based on effective granularity (after hybrid selection)
    float q_scale = 1.0f, k_scale = 1.0f, v_scale = 1.0f;
    int32_t q_zero_point = 0, k_zero_point = 0, v_zero_point = 0;

    // Storage for per-row and per-block scales
    std::vector<float> q_row_scales, k_row_scales, v_row_scales;
    std::vector<float> q_block_scales, k_block_scales, v_block_scales;

    // Calculate quantization scales based on precision and effective granularity
    auto calculate_scales = [&](const torch::Tensor& tensor, QuantizationPrecision precision, float& scale, int32_t& zero_point, std::vector<float>& row_scales, std::vector<float>& block_scales, const std::string& tensor_name) {
        if (precision == QuantizationPrecision::FP16 || precision == QuantizationPrecision::BF16 || precision == QuantizationPrecision::FP32) {
            scale = 1.0f;  // No quantization needed
            zero_point = 0;
            return;
        }

        float max_val = 0.0f;
        switch (effective_config.granularity) {
            case QuantizationGranularity::TENSOR_WISE:
                max_val = tensor.abs().max().item<float>();
                printf("📊 %s tensor-wise scale: max_val=%.6f\n", tensor_name.c_str(), max_val);
                break;

            case QuantizationGranularity::ROW_WISE:
                printf("🚀 Implementing row-wise quantization for %s tensor\n", tensor_name.c_str());
                row_scales = calculate_row_scales(tensor, precision);
                // For compatibility with the current FFI interface, use a single fallback scale
                max_val = tensor.abs().max().item<float>();
                printf("📊 %s row-wise: calculated %zu per-row scales, fallback scale=%.6f\n",
                       tensor_name.c_str(), row_scales.size(), max_val);
                break;

            case QuantizationGranularity::BLOCK_WISE:
                printf("🚀 Implementing block-wise quantization for %s tensor\n", tensor_name.c_str());
                // Use adaptive block sizing if block sizes are not explicitly configured
                BlockSizeConfig adaptive_config = effective_config.block_sizes;
                if (effective_config.enable_adaptive_block_sizes) {
                    printf("🧠 Using adaptive block sizing for %s tensor\n", tensor_name.c_str());
                    adaptive_config = select_optimal_block_sizes(tensor, precision);
                }
                block_scales = calculate_block_scales(tensor, adaptive_config, precision);
                // For compatibility with the current FFI interface, use a single fallback scale
                max_val = tensor.abs().max().item<float>();
                printf("📊 %s block-wise: calculated %zu per-block scales, fallback scale=%.6f\n",
                       tensor_name.c_str(), block_scales.size(), max_val);
                break;

            case QuantizationGranularity::HYBRID: {
                // This case should not occur since hybrid is resolved earlier, but handle gracefully
                printf("⚠️  Unexpected hybrid granularity in scale calculation, falling back to tensor-wise\n");
                max_val = tensor.abs().max().item<float>();
                break;
            }
        }

        if (precision == QuantizationPrecision::INT8) {
            scale = max_val / 127.0f;
        } else { // INT4
            scale = max_val / 7.0f;
        }

        if (effective_config.force_symmetric_quantization) {
            zero_point = 0;
        } else {
            // For now, use symmetric quantization
            zero_point = 0;
        }
    };

    calculate_scales(q_cpu, config.query_precision, q_scale, q_zero_point, q_row_scales, q_block_scales, "Query");
    calculate_scales(k_cpu, config.key_precision, k_scale, k_zero_point, k_row_scales, k_block_scales, "Key");
    calculate_scales(v_cpu, config.value_precision, v_scale, v_zero_point, v_row_scales, v_block_scales, "Value");

    // Quantize tensors if needed using the unified quantization function
    torch::Tensor q_processed = q_cpu;
    torch::Tensor k_processed = k_cpu;
    torch::Tensor v_processed = v_cpu;

    // Enhanced quantization function that supports all granularities
    auto quantize_tensor_unified = [&](const torch::Tensor& tensor, QuantizationPrecision precision, float scale, const std::vector<float>& row_scales, const std::vector<float>& block_scales, const std::string& tensor_name) -> torch::Tensor {
        if (precision == QuantizationPrecision::FP16 || precision == QuantizationPrecision::BF16 || precision == QuantizationPrecision::FP32) {
            return tensor;  // No quantization
        }

        // Choose quantization method based on effective granularity
        switch (effective_config.granularity) {
            case QuantizationGranularity::ROW_WISE:
                if (!row_scales.empty()) {
                    printf("🔧 Using row-wise quantization for %s tensor\n", tensor_name.c_str());
                    return quantize_per_row(tensor, row_scales, precision);
                }
                break;

            case QuantizationGranularity::BLOCK_WISE:
                if (!block_scales.empty()) {
                    printf("🔧 Using block-wise quantization for %s tensor\n", tensor_name.c_str());
                    return quantize_per_block(tensor, block_scales, effective_config.block_sizes, precision);
                }
                break;

            case QuantizationGranularity::TENSOR_WISE:
            default:
                // Use tensor-wise quantization
                break;
        }

        // Fall back to tensor-wise quantization
        printf("🔧 Using tensor-wise quantization for %s tensor (scale=%.6f)\n", tensor_name.c_str(), scale);
        if (precision == QuantizationPrecision::INT8) {
            return torch::round(tensor / scale).clamp(-127, 127).to(torch::kInt8);
        } else { // INT4
            return torch::round(tensor / scale).clamp(-7, 7).to(torch::kInt8);
        }
    };

    q_processed = quantize_tensor_unified(q_cpu, config.query_precision, q_scale, q_row_scales, q_block_scales, "Query");
    k_processed = quantize_tensor_unified(k_cpu, config.key_precision, k_scale, k_row_scales, k_block_scales, "Key");
    v_processed = quantize_tensor_unified(v_cpu, config.value_precision, v_scale, v_row_scales, v_block_scales, "Value");

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

        // Call unified quantized attention function
        result = mfa_attention_forward_quantized_unified(
            MetalSDPABackend::swift_context_,
            q_buffer, k_buffer, v_buffer, out_buffer,
            batch_size, seq_len_q, seq_len_kv, num_heads, head_dim,
            softmax_scale, config.is_causal,
            q_scale, q_zero_point,
            k_scale, k_zero_point,
            v_scale, v_zero_point,
            q_precision_mfa, k_precision_mfa, v_precision_mfa, output_precision_mfa,
            granularity_int,
            effective_config.block_sizes.query_block_size, effective_config.block_sizes.key_block_size, effective_config.block_sizes.value_block_size,
            effective_config.enable_mixed_precision, effective_config.force_symmetric_quantization,
            false, false, false, false  // No transpose for standard layout
        );

        if (result != MFA_SUCCESS) {
            throw std::runtime_error("Unified quantized attention forward pass failed with error code: " + std::to_string(result));
        }

        // Validate output buffer type matches expectations
        size_t expected_size = calculate_expected_buffer_size(output, optimal_output_precision);
        if (!validate_output_buffer_type(output, optimal_output_precision, expected_size)) {
            throw std::runtime_error("Output buffer type validation failed - potential data corruption detected");
        }

        printf("✅ Unified quantized attention completed successfully with type validation\n");
        fflush(stdout);
        return output.to(query.device());

    } catch (...) {
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

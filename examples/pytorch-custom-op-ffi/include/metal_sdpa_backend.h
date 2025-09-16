#pragma once

#include <torch/extension.h>
#include <torch/library.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/core/DeviceType.h>
#include <c10/core/ScalarType.h>
#include <mutex>
#include <optional>

namespace metal_sdpa {

// Supported quantization granularities
enum class QuantizationGranularity {
    TENSOR_WISE,  // Per-tensor quantization (single scale/zero-point for entire tensor)
    ROW_WISE,     // Per-row quantization (scale/zero-point per row)
    BLOCK_WISE,   // Per-block quantization (scale/zero-point per block)
    HYBRID        // Hybrid quantization (different granularities for different tensors)
};

// Supported quantization precisions - MUST match mfa_ffi.h values!
enum class QuantizationPrecision {
    FP16 = 0,   // Float16 (matches MFA_PRECISION_FP16 = 0)
    BF16 = 1,   // BFloat16 (matches MFA_PRECISION_BF16 = 1)
    FP32 = 2,   // Float32 (matches MFA_PRECISION_FP32 = 2)
    INT8 = 3,   // 8-bit integer quantization (matches MFA_PRECISION_INT8 = 3)
    INT4 = 4    // 4-bit integer quantization (matches MFA_PRECISION_INT4 = 4)
};

// Supported output precision types
enum class OutputPrecision {
    FP32,  // Default - Float32
    FP16,  // Float16
    BF16   // BFloat16
};

// Hybrid quantization strategy types
enum class HybridStrategy {
    PERFORMANCE_FIRST,  // Prioritize speed over accuracy
    ACCURACY_FIRST,     // Prioritize accuracy over speed
    BALANCED           // Balance between accuracy and performance
};

// Block size configuration for block-wise quantization
struct BlockSizeConfig {
    uint32_t query_block_size = 128;      // Block size for query tensor (BLKQ)
    uint32_t key_block_size = 64;         // Block size for key tensor (BLKK)
    uint32_t value_block_size = 64;       // Block size for value tensor (BLKV)
    uint32_t head_block_size = 1;         // Block size across head dimension

    // Constructor with default values
    BlockSizeConfig() = default;
    BlockSizeConfig(uint32_t q_blk, uint32_t k_blk, uint32_t v_blk, uint32_t h_blk = 1)
        : query_block_size(q_blk), key_block_size(k_blk), value_block_size(v_blk), head_block_size(h_blk) {}
};

// Configuration for quantized attention operations
struct QuantizationConfig {
    // Legacy string-based precision (for backward compatibility)
    std::string precision = "int8";           // Quantization precision: "int8", "int4"

    // New enum-based precision configuration
    QuantizationPrecision query_precision = QuantizationPrecision::FP16;    // Query tensor precision
    QuantizationPrecision key_precision = QuantizationPrecision::INT8;      // Key tensor precision
    QuantizationPrecision value_precision = QuantizationPrecision::INT8;    // Value tensor precision

    // Granularity configuration
    QuantizationGranularity granularity = QuantizationGranularity::TENSOR_WISE;

    // Block sizes (only used when granularity is BLOCK_WISE or HYBRID)
    BlockSizeConfig block_sizes;

    // Output configuration
    OutputPrecision output_precision = OutputPrecision::FP32;  // Output buffer precision

    // Operation parameters
    bool is_causal = false;                   // Causal masking
    std::optional<double> scale = std::nullopt; // Custom scaling factor

    // Advanced configuration
    bool enable_mixed_precision = true;       // Allow mixed precision optimizations
    bool force_symmetric_quantization = false; // Force symmetric quantization (zero_point = 0)

    // Hybrid quantization configuration
    HybridStrategy hybrid_strategy = HybridStrategy::BALANCED;  // Strategy for hybrid granularity selection
    bool enable_per_tensor_granularity = false; // Allow different granularities for Q, K, V
    bool enable_adaptive_block_sizes = true;    // Use adaptive block size selection

    // Convert string to QuantizationPrecision enum
    static QuantizationPrecision string_to_quantization_precision(const std::string& precision_str) {
        if (precision_str == "int4") {
            return QuantizationPrecision::INT4;
        } else if (precision_str == "int8") {
            return QuantizationPrecision::INT8;
        } else if (precision_str == "fp16" || precision_str == "float16") {
            return QuantizationPrecision::FP16;
        } else if (precision_str == "bf16" || precision_str == "bfloat16") {
            return QuantizationPrecision::BF16;
        } else if (precision_str == "fp32" || precision_str == "float32") {
            return QuantizationPrecision::FP32;
        } else {
            return QuantizationPrecision::INT8;  // Default fallback
        }
    }

    // Convert string to OutputPrecision enum
    static OutputPrecision string_to_precision(const std::string& precision_str) {
        if (precision_str == "fp16" || precision_str == "float16") {
            return OutputPrecision::FP16;
        } else if (precision_str == "bf16" || precision_str == "bfloat16") {
            return OutputPrecision::BF16;
        } else {
            return OutputPrecision::FP32;  // Default fallback
        }
    }

    // Convert OutputPrecision enum to string
    static std::string precision_to_string(OutputPrecision precision) {
        switch (precision) {
            case OutputPrecision::FP16: return "fp16";
            case OutputPrecision::BF16: return "bf16";
            case OutputPrecision::FP32:
            default: return "fp32";
        }
    }

    // Convert OutputPrecision to PyTorch ScalarType
    static torch::ScalarType precision_to_torch_dtype(OutputPrecision precision) {
        switch (precision) {
            case OutputPrecision::FP16: return torch::kFloat16;
            case OutputPrecision::BF16: return torch::kBFloat16;
            case OutputPrecision::FP32:
            default: return torch::kFloat32;
        }
    }

    // Convert PyTorch ScalarType to OutputPrecision
    static OutputPrecision torch_dtype_to_precision(torch::ScalarType dtype) {
        switch (dtype) {
            case torch::kFloat16: return OutputPrecision::FP16;
            case torch::kBFloat16: return OutputPrecision::BF16;
            case torch::kFloat32:
            default: return OutputPrecision::FP32;
        }
    }

    // Validate QuantizationConfig for type safety
    bool validate_config() const {
        // Check output precision is compatible with quantization settings
        if (output_precision == OutputPrecision::FP32 &&
            (key_precision == QuantizationPrecision::INT4 || value_precision == QuantizationPrecision::INT4)) {
            // FP32 output with INT4 inputs may cause precision loss - warn but allow
        }

        // Validate block sizes for block-wise quantization
        if (granularity == QuantizationGranularity::BLOCK_WISE) {
            if (block_sizes.query_block_size == 0 || block_sizes.key_block_size == 0 ||
                block_sizes.value_block_size == 0 || block_sizes.head_block_size == 0) {
                return false; // Invalid block sizes
            }
        }

        // Validate hybrid strategy settings
        if (granularity == QuantizationGranularity::HYBRID && !enable_per_tensor_granularity) {
            // Hybrid granularity requires per-tensor granularity to be meaningful
        }

        return true; // Configuration is valid
    }

    // Get recommended output precision based on input precisions
    OutputPrecision get_recommended_output_precision() const {
        // If explicitly set and not default, use it
        if (output_precision != OutputPrecision::FP32) {
            return output_precision;
        }

        // Intelligent recommendation based on quantization settings
        bool has_int_quantization = (query_precision == QuantizationPrecision::INT4 ||
                                    query_precision == QuantizationPrecision::INT8 ||
                                    key_precision == QuantizationPrecision::INT4 ||
                                    key_precision == QuantizationPrecision::INT8 ||
                                    value_precision == QuantizationPrecision::INT4 ||
                                    value_precision == QuantizationPrecision::INT8);

        if (has_int_quantization) {
            return OutputPrecision::FP16; // FP16 is efficient for quantized scenarios
        }

        // For non-quantized scenarios, maintain precision
        if (query_precision == QuantizationPrecision::FP16) {
            return OutputPrecision::FP16;
        } else if (query_precision == QuantizationPrecision::BF16) {
            return OutputPrecision::BF16;
        }

        return OutputPrecision::FP32; // Default fallback
    }

    // Convert QuantizationPrecision enum to string
    static std::string quantization_precision_to_string(QuantizationPrecision precision) {
        switch (precision) {
            case QuantizationPrecision::INT4: return "int4";
            case QuantizationPrecision::INT8: return "int8";
            case QuantizationPrecision::FP16: return "fp16";
            case QuantizationPrecision::BF16: return "bf16";
            case QuantizationPrecision::FP32: return "fp32";
            default: return "int8";
        }
    }

    // Convert QuantizationGranularity enum to string
    static std::string granularity_to_string(QuantizationGranularity granularity) {
        switch (granularity) {
            case QuantizationGranularity::TENSOR_WISE: return "tensor_wise";
            case QuantizationGranularity::ROW_WISE: return "row_wise";
            case QuantizationGranularity::BLOCK_WISE: return "block_wise";
            case QuantizationGranularity::HYBRID: return "hybrid";
            default: return "tensor_wise";
        }
    }

    // Convert string to QuantizationGranularity enum
    static QuantizationGranularity string_to_granularity(const std::string& granularity_str) {
        if (granularity_str == "row_wise" || granularity_str == "row") {
            return QuantizationGranularity::ROW_WISE;
        } else if (granularity_str == "block_wise" || granularity_str == "block") {
            return QuantizationGranularity::BLOCK_WISE;
        } else if (granularity_str == "hybrid") {
            return QuantizationGranularity::HYBRID;
        } else {
            return QuantizationGranularity::TENSOR_WISE;  // Default fallback
        }
    }
};

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

    // String-based precision interface (safer for FFI)
    mfa_error_t mfa_attention_forward_str(
        mfa_context_t context,
        mfa_buffer_t q, mfa_buffer_t k, mfa_buffer_t v, mfa_buffer_t out,
        uint32_t batch_size, uint32_t seq_len_q, uint32_t seq_len_kv,
        uint32_t num_heads, uint16_t head_dim, float softmax_scale,
        bool causal,
        const char* input_precision, const char* intermediate_precision, const char* output_precision,
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

    // Unified quantized attention with all features - replaces both basic and enhanced variants
    // Set scale arrays for row-wise and block-wise quantization
    mfa_error_t mfa_set_scale_arrays(
        mfa_context_t context,
        const float* q_scales, uint32_t q_scales_count,
        const float* k_scales, uint32_t k_scales_count,
        const float* v_scales, uint32_t v_scales_count
    );

    mfa_error_t mfa_attention_forward_quantized_unified(
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
        int32_t granularity, // 0=tensor_wise, 1=row_wise, 2=block_wise, 3=hybrid
        uint32_t q_block_size, uint32_t k_block_size, uint32_t v_block_size,
        bool enable_mixed_precision, bool force_symmetric_quantization,
        bool transpose_q, bool transpose_k, bool transpose_v, bool transpose_o
    );

    // Backward compatibility functions - implemented as wrappers to unified function
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

    mfa_error_t mfa_attention_forward_quantized_enhanced(
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
        int32_t granularity, // 0=tensor_wise, 1=row_wise, 2=block_wise, 3=hybrid
        uint32_t q_block_size, uint32_t k_block_size, uint32_t v_block_size,
        bool enable_mixed_precision, bool force_symmetric_quantization,
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

    // Unified quantized attention function - replaces all previous variants
    // This is the primary interface that supports all quantization features
    static torch::Tensor quantized_scaled_dot_product_attention_unified(
        const torch::Tensor& query,
        const torch::Tensor& key,
        const torch::Tensor& value,
        const QuantizationConfig& config = QuantizationConfig{}
    );

    // Backward compatibility functions - implemented as wrappers to unified function
    static torch::Tensor quantized_scaled_dot_product_attention(
        const torch::Tensor& query,
        const torch::Tensor& key,
        const torch::Tensor& value,
        const std::string& precision = "int8",
        bool is_causal = false,
        c10::optional<double> scale = c10::nullopt
    );

    static torch::Tensor quantized_scaled_dot_product_attention_with_config(
        const torch::Tensor& query,
        const torch::Tensor& key,
        const torch::Tensor& value,
        const QuantizationConfig& config = QuantizationConfig{}
    );

    static torch::Tensor quantized_scaled_dot_product_attention_enhanced(
        const torch::Tensor& query,
        const torch::Tensor& key,
        const torch::Tensor& value,
        const QuantizationConfig& config = QuantizationConfig{}
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

// Tensor analysis metrics for hybrid selection
struct TensorAnalysisMetrics {
    float dynamic_range;         // Max - Min values
    float variance;              // Statistical variance
    float mean_abs_value;        // Mean absolute value
    int64_t tensor_size;         // Total number of elements
    int64_t memory_footprint;    // Memory usage in bytes
    float sparsity_ratio;        // Ratio of near-zero values
    bool has_outliers;           // Contains extreme values
    float quantization_error_estimate; // Estimated quantization error

    TensorAnalysisMetrics() : dynamic_range(0.0f), variance(0.0f), mean_abs_value(0.0f),
                             tensor_size(0), memory_footprint(0), sparsity_ratio(0.0f),
                             has_outliers(false), quantization_error_estimate(0.0f) {}
};

// Per-tensor granularity configuration for hybrid quantization
struct HybridGranularityConfig {
    QuantizationGranularity query_granularity = QuantizationGranularity::TENSOR_WISE;
    QuantizationGranularity key_granularity = QuantizationGranularity::TENSOR_WISE;
    QuantizationGranularity value_granularity = QuantizationGranularity::TENSOR_WISE;

    // Adaptive block sizes per tensor
    BlockSizeConfig query_blocks;
    BlockSizeConfig key_blocks;
    BlockSizeConfig value_blocks;

    // Selection reasoning for debugging
    std::string selection_reasoning;

    HybridGranularityConfig() = default;
};

// Helper function declarations for quantization
namespace metal_sdpa {
    // Row-wise quantization functions
    std::vector<float> calculate_row_scales(const torch::Tensor& tensor, QuantizationPrecision precision);
    torch::Tensor quantize_per_row(const torch::Tensor& tensor, const std::vector<float>& row_scales, QuantizationPrecision precision);

    // Block-wise quantization functions
    std::vector<float> calculate_block_scales(const torch::Tensor& tensor, const BlockSizeConfig& block_config, QuantizationPrecision precision);
    torch::Tensor quantize_per_block(const torch::Tensor& tensor, const std::vector<float>& block_scales, const BlockSizeConfig& block_config, QuantizationPrecision precision);

    // Adaptive block size selection
    BlockSizeConfig select_optimal_block_sizes(const torch::Tensor& tensor, QuantizationPrecision precision);

    // Hybrid quantization functions
    TensorAnalysisMetrics analyze_tensor_characteristics(const torch::Tensor& tensor, QuantizationPrecision precision);
    QuantizationGranularity select_optimal_granularity(const TensorAnalysisMetrics& metrics,
                                                       QuantizationPrecision precision,
                                                       HybridStrategy strategy = HybridStrategy::BALANCED);
    HybridGranularityConfig select_hybrid_granularities(const torch::Tensor& query,
                                                        const torch::Tensor& key,
                                                        const torch::Tensor& value,
                                                        const QuantizationConfig& config);

    // Utility functions for hybrid quantization
    float estimate_quantization_overhead(QuantizationGranularity granularity,
                                       const TensorAnalysisMetrics& metrics,
                                       QuantizationPrecision precision);
    float estimate_quantization_overhead(QuantizationGranularity granularity,
                                       const torch::Tensor& tensor,
                                       QuantizationPrecision precision);
    float estimate_accuracy_loss(QuantizationGranularity granularity,
                                const TensorAnalysisMetrics& metrics,
                                QuantizationPrecision precision);

    // Return buffer type management functions
    OutputPrecision determine_output_precision(const QuantizationConfig& config,
                                              const torch::Tensor& query,
                                              const torch::Tensor& key,
                                              const torch::Tensor& value);

    torch::Tensor create_typed_output_tensor(const torch::Tensor& reference_tensor,
                                           OutputPrecision output_precision,
                                           bool validate_size = true);

    bool validate_output_buffer_type(const torch::Tensor& output_tensor,
                                   OutputPrecision expected_precision,
                                   size_t expected_size);

    torch::Tensor convert_output_precision(const torch::Tensor& output_tensor,
                                         OutputPrecision source_precision,
                                         OutputPrecision target_precision);

    size_t calculate_expected_buffer_size(const torch::Tensor& reference_tensor,
                                        OutputPrecision precision);

    // Utility functions for Python binding
    bool is_metal_available();
    std::tuple<int, int, int> get_version();
}

} // namespace metal_sdpa

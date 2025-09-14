#ifndef MFA_FFI_H
#define MFA_FFI_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

/**
 * @brief Error codes for MFA operations
 *
 * All functions return 0 on success, positive values on error.
 */
typedef enum {
    MFA_SUCCESS = 0,                        ///< Operation completed successfully
    MFA_ERROR_INVALID_ARGS = 1,            ///< Invalid arguments provided
    MFA_ERROR_MEMORY_ALLOCATION = 2,       ///< Memory allocation failed
    MFA_ERROR_DEVICE_NOT_SUPPORTED = 3,    ///< Metal device not available
    MFA_ERROR_KERNEL_COMPILATION = 4,      ///< Metal kernel compilation failed
    MFA_ERROR_EXECUTION_FAILED = 5         ///< Kernel execution failed
} mfa_error_t;

/**
 * @brief Precision types for tensor operations
 *
 * Controls the precision used for different stages of computation.
 */
typedef enum {
    MFA_PRECISION_FP16 = 0,                ///< Half precision (16-bit)
    MFA_PRECISION_BF16 = 1,                ///< Brain float (16-bit)
    MFA_PRECISION_FP32 = 2                 ///< Single precision (32-bit)
} mfa_precision_t;

/**
 * @brief Opaque handle to MFA context (Metal device/command queue)
 */
typedef void* mfa_context_t;

/**
 * @brief Opaque handle to MFA buffer (Metal buffer)
 */
typedef void* mfa_buffer_t;

// =============================================================================
// Context Management
// =============================================================================

/**
 * @brief Create a new MFA context
 *
 * @param[out] context Pointer to store the created context handle
 * @return MFA_SUCCESS on success, error code on failure
 */
mfa_error_t mfa_create_context(mfa_context_t* context);

/**
 * @brief Destroy an MFA context and release associated resources
 *
 * @param context The context to destroy
 */
void mfa_destroy_context(mfa_context_t context);

// =============================================================================
// Buffer Management
// =============================================================================

/**
 * @brief Create a new buffer with the specified size
 *
 * @param context The MFA context
 * @param size_bytes Size of the buffer in bytes
 * @param[out] buffer Pointer to store the created buffer handle
 * @return MFA_SUCCESS on success, error code on failure
 */
mfa_error_t mfa_create_buffer(
    mfa_context_t context,
    size_t size_bytes,
    mfa_buffer_t* buffer
);

/**
 * @brief Create a buffer from existing data pointer
 *
 * @param context The MFA context
 * @param data_ptr Existing data to wrap
 * @param size_bytes Size of the data in bytes
 * @param[out] buffer Pointer to store the created buffer handle
 * @return MFA_SUCCESS on success, error code on failure
 */
mfa_error_t mfa_buffer_from_ptr(
    mfa_context_t context,
    void* data_ptr,
    size_t size_bytes,
    mfa_buffer_t* buffer
);

/**
 * @brief Get the contents pointer of a buffer
 *
 * @param buffer The buffer to access
 * @return Pointer to buffer contents, or NULL on error
 */
void* mfa_buffer_contents(mfa_buffer_t buffer);

/**
 * @brief Destroy a buffer and release associated resources
 *
 * @param buffer The buffer to destroy
 */
void mfa_destroy_buffer(mfa_buffer_t buffer);

// =============================================================================
// Flash Attention Operations
// =============================================================================

/**
 * @brief Perform Flash Attention forward pass
 *
 * Computes scaled dot-product attention: Attention(Q,K,V) = softmax(QK^T/√d)V
 * Using the Flash Attention algorithm for memory efficiency.
 *
 * @param context The MFA context
 * @param q Query tensor buffer: [batch_size, seq_len_q, num_heads, head_dim]
 * @param k Key tensor buffer: [batch_size, seq_len_kv, num_heads, head_dim]
 * @param v Value tensor buffer: [batch_size, seq_len_kv, num_heads, head_dim]
 * @param out Output tensor buffer: [batch_size, seq_len_q, num_heads, head_dim]
 * @param batch_size Number of sequences in the batch
 * @param seq_len_q Sequence length for queries
 * @param seq_len_kv Sequence length for keys and values
 * @param num_heads Number of attention heads (currently limited to 1)
 * @param head_dim Dimension of each attention head
 * @param softmax_scale Scaling factor for attention scores (typically 1/√head_dim)
 * @param causal Whether to apply causal (lower triangular) mask
 * @param input_precision Precision for Q, K, V tensors
 * @param intermediate_precision Precision for intermediate computations (S, P)
 * @param output_precision Precision for output tensor
 * @param transpose_q Whether Q tensor is transposed (false = row-major)
 * @param transpose_k Whether K tensor is transposed
 * @param transpose_v Whether V tensor is transposed
 * @param transpose_o Whether output tensor is transposed
 *
 * @return MFA_SUCCESS on success, error code on failure
 */
mfa_error_t mfa_attention_forward(
    mfa_context_t context,
    mfa_buffer_t q,
    mfa_buffer_t k,
    mfa_buffer_t v,
    mfa_buffer_t out,
    uint32_t batch_size,
    uint32_t seq_len_q,
    uint32_t seq_len_kv,
    uint32_t num_heads,
    uint16_t head_dim,
    float softmax_scale,
    bool causal,
    mfa_precision_t input_precision,
    mfa_precision_t intermediate_precision,
    mfa_precision_t output_precision,
    bool transpose_q,
    bool transpose_k,
    bool transpose_v,
    bool transpose_o
);

mfa_error_t mfa_attention_backward(
    mfa_context_t context,
    // Input tensors
    mfa_buffer_t dout,        // Gradient w.r.t output
    mfa_buffer_t q,           // Query (from forward pass)
    mfa_buffer_t k,           // Key (from forward pass)
    mfa_buffer_t v,           // Value (from forward pass)
    mfa_buffer_t out,         // Output (from forward pass)
    mfa_buffer_t softmax_lse, // Log-sum-exp from forward pass
    // Output gradients
    mfa_buffer_t dq,          // Gradient w.r.t Q
    mfa_buffer_t dk,          // Gradient w.r.t K
    mfa_buffer_t dv,          // Gradient w.r.t V
    // Dimensions
    uint32_t batch_size,
    uint32_t seq_len_q,
    uint32_t seq_len_kv,
    uint32_t num_heads,
    uint16_t head_dim,
    // Optional parameters
    float softmax_scale,
    bool causal,
    // Precision control
    mfa_precision_t input_precision,
    mfa_precision_t intermediate_precision,
    mfa_precision_t output_precision,
    // Layout control
    bool transpose_q,
    bool transpose_k,
    bool transpose_v,
    bool transpose_o
);

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * @brief Get a human-readable error message for an error code
 *
 * @param error The error code
 * @return Pointer to error message string (caller must free with free())
 */
const char* mfa_error_string(mfa_error_t error);

/**
 * @brief Check if Metal is supported on this device
 *
 * @return true if Metal device is available, false otherwise
 */
bool mfa_is_device_supported(void);

/**
 * @brief Get the version of the MFA library
 *
 * @param[out] major Major version number
 * @param[out] minor Minor version number
 * @param[out] patch Patch version number
 */
void mfa_get_version(int* major, int* minor, int* patch);

/**
 * @brief Get GPU execution time of the last attention operation
 *
 * Returns the pure GPU execution time (excluding CPU overhead) of the
 * most recent attention operation. This provides zero-overhead timing
 * measurements equivalent to native Swift benchmarks.
 *
 * @param context The MFA context
 * @return GPU execution time in seconds, or 0.0 on error
 */
double mfa_get_gpu_latency(mfa_context_t context);

#ifdef __cplusplus
}
#endif

#endif // MFA_FFI_H
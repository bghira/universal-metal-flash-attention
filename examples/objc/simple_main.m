#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>

// C FFI function declarations (same as used by Rust)
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
    MFA_PRECISION_FP32 = 2
} mfa_precision_t;

typedef void* mfa_context_t;
typedef void* mfa_buffer_t;

extern mfa_error_t mfa_create_context(mfa_context_t* context);
extern void mfa_destroy_context(mfa_context_t context);
extern mfa_error_t mfa_create_buffer(mfa_context_t context, size_t size_bytes, mfa_buffer_t* buffer);
extern void mfa_destroy_buffer(mfa_buffer_t buffer);
extern mfa_error_t mfa_attention_forward(mfa_context_t context,
                                        mfa_buffer_t q, mfa_buffer_t k, mfa_buffer_t v, mfa_buffer_t out,
                                        uint32_t batch_size, uint32_t seq_len_q, uint32_t seq_len_kv,
                                        uint32_t num_heads, uint16_t head_dim, float softmax_scale,
                                        bool causal, mfa_precision_t input_precision,
                                        mfa_precision_t intermediate_precision, mfa_precision_t output_precision,
                                        bool transpose_q, bool transpose_k, bool transpose_v, bool transpose_o);
extern double mfa_get_gpu_latency(mfa_context_t context);
extern bool mfa_is_device_supported();

int main() {
    printf("🎯 Simplified Metal Flash Attention - Objective-C C FFI\n");
    printf("======================================================\n");
    printf("(Using same C FFI as Rust→FFI→Swift for identical performance)\n\n");

    // Check Metal device support (same as Rust)
    if (!mfa_is_device_supported()) {
        printf("❌ Metal device not supported\n");
        return 1;
    }
    printf("✅ Metal device is supported\n");

    // Create MFA context (same as Rust)
    mfa_context_t context;
    mfa_error_t result = mfa_create_context(&context);
    if (result != MFA_SUCCESS) {
        printf("❌ Failed to create MFA context: %d\n", result);
        return 1;
    }
    printf("✅ Created MFA context\n");

    // Test configurations matching Rust benchmarks
    struct {
        int seqLen;
        int headDim;
        const char* name;
    } configs[] = {
        {1024, 16, "1024x16"},
        {1024, 64, "1024x64"},
        {1024, 256, "1024x256"}
    };

    printf("\n📊 Apples-to-Apples Performance Comparison\n");
    printf("--------------------------------------------------\n");
    printf("Config         FWD (GINSTRS/s)\n");
    printf("--------------------------------------------------\n");

    for (int i = 0; i < 3; i++) {
        int seqLen = configs[i].seqLen;
        int headDim = configs[i].headDim;

        // Create buffers using C FFI (FP32 like Rust)
        size_t bufferSize = seqLen * headDim * sizeof(float);

        mfa_buffer_t qBuffer, kBuffer, vBuffer, oBuffer;
        if (mfa_create_buffer(context, bufferSize, &qBuffer) != MFA_SUCCESS ||
            mfa_create_buffer(context, bufferSize, &kBuffer) != MFA_SUCCESS ||
            mfa_create_buffer(context, bufferSize, &vBuffer) != MFA_SUCCESS ||
            mfa_create_buffer(context, bufferSize, &oBuffer) != MFA_SUCCESS) {
            printf("❌ Failed to create buffers for %s\n", configs[i].name);
            continue;
        }

        // Warmup runs (3 runs like Rust)
        for (int w = 0; w < 3; w++) {
            mfa_attention_forward(context, qBuffer, kBuffer, vBuffer, oBuffer,
                                1,                        // batch_size
                                seqLen,                   // seq_len_q
                                seqLen,                   // seq_len_kv
                                1,                        // num_heads
                                headDim,                  // head_dim
                                1.0f / sqrtf((float)headDim), // softmax_scale
                                false,                    // causal
                                MFA_PRECISION_FP32,       // input_precision
                                MFA_PRECISION_FP32,       // intermediate_precision
                                MFA_PRECISION_FP32,       // output_precision
                                false, false, false, false); // transpose flags
        }

        // Benchmark runs (5 runs like Rust)
        double totalTime = 0.0;
        for (int b = 0; b < 5; b++) {
            mfa_error_t result = mfa_attention_forward(context, qBuffer, kBuffer, vBuffer, oBuffer,
                                                     1,                        // batch_size
                                                     seqLen,                   // seq_len_q
                                                     seqLen,                   // seq_len_kv
                                                     1,                        // num_heads
                                                     headDim,                  // head_dim
                                                     1.0f / sqrtf((float)headDim), // softmax_scale
                                                     false,                    // causal
                                                     MFA_PRECISION_FP32,       // input_precision
                                                     MFA_PRECISION_FP32,       // intermediate_precision
                                                     MFA_PRECISION_FP32,       // output_precision
                                                     false, false, false, false); // transpose flags

            if (result != MFA_SUCCESS) {
                printf("❌ Attention forward failed for %s: %d\n", configs[i].name, result);
                break;
            }

            // Get GPU timing (same as Rust approach)
            totalTime += mfa_get_gpu_latency(context);
        }

        double meanTime = totalTime / 5.0;

        // Calculate GINSTRS/s using EXACT Swift formula: (2*D + 5) * N² * 5 (for 5x dispatch)
        long operations = (2 * headDim + 5) * (long)seqLen * seqLen * 5;
        double ginstrsPerSec = (operations / 1e9) / meanTime;

        printf("%-14s %8.0f\n", configs[i].name, ginstrsPerSec);

        // Clean up buffers
        mfa_destroy_buffer(qBuffer);
        mfa_destroy_buffer(kBuffer);
        mfa_destroy_buffer(vBuffer);
        mfa_destroy_buffer(oBuffer);
    }

    printf("--------------------------------------------------\n");
    printf("\n📈 Performance Analysis:\n");
    printf("   • Direct C FFI calls (identical to Rust→FFI→Swift)\n");
    printf("   • Uses same caching & dispatch patterns via MFABridge.swift\n");
    printf("   • Zero-copy buffer management\n");
    printf("   • GPU timing eliminates CPU overhead\n");

    // Clean up context
    mfa_destroy_context(context);

    return 0;
}

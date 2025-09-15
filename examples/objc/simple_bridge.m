#import "simple_bridge.h"

// Include the MFA C FFI headers (same as used by Rust)
// Using exact function signatures from bindings.rs
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

// Simplified bridge using the same C FFI as Rust (for exact performance parity)
@interface SimpleBridge()
@property (nonatomic, assign) mfa_context_t context;
@end

@implementation SimpleBridge

- (instancetype)initWithDevice:(id<MTLDevice>)device {
    self = [super init];
    if (self) {
        // Use the same C FFI context as Rust->FFI->Swift
        mfa_context_t context;
        mfa_error_t result = mfa_create_context(&context);
        if (result != MFA_SUCCESS) {
            return nil;
        }
        _context = context;
    }
    return self;
}

- (void)dealloc {
    if (_context) {
        mfa_destroy_context(_context);
        _context = NULL;
    }
}

- (id<MTLBuffer>)createBufferWithSize:(NSUInteger)size {
    mfa_buffer_t buffer;
    mfa_error_t result = mfa_create_buffer(_context, size, &buffer);
    if (result != MFA_SUCCESS) {
        return nil;
    }
    return (__bridge id<MTLBuffer>)buffer;
}

- (double)attentionForwardWithQ:(id<MTLBuffer>)q
                              k:(id<MTLBuffer>)k
                              v:(id<MTLBuffer>)v
                            out:(id<MTLBuffer>)out
                      batchSize:(uint32_t)batchSize
                        seqLenQ:(uint32_t)seqLenQ
                       seqLenKV:(uint32_t)seqLenKV
                       numHeads:(uint32_t)numHeads
                        headDim:(uint16_t)headDim
                   softmaxScale:(float)softmaxScale
                         causal:(BOOL)causal
                 inputPrecision:(int32_t)inputPrecision
          intermediatePrecision:(int32_t)intermediatePrecision
                outputPrecision:(int32_t)outputPrecision
                     transposeQ:(BOOL)transposeQ
                     transposeK:(BOOL)transposeK
                     transposeV:(BOOL)transposeV
                     transposeO:(BOOL)transposeO {

    // Use exactly the same C FFI call as Rust (for identical performance)
    mfa_error_t result = mfa_attention_forward(_context,
                                              (__bridge mfa_buffer_t)q, (__bridge mfa_buffer_t)k,
                                              (__bridge mfa_buffer_t)v, (__bridge mfa_buffer_t)out,
                                              batchSize, seqLenQ, seqLenKV, numHeads, headDim,
                                              softmaxScale, causal ? true : false,
                                              (mfa_precision_t)inputPrecision,
                                              (mfa_precision_t)intermediatePrecision,
                                              (mfa_precision_t)outputPrecision,
                                              transposeQ ? true : false, transposeK ? true : false,
                                              transposeV ? true : false, transposeO ? true : false);

    return result == MFA_SUCCESS ? 0.0 : -1.0;  // Return success/failure
}

- (double)getGpuLatency {
    return mfa_get_gpu_latency(_context);
}

- (NSString*)getVersion {
    return @"1.0.0-c-ffi-bridge";
}

@end

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

// Import Swift FlashAttention types
@interface AttentionDescriptor : NSObject
@property (nonatomic, assign) BOOL lowPrecisionInputs;
@property (nonatomic, assign) BOOL lowPrecisionIntermediates;
@property (nonatomic, assign) struct {
    uint32_t row;
    uint32_t column;
    uint16_t head;
} matrixDimensions;
@property (nonatomic, assign) struct {
    BOOL Q;
    BOOL K;
    BOOL V;
    BOOL O;
} transposeState;
@property (nonatomic, assign) NSInteger maskType; // 0 = none, 1 = causal
- (void*)kernelDescriptorWithType:(NSInteger)type;
- (void)setFunctionConstants:(MTLFunctionConstantValues*)constants;
@end

@interface AttentionKernel : NSObject
@property (nonatomic, readonly) struct {
    uint16_t parallelization;
    uint16_t traversal;
    uint16_t head;
} blockDimensions;
@property (nonatomic, readonly) NSUInteger threadgroupSize;
@property (nonatomic, readonly) NSUInteger threadgroupMemoryAllocation;
- (instancetype)initWithDescriptor:(void*)descriptor;
- (NSString*)createSource;
@end

typedef NS_ENUM(NSInteger, AttentionMaskType) {
    AttentionMaskTypeNone = 0,
    AttentionMaskTypeCausal = 1
};

typedef NS_ENUM(NSInteger, AttentionKernelType) {
    AttentionKernelTypeForward = 0
};

// Objective-C bridge to Swift FlashAttention
@interface MFABridge : NSObject

// Initialize with Metal device
- (instancetype)initWithDevice:(id<MTLDevice>)device;

// Create buffers for attention computation
- (id<MTLBuffer>)createBufferWithSize:(NSUInteger)size;

// Run attention forward pass - returns execution time in seconds
- (double)runAttentionWithQ:(id<MTLBuffer>)qBuffer
                          K:(id<MTLBuffer>)kBuffer
                          V:(id<MTLBuffer>)vBuffer
                          O:(id<MTLBuffer>)oBuffer
                  seqLength:(NSUInteger)seqLength
                   headDim:(NSUInteger)headDim
                      scale:(float)scale
                     causal:(BOOL)causal;

// Get version info
- (NSString*)getVersion;

@end

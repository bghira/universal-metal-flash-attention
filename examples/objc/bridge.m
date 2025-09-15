#import "bridge.h"

// Cache key for deduplicating compiled kernels (matching Swift implementation)
@interface PipelineCacheKey : NSObject <NSCopying>
@property (nonatomic, assign) NSUInteger seqLen;
@property (nonatomic, assign) NSUInteger headDim;
@property (nonatomic, assign) BOOL causal;
@end

@implementation PipelineCacheKey

- (id)copyWithZone:(NSZone *)zone {
    PipelineCacheKey *copy = [[PipelineCacheKey alloc] init];
    copy.seqLen = self.seqLen;
    copy.headDim = self.headDim;
    copy.causal = self.causal;
    return copy;
}

- (BOOL)isEqual:(id)object {
    if (![object isKindOfClass:[PipelineCacheKey class]]) return NO;
    PipelineCacheKey *other = (PipelineCacheKey *)object;
    return self.seqLen == other.seqLen &&
           self.headDim == other.headDim &&
           self.causal == other.causal;
}

- (NSUInteger)hash {
    return self.seqLen ^ (self.headDim << 16) ^ (self.causal ? 1 : 0);
}

@end

@interface MFABridge()
@property (nonatomic, strong) id<MTLDevice> device;
@property (nonatomic, strong) id<MTLCommandQueue> commandQueue;

// Comprehensive caching system (matching Swift MFABridge optimizations)
@property (nonatomic, strong) NSMutableDictionary<PipelineCacheKey*, id<MTLComputePipelineState>> *pipelineCache;
@property (nonatomic, strong) NSMutableDictionary<PipelineCacheKey*, AttentionKernel*> *kernelCache;
@property (nonatomic, strong) NSMutableDictionary<NSNumber*, id<MTLBuffer>> *lBufferCache;
@property (nonatomic, strong) NSMutableDictionary<NSNumber*, id<MTLBuffer>> *dBufferCache;
@property (nonatomic, strong) dispatch_queue_t cacheQueue;
@end

@implementation MFABridge

- (instancetype)initWithDevice:(id<MTLDevice>)device {
    self = [super init];
    if (self) {
        _device = device;
        _commandQueue = [device newCommandQueue];

        // Initialize comprehensive caching system (matching Swift optimizations)
        _pipelineCache = [[NSMutableDictionary alloc] init];
        _kernelCache = [[NSMutableDictionary alloc] init];
        _lBufferCache = [[NSMutableDictionary alloc] init];
        _dBufferCache = [[NSMutableDictionary alloc] init];
        _cacheQueue = dispatch_queue_create("MFABridge.caches", DISPATCH_QUEUE_SERIAL);
    }
    return self;
}

- (id<MTLBuffer>)createBufferWithSize:(NSUInteger)size {
    return [self.device newBufferWithLength:size options:MTLResourceStorageModeShared];
}

// Cache helper methods (matching Swift MFABridge implementation)
- (NSArray*)getCachedPipelineForKey:(PipelineCacheKey*)key {
    __block NSArray *result = nil;
    dispatch_sync(self.cacheQueue, ^{
        id<MTLComputePipelineState> pipeline = self.pipelineCache[key];
        AttentionKernel *kernel = self.kernelCache[key];
        if (pipeline && kernel) {
            result = @[pipeline, kernel];
        }
    });
    return result;
}

- (void)cachePipeline:(id<MTLComputePipelineState>)pipeline kernel:(AttentionKernel*)kernel forKey:(PipelineCacheKey*)key {
    dispatch_sync(self.cacheQueue, ^{
        self.pipelineCache[key] = pipeline;
        self.kernelCache[key] = kernel;
    });
}

- (id<MTLBuffer>)getCachedLBufferForSeqLen:(NSUInteger)seqLen {
    __block id<MTLBuffer> buffer = nil;
    dispatch_sync(self.cacheQueue, ^{
        NSNumber *key = @(seqLen);
        buffer = self.lBufferCache[key];
        if (!buffer) {
            NSUInteger bufferSize = seqLen * sizeof(float);
            buffer = [self.device newBufferWithLength:bufferSize options:MTLResourceStorageModeShared];
            if (buffer) {
                self.lBufferCache[key] = buffer;
            }
        }
    });
    return buffer;
}

- (id<MTLBuffer>)getCachedDBufferForSeqLen:(NSUInteger)seqLen {
    __block id<MTLBuffer> buffer = nil;
    dispatch_sync(self.cacheQueue, ^{
        NSNumber *key = @(seqLen);
        buffer = self.dBufferCache[key];
        if (!buffer) {
            NSUInteger bufferSize = seqLen * sizeof(float);
            buffer = [self.device newBufferWithLength:bufferSize options:MTLResourceStorageModeShared];
            if (buffer) {
                self.dBufferCache[key] = buffer;
            }
        }
    });
    return buffer;
}

- (double)runAttentionWithQ:(id<MTLBuffer>)qBuffer
                          K:(id<MTLBuffer>)kBuffer
                          V:(id<MTLBuffer>)vBuffer
                          O:(id<MTLBuffer>)oBuffer
                  seqLength:(NSUInteger)seqLength
                   headDim:(NSUInteger)headDim
                      scale:(float)scale
                     causal:(BOOL)causal {

    // Create cache key for pipeline deduplication (matching Swift implementation)
    PipelineCacheKey *cacheKey = [[PipelineCacheKey alloc] init];
    cacheKey.seqLen = seqLength;
    cacheKey.headDim = headDim;
    cacheKey.causal = causal;

    // Check if we have cached pipeline and kernel
    id<MTLComputePipelineState> pipeline = nil;
    AttentionKernel *kernel = nil;

    NSArray *cached = [self getCachedPipelineForKey:cacheKey];
    if (cached) {
        pipeline = cached[0];
        kernel = cached[1];
    } else {
        // Create AttentionDescriptor (using native causal masking approach)
        AttentionDescriptor *desc = [[AttentionDescriptor alloc] init];
        desc.lowPrecisionInputs = NO;  // FP32
        desc.lowPrecisionIntermediates = NO;  // FP32
        desc.matrixDimensions = (AttentionMatrixDimensions){
            .row = (uint32_t)seqLength,
            .column = (uint32_t)seqLength,
            .head = (uint16_t)headDim
        };
        desc.transposeState = (AttentionTransposeState){
            .Q = NO, .K = NO, .V = NO, .O = NO
        };

        // Set causal masking using proper native approach (like optimized Swift)
        desc.maskType = causal ? AttentionMaskTypeCausal : AttentionMaskTypeNone;

        // Create kernel descriptor
        AttentionKernelDescriptor *kernelDesc = [desc kernelDescriptorWithType:AttentionKernelTypeForward];
        kernel = [[AttentionKernel alloc] initWithDescriptor:kernelDesc];

        // Set up function constants
        MTLFunctionConstantValues *constants = [[MTLFunctionConstantValues alloc] init];
        [desc setFunctionConstants:constants];

        // Get the Metal function using native kernel source (no string replacement!)
        NSString *source = [kernel createSource];
        NSError *error = nil;
        id<MTLLibrary> library = [self.device newLibraryWithSource:source options:nil error:&error];
        if (error) {
            NSLog(@"Library creation error: %@", error);
            return -1.0;
        }

        id<MTLFunction> function = [library newFunctionWithName:@"attention" constantValues:constants error:&error];
        if (error) {
            NSLog(@"Function creation error: %@", error);
            return -1.0;
        }

        // Create pipeline descriptor with proper settings for Apple Silicon
        MTLComputePipelineDescriptor *pipelineDesc = [[MTLComputePipelineDescriptor alloc] init];
        pipelineDesc.computeFunction = function;
        pipelineDesc.maxTotalThreadsPerThreadgroup = 1024;  // Critical for M1/M2/M3 performance

        pipeline = [self.device newComputePipelineStateWithDescriptor:pipelineDesc
                                                              options:MTLPipelineOptionNone
                                                           reflection:nil
                                                                error:&error];
        if (error) {
            NSLog(@"Pipeline creation error: %@", error);
            return -1.0;
        }

        // Cache the compiled pipeline and kernel
        [self cachePipeline:pipeline kernel:kernel forKey:cacheKey];
    }

    // Create command buffer
    id<MTLCommandBuffer> commandBuffer = [self.commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

    [encoder setComputePipelineState:pipeline];

    // Get cached L and D buffers (avoid reallocation like native Swift)
    id<MTLBuffer> lBuffer = [self getCachedLBufferForSeqLen:seqLength];
    id<MTLBuffer> dBuffer = [self getCachedDBufferForSeqLen:seqLength];
    if (!lBuffer || !dBuffer) {
        NSLog(@"Failed to create cached L/D buffers");
        return -1.0;
    }

    // Set buffers (following MFA test pattern)
    [encoder setBuffer:qBuffer offset:0 atIndex:0];
    [encoder setBuffer:kBuffer offset:0 atIndex:1];
    [encoder setBuffer:vBuffer offset:0 atIndex:2];
    [encoder setBuffer:oBuffer offset:0 atIndex:3];
    [encoder setBuffer:lBuffer offset:0 atIndex:4];  // L buffer (attention statistics)
    [encoder setBuffer:dBuffer offset:0 atIndex:5];  // D buffer (attention statistics)

    // Set threadgroup memory
    [encoder setThreadgroupMemoryLength:kernel.threadgroupMemoryAllocation atIndex:0];

    // Dispatch using MFA's calculation method
    NSUInteger blockCount = (seqLength + kernel.blockDimensions.parallelization - 1) / kernel.blockDimensions.parallelization;
    MTLSize gridSize = MTLSizeMake(blockCount, 1, 1);
    MTLSize groupSize = MTLSizeMake(kernel.threadgroupSize, 1, 1);

    // Multiple dispatches like native Swift (dispatchCount = 5) to amortize GPU setup costs
    NSInteger dispatchCount = 5;
    for (NSInteger i = 0; i < dispatchCount; i++) {
        [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:groupSize];
    }
    [encoder endEncoding];

    // Execute and measure GPU time (not wall-clock time!)
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    if (commandBuffer.error) {
        NSLog(@"Metal execution error: %@", commandBuffer.error);
        return -1.0;
    }

    // Note: We dispatched 5x, so execution time includes 5x work
    // This matches native Swift approach of multiple dispatches per measurement
    return commandBuffer.GPUEndTime - commandBuffer.GPUStartTime;
}

- (NSString*)getVersion {
    return @"1.0.0-objc";
}

@end

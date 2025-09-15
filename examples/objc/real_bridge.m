#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

// Forward declare the Swift class - it will be available at runtime
@interface SwiftAttentionWrapper : NSObject
- (instancetype)initWithDevice:(id<MTLDevice>)device;
- (id<MTLBuffer>)createBufferWithSize:(NSInteger)size;
- (double)runAttentionWithQBuffer:(id<MTLBuffer>)qBuffer
                          kBuffer:(id<MTLBuffer>)kBuffer
                          vBuffer:(id<MTLBuffer>)vBuffer
                          oBuffer:(id<MTLBuffer>)oBuffer
                        seqLength:(NSInteger)seqLength
                          headDim:(NSInteger)headDim
                            scale:(float)scale
                           causal:(BOOL)causal;
- (NSString*)getVersion;
@end

// C functions callable from Rust
void* create_swift_attention_wrapper(void* device_ptr) {
    id<MTLDevice> device = (__bridge id<MTLDevice>)device_ptr;
    SwiftAttentionWrapper* wrapper = [[SwiftAttentionWrapper alloc] initWithDevice:device];
    return (__bridge_retained void*)wrapper;
}

void* swift_create_buffer(void* wrapper_ptr, size_t size) {
    SwiftAttentionWrapper* wrapper = (__bridge SwiftAttentionWrapper*)wrapper_ptr;
    id<MTLBuffer> buffer = [wrapper createBufferWithSize:(NSInteger)size];
    return (__bridge_retained void*)buffer;
}

double swift_run_attention(void* wrapper_ptr,
                          void* q_buffer_ptr,
                          void* k_buffer_ptr,
                          void* v_buffer_ptr,
                          void* o_buffer_ptr,
                          size_t seq_length,
                          size_t head_dim,
                          float scale,
                          bool causal) {
    SwiftAttentionWrapper* wrapper = (__bridge SwiftAttentionWrapper*)wrapper_ptr;
    id<MTLBuffer> qBuffer = (__bridge id<MTLBuffer>)q_buffer_ptr;
    id<MTLBuffer> kBuffer = (__bridge id<MTLBuffer>)k_buffer_ptr;
    id<MTLBuffer> vBuffer = (__bridge id<MTLBuffer>)v_buffer_ptr;
    id<MTLBuffer> oBuffer = (__bridge id<MTLBuffer>)o_buffer_ptr;

    return [wrapper runAttentionWithQBuffer:qBuffer
                                    kBuffer:kBuffer
                                    vBuffer:vBuffer
                                    oBuffer:oBuffer
                                  seqLength:(NSInteger)seq_length
                                    headDim:(NSInteger)head_dim
                                      scale:scale
                                     causal:causal ? YES : NO];
}

const char* swift_get_version(void* wrapper_ptr) {
    SwiftAttentionWrapper* wrapper = (__bridge SwiftAttentionWrapper*)wrapper_ptr;
    NSString* version = [wrapper getVersion];
    return [version UTF8String];
}

// Create Metal device (same as before)
void* create_metal_device() {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    return (__bridge_retained void*)device;
}

// Cleanup
void release_object(void* obj_ptr) {
    if (obj_ptr) {
        CFBridgingRelease(obj_ptr);
    }
}

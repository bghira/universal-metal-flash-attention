#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

// Simplified header - we'll use MFABridge.swift directly
// No need to reimplement the complex caching logic
@interface SimpleBridge : NSObject

- (instancetype)initWithDevice:(id<MTLDevice>)device;
- (id<MTLBuffer>)createBufferWithSize:(NSUInteger)size;
- (NSString*)getVersion;

@end

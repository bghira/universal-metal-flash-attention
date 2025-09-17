#include "mps_utils.h"

#if defined(__APPLE__)
#import <Metal/Metal.h>
#include <ATen/native/mps/OperationUtils.h>

namespace metal_sdpa {
namespace mps_utils {

bool is_mps_tensor(const at::Tensor& tensor) {
    return tensor.device().type() == c10::DeviceType::MPS;
}

void* get_mtl_buffer_handle(const at::Tensor& tensor) {
    if (!is_mps_tensor(tensor)) {
        return nullptr;
    }

    id<MTLBuffer> buffer = at::native::mps::getMTLBufferStorage(tensor);
    if (buffer == nil) {
        return nullptr;
    }
    return (__bridge void*)buffer;
}

} // namespace mps_utils
} // namespace metal_sdpa

#else

namespace metal_sdpa {
namespace mps_utils {

bool is_mps_tensor(const at::Tensor&) {
    return false;
}

void* get_mtl_buffer_handle(const at::Tensor&) {
    return nullptr;
}

} // namespace mps_utils
} // namespace metal_sdpa

#endif


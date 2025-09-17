#pragma once

#include <ATen/Tensor.h>

namespace metal_sdpa {
namespace mps_utils {

bool is_mps_tensor(const at::Tensor& tensor);

// Returns opaque pointer to id<MTLBuffer> (or nullptr on failure)
void* get_mtl_buffer_handle(const at::Tensor& tensor);

} // namespace mps_utils
} // namespace metal_sdpa


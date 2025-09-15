#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include "metal_sdpa_backend.h"

// Forward declarations from the header
extern "C" {
    bool mfa_is_device_supported(void);
    void mfa_get_version(int* major, int* minor, int* patch);
}

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "PyTorch Custom SDPA Backend with Metal Flash Attention via Swift FFI";

    // Register/unregister backend functions
    m.def("register_backend", &metal_sdpa::MetalSDPABackend::register_backend,
          "Register Metal SDPA backend with PyTorch");

    m.def("unregister_backend", &metal_sdpa::MetalSDPABackend::unregister_backend,
          "Unregister Metal SDPA backend from PyTorch");

    // Direct SDPA call (for testing/debugging)
    m.def("metal_scaled_dot_product_attention",
          &metal_sdpa::MetalSDPABackend::scaled_dot_product_attention,
          "Direct call to Metal Flash Attention SDPA",
          py::arg("query"),
          py::arg("key"),
          py::arg("value"),
          py::arg("attn_mask") = py::none(),
          py::arg("dropout_p") = 0.0,
          py::arg("is_causal") = false,
          py::arg("scale") = py::none(),
          py::arg("enable_gqa") = false);

    // Utility functions
    m.def("is_metal_available", []() { return mfa_is_device_supported(); },
          "Check if Metal is available on this device");

    m.def("get_version", []() {
        int major, minor, patch;
        mfa_get_version(&major, &minor, &patch);
        return std::make_tuple(major, minor, patch);
    }, "Get Metal Flash Attention version");

    // Backend context manager helper
    py::class_<metal_sdpa::MetalSDPABackend>(m, "MetalSDPABackend")
        .def_static("register_backend", &metal_sdpa::MetalSDPABackend::register_backend)
        .def_static("unregister_backend", &metal_sdpa::MetalSDPABackend::unregister_backend);
}

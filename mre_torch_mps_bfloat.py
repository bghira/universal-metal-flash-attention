#!/usr/bin/env python3
"""
MRE: importing torch lowers the Metal device's default shading-language
version, which silently undefines `__HAVE_BFLOAT__` for kernels compiled at
runtime via `device.makeLibrary(source:options:)`.

Effect: after `import torch` (which initialises the MPS backend), any Metal
source that uses the `bfloat` type fails to JIT with:
    error: unknown type name 'bfloat'
even on bf16-capable hardware (e.g. Apple M2+), where the identical source
compiles fine in a process that has NOT imported torch.

Before torch:  __HAVE_BFLOAT__ defined  -> bfloat compiles.
After  torch:  __HAVE_BFLOAT__ undefined -> bfloat fails.
After  torch + forcing MSL 3.2 (languageVersion): bfloat compiles again,
confirming the cause is the default language version, not a hardware limit.

Reproduces on GitHub's `macos-15` runner (virtualised Apple M1) and on real
Apple Silicon once torch.mps is initialised.

Dependencies: only `torch` (pip install torch). Uses stdlib ctypes to reach
the Metal framework directly, so no pyobjc/Metal-Python bindings required.
Run: python3 mre_torch_mps_bfloat.py
"""
import ctypes
from ctypes import POINTER, byref, c_char_p, c_uint32, c_void_p

OBJC = ctypes.CDLL("/usr/lib/libobjc.dylib")
METAL = ctypes.CDLL("/System/Library/Frameworks/Metal.framework/Metal")
CF = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")

METAL.MTLCreateSystemDefaultDevice.restype = c_void_p
METAL.MTLCreateSystemDefaultDevice.argtypes = []

OBJC.sel_registerName.restype = c_void_p
OBJC.sel_registerName.argtypes = [c_char_p]
OBJC.objc_getClass.restype = c_void_p
OBJC.objc_getClass.argtypes = [c_char_p]


def _send(nargs):
    # Distinct CFUNCTYPE per arity so the typed variants don't clobber each
    # other (they all wrap the same objc_msgSend symbol).
    proto = ctypes.CFUNCTYPE(c_void_p, c_void_p, c_void_p, *([c_void_p] * nargs))
    return proto(("objc_msgSend", OBJC))


send0 = _send(0)
send1 = _send(1)

# newLibraryWithSource:options:error: -- error param is NSError**
_make_lib = ctypes.CFUNCTYPE(
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, POINTER(c_void_p)
)(("objc_msgSend", OBJC))

CF.CFStringCreateWithCString.restype = c_void_p
CF.CFStringCreateWithCString.argtypes = [c_void_p, c_char_p, c_uint32]
_K_UTF8 = 0x08000100  # kCFStringEncodingUTF8

# MTLLanguageVersion3_2 = (3 << 16) + 2 = 0x30002  (Metal SDK enum value)
_MSL_3_2 = 0x30002

# Minimal Metal source that only compiles when __HAVE_BFLOAT__ is defined.
_SOURCE = """
#include <metal_stdlib>
using namespace metal;

#if !defined(__HAVE_BFLOAT__)
#error "NO_NATIVE_BFLOAT"
#endif

kernel void __bfloat_probe(device float* x [[buffer(0)]],
                           uint i [[thread_position_in_grid]]) {
  x[i] = 0;
}
"""


def _sel(name):
    return OBJC.sel_registerName(name.encode())


def _nsstring(s):
    return CF.CFStringCreateWithCString(None, s.encode("utf-8"), _K_UTF8)


def bfloat_compiles(force_msl32=False):
    """Compile the bfloat probe via the runtime Metal compiler (device.makeLibrary)."""
    dev = METAL.MTLCreateSystemDefaultDevice()
    if not dev:
        return False, "no Metal device"

    cls = OBJC.objc_getClass(b"MTLCompileOptions")
    opts = send0(send0(cls, _sel("alloc")), _sel("init"))
    if force_msl32:
        send1(opts, _sel("setLanguageVersion:"), c_void_p(_MSL_3_2))

    src = _nsstring(_SOURCE)
    err = c_void_p(0)
    lib = _make_lib(
        dev, _sel("newLibraryWithSource:options:error:"), src, opts, byref(err)
    )
    return bool(lib), None if lib else "compile error (bfloat unavailable)"


def main():
    print("Probing the runtime Metal compiler (device.makeLibrary):\n")

    before, _ = bfloat_compiles(force_msl32=False)
    print(f"bfloat compiles BEFORE import torch (default options): {before}")

    import torch  # noqa: F401  -- importing torch is exactly what triggers this

    print(f"\nimported torch {torch.__version__}")
    print(f"torch.mps available: {torch.backends.mps.is_available()}\n")

    after, _ = bfloat_compiles(force_msl32=False)
    after_forced, _ = bfloat_compiles(force_msl32=True)
    print(f"bfloat compiles AFTER  import torch (default options): {after}")
    print(f"bfloat compiles AFTER  import torch (forced MSL 3.2):  {after_forced}")

    print()
    if before and not after:
        print(">>> REPRODUCED: importing torch disabled __HAVE_BFLOAT__. <<<")
    if after and not after_forced:
        print(">>> unexpected: forcing MSL 3.2 did not restore bfloat. <<<")
    print()
    print("Diagnosis: torch's MPS initialisation lowers the Metal device's")
    print("default shading-language version, so __HAVE_BFLOAT__ is no longer")
    print("defined for kernels JIT-compiled at runtime -> 'unknown type name")
    print("bfloat'. Pinning languageVersion = .version3_2 overrides it.")
    print("Ref: https://github.com/bghira/universal-metal-flash-attention/issues/10")


if __name__ == "__main__":
    main()

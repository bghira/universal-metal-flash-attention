// This file provides a C interface that can be linked by other languages
// The actual implementation is in MFABridge.swift

#include "mfa_ffi.h"

// The Swift functions are declared with @_cdecl, so they are available as C symbols
// This file exists mainly to satisfy the build system and provide a linkable target

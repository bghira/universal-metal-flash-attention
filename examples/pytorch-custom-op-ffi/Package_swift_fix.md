# Swift Compiler Configuration Fix

**Author: bghira**

## Issue

The current Package.swift has unsafe compiler flags that can cause incorrect results:

```swift
// PROBLEMATIC - Current configuration
swiftSettings: [
    .unsafeFlags(["-O"], .when(configuration: .release)),
    .unsafeFlags(["-Ounchecked"], .when(configuration: .debug))  // ⚠️ DANGEROUS
]
```

## Recommended Fix

Replace the MFABridge target configuration with:

```swift
.target(
    name: "MFABridge",
    dependencies: [
        .product(name: "FlashAttention", package: "metal-flash-attention")
    ],
    publicHeadersPath: "include",
    swiftSettings: [
        // Release: Aggressive optimization with safety
        .unsafeFlags(["-O"], .when(configuration: .release)),
        .unsafeFlags(["-whole-module-optimization"], .when(configuration: .release)),
        .unsafeFlags(["-enable-library-evolution"], .when(configuration: .release)),

        // Debug: Safe debugging with assertions
        .unsafeFlags(["-Onone"], .when(configuration: .debug)),
        .unsafeFlags(["-g"], .when(configuration: .debug)),
        .unsafeFlags(["-DDEBUG"], .when(configuration: .debug)),

        // Metal-specific optimizations for both modes
        .define("METAL_SIMD_OPTIMIZATION"),
        .define("FAST_METAL_MATH")
    ]
)
```

## Why This Matters

1. **-Ounchecked removes ALL safety checks** - can cause silent corruption
2. **Metal operations need consistent precision** - unsafe math can cause NaN/Inf
3. **Debug builds should be debuggable** - not optimized unsafely

## Verification

After fixing, rebuild and verify:

```bash
cd /path/to/universal-metal-flash-attention
swift build --configuration release --product MFAFFI
swift build --configuration debug --product MFAFFI

# Test both configurations produce same results
python examples/pytorch-custom-op-ffi/simple_perf.py
```

## Impact

This fix should:

- ✅ Eliminate potential numerical instability
- ✅ Ensure consistent results across debug/release
- ✅ Maintain high performance in release builds
- ✅ Enable proper debugging in debug builds

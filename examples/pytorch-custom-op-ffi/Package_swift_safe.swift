// swift-tools-version: 5.10

import PackageDescription

let package = Package(
  name: "universal-metal-flash-attention",
  platforms: [
    .iOS(.v17),
    .macOS(.v14),
    .tvOS(.v17),
    .visionOS(.v1),
  ],
  products: [
    .library(
      name: "MFAFFI",
      type: .dynamic,
      targets: ["MFAFFI"]
    ),
  ],
  dependencies: [
    .package(path: "./metal-flash-attention"),
  ],
  targets: [
    .target(
      name: "MFABridge",
      dependencies: [
        .product(name: "FlashAttention", package: "metal-flash-attention"),
      ],
      publicHeadersPath: "include",
      swiftSettings: [
        // SAFE CONFIGURATION - Prioritize correctness over performance
        .unsafeFlags(["-O"], .when(configuration: .release)), // Safe optimization
        .unsafeFlags(["-whole-module-optimization"], .when(configuration: .release)),
        .unsafeFlags(["-enable-testing"], .when(configuration: .debug)), // Enable testing in debug
        .unsafeFlags(["-g"], .when(configuration: .debug)), // Debug symbols
        .unsafeFlags(["-Onone"], .when(configuration: .debug)), // No optimization in debug

        // Metal-specific safe optimizations
        .define("METAL_SAFE_MATH"),
        .define("SWIFT_PACKAGE"),
      ]
    ),
    .target(
      name: "MFAFFI",
      dependencies: ["MFABridge"],
      publicHeadersPath: "include",
      cSettings: [
        .unsafeFlags(["-O2"]), // Safe C optimization (not -O3)
        .unsafeFlags(["-fno-fast-math"]), // DISABLE fast math for correctness
        .unsafeFlags(["-fno-unsafe-math-optimizations"]), // DISABLE unsafe math
        .define("NDEBUG", .when(configuration: .release)),
      ]
    ),
    .testTarget(
      name: "MFAFFITests",
      dependencies: ["MFAFFI"]
    ),
    .testTarget(
      name: "FlashAttentionTests",
      dependencies: [
        "MFABridge",
        .product(name: "FlashAttention", package: "metal-flash-attention"),
      ]
    ),
    .testTarget(
      name: "QuantizationTests",
      dependencies: [
        "MFABridge",
        .product(name: "FlashAttention", package: "metal-flash-attention"),
      ]
    ),
    .testTarget(
      name: "PerformanceTests",
      dependencies: [
        "MFABridge",
        .product(name: "FlashAttention", package: "metal-flash-attention"),
      ]
    ),
  ]
)

// NOTES:
// 1. Removed -Ounchecked (DANGEROUS)
// 2. Removed -ffast-math (can cause numerical issues)
// 3. Added -fno-unsafe-math-optimizations (explicit safety)
// 4. Using -O2 instead of -O3 for C code (more conservative)
// 5. Safe Swift optimization flags only

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
        .unsafeFlags(["-O"], .when(configuration: .release)),
        .unsafeFlags(["-Ounchecked"], .when(configuration: .debug)),
      ]
    ),
    .target(
      name: "MFAFFI",
      dependencies: ["MFABridge"],
      publicHeadersPath: "include",
      cSettings: [
        .unsafeFlags(["-O3"]),
        .unsafeFlags(["-ffast-math"]),
        .unsafeFlags(["-funroll-loops"]),
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

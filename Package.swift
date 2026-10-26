// swift-tools-version: 6.0

import PackageDescription

let package = Package(
  name: "universal-metal-flash-attention",
  platforms: [
    .iOS(.v17),
    .macOS(.v15),
    .tvOS(.v17),
    .visionOS(.v1),
  ],
  products: [
    .library(
      name: "MFAFFI",
      type: .dynamic,
      targets: ["MFAFFI"]
    ),
    .executable(
      name: "AttentionBenchmark",
      targets: ["AttentionBenchmark"]
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
        .enableUpcomingFeature("StrictConcurrency"),
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
    .executableTarget(
      name: "AttentionBenchmark",
      dependencies: [
        .product(name: "FlashAttention", package: "metal-flash-attention"),
      ],
      path: "benchmarks",
      exclude: [
        "BenchmarkQuantizedAttention.swift",
        "GEMMBenchmark.swift",
        "GluonOptimizationBenchmark.swift",
        "MemoryBandwidthBenchmark.swift",
        "QuantizedBackwardBenchmark.swift",
        "SimpleGEMMBenchmark.swift",
      ],
      swiftSettings: [
        .enableUpcomingFeature("StrictConcurrency"),
      ]
    ),
    .testTarget(
      name: "MFAFFITests",
      dependencies: ["MFAFFI"],
      swiftSettings: [
        .enableUpcomingFeature("StrictConcurrency"),
      ]
    ),
    .testTarget(
      name: "FlashAttentionTests",
      dependencies: [
        "MFABridge",
        .product(name: "FlashAttention", package: "metal-flash-attention"),
      ],
      swiftSettings: [
        .enableUpcomingFeature("StrictConcurrency"),
      ]
    ),
    .testTarget(
      name: "QuantizationTests",
      dependencies: [
        "MFABridge",
        .product(name: "FlashAttention", package: "metal-flash-attention"),
      ],
      swiftSettings: [
        .enableUpcomingFeature("StrictConcurrency"),
      ]
    ),
    .testTarget(
      name: "PerformanceTests",
      dependencies: [
        "MFABridge",
        .product(name: "FlashAttention", package: "metal-flash-attention"),
      ],
      swiftSettings: [
        .enableUpcomingFeature("StrictConcurrency"),
      ]
    ),
  ]
)

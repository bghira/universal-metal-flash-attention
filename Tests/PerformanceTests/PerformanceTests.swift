import FlashAttention
import Metal
import XCTest

@testable import MFABridge

final class PerformanceTests: XCTestCase {

  func testQuantizationPerformance() throws {
    let sizes = [1024, 4096, 16384, 65536]

    for size in sizes {
      let testData = generateTestData(size: size)

      // Test INT8 quantization performance
      let int8Result = measureTime {
        quantizeInt8Fast(testData)
      }

      // Test INT4 quantization performance
      let int4Result = measureTime {
        quantizeInt4Fast(testData)
      }

      print("Size: \(size)")
      print("  INT8: \(String(format: "%.2f", int8Result.time * 1000)) ms")
      print("  INT4: \(String(format: "%.2f", int4Result.time * 1000)) ms")

      // Performance assertions
      XCTAssertLessThan(int8Result.time, 0.1, "INT8 quantization should be fast for size \(size)")
      XCTAssertLessThan(int4Result.time, 0.1, "INT4 quantization should be fast for size \(size)")

      // Verify compression ratios
      let originalSize = testData.count * MemoryLayout<Float>.size
      let int8Size = int8Result.result.0.count * MemoryLayout<Int8>.size
      let int4Size = int4Result.result.0.count * MemoryLayout<UInt8>.size

      XCTAssertEqual(int8Size, originalSize / 4, "INT8 should use 4x less memory")
      XCTAssertEqual(int4Size, (originalSize + 7) / 8, "INT4 should use ~8x less memory")
    }
  }

  func testQuantizedAttentionBenchmark() throws {
    guard let device = MTLCreateSystemDefaultDevice() else {
      throw XCTSkip("Metal device not available")
    }

    let quantizedAttention = QuantizedAttention(device: device)

    // Test multiple configurations to find where vectorization shows benefits
    let configs = [
      (batchSize: 1, sequenceLength: 512, headDim: 64, name: "Small"),
      (batchSize: 1, sequenceLength: 1024, headDim: 64, name: "Medium"),
      (batchSize: 1, sequenceLength: 2048, headDim: 128, name: "Large"),
    ]

    for config in configs {
      print("\n=== \(config.name) Configuration ===")
      let benchmarkResults = quantizedAttention.benchmark(
        batchSize: config.batchSize,
        sequenceLength: config.sequenceLength,
        headDim: config.headDim,
        iterations: 50  // Reduced since we're doing multiple configs
      )

      print("Results for \(config.name) (\(config.sequenceLength)x\(config.headDim)):")
      for (key, value) in benchmarkResults.sorted(by: { $0.key < $1.key }) {
        print("  \(key): \(String(format: "%.3f", value))")
      }
    }

    // Use the small config for the test assertions
    let benchmarkResults = quantizedAttention.benchmark(
      batchSize: 1,
      sequenceLength: 512,
      headDim: 64,
      iterations: 50
    )

    // Verify we get results for different precisions
    XCTAssertNotNil(benchmarkResults["FP16_avg_ms"])
    XCTAssertNotNil(benchmarkResults["INT8_avg_ms"])
    XCTAssertNotNil(benchmarkResults["INT4_avg_ms"])

    // Performance should be reasonable (not infinite/NaN)
    for (key, value) in benchmarkResults {
      XCTAssertFalse(value.isNaN, "\(key) should not be NaN")
      XCTAssertFalse(value.isInfinite, "\(key) should not be infinite")
      if key.contains("avg_ms") {
        XCTAssertGreaterThan(value, 0, "\(key) should be positive")
        XCTAssertLessThan(value, 1000, "\(key) should be reasonable (< 1 second)")
      }
    }

    print("Quantized Attention Benchmark Results:")
    for (key, value) in benchmarkResults.sorted(by: { $0.key < $1.key }) {
      print("  \(key): \(String(format: "%.2f", value))")
    }
  }

  func testMemoryEfficiencyComparison() throws {
    let elementCount = 65536
    let originalSize = elementCount * MemoryLayout<Float>.size

    let testData: [Float] = (0..<elementCount).map { Float($0) / Float(elementCount) }

    // INT8 quantization
    let (quantizedInt8, _) = quantizeInt8Fast(testData)
    let int8Size = quantizedInt8.count * MemoryLayout<Int8>.size
    let int8Ratio = Float(originalSize) / Float(int8Size)

    // INT4 quantization
    let (packedInt4, _) = quantizeInt4Fast(testData)
    let int4Size = packedInt4.count * MemoryLayout<UInt8>.size
    let int4Ratio = Float(originalSize) / Float(int4Size)

    print("Memory Efficiency Comparison:")
    print("  Original (FP32): \(originalSize) bytes")
    print("  INT8: \(int8Size) bytes (\(String(format: "%.1f", int8Ratio))x compression)")
    print("  INT4: \(int4Size) bytes (\(String(format: "%.1f", int4Ratio))x compression)")

    // Assert compression ratios
    XCTAssertGreaterThanOrEqual(int8Ratio, 3.5, "INT8 should achieve at least 3.5x compression")
    XCTAssertGreaterThanOrEqual(int4Ratio, 7.0, "INT4 should achieve at least 7x compression")
  }
}

// MARK: - Helper Functions

private func measureTime<T>(_ operation: () throws -> T) rethrows -> (result: T, time: Double) {
  let startTime = CFAbsoluteTimeGetCurrent()
  let result = try operation()
  let endTime = CFAbsoluteTimeGetCurrent()
  return (result, endTime - startTime)
}

private func generateTestData(size: Int) -> [Float] {
  return (0..<size).map { Float($0) * 0.001 - Float(size / 2) * 0.001 }
}

private func quantizeInt8Fast(_ input: [Float]) -> ([Int8], Float) {
  let maxAbs = input.reduce(0) { max(abs($0), abs($1)) }
  let scale = maxAbs > 0 ? maxAbs / 127.0 : 1.0  // Avoid division by zero

  let quantized = input.map { value in
    Int8(clamping: Int32(round(value / scale)))
  }

  return (quantized, scale)
}

private func quantizeInt4Fast(_ input: [Float]) -> ([UInt8], Float) {
  let maxAbs = input.reduce(0) { max(abs($0), abs($1)) }
  let scale = maxAbs > 0 ? maxAbs / 7.0 : 1.0  // Avoid division by zero

  var packed = [UInt8]()
  packed.reserveCapacity((input.count + 1) / 2)

  for i in stride(from: 0, to: input.count, by: 2) {
    let val1 = Int32(round(input[i] / scale))
    let val2 = i + 1 < input.count ? Int32(round(input[i + 1] / scale)) : 0

    let packed1 = UInt8(max(0, min(15, val1 + 8)))  // Clamp [-8,7] to [0,15] before UInt8 conversion
    let packed2 = UInt8(max(0, min(15, val2 + 8)))

    packed.append((packed2 << 4) | packed1)
  }

  return (packed, scale)
}

// MARK: - Extensions

// Note: Int8(clamping:) and UInt8(clamping:) are provided by Swift standard library

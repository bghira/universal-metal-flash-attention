import FlashAttention
import Metal
import XCTest

@testable import MFABridge

final class QuantizedBenchmarkDebugTests: XCTestCase {

  func testIsolatedQuantizedBenchmark() throws {
    guard let device = MTLCreateSystemDefaultDevice() else {
      throw XCTSkip("Metal device not available")
    }

    let quantizedAttention = QuantizedAttention(device: device)

    print("Starting isolated quantized attention benchmark...")

    // Try the smallest possible test case to isolate the error
    let results = quantizedAttention.benchmark(
      batchSize: 1,
      sequenceLength: 32,  // Very small size
      headDim: 16,  // Very small head dimension
      iterations: 1  // Just one iteration to see the error
    )

    print("Benchmark completed successfully")
    print("Results:")
    for (key, value) in results.sorted(by: { $0.key < $1.key }) {
      print("  \(key): \(String(format: "%.4f", value))")
    }

    // Basic validation - should have at least some results
    XCTAssertGreaterThan(results.count, 0, "Should have some benchmark results")

    // Check that we got results for different precisions
    let keys = results.keys
    let hasValidResults = keys.contains { $0.contains("avg_ms") }
    XCTAssertTrue(hasValidResults, "Should have timing results")
  }
}

//
//  MultiHeadFFITests.swift
//  MFAFFITests
//
//  Created by bghira on 9/15/24.
//

import MFAFFI
import XCTest

final class MultiHeadFFITests: XCTestCase {
  private var context: UnsafeMutableRawPointer?

  override func setUp() {
    super.setUp()
    // Create MFA context
    let result = mfa_create_context(&context)
    XCTAssertEqual(result, MFA_SUCCESS, "Failed to create MFA context")
    XCTAssertNotNil(context, "Context should not be nil")
  }

  override func tearDown() {
    if let context {
      mfa_destroy_context(context)
    }
    super.tearDown()
  }

  func testMultiHeadAttentionForward() throws {
    // Test multi-head attention through FFI
    let batchSize: UInt32 = 1
    let numHeads: UInt32 = 4
    let seqLen: UInt32 = 32
    let headDim: UInt16 = 16

    let totalElements = Int(batchSize * numHeads * seqLen * UInt32(headDim))

    // Create test data
    var queryData = generateRandomData(count: totalElements)
    var keyData = generateRandomData(count: totalElements)
    var valueData = generateRandomData(count: totalElements)
    var outputData = [Float](repeating: 0.0, count: totalElements)

    // Create MFA buffers
    var qBuffer: UnsafeMutableRawPointer?
    var kBuffer: UnsafeMutableRawPointer?
    var vBuffer: UnsafeMutableRawPointer?
    var oBuffer: UnsafeMutableRawPointer?

    let dataSize = totalElements * MemoryLayout<Float>.size

    // Create buffers from existing data
    let result1 = queryData.withUnsafeMutableBytes { ptr in
      mfa_buffer_from_ptr(context, ptr.baseAddress, dataSize, &qBuffer)
    }
    let result2 = keyData.withUnsafeMutableBytes { ptr in
      mfa_buffer_from_ptr(context, ptr.baseAddress, dataSize, &kBuffer)
    }
    let result3 = valueData.withUnsafeMutableBytes { ptr in
      mfa_buffer_from_ptr(context, ptr.baseAddress, dataSize, &vBuffer)
    }
    let result4 = outputData.withUnsafeMutableBytes { ptr in
      mfa_buffer_from_ptr(context, ptr.baseAddress, dataSize, &oBuffer)
    }

    XCTAssertEqual(result1, MFA_SUCCESS, "Failed to create query buffer")
    XCTAssertEqual(result2, MFA_SUCCESS, "Failed to create key buffer")
    XCTAssertEqual(result3, MFA_SUCCESS, "Failed to create value buffer")
    XCTAssertEqual(result4, MFA_SUCCESS, "Failed to create output buffer")

    // Execute multi-head attention
    let result = mfa_attention_forward(
      context, qBuffer, kBuffer, vBuffer, oBuffer,
      batchSize, seqLen, seqLen, numHeads, headDim,
      1.0 / Float(headDim).squareRoot(), // softmax scale
      false, // causal
      MFA_PRECISION_FP32, // input precision
      MFA_PRECISION_FP32, // intermediate precision
      MFA_PRECISION_FP32, // output precision
      false, false, false, false // transpose flags
    )

    XCTAssertEqual(result, MFA_SUCCESS, "Multi-head attention execution failed")

    // Validate output
    XCTAssertFalse(outputData.contains { $0.isNaN }, "Output contains NaN values")
    XCTAssertFalse(outputData.contains { $0.isInfinite }, "Output contains infinite values")

    // Check that output is not all zeros (should have computed something)
    let nonZeroCount = outputData.filter { $0 != 0.0 }.count
    XCTAssertGreaterThanOrEqual(
      nonZeroCount,
      totalElements / 4,
      "Output should have computed non-zero values"
    )

    // Clean up buffers
    mfa_destroy_buffer(qBuffer)
    mfa_destroy_buffer(kBuffer)
    mfa_destroy_buffer(vBuffer)
    mfa_destroy_buffer(oBuffer)

    print("✅ Multi-head attention (H=\(numHeads), S=\(seqLen), D=\(headDim)) executed successfully")
  }

  func testVariousHeadCounts() throws {
    // Test different numbers of heads
    let headCounts: [UInt32] = [1, 2, 4, 8]
    let seqLen: UInt32 = 16
    let headDim: UInt16 = 32

    for numHeads in headCounts {
      let batchSize: UInt32 = 1
      let totalElements = Int(batchSize * numHeads * seqLen * UInt32(headDim))

      // Create test data
      var queryData = generateRandomData(count: totalElements)
      var keyData = generateRandomData(count: totalElements)
      var valueData = generateRandomData(count: totalElements)
      var outputData = [Float](repeating: 0.0, count: totalElements)

      // Create MFA buffers
      var qBuffer: UnsafeMutableRawPointer?
      var kBuffer: UnsafeMutableRawPointer?
      var vBuffer: UnsafeMutableRawPointer?
      var oBuffer: UnsafeMutableRawPointer?

      let dataSize = totalElements * MemoryLayout<Float>.size

      let result1 = queryData.withUnsafeMutableBytes { ptr in
        mfa_buffer_from_ptr(context, ptr.baseAddress, dataSize, &qBuffer)
      }
      let result2 = keyData.withUnsafeMutableBytes { ptr in
        mfa_buffer_from_ptr(context, ptr.baseAddress, dataSize, &kBuffer)
      }
      let result3 = valueData.withUnsafeMutableBytes { ptr in
        mfa_buffer_from_ptr(context, ptr.baseAddress, dataSize, &vBuffer)
      }
      let result4 = outputData.withUnsafeMutableBytes { ptr in
        mfa_buffer_from_ptr(context, ptr.baseAddress, dataSize, &oBuffer)
      }

      XCTAssertEqual(result1, MFA_SUCCESS, "Failed to create query buffer for \(numHeads) heads")
      XCTAssertEqual(result2, MFA_SUCCESS, "Failed to create key buffer for \(numHeads) heads")
      XCTAssertEqual(result3, MFA_SUCCESS, "Failed to create value buffer for \(numHeads) heads")
      XCTAssertEqual(result4, MFA_SUCCESS, "Failed to create output buffer for \(numHeads) heads")

      // Execute multi-head attention
      let result = mfa_attention_forward(
        context, qBuffer, kBuffer, vBuffer, oBuffer,
        batchSize, seqLen, seqLen, numHeads, headDim,
        1.0 / Float(headDim).squareRoot(),
        false, MFA_PRECISION_FP32, MFA_PRECISION_FP32, MFA_PRECISION_FP32,
        false, false, false, false
      )

      XCTAssertEqual(result, MFA_SUCCESS, "Multi-head attention failed for \(numHeads) heads")

      // Validate output
      XCTAssertFalse(outputData.contains { $0.isNaN }, "Output contains NaN for \(numHeads) heads")
      XCTAssertFalse(
        outputData.contains { $0.isInfinite },
        "Output contains Inf for \(numHeads) heads"
      )

      // Clean up buffers
      mfa_destroy_buffer(qBuffer)
      mfa_destroy_buffer(kBuffer)
      mfa_destroy_buffer(vBuffer)
      mfa_destroy_buffer(oBuffer)

      print("✅ H=\(numHeads) test passed")
    }
  }

  func testPerformanceComparison() throws {
    // Test multiple problem sizes to find optimal scaling
    let testConfigs = [
      (seqLen: UInt32(64), headDim: UInt16(32)), // Small
      (seqLen: UInt32(128), headDim: UInt16(64)), // Medium
      (seqLen: UInt32(256), headDim: UInt16(64)), // Large
    ]

    for (seqLen, headDim) in testConfigs {
      let iterations = 5

      // Test single head
      let singleHeadTime = measureExecutionTime {
        for _ in 0..<iterations {
          executeSingleTest(numHeads: 1, seqLen: seqLen, headDim: headDim)
        }
      }

      // Test multi-head
      let multiHeadTime = measureExecutionTime {
        for _ in 0..<iterations {
          executeSingleTest(numHeads: 4, seqLen: seqLen, headDim: headDim)
        }
      }

      let ratio = multiHeadTime / singleHeadTime
      print("Performance (S=\(seqLen), D=\(headDim)):")
      print(
        "  Single-head (1): \(String(format: "%.3f", singleHeadTime * 1000 / Double(iterations))) ms/iter"
      )
      print(
        "  Multi-head (4):  \(String(format: "%.3f", multiHeadTime * 1000 / Double(iterations))) ms/iter"
      )
      print("  Overhead: \(String(format: "%.1f", ratio))x")
    }

    // Use medium size for the main test
    let seqLen: UInt32 = 128
    let headDim: UInt16 = 64
    let iterations = 5

    let singleHeadTime = measureExecutionTime {
      for _ in 0..<iterations {
        executeSingleTest(numHeads: 1, seqLen: seqLen, headDim: headDim)
      }
    }

    let multiHeadTime = measureExecutionTime {
      for _ in 0..<iterations {
        executeSingleTest(numHeads: 4, seqLen: seqLen, headDim: headDim)
      }
    }

    // Multi-head should be reasonably close to 4x single-head time
    let actualRatio = multiHeadTime / singleHeadTime
    XCTAssertLessThan(actualRatio, 5.0, "Multi-head overhead too high")
  }

  func testCausalMasking() throws {
    // Test causal masking with multi-head attention
    let numHeads: UInt32 = 2
    let seqLen: UInt32 = 8
    let headDim: UInt16 = 16

    let totalElements = Int(numHeads * seqLen * UInt32(headDim))

    // Create test data
    var queryData = generateRandomData(count: totalElements)
    var keyData = generateRandomData(count: totalElements)
    var valueData = generateRandomData(count: totalElements)
    var outputData = [Float](repeating: 0.0, count: totalElements)

    // Create MFA buffers
    var qBuffer: UnsafeMutableRawPointer?
    var kBuffer: UnsafeMutableRawPointer?
    var vBuffer: UnsafeMutableRawPointer?
    var oBuffer: UnsafeMutableRawPointer?

    let dataSize = totalElements * MemoryLayout<Float>.size

    let result1 = queryData.withUnsafeMutableBytes { ptr in
      mfa_buffer_from_ptr(context, ptr.baseAddress, dataSize, &qBuffer)
    }
    let result2 = keyData.withUnsafeMutableBytes { ptr in
      mfa_buffer_from_ptr(context, ptr.baseAddress, dataSize, &kBuffer)
    }
    let result3 = valueData.withUnsafeMutableBytes { ptr in
      mfa_buffer_from_ptr(context, ptr.baseAddress, dataSize, &vBuffer)
    }
    let result4 = outputData.withUnsafeMutableBytes { ptr in
      mfa_buffer_from_ptr(context, ptr.baseAddress, dataSize, &oBuffer)
    }

    XCTAssertEqual(result1, MFA_SUCCESS)
    XCTAssertEqual(result2, MFA_SUCCESS)
    XCTAssertEqual(result3, MFA_SUCCESS)
    XCTAssertEqual(result4, MFA_SUCCESS)

    // Execute with causal masking
    let result = mfa_attention_forward(
      context, qBuffer, kBuffer, vBuffer, oBuffer,
      1, seqLen, seqLen, numHeads, headDim,
      1.0 / Float(headDim).squareRoot(),
      true, // causal masking enabled
      MFA_PRECISION_FP32, MFA_PRECISION_FP32, MFA_PRECISION_FP32,
      false, false, false, false
    )

    XCTAssertEqual(result, MFA_SUCCESS, "Causal multi-head attention execution failed")

    // Validate output
    XCTAssertFalse(outputData.contains { $0.isNaN }, "Causal output contains NaN values")
    XCTAssertFalse(outputData.contains { $0.isInfinite }, "Causal output contains infinite values")

    // Clean up buffers
    mfa_destroy_buffer(qBuffer)
    mfa_destroy_buffer(kBuffer)
    mfa_destroy_buffer(vBuffer)
    mfa_destroy_buffer(oBuffer)

    print("✅ Causal multi-head attention test passed")
  }

  // MARK: - Helper Methods

  private func executeSingleTest(numHeads: UInt32, seqLen: UInt32, headDim: UInt16) {
    let totalElements = Int(numHeads * seqLen * UInt32(headDim))

    var queryData = generateRandomData(count: totalElements)
    var keyData = generateRandomData(count: totalElements)
    var valueData = generateRandomData(count: totalElements)
    var outputData = [Float](repeating: 0.0, count: totalElements)

    var qBuffer: UnsafeMutableRawPointer?
    var kBuffer: UnsafeMutableRawPointer?
    var vBuffer: UnsafeMutableRawPointer?
    var oBuffer: UnsafeMutableRawPointer?

    let dataSize = totalElements * MemoryLayout<Float>.size

    _ = queryData.withUnsafeMutableBytes { ptr in
      mfa_buffer_from_ptr(context, ptr.baseAddress, dataSize, &qBuffer)
    }
    _ = keyData.withUnsafeMutableBytes { ptr in
      mfa_buffer_from_ptr(context, ptr.baseAddress, dataSize, &kBuffer)
    }
    _ = valueData.withUnsafeMutableBytes { ptr in
      mfa_buffer_from_ptr(context, ptr.baseAddress, dataSize, &vBuffer)
    }
    _ = outputData.withUnsafeMutableBytes { ptr in
      mfa_buffer_from_ptr(context, ptr.baseAddress, dataSize, &oBuffer)
    }

    _ = mfa_attention_forward(
      context, qBuffer, kBuffer, vBuffer, oBuffer,
      1, seqLen, seqLen, numHeads, headDim,
      1.0 / Float(headDim).squareRoot(),
      false, MFA_PRECISION_FP32, MFA_PRECISION_FP32, MFA_PRECISION_FP32,
      false, false, false, false
    )

    mfa_destroy_buffer(qBuffer)
    mfa_destroy_buffer(kBuffer)
    mfa_destroy_buffer(vBuffer)
    mfa_destroy_buffer(oBuffer)
  }

  private func generateRandomData(count: Int) -> [Float] {
    (0..<count).map { _ in Float.random(in: -1...1) }
  }

  private func measureExecutionTime(_ block: () -> Void) -> Double {
    let startTime = CFAbsoluteTimeGetCurrent()
    block()
    let endTime = CFAbsoluteTimeGetCurrent()
    return endTime - startTime
  }
}

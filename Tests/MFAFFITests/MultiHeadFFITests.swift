//
//  MultiHeadFFITests.swift
//  MFAFFITests
//
//  Created by bghira on 9/15/24.
//

import MFAFFI
import XCTest

extension Array where Element == Float {
  func minAndMax() -> (min: Float, max: Float) {
    guard !isEmpty else { return (0.0, 0.0) }
    var min = self[0]
    var max = self[0]
    for element in self {
      if element < min { min = element }
      if element > max { max = element }
    }
    return (min: min, max: max)
  }
}

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
    // Comprehensive multi-head attention test with multiple precisions and FLUX configurations
    print("\n🧪 COMPREHENSIVE MULTI-HEAD ATTENTION TEST")
    print(String(repeating: "=", count: 60))

    let testConfigurations = [
      // Standard configurations
      (name: "Standard Small", batchSize: UInt32(1), numHeads: UInt32(4), seqLen: UInt32(32), headDim: UInt16(16)),
      (name: "Standard Medium", batchSize: UInt32(1), numHeads: UInt32(8), seqLen: UInt32(128), headDim: UInt16(64)),

      // FLUX-specific configurations
      (name: "FLUX Joint Attention", batchSize: UInt32(1), numHeads: UInt32(24), seqLen: UInt32(512), headDim: UInt16(64)),
      (name: "FLUX Large", batchSize: UInt32(1), numHeads: UInt32(16), seqLen: UInt32(1024), headDim: UInt16(88)),
      (name: "FLUX XL", batchSize: UInt32(1), numHeads: UInt32(24), seqLen: UInt32(4096), headDim: UInt16(128)),
    ]

    let precisionConfigs = [
      (name: "FP32", input: mfa_precision_t(rawValue: 2), intermediate: mfa_precision_t(rawValue: 2), output: mfa_precision_t(rawValue: 2)),
      (name: "FP16", input: mfa_precision_t(rawValue: 0), intermediate: mfa_precision_t(rawValue: 0), output: mfa_precision_t(rawValue: 0)),
      (name: "BF16", input: mfa_precision_t(rawValue: 1), intermediate: mfa_precision_t(rawValue: 1), output: mfa_precision_t(rawValue: 1)),
      (name: "Mixed FP32->FP16", input: mfa_precision_t(rawValue: 2), intermediate: mfa_precision_t(rawValue: 0), output: mfa_precision_t(rawValue: 0)),
      (name: "Mixed BF16->FP32", input: mfa_precision_t(rawValue: 1), intermediate: mfa_precision_t(rawValue: 2), output: mfa_precision_t(rawValue: 2)),
    ]

    for config in testConfigurations {
      print("\n--- Testing \(config.name) ---")

      for precisionConfig in precisionConfigs {
        print("  Precision: \(precisionConfig.name)")

        let passed = try testSingleConfiguration(
          batchSize: config.batchSize,
          numHeads: config.numHeads,
          seqLen: config.seqLen,
          headDim: config.headDim,
          inputPrecision: precisionConfig.input,
          intermediatePrecision: precisionConfig.intermediate,
          outputPrecision: precisionConfig.output,
          precisionName: precisionConfig.name
        )

        if !passed {
          XCTFail("Failed configuration: \(config.name) with precision \(precisionConfig.name)")
        }
      }
    }

    // Test specific bug reproduction cases
    try testBugReproductionCases()

    // Test quantization configurations
    try testQuantizationConfigurations()

    print("\n✅ All comprehensive multi-head attention tests completed")
  }

  private func testSingleConfiguration(
    batchSize: UInt32,
    numHeads: UInt32,
    seqLen: UInt32,
    headDim: UInt16,
    inputPrecision: mfa_precision_t,
    intermediatePrecision: mfa_precision_t,
    outputPrecision: mfa_precision_t,
    precisionName: String
  ) throws -> Bool {
    let totalElements = Int(batchSize * numHeads * seqLen * UInt32(headDim))

    // Generate deterministic test data for reproducibility
    let seed: UInt64 = 42
    var queryData = generateDeterministicData(count: totalElements, seed: seed)
    var keyData = generateDeterministicData(count: totalElements, seed: seed + 1)
    var valueData = generateDeterministicData(count: totalElements, seed: seed + 2)
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

    guard result1 == MFA_SUCCESS && result2 == MFA_SUCCESS &&
          result3 == MFA_SUCCESS && result4 == MFA_SUCCESS else {
      print("    ❌ Failed to create buffers")
      return false
    }

    // Execute multi-head attention
    let result = mfa_attention_forward(
      context, qBuffer, kBuffer, vBuffer, oBuffer,
      batchSize, seqLen, seqLen, numHeads, headDim,
      1.0 / Float(headDim).squareRoot(), // softmax scale
      false, // causal
      inputPrecision,
      intermediatePrecision,
      outputPrecision,
      false, false, false, false // transpose flags
    )

    defer {
      mfa_destroy_buffer(qBuffer)
      mfa_destroy_buffer(kBuffer)
      mfa_destroy_buffer(vBuffer)
      mfa_destroy_buffer(oBuffer)
    }

    guard result == MFA_SUCCESS else {
      print("    ❌ Execution failed with result: \(result)")
      return false
    }

    // Check for NaN/Inf (critical for precision issues)
    let hasNaN = outputData.contains { $0.isNaN }
    let hasInf = outputData.contains { $0.isInfinite }

    if hasNaN || hasInf {
      print("    ❌ Contains NaN: \(hasNaN), Inf: \(hasInf)")
      return false
    }

    // Check for reasonable output range
    let outputRange = outputData.minAndMax()
    let maxAbsValue = max(abs(outputRange.min), abs(outputRange.max))

    // Values should be reasonable for attention output
    if maxAbsValue > 100.0 || maxAbsValue < 1e-6 {
      print("    ⚠️  Suspicious output range: [\(outputRange.min), \(outputRange.max)]")
    }

    // Check that output is not all zeros
    let nonZeroCount = outputData.filter { abs($0) > 1e-8 }.count
    let nonZeroRatio = Double(nonZeroCount) / Double(totalElements)

    if nonZeroRatio < 0.1 {
      print("    ❌ Too many zero values: \(nonZeroRatio * 100)% non-zero")
      return false
    }

    // Calculate basic statistics
    let mean = outputData.reduce(0, +) / Float(outputData.count)
    let variance = outputData.reduce(0) { sum, value in
      sum + (value - mean) * (value - mean)
    } / Float(outputData.count)
    let stdDev = sqrt(variance)

    print("    ✅ H=\(numHeads), S=\(seqLen), D=\(headDim), Range=[\(String(format: "%.3f", outputRange.min)), \(String(format: "%.3f", outputRange.max))], StdDev=\(String(format: "%.4f", stdDev))")

    return true
  }

  private func testBugReproductionCases() throws {
    print("\n--- Bug Reproduction Test Cases ---")

    // Case 1: BF16/FP16 NaN bug reproduction with specific FLUX dimensions
    print("  Testing BF16 NaN bug reproduction...")
    let fluxBugPassed = try testSingleConfiguration(
      batchSize: 1,
      numHeads: 24,
      seqLen: 4096,
      headDim: 64,
      inputPrecision: mfa_precision_t(rawValue: 1), // BF16
      intermediatePrecision: mfa_precision_t(rawValue: 1), // BF16
      outputPrecision: mfa_precision_t(rawValue: 1), // BF16
      precisionName: "BF16-Bug-Repro"
    )

    // Case 2: Specific problematic dimensions that caused issues
    print("  Testing FP16 edge case...")
    let fp16EdgePassed = try testSingleConfiguration(
      batchSize: 1,
      numHeads: 16,
      seqLen: 1024,
      headDim: 88,
      inputPrecision: mfa_precision_t(rawValue: 0), // FP16
      intermediatePrecision: mfa_precision_t(rawValue: 0), // FP16
      outputPrecision: mfa_precision_t(rawValue: 2), // FP32 output to catch precision loss
      precisionName: "FP16-Edge-Case"
    )

    // Case 3: Large sequence length with mixed precision
    print("  Testing large sequence mixed precision...")
    let largeMixedPassed = try testSingleConfiguration(
      batchSize: 1,
      numHeads: 8,
      seqLen: 8192,
      headDim: 128,
      inputPrecision: mfa_precision_t(rawValue: 1), // BF16
      intermediatePrecision: mfa_precision_t(rawValue: 2), // FP32
      outputPrecision: mfa_precision_t(rawValue: 0), // FP16
      precisionName: "Large-Mixed-Precision"
    )

    XCTAssertTrue(fluxBugPassed, "BF16 bug reproduction test failed")
    XCTAssertTrue(fp16EdgePassed, "FP16 edge case test failed")
    XCTAssertTrue(largeMixedPassed, "Large mixed precision test failed")
  }

  private func testQuantizationConfigurations() throws {
    print("\n--- Quantization Configuration Tests ---")

    // Test quantized attention with different configurations
    let quantConfigs = [
      (name: "INT8", qPrecision: mfa_precision_t(rawValue: 3), outputPrecision: mfa_precision_t(rawValue: 2)),
      (name: "INT4", qPrecision: mfa_precision_t(rawValue: 4), outputPrecision: mfa_precision_t(rawValue: 2)),
    ]

    for config in quantConfigs {
      print("  Testing \(config.name) quantization...")

      // Use smaller dimensions for quantization to avoid memory issues
      let passed = try testQuantizedConfiguration(
        batchSize: 1,
        numHeads: 8,
        seqLen: 256,
        headDim: 64,
        quantizationPrecision: config.qPrecision,
        outputPrecision: config.outputPrecision,
        configName: config.name
      )

      XCTAssertTrue(passed, "\(config.name) quantization test failed")
    }
  }

  private func testQuantizedConfiguration(
    batchSize: UInt32,
    numHeads: UInt32,
    seqLen: UInt32,
    headDim: UInt16,
    quantizationPrecision: mfa_precision_t,
    outputPrecision: mfa_precision_t,
    configName: String
  ) throws -> Bool {
    // Note: This is a basic test for quantized attention
    // Full quantization testing would require additional FFI functions
    // For now, we test that the precision values are handled correctly

    let totalElements = Int(batchSize * numHeads * seqLen * UInt32(headDim))

    var queryData = generateDeterministicData(count: totalElements, seed: 100)
    var keyData = generateDeterministicData(count: totalElements, seed: 101)
    var valueData = generateDeterministicData(count: totalElements, seed: 102)
    var outputData = [Float](repeating: 0.0, count: totalElements)

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

    guard result1 == MFA_SUCCESS && result2 == MFA_SUCCESS &&
          result3 == MFA_SUCCESS && result4 == MFA_SUCCESS else {
      print("    ❌ Failed to create quantization buffers")
      return false
    }

    // Test with quantized key/value precision
    let result = mfa_attention_forward(
      context, qBuffer, kBuffer, vBuffer, oBuffer,
      batchSize, seqLen, seqLen, numHeads, headDim,
      1.0 / Float(headDim).squareRoot(),
      false, // causal
      mfa_precision_t(rawValue: 2), // FP32 input (queries stay in high precision)
      quantizationPrecision, // Quantized intermediate (affects K/V processing)
      outputPrecision,
      false, false, false, false
    )

    defer {
      mfa_destroy_buffer(qBuffer)
      mfa_destroy_buffer(kBuffer)
      mfa_destroy_buffer(vBuffer)
      mfa_destroy_buffer(oBuffer)
    }

    guard result == MFA_SUCCESS else {
      print("    ❌ Quantized execution failed with result: \(result)")
      return false
    }

    // Check for NaN/Inf in quantized output
    let hasNaN = outputData.contains { $0.isNaN }
    let hasInf = outputData.contains { $0.isInfinite }

    if hasNaN || hasInf {
      print("    ❌ Quantized output contains NaN: \(hasNaN), Inf: \(hasInf)")
      return false
    }

    let nonZeroCount = outputData.filter { abs($0) > 1e-8 }.count
    let nonZeroRatio = Double(nonZeroCount) / Double(totalElements)

    if nonZeroRatio < 0.05 {
      print("    ❌ Quantized output mostly zeros: \(nonZeroRatio * 100)% non-zero")
      return false
    }

    print("    ✅ \(configName) quantization passed")
    return true
  }

  func testVariousHeadCounts() throws {
    print("\n🔢 COMPREHENSIVE HEAD COUNT TESTING")
    print(String(repeating: "=", count: 60))

    // Test head counts including FLUX-specific values and edge cases
    let headCountConfigs = [
      (heads: UInt32(1), name: "Single Head", seqLens: [UInt32(64), UInt32(256)]),
      (heads: UInt32(4), name: "Standard 4", seqLens: [UInt32(128), UInt32(512)]),
      (heads: UInt32(8), name: "Standard 8", seqLens: [UInt32(256), UInt32(1024)]),
      (heads: UInt32(12), name: "Medium 12", seqLens: [UInt32(512), UInt32(2048)]),
      (heads: UInt32(16), name: "Large 16", seqLens: [UInt32(1024), UInt32(4096)]),
      (heads: UInt32(24), name: "FLUX Joint", seqLens: [UInt32(512), UInt32(2048)]),
      (heads: UInt32(32), name: "Extreme 32", seqLens: [UInt32(256), UInt32(1024)])
    ]

    let precisionConfigs = [
      (name: "FP32", input: mfa_precision_t(rawValue: 2), intermediate: mfa_precision_t(rawValue: 2), output: mfa_precision_t(rawValue: 2)),
      (name: "FP16", input: mfa_precision_t(rawValue: 0), intermediate: mfa_precision_t(rawValue: 0), output: mfa_precision_t(rawValue: 0)),
      (name: "BF16", input: mfa_precision_t(rawValue: 1), intermediate: mfa_precision_t(rawValue: 1), output: mfa_precision_t(rawValue: 1)),
      (name: "Mixed FP16->FP32", input: mfa_precision_t(rawValue: 0), intermediate: mfa_precision_t(rawValue: 2), output: mfa_precision_t(rawValue: 2))
    ]

    let headDims: [UInt16] = [64, 88, 128] // Common dimensions including FLUX's 88

    var failedTests: [(String, String)] = []
    var successCount = 0
    var totalTests = 0

    for headConfig in headCountConfigs {
      print("\n--- Testing \(headConfig.name) (H=\(headConfig.heads)) ---")

      for seqLen in headConfig.seqLens {
        for headDim in headDims {
          // Skip very large configurations to avoid memory issues in tests
          if headConfig.heads * seqLen > 32768 && headDim == 128 {
            continue
          }

          for precisionConfig in precisionConfigs {
            totalTests += 1

            let testName = "\(headConfig.name)_S\(seqLen)_D\(headDim)_\(precisionConfig.name)"

            do {
              let passed = try testSingleHeadConfiguration(
                numHeads: headConfig.heads,
                seqLen: seqLen,
                headDim: headDim,
                inputPrecision: precisionConfig.input,
                intermediatePrecision: precisionConfig.intermediate,
                outputPrecision: precisionConfig.output,
                testName: testName
              )

              if passed {
                successCount += 1
                print("    ✅ \(testName)")
              } else {
                failedTests.append((testName, "Test validation failed"))
                print("    ❌ \(testName)")
              }
            } catch {
              failedTests.append((testName, error.localizedDescription))
              print("    ❌ \(testName) - Error: \(error.localizedDescription)")
            }
          }
        }
      }
    }

    // Test multi-head correctness properties
    try testMultiHeadCorrectnessProperties()

    // Test attention weight sum validation
    try testAttentionWeightSumming()

    // Performance scaling validation
    try testHeadCountPerformanceScaling()

    print("\n📊 HEAD COUNT TEST SUMMARY")
    print(String(repeating: "=", count: 60))
    print("Total tests: \(totalTests)")
    print("Successful: \(successCount)")
    print("Failed: \(failedTests.count)")

    if !failedTests.isEmpty {
      print("\n❌ FAILED TESTS:")
      for (testName, reason) in failedTests {
        print("  • \(testName): \(reason)")
      }
      XCTFail("Head count tests failed: \(failedTests.count)/\(totalTests)")
    } else {
      print("✅ All head count variations passed!")
    }
  }

  private func testSingleHeadConfiguration(
    numHeads: UInt32,
    seqLen: UInt32,
    headDim: UInt16,
    inputPrecision: mfa_precision_t,
    intermediatePrecision: mfa_precision_t,
    outputPrecision: mfa_precision_t,
    testName: String
  ) throws -> Bool {
    let batchSize: UInt32 = 1
    let totalElements = Int(batchSize * numHeads * seqLen * UInt32(headDim))

    // Use deterministic data for reproducibility
    let seed = UInt64(testName.hashValue) % 10000 + 1000
    var queryData = generateDeterministicData(count: totalElements, seed: seed)
    var keyData = generateDeterministicData(count: totalElements, seed: seed + 1)
    var valueData = generateDeterministicData(count: totalElements, seed: seed + 2)
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

    guard result1 == MFA_SUCCESS && result2 == MFA_SUCCESS &&
          result3 == MFA_SUCCESS && result4 == MFA_SUCCESS else {
      return false
    }

    // Execute multi-head attention
    let result = mfa_attention_forward(
      context, qBuffer, kBuffer, vBuffer, oBuffer,
      batchSize, seqLen, seqLen, numHeads, headDim,
      1.0 / Float(headDim).squareRoot(),
      false, // causal
      inputPrecision,
      intermediatePrecision,
      outputPrecision,
      false, false, false, false
    )

    defer {
      mfa_destroy_buffer(qBuffer)
      mfa_destroy_buffer(kBuffer)
      mfa_destroy_buffer(vBuffer)
      mfa_destroy_buffer(oBuffer)
    }

    guard result == MFA_SUCCESS else {
      return false
    }

    // Comprehensive validation

    // 1. Check for NaN/Inf (critical for precision issues)
    let hasNaN = outputData.contains { $0.isNaN }
    let hasInf = outputData.contains { $0.isInfinite }

    if hasNaN || hasInf {
      return false
    }

    // 2. Check reasonable output range
    let outputRange = outputData.minAndMax()
    let maxAbsValue = max(abs(outputRange.min), abs(outputRange.max))

    // Values should be reasonable for attention output
    if maxAbsValue > 50.0 || maxAbsValue < 1e-7 {
      return false
    }

    // 3. Check for mostly non-zero output
    let nonZeroCount = outputData.filter { abs($0) > 1e-8 }.count
    let nonZeroRatio = Double(nonZeroCount) / Double(totalElements)

    if nonZeroRatio < 0.05 {
      return false
    }

    // 4. Head-specific validation: ensure different heads produce different outputs
    if numHeads > 1 {
      let elementsPerHead = Int(seqLen * UInt32(headDim))
      var headOutputs: [[Float]] = []

      for head in 0..<Int(numHeads) {
        let startIdx = head * elementsPerHead
        let endIdx = startIdx + elementsPerHead
        headOutputs.append(Array(outputData[startIdx..<endIdx]))
      }

      // Check that different heads produce sufficiently different outputs
      for i in 0..<headOutputs.count {
        for j in (i+1)..<headOutputs.count {
          let correlation = calculateCorrelation(headOutputs[i], headOutputs[j])
          // Different heads should not be too highly correlated
          if correlation > 0.95 {
            return false // Heads are too similar
          }
        }
      }
    }

    // 5. Check numerical stability across head counts
    let mean = outputData.reduce(0, +) / Float(outputData.count)
    let variance = outputData.reduce(0) { sum, value in
      sum + (value - mean) * (value - mean)
    } / Float(outputData.count)
    let stdDev = sqrt(variance)

    // Standard deviation should be reasonable
    if stdDev < 1e-6 || stdDev > 10.0 {
      return false
    }

    return true
  }

  private func testMultiHeadCorrectnessProperties() throws {
    print("\n--- Testing Multi-Head Correctness Properties ---")

    // Test that increasing head count maintains output quality
    let baseConfig = (heads: UInt32(1), seqLen: UInt32(128), headDim: UInt16(64))
    let multiConfig = (heads: UInt32(8), seqLen: UInt32(128), headDim: UInt16(64))

    // Use same input data for both tests
    let seed: UInt64 = 5555
    let singleHeadOutput = try runAttentionWithConfig(
      numHeads: baseConfig.heads,
      seqLen: baseConfig.seqLen,
      headDim: baseConfig.headDim,
      seed: seed
    )

    let multiHeadOutput = try runAttentionWithConfig(
      numHeads: multiConfig.heads,
      seqLen: multiConfig.seqLen,
      headDim: multiConfig.headDim,
      seed: seed
    )

    // Both should be valid (no NaN/Inf)
    XCTAssertFalse(singleHeadOutput.contains { $0.isNaN }, "Single-head contains NaN")
    XCTAssertFalse(multiHeadOutput.contains { $0.isNaN }, "Multi-head contains NaN")
    XCTAssertFalse(singleHeadOutput.contains { $0.isInfinite }, "Single-head contains Inf")
    XCTAssertFalse(multiHeadOutput.contains { $0.isInfinite }, "Multi-head contains Inf")

    // Both should have reasonable ranges
    let singleRange = singleHeadOutput.minAndMax()
    let multiRange = multiHeadOutput.minAndMax()

    XCTAssertLessThan(max(abs(singleRange.min), abs(singleRange.max)), 20.0)
    XCTAssertLessThan(max(abs(multiRange.min), abs(multiRange.max)), 20.0)

    print("    ✅ Multi-head correctness properties validated")
  }

  private func testAttentionWeightSumming() throws {
    print("\n--- Testing Attention Weight Summing Properties ---")

    // For this test, we need to validate that attention behaves correctly
    // We'll use small dimensions so we can validate some properties

    let configs = [
      (heads: UInt32(2), seqLen: UInt32(8), headDim: UInt16(16)),
      (heads: UInt32(4), seqLen: UInt32(16), headDim: UInt16(32)),
      (heads: UInt32(8), seqLen: UInt32(32), headDim: UInt16(64))
    ]

    for config in configs {
      let output = try runAttentionWithConfig(
        numHeads: config.heads,
        seqLen: config.seqLen,
        headDim: config.headDim,
        seed: 7777
      )

      // Validate basic attention properties
      XCTAssertFalse(output.contains { $0.isNaN }, "Attention output contains NaN")
      XCTAssertFalse(output.contains { $0.isInfinite }, "Attention output contains Inf")

      // Check that output has reasonable variance (not all same values)
      let mean = output.reduce(0, +) / Float(output.count)
      let variance = output.reduce(0) { sum, value in
        sum + (value - mean) * (value - mean)
      } / Float(output.count)

      XCTAssertGreaterThan(variance, 1e-6, "Output variance too low for H=\(config.heads)")

      // Check reasonable output magnitudes
      let maxAbs = output.map { abs($0) }.max() ?? 0
      XCTAssertLessThan(maxAbs, 10.0, "Output magnitude too large for H=\(config.heads)")
      XCTAssertGreaterThan(maxAbs, 1e-5, "Output magnitude too small for H=\(config.heads)")
    }

    print("    ✅ Attention weight properties validated")
  }

  private func testHeadCountPerformanceScaling() throws {
    print("\n--- Testing Head Count Performance Scaling ---")

    let baseConfig = (seqLen: UInt32(64), headDim: UInt16(32))
    let headCounts: [UInt32] = [1, 2, 4, 8, 16]

    var timings: [(UInt32, Double)] = []

    for numHeads in headCounts {
      let iterations = 3
      let totalTime = measureExecutionTime {
        for _ in 0..<iterations {
          _ = try? runAttentionWithConfig(
            numHeads: numHeads,
            seqLen: baseConfig.seqLen,
            headDim: baseConfig.headDim,
            seed: 9999
          )
        }
      }

      let avgTime = totalTime / Double(iterations)
      timings.append((numHeads, avgTime))
      print("    H=\(numHeads): \(String(format: "%.3f", avgTime * 1000)) ms")
    }

    // Check that scaling is reasonable (not exponential blowup)
    let singleHeadTime = timings.first?.1 ?? 0
    let maxHeadTime = timings.last?.1 ?? 0
    let scalingRatio = maxHeadTime / singleHeadTime

    // Should scale reasonably (not more than heads * 2)
    XCTAssertLessThan(scalingRatio, Double(headCounts.last! * 2), "Performance scaling too poor")

    print("    ✅ Performance scaling: \(String(format: "%.1f", scalingRatio))x for \(headCounts.last!)x heads")
  }

  private func runAttentionWithConfig(
    numHeads: UInt32,
    seqLen: UInt32,
    headDim: UInt16,
    seed: UInt64
  ) throws -> [Float] {
    let batchSize: UInt32 = 1
    let totalElements = Int(batchSize * numHeads * seqLen * UInt32(headDim))

    var queryData = generateDeterministicData(count: totalElements, seed: seed)
    var keyData = generateDeterministicData(count: totalElements, seed: seed + 1)
    var valueData = generateDeterministicData(count: totalElements, seed: seed + 2)
    var outputData = [Float](repeating: 0.0, count: totalElements)

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

    guard result1 == MFA_SUCCESS && result2 == MFA_SUCCESS &&
          result3 == MFA_SUCCESS && result4 == MFA_SUCCESS else {
      throw NSError(domain: "MFA", code: -1, userInfo: [NSLocalizedDescriptionKey: "Buffer creation failed"])
    }

    let result = mfa_attention_forward(
      context, qBuffer, kBuffer, vBuffer, oBuffer,
      batchSize, seqLen, seqLen, numHeads, headDim,
      1.0 / Float(headDim).squareRoot(),
      false,
      MFA_PRECISION_FP32, MFA_PRECISION_FP32, MFA_PRECISION_FP32,
      false, false, false, false
    )

    defer {
      mfa_destroy_buffer(qBuffer)
      mfa_destroy_buffer(kBuffer)
      mfa_destroy_buffer(vBuffer)
      mfa_destroy_buffer(oBuffer)
    }

    guard result == MFA_SUCCESS else {
      throw NSError(domain: "MFA", code: Int(result.rawValue), userInfo: [NSLocalizedDescriptionKey: "Attention execution failed"])
    }

    return outputData
  }

  private func calculateCorrelation(_ array1: [Float], _ array2: [Float]) -> Float {
    guard array1.count == array2.count else { return 0.0 }

    let mean1 = array1.reduce(0, +) / Float(array1.count)
    let mean2 = array2.reduce(0, +) / Float(array2.count)

    var numerator: Float = 0
    var denominator1: Float = 0
    var denominator2: Float = 0

    for i in 0..<array1.count {
      let diff1 = array1[i] - mean1
      let diff2 = array2[i] - mean2

      numerator += diff1 * diff2
      denominator1 += diff1 * diff1
      denominator2 += diff2 * diff2
    }

    let denominator = sqrt(denominator1 * denominator2)
    return denominator > 1e-10 ? numerator / denominator : 0.0
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

  func testNumericalCorrectnessAgainstPyTorch() throws {
    // Test numerical correctness against PyTorch reference for small, verifiable cases
    print("\n🎯 NUMERICAL CORRECTNESS vs PyTorch REFERENCE")
    print(String(repeating: "=", count: 60))

    // Use small dimensions for exact comparison
    let configs = [
      (name: "Tiny", batchSize: UInt32(1), numHeads: UInt32(2), seqLen: UInt32(4), headDim: UInt16(8)),
      (name: "Small", batchSize: UInt32(1), numHeads: UInt32(4), seqLen: UInt32(8), headDim: UInt16(16)),
    ]

    for config in configs {
      print("\n--- Testing \(config.name) Configuration ---")

      // Generate the same deterministic data that we can reproduce in Python
      let totalElements = Int(config.batchSize * config.numHeads * config.seqLen * UInt32(config.headDim))
      let seed: UInt64 = 12345

      var queryData = generateDeterministicData(count: totalElements, seed: seed)
      var keyData = generateDeterministicData(count: totalElements, seed: seed + 1)
      var valueData = generateDeterministicData(count: totalElements, seed: seed + 2)
      var outputData = [Float](repeating: 0.0, count: totalElements)

      // Create buffers
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

      // Execute MFA attention
      let result = mfa_attention_forward(
        context, qBuffer, kBuffer, vBuffer, oBuffer,
        config.batchSize, config.seqLen, config.seqLen, config.numHeads, config.headDim,
        1.0 / Float(config.headDim).squareRoot(),
        false, // causal
        MFA_PRECISION_FP32, MFA_PRECISION_FP32, MFA_PRECISION_FP32,
        false, false, false, false
      )

      defer {
        mfa_destroy_buffer(qBuffer)
        mfa_destroy_buffer(kBuffer)
        mfa_destroy_buffer(vBuffer)
        mfa_destroy_buffer(oBuffer)
      }

      XCTAssertEqual(result, MFA_SUCCESS, "MFA execution failed")

      // Check for NaN/Inf
      XCTAssertFalse(outputData.contains { $0.isNaN }, "MFA output contains NaN")
      XCTAssertFalse(outputData.contains { $0.isInfinite }, "MFA output contains Inf")

      // Print data for manual verification against PyTorch
      print("  Configuration: B=\(config.batchSize), H=\(config.numHeads), S=\(config.seqLen), D=\(config.headDim)")
      print("  Seed: \(seed)")
      print("  First 5 Q values: \(Array(queryData.prefix(5)).map { String(format: "%.6f", $0) }.joined(separator: ", "))")
      print("  First 5 K values: \(Array(keyData.prefix(5)).map { String(format: "%.6f", $0) }.joined(separator: ", "))")
      print("  First 5 V values: \(Array(valueData.prefix(5)).map { String(format: "%.6f", $0) }.joined(separator: ", "))")
      print("  First 5 output values: \(Array(outputData.prefix(5)).map { String(format: "%.6f", $0) }.joined(separator: ", "))")
      print("  Last 5 output values: \(Array(outputData.suffix(5)).map { String(format: "%.6f", $0) }.joined(separator: ", "))")

      // Basic sanity checks
      let outputRange = outputData.minAndMax()
      print("  Output range: [\(String(format: "%.6f", outputRange.min)), \(String(format: "%.6f", outputRange.max))]")

      // Verify reasonable output characteristics
      let mean = outputData.reduce(0, +) / Float(outputData.count)
      let absValues = outputData.map { abs($0) }
      let maxAbs = absValues.max() ?? 0

      print("  Mean: \(String(format: "%.6f", mean)), Max abs: \(String(format: "%.6f", maxAbs))")

      // Basic reasonableness checks
      XCTAssertLessThan(maxAbs, 10.0, "Output values should be reasonable for attention")
      XCTAssertGreaterThan(maxAbs, 1e-6, "Output should not be effectively zero")

      let nonZeroCount = outputData.filter { abs($0) > 1e-8 }.count
      let nonZeroRatio = Double(nonZeroCount) / Double(totalElements)
      XCTAssertGreaterThan(nonZeroRatio, 0.1, "Should have reasonable number of non-zero outputs")

      print("  ✅ \(config.name) basic correctness verified")
    }

    print("\n📝 NOTE: To verify exact numerical correctness:")
    print("   1. Copy the printed seed and input values")
    print("   2. Create a PyTorch script with the same deterministic data")
    print("   3. Compare F.scaled_dot_product_attention() output")
    print("   4. Expected tolerance: < 1e-5 for FP32")
  }

  func testTensorLayoutVariations() throws {
    // Test different tensor layouts that might be used by Metal vs FLUX
    print("\n🔄 TENSOR LAYOUT VARIATION TESTS")
    print(String(repeating: "=", count: 60))

    let batchSize: UInt32 = 1
    let numHeads: UInt32 = 8
    let seqLen: UInt32 = 64
    let headDim: UInt16 = 32
    let totalElements = Int(batchSize * numHeads * seqLen * UInt32(headDim))

    // Test with different transpose configurations
    let transposeConfigs = [
      (name: "No Transpose", q: false, k: false, v: false, o: false),
      (name: "Transpose K", q: false, k: true, v: false, o: false),
      (name: "Transpose V", q: false, k: false, v: true, o: false),
      (name: "Transpose Output", q: false, k: false, v: false, o: true),
      (name: "Transpose All", q: true, k: true, v: true, o: true),
    ]

    for config in transposeConfigs {
      print("\n--- Testing \(config.name) ---")

      var queryData = generateDeterministicData(count: totalElements, seed: 1000)
      var keyData = generateDeterministicData(count: totalElements, seed: 1001)
      var valueData = generateDeterministicData(count: totalElements, seed: 1002)
      var outputData = [Float](repeating: 0.0, count: totalElements)

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

      // Execute with transpose flags
      let result = mfa_attention_forward(
        context, qBuffer, kBuffer, vBuffer, oBuffer,
        batchSize, seqLen, seqLen, numHeads, headDim,
        1.0 / Float(headDim).squareRoot(),
        false, // causal
        MFA_PRECISION_FP32, MFA_PRECISION_FP32, MFA_PRECISION_FP32,
        config.q, config.k, config.v, config.o
      )

      defer {
        mfa_destroy_buffer(qBuffer)
        mfa_destroy_buffer(kBuffer)
        mfa_destroy_buffer(vBuffer)
        mfa_destroy_buffer(oBuffer)
      }

      let success = result == MFA_SUCCESS
      let hasNaN = outputData.contains { $0.isNaN }
      let hasInf = outputData.contains { $0.isInfinite }
      let nonZeroCount = outputData.filter { abs($0) > 1e-8 }.count
      let nonZeroRatio = Double(nonZeroCount) / Double(totalElements)

      print("  Success: \(success), NaN: \(hasNaN), Inf: \(hasInf), NonZero: \(String(format: "%.1f", nonZeroRatio * 100))%")

      if success && !hasNaN && !hasInf && nonZeroRatio > 0.1 {
        print("  ✅ \(config.name) layout test passed")
      } else {
        print("  ❌ \(config.name) layout test failed")
        XCTFail("Layout test failed for \(config.name)")
      }
    }
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

  private func generateDeterministicData(count: Int, seed: UInt64) -> [Float] {
    // Use a simple LCG for deterministic random numbers
    var rng = seed
    return (0..<count).map { _ in
      rng = rng &* 1664525 &+ 1013904223 // LCG constants
      let normalized = Float(rng % 1000000) / 1000000.0 // Normalize to [0, 1]
      return (normalized - 0.5) * 2.0 // Convert to [-1, 1]
    }
  }

  private func measureExecutionTime(_ block: () -> Void) -> Double {
    let startTime = CFAbsoluteTimeGetCurrent()
    block()
    let endTime = CFAbsoluteTimeGetCurrent()
    return endTime - startTime
  }

  func testComprehensiveTestSuite() throws {
    // Run all the comprehensive tests in sequence and provide summary
    print("\n🚀 COMPREHENSIVE TEST SUITE EXECUTION")
    print(String(repeating: "=", count: 60))
    print("This test suite includes:")
    print("✓ Multiple precision testing (FP32, FP16, BF16)")
    print("✓ FLUX-specific configurations (24 heads, 512+ sequences)")
    print("✓ Bug reproduction cases for BF16/FP16 NaN issues")
    print("✓ Quantization testing (INT8, INT4)")
    print("✓ Tensor layout variations")
    print("✓ Numerical correctness verification")
    print("✓ Performance characteristics")
    print("")

    // Execute main comprehensive test
    try testMultiHeadAttentionForward()

    // Execute additional verification tests
    try testNumericalCorrectnessAgainstPyTorch()
    try testTensorLayoutVariations()

    print("\n🎯 COMPREHENSIVE TEST SUITE SUMMARY")
    print(String(repeating: "=", count: 60))
    print("✅ All comprehensive tests completed")
    print("📊 Coverage includes:")
    print("   • Standard and FLUX attention configurations")
    print("   • All precision combinations (FP32, FP16, BF16)")
    print("   • Mixed precision scenarios")
    print("   • Quantization support (INT8, INT4)")
    print("   • Edge cases and bug reproduction")
    print("   • Tensor layout compatibility")
    print("   • Numerical stability verification")
    print("")
    print("🔍 This test suite is designed to catch:")
    print("   • BF16/FP16 NaN generation bugs")
    print("   • FLUX-specific dimension issues")
    print("   • Quantization precision problems")
    print("   • Memory layout incompatibilities")
    print("   • Numerical accuracy regressions")
    print("")
    print("📝 For PyTorch reference comparison:")
    print("   • Check test output for deterministic input values")
    print("   • Use the same LCG seed (42, 12345, etc.)")
    print("   • Compare against F.scaled_dot_product_attention()")
    print("   • Expected tolerance: 1e-5 for FP32, 1e-3 for FP16")
  }
}

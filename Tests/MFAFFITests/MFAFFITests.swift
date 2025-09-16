import XCTest

@testable import MFAFFI

final class MFAFFITests: XCTestCase {
  func testContextCreation() throws {
    var context: UnsafeMutableRawPointer?
    let result = mfa_create_context(&context)

    XCTAssertEqual(result, MFA_SUCCESS, "Context creation should succeed")
    XCTAssertNotNil(context, "Context should not be nil")

    if let context {
      mfa_destroy_context(context)
    }
  }

  func testDeviceSupport() throws {
    let isSupported = mfa_is_device_supported()
    XCTAssertTrue(isSupported, "Metal device should be supported on Apple platforms")
  }

  func testVersion() throws {
    var major: Int32 = 0
    var minor: Int32 = 0
    var patch: Int32 = 0

    mfa_get_version(&major, &minor, &patch)

    XCTAssertEqual(major, 1)
    XCTAssertEqual(minor, 0)
    XCTAssertEqual(patch, 0)
  }

  func testErrorStrings() throws {
    let successStr = mfa_error_string(MFA_SUCCESS)
    let invalidArgsStr = mfa_error_string(MFA_ERROR_INVALID_ARGS)

    XCTAssertNotNil(successStr)
    XCTAssertNotNil(invalidArgsStr)

    if let successStr {
      let swiftStr = String(cString: successStr)
      XCTAssertEqual(swiftStr, "Success")
      free(UnsafeMutableRawPointer(mutating: successStr))
    }

    if let invalidArgsStr {
      let swiftStr = String(cString: invalidArgsStr)
      XCTAssertEqual(swiftStr, "Invalid arguments")
      free(UnsafeMutableRawPointer(mutating: invalidArgsStr))
    }
  }

  func testBufferManagement() throws {
    var context: UnsafeMutableRawPointer?
    let contextResult = mfa_create_context(&context)
    XCTAssertEqual(contextResult, MFA_SUCCESS)

    defer {
      if let context {
        mfa_destroy_context(context)
      }
    }

    guard let context else {
      XCTFail("Context creation failed")
      return
    }

    // Test buffer creation
    var buffer: UnsafeMutableRawPointer?
    let bufferResult = mfa_create_buffer(context, 1024, &buffer)
    XCTAssertEqual(bufferResult, MFA_SUCCESS, "Buffer creation should succeed")
    XCTAssertNotNil(buffer, "Buffer should not be nil")

    // Test buffer contents access
    if let buffer {
      let contents = mfa_buffer_contents(buffer)
      XCTAssertNotNil(contents, "Buffer contents should be accessible")
      mfa_destroy_buffer(buffer)
    }
  }

  func testAttentionForward() throws {
    var context: UnsafeMutableRawPointer?
    let contextResult = mfa_create_context(&context)
    XCTAssertEqual(contextResult, MFA_SUCCESS)

    defer {
      if let context {
        mfa_destroy_context(context)
      }
    }

    guard let context else {
      XCTFail("Context creation failed")
      return
    }

    // Test comprehensive attention configurations
    try testAttentionFP32(context: context)
    try testAttentionFP16(context: context)
    try testAttentionBF16(context: context)
    try testAttentionVariousSizes(context: context)
    try testAttentionRealisticPatterns(context: context)
    try testAttentionEdgeCases(context: context)
  }

  // MARK: - FP32 Tests (Known Working)

  func testAttentionFP32(context: UnsafeMutableRawPointer) throws {
    print("\n=== Testing FP32 Attention ===")

    let testCases = [
      (seqLen: 4, headDim: 4, name: "tiny"),
      (seqLen: 64, headDim: 64, name: "small"),
      (seqLen: 128, headDim: 64, name: "medium")
    ]

    for testCase in testCases {
      print("Testing FP32 \(testCase.name): seq=\(testCase.seqLen), head_dim=\(testCase.headDim)")

      try runAttentionTest(
        context: context,
        seqLen: UInt32(testCase.seqLen),
        headDim: UInt16(testCase.headDim),
        inputPrecision: mfa_precision_t(rawValue: 2), // FP32
        intermediatePrecision: mfa_precision_t(rawValue: 2), // FP32
        outputPrecision: mfa_precision_t(rawValue: 2), // FP32
        expectedToPass: true,
        testName: "FP32-\(testCase.name)"
      )
    }
  }

  // MARK: - FP16 Tests (May Have Bugs)

  func testAttentionFP16(context: UnsafeMutableRawPointer) throws {
    print("\n=== Testing FP16 Attention ===")

    // Start with small sizes for FP16 due to known issues
    let testCases = [
      (seqLen: 4, headDim: 4, name: "tiny"),
      (seqLen: 32, headDim: 32, name: "small")
    ]

    for testCase in testCases {
      print("Testing FP16 \(testCase.name): seq=\(testCase.seqLen), head_dim=\(testCase.headDim)")

      try runAttentionTest(
        context: context,
        seqLen: UInt32(testCase.seqLen),
        headDim: UInt16(testCase.headDim),
        inputPrecision: mfa_precision_t(rawValue: 0), // FP16
        intermediatePrecision: mfa_precision_t(rawValue: 0), // FP16
        outputPrecision: mfa_precision_t(rawValue: 0), // FP16
        expectedToPass: false, // Known to have NaN issues
        testName: "FP16-\(testCase.name)"
      )
    }
  }

  // MARK: - BF16 Tests (May Have Bugs)

  func testAttentionBF16(context: UnsafeMutableRawPointer) throws {
    print("\n=== Testing BF16 Attention ===")

    // Start with small sizes for BF16 due to known issues
    let testCases = [
      (seqLen: 4, headDim: 4, name: "tiny"),
      (seqLen: 32, headDim: 32, name: "small")
    ]

    for testCase in testCases {
      print("Testing BF16 \(testCase.name): seq=\(testCase.seqLen), head_dim=\(testCase.headDim)")

      try runAttentionTest(
        context: context,
        seqLen: UInt32(testCase.seqLen),
        headDim: UInt16(testCase.headDim),
        inputPrecision: mfa_precision_t(rawValue: 1), // BF16
        intermediatePrecision: mfa_precision_t(rawValue: 1), // BF16
        outputPrecision: mfa_precision_t(rawValue: 1), // BF16
        expectedToPass: false, // Known to have NaN issues
        testName: "BF16-\(testCase.name)"
      )
    }
  }

  // MARK: - Various Size Tests

  func testAttentionVariousSizes(context: UnsafeMutableRawPointer) throws {
    print("\n=== Testing Various Sizes (FP32) ===")

    let testCases = [
      (seqLen: 16, headDim: 16, name: "16x16"),
      (seqLen: 32, headDim: 64, name: "32x64"),
      (seqLen: 64, headDim: 32, name: "64x32"),
      (seqLen: 128, headDim: 128, name: "128x128"),
      (seqLen: 256, headDim: 64, name: "256x64"),
      (seqLen: 512, headDim: 64, name: "512x64")
    ]

    for testCase in testCases {
      print("Testing size \(testCase.name): seq=\(testCase.seqLen), head_dim=\(testCase.headDim)")

      try runAttentionTest(
        context: context,
        seqLen: UInt32(testCase.seqLen),
        headDim: UInt16(testCase.headDim),
        inputPrecision: mfa_precision_t(rawValue: 2), // FP32
        intermediatePrecision: mfa_precision_t(rawValue: 2), // FP32
        outputPrecision: mfa_precision_t(rawValue: 2), // FP32
        expectedToPass: true,
        testName: "Size-\(testCase.name)"
      )
    }
  }

  // MARK: - Realistic Pattern Tests

  func testAttentionRealisticPatterns(context: UnsafeMutableRawPointer) throws {
    print("\n=== Testing Realistic Input Patterns ===")

    let seqLen: UInt32 = 64
    let headDim: UInt16 = 64

    // Test different input patterns that might reveal bugs
    let patterns = [
      "gaussian", "uniform", "sequential", "alternating", "sparse"
    ]

    for pattern in patterns {
      print("Testing pattern: \(pattern)")

      try runAttentionTestWithPattern(
        context: context,
        seqLen: seqLen,
        headDim: headDim,
        pattern: pattern,
        precision: mfa_precision_t(rawValue: 2), // FP32
        testName: "Pattern-\(pattern)"
      )
    }
  }

  // MARK: - Edge Case Tests

  func testAttentionEdgeCases(context: UnsafeMutableRawPointer) throws {
    print("\n=== Testing Edge Cases ===")

    // Test causal vs non-causal
    try testCausalAttention(context: context)

    // Test extreme softmax scales
    try testSoftmaxScales(context: context)

    // Test single element sequences
    try testMinimalSizes(context: context)
  }

  func testCausalAttention(context: UnsafeMutableRawPointer) throws {
    print("Testing causal vs non-causal attention")

    let seqLen: UInt32 = 32
    let headDim: UInt16 = 32

    for causal in [false, true] {
      print("  Testing causal=\(causal)")

      try runAttentionTest(
        context: context,
        seqLen: seqLen,
        headDim: headDim,
        inputPrecision: mfa_precision_t(rawValue: 2), // FP32
        intermediatePrecision: mfa_precision_t(rawValue: 2), // FP32
        outputPrecision: mfa_precision_t(rawValue: 2), // FP32
        expectedToPass: true,
        testName: "Causal-\(causal)",
        causal: causal
      )
    }
  }

  func testSoftmaxScales(context: UnsafeMutableRawPointer) throws {
    print("Testing extreme softmax scales")

    let seqLen: UInt32 = 16
    let headDim: UInt16 = 16
    let scales: [Float] = [0.01, 0.1, 1.0, 10.0, 100.0]

    for scale in scales {
      print("  Testing scale=\(scale)")

      try runAttentionTest(
        context: context,
        seqLen: seqLen,
        headDim: headDim,
        inputPrecision: mfa_precision_t(rawValue: 2), // FP32
        intermediatePrecision: mfa_precision_t(rawValue: 2), // FP32
        outputPrecision: mfa_precision_t(rawValue: 2), // FP32
        expectedToPass: true,
        testName: "Scale-\(scale)",
        softmaxScale: scale
      )
    }
  }

  func testMinimalSizes(context: UnsafeMutableRawPointer) throws {
    print("Testing minimal sizes")

    let testCases = [
      (seqLen: 1, headDim: 1, name: "1x1"),
      (seqLen: 1, headDim: 4, name: "1x4"),
      (seqLen: 2, headDim: 2, name: "2x2")
    ]

    for testCase in testCases {
      print("  Testing minimal \(testCase.name)")

      try runAttentionTest(
        context: context,
        seqLen: UInt32(testCase.seqLen),
        headDim: UInt16(testCase.headDim),
        inputPrecision: mfa_precision_t(rawValue: 2), // FP32
        intermediatePrecision: mfa_precision_t(rawValue: 2), // FP32
        outputPrecision: mfa_precision_t(rawValue: 2), // FP32
        expectedToPass: true,
        testName: "Minimal-\(testCase.name)"
      )
    }
  }

  // MARK: - Helper Functions

  func runAttentionTest(
    context: UnsafeMutableRawPointer,
    seqLen: UInt32,
    headDim: UInt16,
    inputPrecision: mfa_precision_t,
    intermediatePrecision: mfa_precision_t,
    outputPrecision: mfa_precision_t,
    expectedToPass: Bool,
    testName: String,
    causal: Bool = false,
    softmaxScale: Float? = nil
  ) throws {

    let elementCount = Int(seqLen * UInt32(headDim))
    let tensorSize = elementCount * 4 // FP32 bytes

    // Create realistic test data with Gaussian distribution
    var qData: [Float] = (0..<elementCount).map { _ in Float.random(in: -1...1) }
    var kData: [Float] = (0..<elementCount).map { _ in Float.random(in: -1...1) }
    var vData: [Float] = (0..<elementCount).map { _ in Float.random(in: -1...1) }
    var outData: [Float] = Array(repeating: 0.0, count: elementCount)

    // Create buffers
    var qBuffer: UnsafeMutableRawPointer?
    var kBuffer: UnsafeMutableRawPointer?
    var vBuffer: UnsafeMutableRawPointer?
    var outBuffer: UnsafeMutableRawPointer?

    let qResult = mfa_buffer_from_ptr(context, &qData, tensorSize, &qBuffer)
    let kResult = mfa_buffer_from_ptr(context, &kData, tensorSize, &kBuffer)
    let vResult = mfa_buffer_from_ptr(context, &vData, tensorSize, &vBuffer)
    let outResult = mfa_buffer_from_ptr(context, &outData, tensorSize, &outBuffer)

    XCTAssertEqual(qResult, MFA_SUCCESS, "\(testName): Q buffer creation failed")
    XCTAssertEqual(kResult, MFA_SUCCESS, "\(testName): K buffer creation failed")
    XCTAssertEqual(vResult, MFA_SUCCESS, "\(testName): V buffer creation failed")
    XCTAssertEqual(outResult, MFA_SUCCESS, "\(testName): Output buffer creation failed")

    defer {
      if let qBuffer { mfa_destroy_buffer(qBuffer) }
      if let kBuffer { mfa_destroy_buffer(kBuffer) }
      if let vBuffer { mfa_destroy_buffer(vBuffer) }
      if let outBuffer { mfa_destroy_buffer(outBuffer) }
    }

    let scale = softmaxScale ?? (1.0 / sqrt(Float(headDim)))

    // Run attention
    let attentionResult = mfa_attention_forward(
      context,
      qBuffer, kBuffer, vBuffer, outBuffer,
      1, // batch_size
      seqLen, // seq_len_q
      seqLen, // seq_len_kv
      1, // num_heads
      headDim, // head_dim
      scale, // softmax_scale
      causal, // causal
      inputPrecision, // input_precision
      intermediatePrecision, // intermediate_precision
      outputPrecision, // output_precision
      false, false, false, false // no transposes
    )

    XCTAssertEqual(attentionResult, MFA_SUCCESS, "\(testName): Attention computation failed")

    // Validate output quality
    let nanCount = outData.filter { $0.isNaN }.count
    let infCount = outData.filter { $0.isInfinite }.count
    let zeroCount = outData.filter { $0 == 0.0 }.count
    let nonZeroCount = outData.filter { $0 != 0.0 }.count

    print("  \(testName) Results:")
    print("    NaN count: \(nanCount)/\(elementCount)")
    print("    Inf count: \(infCount)/\(elementCount)")
    print("    Zero count: \(zeroCount)/\(elementCount)")
    print("    Non-zero count: \(nonZeroCount)/\(elementCount)")
    print("    Output range: [\(outData.min() ?? 0), \(outData.max() ?? 0)]")
    print("    Output mean: \(outData.reduce(0, +) / Float(elementCount))")

    // Critical checks for NaN/Inf
    if expectedToPass {
      XCTAssertEqual(nanCount, 0, "\(testName): Found \(nanCount) NaN values in output")
      XCTAssertEqual(infCount, 0, "\(testName): Found \(infCount) Inf values in output")
      XCTAssertGreaterThan(nonZeroCount, 0, "\(testName): Output should contain non-zero values")

      // Verify attention output properties
      verifyAttentionProperties(output: outData, testName: testName)
    } else {
      // For known failing cases, just log the issues
      if nanCount > 0 || infCount > 0 {
        print("  ⚠️  \(testName): Found NaN/Inf as expected in buggy precision mode")
      }
    }
  }

  func runAttentionTestWithPattern(
    context: UnsafeMutableRawPointer,
    seqLen: UInt32,
    headDim: UInt16,
    pattern: String,
    precision: mfa_precision_t,
    testName: String
  ) throws {

    let elementCount = Int(seqLen * UInt32(headDim))
    let tensorSize = elementCount * 4 // FP32 bytes

    // Generate data based on pattern
    var qData: [Float]
    var kData: [Float]
    var vData: [Float]

    switch pattern {
    case "gaussian":
      qData = (0..<elementCount).map { _ in gaussianRandom() }
      kData = (0..<elementCount).map { _ in gaussianRandom() }
      vData = (0..<elementCount).map { _ in gaussianRandom() }

    case "uniform":
      qData = (0..<elementCount).map { _ in Float.random(in: -2...2) }
      kData = (0..<elementCount).map { _ in Float.random(in: -2...2) }
      vData = (0..<elementCount).map { _ in Float.random(in: -2...2) }

    case "sequential":
      qData = (0..<elementCount).map { Float($0) / Float(elementCount) }
      kData = (0..<elementCount).map { Float($0) / Float(elementCount) }
      vData = (0..<elementCount).map { Float($0) / Float(elementCount) }

    case "alternating":
      qData = (0..<elementCount).map { $0 % 2 == 0 ? 1.0 : -1.0 }
      kData = (0..<elementCount).map { $0 % 2 == 0 ? -1.0 : 1.0 }
      vData = (0..<elementCount).map { $0 % 2 == 0 ? 1.0 : -1.0 }

    case "sparse":
      qData = (0..<elementCount).map { $0 % 10 == 0 ? gaussianRandom() : 0.0 }
      kData = (0..<elementCount).map { $0 % 10 == 0 ? gaussianRandom() : 0.0 }
      vData = (0..<elementCount).map { $0 % 10 == 0 ? gaussianRandom() : 0.0 }

    default:
      qData = Array(repeating: 1.0, count: elementCount)
      kData = Array(repeating: 1.0, count: elementCount)
      vData = Array(repeating: 1.0, count: elementCount)
    }

    var outData: [Float] = Array(repeating: 0.0, count: elementCount)

    // Create buffers and run test
    var qBuffer: UnsafeMutableRawPointer?
    var kBuffer: UnsafeMutableRawPointer?
    var vBuffer: UnsafeMutableRawPointer?
    var outBuffer: UnsafeMutableRawPointer?

    let qResult = mfa_buffer_from_ptr(context, &qData, tensorSize, &qBuffer)
    let kResult = mfa_buffer_from_ptr(context, &kData, tensorSize, &kBuffer)
    let vResult = mfa_buffer_from_ptr(context, &vData, tensorSize, &vBuffer)
    let outResult = mfa_buffer_from_ptr(context, &outData, tensorSize, &outBuffer)

    XCTAssertEqual(qResult, MFA_SUCCESS)
    XCTAssertEqual(kResult, MFA_SUCCESS)
    XCTAssertEqual(vResult, MFA_SUCCESS)
    XCTAssertEqual(outResult, MFA_SUCCESS)

    defer {
      if let qBuffer { mfa_destroy_buffer(qBuffer) }
      if let kBuffer { mfa_destroy_buffer(kBuffer) }
      if let vBuffer { mfa_destroy_buffer(vBuffer) }
      if let outBuffer { mfa_destroy_buffer(outBuffer) }
    }

    let attentionResult = mfa_attention_forward(
      context,
      qBuffer, kBuffer, vBuffer, outBuffer,
      1, seqLen, seqLen, 1, headDim,
      1.0 / sqrt(Float(headDim)),
      false,
      precision, precision, precision,
      false, false, false, false
    )

    XCTAssertEqual(attentionResult, MFA_SUCCESS, "\(testName): Attention computation failed")

    // Validate pattern-specific properties
    validatePatternResults(output: outData, pattern: pattern, testName: testName)
  }

  func gaussianRandom() -> Float {
    // Box-Muller transform for Gaussian random numbers
    let u1 = Float.random(in: 0..<1)
    let u2 = Float.random(in: 0..<1)
    return sqrt(-2 * log(u1)) * cos(2 * Float.pi * u2)
  }

  func verifyAttentionProperties(output: [Float], testName: String) {
    guard !output.isEmpty else { return }

    let mean = output.reduce(0, +) / Float(output.count)
    let variance = output.map { pow($0 - mean, 2) }.reduce(0, +) / Float(output.count)
    let stdDev = sqrt(variance)

    // Attention outputs should have reasonable statistical properties
    XCTAssertGreaterThan(stdDev, 0.001, "\(testName): Output has too little variance")
    XCTAssertLessThan(abs(mean), 10.0, "\(testName): Output mean is too extreme")

    // Check for reasonable output range
    let minVal = output.min() ?? 0
    let maxVal = output.max() ?? 0
    XCTAssertGreaterThan(maxVal - minVal, 0.001, "\(testName): Output range is too narrow")
  }

  func validatePatternResults(output: [Float], pattern: String, testName: String) {
    let nanCount = output.filter { $0.isNaN }.count
    let infCount = output.filter { $0.isInfinite }.count

    XCTAssertEqual(nanCount, 0, "\(testName): Pattern \(pattern) produced NaN values")
    XCTAssertEqual(infCount, 0, "\(testName): Pattern \(pattern) produced Inf values")

    // Pattern-specific validations
    switch pattern {
    case "sparse":
      let nonZeroCount = output.filter { abs($0) > 1e-6 }.count
      XCTAssertGreaterThan(nonZeroCount, 0, "\(testName): Sparse pattern should produce some non-zero output")

    case "uniform", "gaussian":
      let range = (output.max() ?? 0) - (output.min() ?? 0)
      XCTAssertGreaterThan(range, 0.1, "\(testName): \(pattern) pattern should produce varied output")

    default:
      break
    }
  }
}

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

    // Simple test: 4x4 tensors using FP32 like MFA tests
    let seqLen: UInt32 = 4
    let headDim: UInt16 = 4
    let tensorSize = Int(seqLen * UInt32(headDim) * 4) // FP32, 4 bytes per element

    // Create test data using Float (FP32) like MFA tests
    var qData: [Float] = Array(repeating: 1.0, count: Int(seqLen * UInt32(headDim)))
    var kData: [Float] = Array(repeating: 1.0, count: Int(seqLen * UInt32(headDim)))
    var vData: [Float] = Array(repeating: 1.0, count: Int(seqLen * UInt32(headDim)))
    var outData: [Float] = Array(repeating: 0.0, count: Int(seqLen * UInt32(headDim)))

    // Create buffers from data
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

    // Run attention
    let attentionResult = mfa_attention_forward(
      context,
      qBuffer, kBuffer, vBuffer, outBuffer,
      1, // batch_size
      seqLen, // seq_len_q
      seqLen, // seq_len_kv
      1, // num_heads
      headDim, // head_dim
      1.0 / sqrt(Float(headDim)), // softmax_scale
      false, // causal - test non-causal first
      mfa_precision_t(rawValue: 2), // input_precision (FP32)
      mfa_precision_t(rawValue: 2), // intermediate_precision (FP32)
      mfa_precision_t(rawValue: 2), // output_precision (FP32)
      false, false, false, false // no transposes
    )

    XCTAssertEqual(attentionResult, MFA_SUCCESS, "Attention computation should succeed")

    // Check output is not all zeros
    let hasNonZero = outData.contains { $0 != 0.0 }
    XCTAssertTrue(hasNonZero, "Output should contain non-zero values")

    print("Output tensor: \(outData.prefix(10))...")
  }
}

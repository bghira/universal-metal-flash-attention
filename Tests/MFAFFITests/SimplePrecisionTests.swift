import XCTest
@testable import MFABridge

final class SimplePrecisionTests: XCTestCase {
  func testFP32PrecisionDoesNotProduceNaN() throws {
    let batch: UInt32 = 1
    let heads: UInt32 = 4
    let seqLen: UInt32 = 32
    let headDim: UInt16 = 16
    let elements = Int(batch * heads * seqLen * UInt32(headDim))

    // Generate test data
    var seed = 12345
    var qData = [Float](repeating: 0, count: elements)
    var kData = [Float](repeating: 0, count: elements)
    var vData = [Float](repeating: 0, count: elements)

    for i in 0..<elements {
      seed = (seed &* 1_664_525 &+ 1_013_904_223) & 0xFFFF_FFFF
      let value = Float(seed % 1_000_000) / 5_000_000.0 - 0.1 // Small values
      qData[i] = value

      seed = (seed &* 1_664_525 &+ 1_013_904_223) & 0xFFFF_FFFF
      kData[i] = Float(seed % 1_000_000) / 5_000_000.0 - 0.1

      seed = (seed &* 1_664_525 &+ 1_013_904_223) & 0xFFFF_FFFF
      vData[i] = Float(seed % 1_000_000) / 5_000_000.0 - 0.1
    }

    // Create context
    var context: UnsafeMutableRawPointer?
    let createResult = mfa_create_context(&context)
    XCTAssertEqual(createResult, 0, "Failed to create context")
    defer { mfa_destroy_context(context) }

    // Create buffers
    var q: UnsafeMutableRawPointer?
    var k: UnsafeMutableRawPointer?
    var v: UnsafeMutableRawPointer?
    var out: UnsafeMutableRawPointer?

    let bytes = elements * MemoryLayout<Float>.size

    let qResult = mfa_create_buffer(context, bytes, &q)
    XCTAssertEqual(qResult, 0, "Failed to create Q buffer")
    defer { mfa_destroy_buffer(q) }

    let kResult = mfa_create_buffer(context, bytes, &k)
    XCTAssertEqual(kResult, 0, "Failed to create K buffer")
    defer { mfa_destroy_buffer(k) }

    let vResult = mfa_create_buffer(context, bytes, &v)
    XCTAssertEqual(vResult, 0, "Failed to create V buffer")
    defer { mfa_destroy_buffer(v) }

    let outResult = mfa_create_buffer(context, bytes, &out)
    XCTAssertEqual(outResult, 0, "Failed to create output buffer")
    defer { mfa_destroy_buffer(out) }

    // Copy data to buffers
    if let qContents = mfa_buffer_contents(q) {
      qData.withUnsafeBytes { ptr in
        qContents.copyMemory(from: ptr.baseAddress!, byteCount: bytes)
      }
    }

    if let kContents = mfa_buffer_contents(k) {
      kData.withUnsafeBytes { ptr in
        kContents.copyMemory(from: ptr.baseAddress!, byteCount: bytes)
      }
    }

    if let vContents = mfa_buffer_contents(v) {
      vData.withUnsafeBytes { ptr in
        vContents.copyMemory(from: ptr.baseAddress!, byteCount: bytes)
      }
    }

    // Call attention with FP32 precision (C FFI enum: FP16=0, BF16=1, FP32=2)
    let result = mfa_attention_forward_nomask(
      context, q, k, v, out,
      batch, seqLen, seqLen, heads, headDim,
      1.0 / sqrtf(Float(headDim)), // softmax scale
      false, // not causal
      2, // FP32 input precision
      2, // FP32 intermediate precision
      2, // FP32 output precision
      false, false, false, false // no transpose
    )

    XCTAssertEqual(result, 0, "FP32 attention failed")

    // Check output for NaN
    if let outContents = mfa_buffer_contents(out) {
      let outPointer = outContents.bindMemory(to: Float.self, capacity: elements)
      var nanCount = 0
      var nonZeroCount = 0

      for i in 0..<elements {
        if outPointer[i].isNaN {
          nanCount += 1
        }
        if abs(outPointer[i]) > 1e-8 {
          nonZeroCount += 1
        }
      }

      XCTAssertEqual(nanCount, 0, "Found \(nanCount) NaN values in FP32 output")
      XCTAssertGreaterThan(nonZeroCount, 0, "Output is all zeros")

      print("✅ FP32 test: \(nonZeroCount)/\(elements) non-zero values, no NaNs")
    }
  }

  func testFP16PrecisionDoesNotProduceNaN() throws {
    let batch: UInt32 = 1
    let heads: UInt32 = 4
    let seqLen: UInt32 = 32
    let headDim: UInt16 = 16
    let elements = Int(batch * heads * seqLen * UInt32(headDim))

    // Generate test data as Float16
    var seed = 12345
    var qData = [Float16](repeating: 0, count: elements)
    var kData = [Float16](repeating: 0, count: elements)
    var vData = [Float16](repeating: 0, count: elements)

    for i in 0..<elements {
      seed = (seed &* 1_664_525 &+ 1_013_904_223) & 0xFFFF_FFFF
      let value = Float16(Float(seed % 1_000_000) / 5_000_000.0 - 0.1)
      qData[i] = value

      seed = (seed &* 1_664_525 &+ 1_013_904_223) & 0xFFFF_FFFF
      kData[i] = Float16(Float(seed % 1_000_000) / 5_000_000.0 - 0.1)

      seed = (seed &* 1_664_525 &+ 1_013_904_223) & 0xFFFF_FFFF
      vData[i] = Float16(Float(seed % 1_000_000) / 5_000_000.0 - 0.1)
    }

    // Create context
    var context: UnsafeMutableRawPointer?
    let createResult = mfa_create_context(&context)
    XCTAssertEqual(createResult, 0, "Failed to create context")
    defer { mfa_destroy_context(context) }

    // Create buffers (FP16 uses half the bytes)
    var q: UnsafeMutableRawPointer?
    var k: UnsafeMutableRawPointer?
    var v: UnsafeMutableRawPointer?
    var out: UnsafeMutableRawPointer?

    let bytes = elements * MemoryLayout<Float16>.size
    // The kernel always stores O as FP32 regardless of the output label.
    let outBytes = elements * MemoryLayout<Float>.size

    let qResult = mfa_create_buffer(context, bytes, &q)
    XCTAssertEqual(qResult, 0, "Failed to create Q buffer")
    defer { mfa_destroy_buffer(q) }

    let kResult = mfa_create_buffer(context, bytes, &k)
    XCTAssertEqual(kResult, 0, "Failed to create K buffer")
    defer { mfa_destroy_buffer(k) }

    let vResult = mfa_create_buffer(context, bytes, &v)
    XCTAssertEqual(vResult, 0, "Failed to create V buffer")
    defer { mfa_destroy_buffer(v) }

    let outResult = mfa_create_buffer(context, outBytes, &out)
    XCTAssertEqual(outResult, 0, "Failed to create output buffer")
    defer { mfa_destroy_buffer(out) }

    // Copy data to buffers
    if let qContents = mfa_buffer_contents(q) {
      qData.withUnsafeBytes { ptr in
        qContents.copyMemory(from: ptr.baseAddress!, byteCount: bytes)
      }
    }

    if let kContents = mfa_buffer_contents(k) {
      kData.withUnsafeBytes { ptr in
        kContents.copyMemory(from: ptr.baseAddress!, byteCount: bytes)
      }
    }

    if let vContents = mfa_buffer_contents(v) {
      vData.withUnsafeBytes { ptr in
        vContents.copyMemory(from: ptr.baseAddress!, byteCount: bytes)
      }
    }

    // Call attention with FP16 precision (C FFI enum: FP16=0, BF16=1, FP32=2)
    let result = mfa_attention_forward_nomask(
      context, q, k, v, out,
      batch, seqLen, seqLen, heads, headDim,
      1.0 / sqrtf(Float(headDim)), // softmax scale
      false, // not causal
      0, // FP16 input precision
      0, // FP16 intermediate precision
      0, // FP16 output precision
      false, false, false, false // no transpose
    )

    XCTAssertEqual(result, 0, "FP16 attention failed")

    // Check output for NaN
    if let outContents = mfa_buffer_contents(out) {
      let outPointer = outContents.bindMemory(to: Float.self, capacity: elements)
      var nanCount = 0
      var nonZeroCount = 0

      for i in 0..<elements {
        if outPointer[i].isNaN {
          nanCount += 1
        }
        if abs(outPointer[i]) > 1e-8 {
          nonZeroCount += 1
        }
      }

      XCTAssertEqual(nanCount, 0, "Found \(nanCount) NaN values in FP16 output")
      XCTAssertGreaterThan(nonZeroCount, 0, "Output is all zeros")

      print("✅ FP16 test: \(nonZeroCount)/\(elements) non-zero values, no NaNs")
    }
  }
}

import XCTest
import Metal
import MetalPerformanceShaders
import MFAFFI
import MFABridge

/// Comprehensive precision tests for Metal Flash Attention
/// Tests FP32, FP16, and BF16 precision handling to catch NaN issues
final class PrecisionTests: XCTestCase {

    var device: MTLDevice!
    var commandQueue: MTLCommandQueue!

    override func setUp() {
        super.setUp()
        device = MTLCreateSystemDefaultDevice()
        XCTAssertNotNil(device, "Metal device not available")
        commandQueue = device.makeCommandQueue()
        XCTAssertNotNil(commandQueue, "Failed to create command queue")
    }

    // MARK: - Helper Functions

    /// Generate deterministic test data for reproducible results
    private func generateTestData(count: Int, seed: Int = 12345) -> [Float] {
        var rng = seed
        var data: [Float] = []

        for _ in 0..<count {
            // LCG for deterministic pseudo-random numbers
            rng = (rng &* 1664525 &+ 1013904223) & 0xFFFFFFFF
            let normalized = Float(rng % 1000000) / 1000000.0
            let value = (normalized - 0.5) * 0.2  // Small values to avoid overflow
            data.append(value)
        }

        return data
    }

    /// Check for NaN values in output buffer
    private func checkForNaN(_ buffer: MTLBuffer, count: Int, precision: String) -> (hasNaN: Bool, nanCount: Int) {
        let pointer = buffer.contents().bindMemory(to: Float.self, capacity: count)
        var nanCount = 0

        for i in 0..<count {
            if pointer[i].isNaN {
                nanCount += 1
            }
        }

        return (nanCount > 0, nanCount)
    }

    // MARK: - Precision Tests

    func testFP32Precision() throws {
        let batchSize: Int32 = 1
        let numHeads: Int32 = 4
        let seqLen: Int32 = 32
        let headDim: Int32 = 16
        let elementCount = Int(batchSize * numHeads * seqLen * headDim)

        // Generate test data
        let qData = generateTestData(count: elementCount, seed: 1001)
        let kData = generateTestData(count: elementCount, seed: 2002)
        let vData = generateTestData(count: elementCount, seed: 3003)

        // Create Metal buffers
        let qBuffer = device.makeBuffer(bytes: qData, length: qData.count * MemoryLayout<Float>.size, options: .storageModeShared)!
        let kBuffer = device.makeBuffer(bytes: kData, length: kData.count * MemoryLayout<Float>.size, options: .storageModeShared)!
        let vBuffer = device.makeBuffer(bytes: vData, length: vData.count * MemoryLayout<Float>.size, options: .storageModeShared)!
        let outBuffer = device.makeBuffer(length: elementCount * MemoryLayout<Float>.size, options: .storageModeShared)!

        // Create context
        var context: UnsafeMutableRawPointer?
        let createResult = mfa_create_context(&context)
        XCTAssertEqual(createResult, mfa_error_t(rawValue: 0), "Failed to create context")
        defer { mfa_destroy_context(context) }

        // Call attention with FP32 precision
        let result = mfa_attention_forward_str(
            context,
            UnsafeMutableRawPointer(qBuffer),
            UnsafeMutableRawPointer(kBuffer),
            UnsafeMutableRawPointer(vBuffer),
            UnsafeMutableRawPointer(outBuffer),
            UInt32(batchSize),
            UInt32(seqLen),
            UInt32(seqLen),
            UInt32(numHeads),
            UInt16(headDim),
            1.0 / sqrtf(Float(headDim)),  // softmax scale
            false,  // not causal
            "fp32",  // input precision
            "fp32",  // intermediate precision
            "fp32",  // output precision
            false, false, false, false  // no transpose
        )

        XCTAssertEqual(result, 0, "FP32 attention failed with error code \(result)")

        // Check for NaN values
        let nanCheck = checkForNaN(outBuffer, count: elementCount, precision: "FP32")
        XCTAssertFalse(nanCheck.hasNaN, "Found \(nanCheck.nanCount) NaN values in FP32 output")

        // Verify output is not all zeros
        let outPointer = outBuffer.contents().bindMemory(to: Float.self, capacity: elementCount)
        var nonZeroCount = 0
        for i in 0..<elementCount {
            if abs(outPointer[i]) > 1e-8 {
                nonZeroCount += 1
            }
        }
        XCTAssertGreaterThan(nonZeroCount, 0, "FP32 output is all zeros")

        print("✅ FP32 test passed: \(nonZeroCount)/\(elementCount) non-zero values")
    }

    func testFP16Precision() throws {
        let batchSize: Int32 = 1
        let numHeads: Int32 = 4
        let seqLen: Int32 = 32
        let headDim: Int32 = 16
        let elementCount = Int(batchSize * numHeads * seqLen * headDim)

        // Generate test data (as Float, will be converted)
        let qData = generateTestData(count: elementCount, seed: 1001)
        let kData = generateTestData(count: elementCount, seed: 2002)
        let vData = generateTestData(count: elementCount, seed: 3003)

        // Convert to Float16
        var qData16 = qData.map { Float16($0) }
        var kData16 = kData.map { Float16($0) }
        var vData16 = vData.map { Float16($0) }

        // Create Metal buffers
        let qBuffer = device.makeBuffer(bytes: &qData16, length: qData16.count * MemoryLayout<Float16>.size, options: .storageModeShared)!
        let kBuffer = device.makeBuffer(bytes: &kData16, length: kData16.count * MemoryLayout<Float16>.size, options: .storageModeShared)!
        let vBuffer = device.makeBuffer(bytes: &vData16, length: vData16.count * MemoryLayout<Float16>.size, options: .storageModeShared)!
        let outBuffer = device.makeBuffer(length: elementCount * MemoryLayout<Float16>.size, options: .storageModeShared)!

        // Create context
        var context: UnsafeMutableRawPointer?
        let createResult = mfa_create_context(&context)
        XCTAssertEqual(createResult, mfa_error_t(rawValue: 0), "Failed to create context")
        defer { mfa_destroy_context(context) }

        // Call attention with FP16 precision
        let result = mfa_attention_forward_str(
            context,
            UnsafeMutableRawPointer(qBuffer),
            UnsafeMutableRawPointer(kBuffer),
            UnsafeMutableRawPointer(vBuffer),
            UnsafeMutableRawPointer(outBuffer),
            UInt32(batchSize),
            UInt32(seqLen),
            UInt32(seqLen),
            UInt32(numHeads),
            UInt16(headDim),
            1.0 / sqrtf(Float(headDim)),  // softmax scale
            false,  // not causal
            "fp16",  // input precision
            "fp16",  // intermediate precision
            "fp16",  // output precision
            false, false, false, false  // no transpose
        )

        XCTAssertEqual(result, 0, "FP16 attention failed with error code \(result)")

        // Check for NaN values (convert FP16 to Float for checking)
        let outPointer16 = outBuffer.contents().bindMemory(to: Float16.self, capacity: elementCount)
        var nanCount = 0
        var nonZeroCount = 0

        for i in 0..<elementCount {
            let value = Float(outPointer16[i])
            if value.isNaN {
                nanCount += 1
            }
            if abs(value) > 1e-8 {
                nonZeroCount += 1
            }
        }

        XCTAssertEqual(nanCount, 0, "Found \(nanCount) NaN values in FP16 output")
        XCTAssertGreaterThan(nonZeroCount, 0, "FP16 output is all zeros")

        print("✅ FP16 test passed: \(nonZeroCount)/\(elementCount) non-zero values")
    }

    func testBF16Precision() throws {
        // Note: BFloat16 requires iOS 14+ / macOS 11+
        guard #available(iOS 14.0, macOS 11.0, *) else {
            throw XCTSkip("BFloat16 requires iOS 14+ / macOS 11+")
        }

        let batchSize: Int32 = 1
        let numHeads: Int32 = 4
        let seqLen: Int32 = 32
        let headDim: Int32 = 16
        let elementCount = Int(batchSize * numHeads * seqLen * headDim)

        // Generate test data
        let qData = generateTestData(count: elementCount, seed: 1001)
        let kData = generateTestData(count: elementCount, seed: 2002)
        let vData = generateTestData(count: elementCount, seed: 3003)

        // Note: Swift doesn't have built-in BFloat16, so we'd need to handle this specially
        // For now, we'll test with the string interface and Float32 data

        // Create Metal buffers
        let qBuffer = device.makeBuffer(bytes: qData, length: qData.count * MemoryLayout<Float>.size, options: .storageModeShared)!
        let kBuffer = device.makeBuffer(bytes: kData, length: kData.count * MemoryLayout<Float>.size, options: .storageModeShared)!
        let vBuffer = device.makeBuffer(bytes: vData, length: vData.count * MemoryLayout<Float>.size, options: .storageModeShared)!
        let outBuffer = device.makeBuffer(length: elementCount * MemoryLayout<Float>.size, options: .storageModeShared)!

        // Create context
        var context: UnsafeMutableRawPointer?
        let createResult = mfa_create_context(&context)
        XCTAssertEqual(createResult, mfa_error_t(rawValue: 0), "Failed to create context")
        defer { mfa_destroy_context(context) }

        // Call attention with BF16 precision specification
        let result = mfa_attention_forward_str(
            context,
            UnsafeMutableRawPointer(qBuffer),
            UnsafeMutableRawPointer(kBuffer),
            UnsafeMutableRawPointer(vBuffer),
            UnsafeMutableRawPointer(outBuffer),
            UInt32(batchSize),
            UInt32(seqLen),
            UInt32(seqLen),
            UInt32(numHeads),
            UInt16(headDim),
            1.0 / sqrtf(Float(headDim)),  // softmax scale
            false,  // not causal
            "bf16",  // input precision
            "bf16",  // intermediate precision
            "bf16",  // output precision
            false, false, false, false  // no transpose
        )

        XCTAssertEqual(result, 0, "BF16 attention failed with error code \(result)")

        // Check for NaN values
        let nanCheck = checkForNaN(outBuffer, count: elementCount, precision: "BF16")
        XCTAssertFalse(nanCheck.hasNaN, "Found \(nanCheck.nanCount) NaN values in BF16 output")

        print("✅ BF16 test passed")
    }

    func testMixedPrecision() throws {
        // Test with mixed precision: FP16 input, FP32 intermediate, FP16 output
        let batchSize: Int32 = 1
        let numHeads: Int32 = 4
        let seqLen: Int32 = 32
        let headDim: Int32 = 16
        let elementCount = Int(batchSize * numHeads * seqLen * headDim)

        // Generate test data
        let qData = generateTestData(count: elementCount, seed: 1001)
        var qData16 = qData.map { Float16($0) }

        let kData = generateTestData(count: elementCount, seed: 2002)
        var kData16 = kData.map { Float16($0) }

        let vData = generateTestData(count: elementCount, seed: 3003)
        var vData16 = vData.map { Float16($0) }

        // Create Metal buffers
        let qBuffer = device.makeBuffer(bytes: &qData16, length: qData16.count * MemoryLayout<Float16>.size, options: .storageModeShared)!
        let kBuffer = device.makeBuffer(bytes: &kData16, length: kData16.count * MemoryLayout<Float16>.size, options: .storageModeShared)!
        let vBuffer = device.makeBuffer(bytes: &vData16, length: vData16.count * MemoryLayout<Float16>.size, options: .storageModeShared)!
        let outBuffer = device.makeBuffer(length: elementCount * MemoryLayout<Float16>.size, options: .storageModeShared)!

        // Create context
        var context: UnsafeMutableRawPointer?
        let createResult = mfa_create_context(&context)
        XCTAssertEqual(createResult, mfa_error_t(rawValue: 0), "Failed to create context")
        defer { mfa_destroy_context(context) }

        // Call attention with mixed precision
        let result = mfa_attention_forward_str(
            context,
            UnsafeMutableRawPointer(qBuffer),
            UnsafeMutableRawPointer(kBuffer),
            UnsafeMutableRawPointer(vBuffer),
            UnsafeMutableRawPointer(outBuffer),
            UInt32(batchSize),
            UInt32(seqLen),
            UInt32(seqLen),
            UInt32(numHeads),
            UInt16(headDim),
            1.0 / sqrtf(Float(headDim)),  // softmax scale
            false,  // not causal
            "fp16",  // input precision
            "fp32",  // intermediate precision (higher for stability)
            "fp16",  // output precision
            false, false, false, false  // no transpose
        )

        XCTAssertEqual(result, 0, "Mixed precision attention failed with error code \(result)")

        // Check for NaN values
        let outPointer16 = outBuffer.contents().bindMemory(to: Float16.self, capacity: elementCount)
        var nanCount = 0

        for i in 0..<elementCount {
            if Float(outPointer16[i]).isNaN {
                nanCount += 1
            }
        }

        XCTAssertEqual(nanCount, 0, "Found \(nanCount) NaN values in mixed precision output")

        print("✅ Mixed precision test passed")
    }
}
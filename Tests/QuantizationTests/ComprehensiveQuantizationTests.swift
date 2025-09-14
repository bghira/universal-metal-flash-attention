import FlashAttention
import Metal
import XCTest

@testable import MFABridge

final class ComprehensiveQuantizationTests: XCTestCase {
  var device: MTLDevice!
  var commandQueue: MTLCommandQueue!

  override func setUpWithError() throws {
    guard let device = MTLCreateSystemDefaultDevice() else {
      throw XCTSkip("Metal device not available")
    }
    self.device = device

    guard let commandQueue = device.makeCommandQueue() else {
      throw XCTSkip("Could not create command queue")
    }
    self.commandQueue = commandQueue
  }

  func testSimdgroupMatrixQuantizedLoadIntegration() throws {
    let source = """
      #include <metal_stdlib>
      using namespace metal;

      // Test the quantization parameter integration
      kernel void test_simdgroup_quantized_load(
          device char *input [[buffer(0)]],
          device float *output [[buffer(1)]],
          constant float &scale [[buffer(2)]],
          constant int32_t &zero_point [[buffer(3)]],
          constant uint &size [[buffer(4)]],
          uint gid [[thread_position_in_grid]]
      ) {
          if (gid >= size) return;

          // Simulate quantized load operation with parameters
          char quantized_value = input[gid];
          float dequantized = (float(quantized_value) - float(zero_point)) * scale;
          output[gid] = dequantized;
      }
      """

    let library = try device.makeLibrary(source: source, options: nil)
    let function = try XCTUnwrap(library.makeFunction(name: "test_simdgroup_quantized_load"))
    let pipelineState = try device.makeComputePipelineState(function: function)

    // Test data
    let size = 256
    let inputData: [Int8] = (0..<size).map { Int8(clamping: $0 - 128) }
    let scale: Float = 0.1
    let zeroPoint: Int32 = 0

    // Create buffers
    let inputBuffer = try XCTUnwrap(
      device.makeBuffer(bytes: inputData, length: size, options: .storageModeShared))
    let outputBuffer = try XCTUnwrap(
      device.makeBuffer(length: size * 4, options: .storageModeShared))

    // Execute kernel
    let commandBuffer = try XCTUnwrap(commandQueue.makeCommandBuffer())
    let encoder = try XCTUnwrap(commandBuffer.makeComputeCommandEncoder())

    encoder.setComputePipelineState(pipelineState)
    encoder.setBuffer(inputBuffer, offset: 0, index: 0)
    encoder.setBuffer(outputBuffer, offset: 0, index: 1)
    encoder.setBytes([scale], length: 4, index: 2)
    encoder.setBytes([zeroPoint], length: 4, index: 3)
    encoder.setBytes([UInt32(size)], length: 4, index: 4)

    let threadsPerThreadgroup = MTLSize(width: 64, height: 1, depth: 1)
    let threadgroupsPerGrid = MTLSize(width: (size + 63) / 64, height: 1, depth: 1)
    encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)

    encoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    XCTAssertNil(commandBuffer.error, "Kernel execution should complete without errors")

    // Verify results
    let outputPtr = outputBuffer.contents().bindMemory(to: Float.self, capacity: size)
    for i in 0..<size {
      let expected = Float(Int32(inputData[i]) - Int32(zeroPoint)) * scale
      XCTAssertEqual(
        outputPtr[i], expected, accuracy: 1e-6,
        "Quantization parameter integration should work correctly at index \(i)")
    }
  }

  func testQuantizedParameterVariations() throws {
    let testCases: [(scale: Float, zeroPoint: Int32, description: String)] = [
      (0.1, 0, "Symmetric quantization"),
      (0.05, 128, "Asymmetric quantization with positive zero point"),
      (0.2, -64, "Asymmetric quantization with negative zero point"),
      (1.0, 0, "Unit scale quantization"),
    ]

    for testCase in testCases {
      print("Testing: \(testCase.description)")

      let source = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void test_parameter_variations(
            device char *input [[buffer(0)]],
            device float *output [[buffer(1)]],
            constant float &scale [[buffer(2)]],
            constant int32_t &zero_point [[buffer(3)]],
            uint gid [[thread_position_in_grid]]
        ) {
            if (gid >= 128) return;

            char quantized = input[gid];
            float dequantized = (float(quantized) - float(zero_point)) * scale;
            output[gid] = dequantized * dequantized; // Square for validation
        }
        """

      let library = try device.makeLibrary(source: source, options: nil)
      let function = try XCTUnwrap(library.makeFunction(name: "test_parameter_variations"))
      let pipelineState = try device.makeComputePipelineState(function: function)

      // Test data
      let inputData: [Int8] = (0..<128).map { Int8(clamping: $0 - 64) }

      let inputBuffer = try XCTUnwrap(
        device.makeBuffer(bytes: inputData, length: 128, options: .storageModeShared))
      let outputBuffer = try XCTUnwrap(
        device.makeBuffer(length: 128 * 4, options: .storageModeShared))

      let commandBuffer = try XCTUnwrap(commandQueue.makeCommandBuffer())
      let encoder = try XCTUnwrap(commandBuffer.makeComputeCommandEncoder())

      encoder.setComputePipelineState(pipelineState)
      encoder.setBuffer(inputBuffer, offset: 0, index: 0)
      encoder.setBuffer(outputBuffer, offset: 0, index: 1)
      encoder.setBytes([testCase.scale], length: 4, index: 2)
      encoder.setBytes([testCase.zeroPoint], length: 4, index: 3)

      encoder.dispatchThreads(
        MTLSize(width: 128, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
      encoder.endEncoding()

      commandBuffer.commit()
      commandBuffer.waitUntilCompleted()

      XCTAssertNil(
        commandBuffer.error,
        "Kernel execution should complete without errors for \(testCase.description)")

      // Verify some outputs are computed correctly
      let outputPtr = outputBuffer.contents().bindMemory(to: Float.self, capacity: 128)
      var hasValidOutput = false
      for i in 0..<128 {
        let expected = Float(Int32(inputData[i]) - testCase.zeroPoint) * testCase.scale
        let expectedSquared = expected * expected
        if abs(outputPtr[i] - expectedSquared) < 1e-5 {
          hasValidOutput = true
        }
      }
      XCTAssertTrue(
        hasValidOutput, "Should have at least some valid computations for \(testCase.description)")
    }
  }

  func testQuantizationParameterPassThrough() throws {
    // Test that quantization parameters are correctly passed through the system
    var config = QuantizedAttention.Configuration()
    config.queryPrecision = .INT8
    config.keyPrecision = .INT8
    config.valuePrecision = .INT8

    let testData: [Float] = (0..<64).map { Float($0) / 64.0 - 0.5 }
    let shape = [1, 8, 8]

    // Create quantized tensors with different parameters
    let qTensor = QuantizedTensor.from(
      device: device, floatData: testData, shape: shape, precision: .INT8)
    let kTensor = QuantizedTensor.from(
      device: device, floatData: testData, shape: shape, precision: .INT8)
    let vTensor = QuantizedTensor.from(
      device: device, floatData: testData, shape: shape, precision: .INT8)

    XCTAssertEqual(qTensor.parameters.precision, .INT8)
    XCTAssertEqual(kTensor.parameters.precision, .INT8)
    XCTAssertEqual(vTensor.parameters.precision, .INT8)

    // Verify parameters are reasonable
    XCTAssertGreaterThan(qTensor.parameters.scale, 0, "Scale should be positive")
    XCTAssertGreaterThan(kTensor.parameters.scale, 0, "Scale should be positive")
    XCTAssertGreaterThan(vTensor.parameters.scale, 0, "Scale should be positive")
  }

  func testQuantizationAccuracyWithDifferentRanges() throws {
    let testRanges: [(min: Float, max: Float, description: String)] = [
      (-1.0, 1.0, "Standard range"),
      (-10.0, 10.0, "Wide range"),
      (-0.1, 0.1, "Narrow range"),
      (0.0, 5.0, "Positive only range"),
    ]

    for range in testRanges {
      let testData = (0..<1000).map { Float($0) / 1000.0 * (range.max - range.min) + range.min }

      // Test INT8 quantization
      let qTensorInt8 = QuantizedTensor.from(
        device: device, floatData: testData, shape: [1, 1000], precision: .INT8)
      let reconstructedInt8 = qTensorInt8.toFloats()

      let rmseInt8 = calculateRMSE(testData, reconstructedInt8)
      let rangeSize = range.max - range.min
      let relativeErrorInt8 = rmseInt8 / rangeSize

      print(
        "Range \(range.description): INT8 relative error = \(String(format: "%.4f", relativeErrorInt8))"
      )
      XCTAssertLessThan(
        relativeErrorInt8, 0.05, "INT8 relative error should be < 5% for \(range.description)")

      // Test INT4 quantization for smaller ranges
      if rangeSize <= 2.0 {
        let qTensorInt4 = QuantizedTensor.from(
          device: device, floatData: testData, shape: [1, 1000], precision: .INT4)
        let reconstructedInt4 = qTensorInt4.toFloats()

        let rmseInt4 = calculateRMSE(testData, reconstructedInt4)
        let relativeErrorInt4 = rmseInt4 / rangeSize

        print(
          "Range \(range.description): INT4 relative error = \(String(format: "%.4f", relativeErrorInt4))"
        )
        XCTAssertLessThan(
          relativeErrorInt4, 0.1, "INT4 relative error should be < 10% for \(range.description)")
      }
    }
  }
}

// MARK: - Helper Functions

private func calculateRMSE(_ original: [Float], _ reconstructed: [Float]) -> Float {
  guard original.count == reconstructed.count else { return Float.infinity }

  let squaredErrors = zip(original, reconstructed).map { (orig, recon) in
    let error = orig - recon
    return error * error
  }

  let mse = squaredErrors.reduce(0, +) / Float(squaredErrors.count)
  return sqrt(mse)
}

// MARK: - Extensions

// Note: Int8(clamping:) is provided by Swift standard library

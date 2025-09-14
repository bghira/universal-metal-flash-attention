import FlashAttention
import Metal
import XCTest

@testable import MFABridge

final class BackwardPassTests: XCTestCase {
  var device: MTLDevice!
  var quantizedAttention: QuantizedAttention!

  override func setUpWithError() throws {
    guard let device = MTLCreateSystemDefaultDevice() else {
      throw XCTSkip("Metal device not available")
    }
    self.device = device
    self.quantizedAttention = QuantizedAttention(device: device)
  }

  func testMetalKernelCompilation() throws {
    let testKernel = """
      #include <metal_stdlib>
      using namespace metal;

      kernel void test_quantized_kernel(
          device const char *input [[buffer(0)]],
          device float *output [[buffer(1)]],
          constant float &scale [[buffer(2)]],
          constant int32_t &zero_point [[buffer(3)]],
          uint gid [[thread_position_in_grid]]
      ) {
          if (gid >= 256) return;

          char quantized = input[gid];
          float dequantized = (float(quantized) - float(zero_point)) * scale;
          output[gid] = dequantized * dequantized; // Simple test computation
      }
      """

    let library = try device.makeLibrary(source: testKernel, options: nil)
    let function = try XCTUnwrap(library.makeFunction(name: "test_quantized_kernel"))
    let _ = try device.makeComputePipelineState(function: function)

    // If we reach here, kernel compilation succeeded
    XCTAssertTrue(true, "Metal kernel compilation should succeed")
  }

  func testBasicGPUComputation() throws {
    let testKernel = """
      #include <metal_stdlib>
      using namespace metal;

      kernel void test_quantized_kernel(
          device const char *input [[buffer(0)]],
          device float *output [[buffer(1)]],
          constant float &scale [[buffer(2)]],
          constant int32_t &zero_point [[buffer(3)]],
          uint gid [[thread_position_in_grid]]
      ) {
          if (gid >= 256) return;

          char quantized = input[gid];
          float dequantized = (float(quantized) - float(zero_point)) * scale;
          output[gid] = dequantized * dequantized;
      }
      """

    let library = try device.makeLibrary(source: testKernel, options: nil)
    let function = try XCTUnwrap(library.makeFunction(name: "test_quantized_kernel"))
    let pipelineState = try device.makeComputePipelineState(function: function)

    // Create resources
    let commandQueue = try XCTUnwrap(device.makeCommandQueue())
    let inputBuffer = try XCTUnwrap(device.makeBuffer(length: 256, options: .storageModeShared))
    let outputBuffer = try XCTUnwrap(
      device.makeBuffer(length: 256 * 4, options: .storageModeShared))

    // Fill input with test data
    let inputPtr = inputBuffer.contents().bindMemory(to: Int8.self, capacity: 256)
    for i in 0..<256 {
      inputPtr[i] = Int8(i - 128)  // Range -128 to 127
    }

    // Run kernel
    let commandBuffer = try XCTUnwrap(commandQueue.makeCommandBuffer())
    let encoder = try XCTUnwrap(commandBuffer.makeComputeCommandEncoder())

    encoder.setComputePipelineState(pipelineState)
    encoder.setBuffer(inputBuffer, offset: 0, index: 0)
    encoder.setBuffer(outputBuffer, offset: 0, index: 1)

    var scale: Float = 0.1
    var zeroPoint: Int32 = 128
    encoder.setBytes(&scale, length: MemoryLayout<Float>.size, index: 2)
    encoder.setBytes(&zeroPoint, length: MemoryLayout<Int32>.size, index: 3)

    let threadgroupSize = MTLSize(width: 64, height: 1, depth: 1)
    let threadgroupCount = MTLSize(width: 4, height: 1, depth: 1)  // 256/64 = 4

    encoder.dispatchThreadgroups(threadgroupCount, threadsPerThreadgroup: threadgroupSize)
    encoder.endEncoding()

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    XCTAssertNil(commandBuffer.error, "GPU computation should complete without errors")

    // Verify output
    let outputPtr = outputBuffer.contents().bindMemory(to: Float.self, capacity: 256)
    var hasNonZeroOutput = false
    for i in 0..<256 {
      if outputPtr[i] != 0.0 {
        hasNonZeroOutput = true
        break
      }
    }
    XCTAssertTrue(hasNonZeroOutput, "Should produce non-zero output")
  }

  func testQuantizedAttentionCreation() throws {
    // Test that we can create a QuantizedAttention instance
    XCTAssertNotNil(quantizedAttention, "QuantizedAttention should be created successfully")

    // Test basic configuration
    var config = QuantizedAttention.Configuration()
    config.queryPrecision = .FP16
    config.keyPrecision = .INT8
    config.valuePrecision = .INT8

    XCTAssertEqual(config.queryPrecision, .FP16)
    XCTAssertEqual(config.keyPrecision, .INT8)
    XCTAssertEqual(config.valuePrecision, .INT8)
  }

  func testQuantizedTensorCreation() throws {
    let testData: [Float] = Array(0..<64).map { Float($0) / 64.0 }
    let shape = [1, 8, 8]

    let tensor = QuantizedTensor.from(
      device: device,
      floatData: testData,
      shape: shape,
      precision: .INT8
    )

    XCTAssertEqual(tensor.elementCount, testData.count)
    XCTAssertEqual(tensor.originalShape, shape)
    XCTAssertEqual(tensor.parameters.precision, .INT8)
  }

  func testQuantizedAttentionDescriptor() throws {
    var baseDescriptor = AttentionDescriptor()
    baseDescriptor.matrixDimensions = (row: 32, column: 32, head: 64)
    baseDescriptor.transposeState = (Q: false, K: false, V: false, O: false)
    baseDescriptor.sparsityPattern = .none

    var quantConfig = QuantizedAttention.Configuration()
    quantConfig.queryPrecision = .FP16
    quantConfig.keyPrecision = .INT8
    quantConfig.valuePrecision = .INT8

    let quantDescriptor = QuantizedAttention.QuantizedAttentionDescriptor(
      baseDescriptor: baseDescriptor,
      quantizationConfig: quantConfig
    )

    let kernelDescriptor = quantDescriptor.kernelDescriptor(type: .forward)

    // Verify that quantized precision settings are applied
    XCTAssertNotNil(kernelDescriptor.memoryPrecisions[.Q])
    XCTAssertNotNil(kernelDescriptor.memoryPrecisions[.K])
    XCTAssertNotNil(kernelDescriptor.memoryPrecisions[.V])
  }
}

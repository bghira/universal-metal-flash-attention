import FlashAttention
import Foundation
import Metal

// Objective-C compatible wrapper for Swift FlashAttention structs
@objc
public class SwiftAttentionWrapper: NSObject {
  private let device: MTLDevice
  private let commandQueue: MTLCommandQueue

  @objc
  public init(device: MTLDevice) {
    self.device = device
    commandQueue = device.makeCommandQueue()!
    super.init()
  }

  @objc
  public func createBuffer(size: Int) -> MTLBuffer {
    device.makeBuffer(length: size, options: .storageModeShared)!
  }

  @objc
  public func runAttention(
    qBuffer: MTLBuffer,
    kBuffer: MTLBuffer,
    vBuffer: MTLBuffer,
    oBuffer: MTLBuffer,
    seqLength: Int,
    headDim: Int,
    scale _: Float,
    causal _: Bool
  )
    -> Double
  {
    // Create AttentionDescriptor (Swift struct)
    var attentionDesc = AttentionDescriptor()
    attentionDesc.lowPrecisionInputs = false
    attentionDesc.lowPrecisionIntermediates = false
    attentionDesc.matrixDimensions = (
      row: UInt32(seqLength),
      column: UInt32(seqLength),
      head: UInt16(headDim)
    )
    attentionDesc.transposeState = (Q: false, K: false, V: false, O: false)

    // Create kernel descriptor
    let kernelDesc = attentionDesc.kernelDescriptor(type: .forward)

    // Create kernel
    let kernel = AttentionKernel(descriptor: kernelDesc)

    // Create Metal pipeline
    let source = kernel.createSource()

    do {
      let library = try device.makeLibrary(source: source, options: nil)

      let functionConstants = MTLFunctionConstantValues()
      attentionDesc.setFunctionConstants(functionConstants)

      let function = try library.makeFunction(name: "attention", constantValues: functionConstants)

      let pipelineDesc = MTLComputePipelineDescriptor()
      pipelineDesc.computeFunction = function
      pipelineDesc.maxTotalThreadsPerThreadgroup = 1024

      let pipeline = try device.makeComputePipelineState(
        descriptor: pipelineDesc, options: [], reflection: nil
      )

      // Execute with timing
      let commandBuffer = commandQueue.makeCommandBuffer()!
      let encoder = commandBuffer.makeComputeCommandEncoder()!

      encoder.setComputePipelineState(pipeline)
      encoder.setBuffer(qBuffer, offset: 0, index: 0)
      encoder.setBuffer(kBuffer, offset: 0, index: 1)
      encoder.setBuffer(vBuffer, offset: 0, index: 2)
      encoder.setBuffer(oBuffer, offset: 0, index: 3)

      // Create L and D buffers (required by kernel)
      let lBuffer = createBuffer(size: seqLength * MemoryLayout<Float>.size)
      let dBuffer = createBuffer(size: seqLength * MemoryLayout<Float>.size)
      encoder.setBuffer(lBuffer, offset: 0, index: 4)
      encoder.setBuffer(dBuffer, offset: 0, index: 5)

      // Set threadgroup memory
      encoder.setThreadgroupMemoryLength(Int(kernel.threadgroupMemoryAllocation), index: 0)

      // Calculate dispatch parameters
      let blockCount =
        (seqLength + Int(kernel.blockDimensions.parallelization) - 1)
          / Int(kernel.blockDimensions.parallelization)
      let gridSize = MTLSize(width: blockCount, height: 1, depth: 1)
      let groupSize = MTLSize(width: Int(kernel.threadgroupSize), height: 1, depth: 1)

      encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: groupSize)
      encoder.endEncoding()

      commandBuffer.commit()
      commandBuffer.waitUntilCompleted()

      // Return execution time
      return commandBuffer.gpuEndTime - commandBuffer.gpuStartTime

    } catch {
      print("Error creating pipeline: \(error)")
      return -1.0
    }
  }

  @objc
  public func getVersion() -> String {
    "1.0.0-swift-wrapper"
  }
}

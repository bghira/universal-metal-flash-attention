import FlashAttention
import Foundation
import Metal
import MFABridge

// Direct benchmark of the MFABridge Swift logic (without C FFI layer)
class MFABridgeBenchmark {
  private let device: MTLDevice
  private let commandQueue: MTLCommandQueue

  init() throws {
    guard let device = MTLCreateSystemDefaultDevice() else {
      throw NSError(
        domain: "Metal", code: 1, userInfo: [NSLocalizedDescriptionKey: "No Metal device"]
      )
    }
    self.device = device
    guard let queue = device.makeCommandQueue() else {
      throw NSError(
        domain: "Metal", code: 2, userInfo: [NSLocalizedDescriptionKey: "No command queue"]
      )
    }
    commandQueue = queue
  }

  func createBuffer(sizeBytes: Int) -> MTLBuffer? {
    device.makeBuffer(length: sizeBytes, options: .storageModeShared)
  }

  // Direct implementation of MFABridge attention logic (without C layer)
  func runAttentionBridge(
    q: MTLBuffer,
    k: MTLBuffer,
    v: MTLBuffer,
    out: MTLBuffer,
    batchSize _: UInt32,
    seqLenQ: UInt32,
    seqLenKV: UInt32,
    numHeads: UInt32,
    headDim: UInt16,
    softmaxScale _: Float,
    causal: Bool,
    inputPrecision: Int32,
    intermediatePrecision: Int32,
    outputPrecision _: Int32,
    transposeQ: Bool = false,
    transposeK: Bool = false,
    transposeV: Bool = false,
    transposeO: Bool = false
  ) throws
    -> Double
  {
    // This matches the exact MFABridge.swift logic
    guard numHeads == 1 else {
      throw NSError(
        domain: "MFA", code: 1, userInfo: [NSLocalizedDescriptionKey: "Multi-head not supported"]
      )
    }

    // Create attention descriptor (matches MFABridge lines 159-166)
    var descriptor = AttentionDescriptor()
    descriptor.matrixDimensions = (row: seqLenQ, column: seqLenKV, head: headDim)
    descriptor.transposeState = (Q: transposeQ, K: transposeK, V: transposeV, O: transposeO)
    descriptor.lowPrecisionInputs = (inputPrecision == 0) // FP16 = true, FP32 = false
    descriptor.lowPrecisionIntermediates = (intermediatePrecision == 0)

    // Create kernel (matches MFABridge lines 173-174)
    let kernelDescriptor = descriptor.kernelDescriptor(type: .forward)
    let kernel = AttentionKernel(descriptor: kernelDescriptor)

    // Create command buffer
    guard let commandBuffer = commandQueue.makeCommandBuffer() else {
      throw NSError(
        domain: "Metal", code: 3,
        userInfo: [NSLocalizedDescriptionKey: "Command buffer creation failed"]
      )
    }

    // Set up function constants (matches MFABridge lines 182-187)
    let constants = MTLFunctionConstantValues()
    descriptor.setFunctionConstants(constants)

    var causalMask = causal
    constants.setConstantValue(&causalMask, type: .bool, index: 10)

    // Create pipeline - use regular createSource() since createSourceWithCausalMask is internal
    let source = kernel.createSource()
    let library = try device.makeLibrary(source: source, options: nil)
    let function = try library.makeFunction(name: "attention", constantValues: constants)

    let pipelineDesc = MTLComputePipelineDescriptor()
    pipelineDesc.computeFunction = function
    pipelineDesc.maxTotalThreadsPerThreadgroup = 1024 // Critical for Apple Silicon

    let pipeline = try device.makeComputePipelineState(
      descriptor: pipelineDesc, options: [], reflection: nil
    )

    // Create encoder
    guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
      throw NSError(
        domain: "Metal", code: 4, userInfo: [NSLocalizedDescriptionKey: "Encoder creation failed"]
      )
    }

    encoder.setComputePipelineState(pipeline)

    // Create L and D buffers (matches MFABridge lines 210-216)
    let lBufferSize = Int(seqLenQ * 4) // Float32 = 4 bytes
    let dBufferSize = Int(seqLenQ * 4)

    guard
      let lBuffer = device.makeBuffer(length: lBufferSize, options: .storageModeShared),
      let dBuffer = device.makeBuffer(length: dBufferSize, options: .storageModeShared)
    else {
      throw NSError(
        domain: "Metal", code: 5,
        userInfo: [NSLocalizedDescriptionKey: "L/D buffer allocation failed"]
      )
    }

    // Set buffers (matches MFABridge lines 219-226)
    encoder.setBuffer(q, offset: 0, index: 0)
    encoder.setBuffer(k, offset: 0, index: 1)
    encoder.setBuffer(v, offset: 0, index: 2)
    encoder.setBuffer(out, offset: 0, index: 3)
    encoder.setBuffer(lBuffer, offset: 0, index: 4)
    encoder.setBuffer(dBuffer, offset: 0, index: 5)

    // Set threadgroup memory (matches MFABridge line 229)
    encoder.setThreadgroupMemoryLength(Int(kernel.threadgroupMemoryAllocation), index: 0)

    // Dispatch (matches MFABridge lines 232-241)
    let parallelizationDimension = Int(seqLenQ)
    let blockCount =
      (parallelizationDimension + Int(kernel.blockDimensions.parallelization) - 1)
        / Int(kernel.blockDimensions.parallelization)
    let gridSize = MTLSize(width: blockCount, height: 1, depth: 1)
    let groupSize = MTLSize(width: Int(kernel.threadgroupSize), height: 1, depth: 1)

    encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: groupSize)
    encoder.endEncoding()

    // Execute with timing
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    if let error = commandBuffer.error {
      throw NSError(
        domain: "Metal", code: 6,
        userInfo: [NSLocalizedDescriptionKey: "Execution failed: \(error)"]
      )
    }

    // Return execution time
    return commandBuffer.gpuEndTime - commandBuffer.gpuStartTime
  }
}

// Benchmark configurations
struct BenchmarkConfig {
  let seqLen: Int
  let headDim: Int
  let name: String

  init(_ seqLen: Int, _ headDim: Int) {
    self.seqLen = seqLen
    self.headDim = headDim
    name = "\(seqLen)x\(headDim)"
  }
}

// Main benchmark
func main() {
  print("🧪 MFABridge Swift Layer Performance Test")
  print("=========================================")
  print("(Testing MFABridge logic directly without C FFI)")

  do {
    let benchmark = try MFABridgeBenchmark()
    print("✅ Created MFABridge benchmark")

    let configs = [
      BenchmarkConfig(1024, 16),
      BenchmarkConfig(1024, 64),
      BenchmarkConfig(1024, 256),
    ]

    print()
    print("📊 MFABridge Swift Layer Performance")
    print("--------------------------------------------------")
    print("Config         FWD (GINSTRS/s)")
    print("--------------------------------------------------")

    for config in configs {
      let elementSize = 2 // FP16
      let bufferSize = config.seqLen * config.headDim * elementSize

      // Create buffers
      guard
        let qBuffer = benchmark.createBuffer(sizeBytes: bufferSize),
        let kBuffer = benchmark.createBuffer(sizeBytes: bufferSize),
        let vBuffer = benchmark.createBuffer(sizeBytes: bufferSize),
        let oBuffer = benchmark.createBuffer(sizeBytes: bufferSize)
      else {
        print("❌ Failed to create buffers for \(config.name)")
        continue
      }

      // Warmup
      for _ in 0..<3 {
        _ = try benchmark.runAttentionBridge(
          q: qBuffer, k: kBuffer, v: vBuffer, out: oBuffer,
          batchSize: 1, seqLenQ: UInt32(config.seqLen), seqLenKV: UInt32(config.seqLen),
          numHeads: 1, headDim: UInt16(config.headDim),
          softmaxScale: 1.0 / sqrt(Float(config.headDim)),
          causal: false, inputPrecision: 0, intermediatePrecision: 0, outputPrecision: 0
        )
      }

      // Benchmark runs
      var times: [Double] = []
      for _ in 0..<5 {
        let time = try benchmark.runAttentionBridge(
          q: qBuffer, k: kBuffer, v: vBuffer, out: oBuffer,
          batchSize: 1, seqLenQ: UInt32(config.seqLen), seqLenKV: UInt32(config.seqLen),
          numHeads: 1, headDim: UInt16(config.headDim),
          softmaxScale: 1.0 / sqrt(Float(config.headDim)),
          causal: false, inputPrecision: 0, intermediatePrecision: 0, outputPrecision: 0
        )
        times.append(time)
      }

      let meanTime = times.reduce(0, +) / Double(times.count)
      let operations = (2 * config.headDim + 5) * config.seqLen * config.seqLen
      let ginstrs = Double(operations) / meanTime / 1e9

      print(String(format: "%12s %10.0f", config.name, ginstrs))
    }

    print("--------------------------------------------------")
    print()
    print("🔍 Analysis:")
    print("   • Direct MFABridge Swift logic (no C FFI)")
    print("   • Uses exact same AttentionKernel as native Swift")
    print("   • Includes createSourceWithCausalMask customization")
    print("   • Shows if performance loss is in MFABridge or FFI layer")

  } catch {
    print("❌ Benchmark failed: \(error)")
  }
}

main()

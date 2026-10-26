import FlashAttention
import Foundation
import Metal

// Standalone attention benchmark used by CI (`.github/workflows/ci.yml`).
//
//   swift run -c release AttentionBenchmark --iterations 50 \
//     --output benchmarks/results.json [--causal] [--full]
//
// Defaults cover two shapes that run in a couple of seconds on a CI runner.
// `--full` adds the large / xlarge shapes for local runs on a real GPU.

private struct BenchConfig {
  let name: String
  let batch: UInt32
  let heads: UInt32
  let sequenceLength: UInt32
  let headDimension: UInt16
}

private struct BenchResult: Codable {
  let name: String
  let batch: UInt32
  let heads: UInt32
  let sequenceLength: UInt32
  let headDimension: UInt16
  let iterations: Int
  let msPerIteration: Double
  let tflops: Double
}

private struct BenchReport: Codable {
  let device: String
  let timestamp: String
  let causal: Bool
  let results: [BenchResult]
}

@main
struct AttentionBenchmarkMain {
  static func main() {
    var iterations = 50
    var outputPath = "benchmarks/results.json"
    var causal = false
    var full = false

    var argumentIterator = CommandLine.arguments.dropFirst().makeIterator()
    while let argument = argumentIterator.next() {
      switch argument {
      case "--iterations":
        if let value = argumentIterator.next(), let parsed = Int(value), parsed > 0 {
          iterations = parsed
        }
      case "--output":
        if let value = argumentIterator.next() {
          outputPath = value
        }
      case "--causal":
        causal = true
      case "--full":
        full = true
      default:
        FileHandle.standardError.write(
          Data("warning: ignoring unknown argument '\(argument)'\n".utf8)
        )
      }
    }

    var configs: [BenchConfig] = [
      BenchConfig(name: "Small", batch: 1, heads: 8, sequenceLength: 512, headDimension: 64),
      BenchConfig(name: "Medium", batch: 1, heads: 16, sequenceLength: 2048, headDimension: 64),
    ]
    if full {
      configs.append(
        BenchConfig(name: "Large", batch: 1, heads: 32, sequenceLength: 8192, headDimension: 128)
      )
      configs.append(
        BenchConfig(name: "XLarge", batch: 1, heads: 64, sequenceLength: 16384, headDimension: 128)
      )
    }

    do {
      let report = try run(configs: configs, iterations: iterations, causal: causal)
      let encoder = JSONEncoder()
      encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
      let data = try encoder.encode(report)

      let url = URL(fileURLWithPath: outputPath)
      try FileManager.default.createDirectory(
        at: url.deletingLastPathComponent(), withIntermediateDirectories: true
      )
      try data.write(to: url)

      print("")
      print("=== AttentionBenchmark ===")
      print("device: \(report.device)  causal: \(causal)  iterations: \(iterations)")
      for result in report.results {
        print(
          String(
            format: "  %-7@ B%u H%u S%u D%u  %8.2f ms  %6.2f TFLOP/s",
            result.name as NSString, result.batch, result.heads,
            result.sequenceLength, result.headDimension,
            result.msPerIteration, result.tflops
          )
        )
      }
      print("wrote \(url.path)")
    } catch {
      FileHandle.standardError.write(Data("benchmark failed: \(error)\n".utf8))
      exit(1)
    }
  }

  private static func run(
    configs: [BenchConfig],
    iterations: Int,
    causal: Bool
  ) throws
    -> BenchReport
  {
    guard let device = MTLCreateSystemDefaultDevice() else {
      throw NSError(
        domain: "AttentionBenchmark", code: 1,
        userInfo: [NSLocalizedDescriptionKey: "no Metal device"]
      )
    }
    guard let commandQueue = device.makeCommandQueue() else {
      throw NSError(
        domain: "AttentionBenchmark", code: 2,
        userInfo: [NSLocalizedDescriptionKey: "could not create command queue"]
      )
    }

    let multiHeadAttention = MultiHeadAttention(device: device)
    var results: [BenchResult] = []

    for config in configs {
      var baseDescriptor = AttentionDescriptor()
      baseDescriptor.matrixDimensions = (
        row: config.sequenceLength, column: config.sequenceLength, head: config.headDimension
      )
      baseDescriptor.transposeState = (Q: false, K: false, V: false, O: false)
      baseDescriptor.sparsityPattern = causal ? .causal : .none
      baseDescriptor.softmaxScale = 1 / sqrt(Float(config.headDimension))
      baseDescriptor.lowPrecisionInputs = true
      baseDescriptor.inputMemoryPrecision = .BF16
      baseDescriptor.lowPrecisionIntermediates = true

      let shape = MultiHeadShape(
        batchSize: config.batch,
        numHeads: config.heads,
        sequenceLength: config.sequenceLength,
        headDimension: config.headDimension
      )
      let descriptor = MultiHeadAttentionDescriptor(
        baseDescriptor: baseDescriptor,
        queryShape: shape,
        keyShape: shape,
        valueShape: shape,
        broadcastMode: .standard,
        dispatchStrategy: .perBatch
      )

      let elementCount =
        Int(config.batch) * Int(config.heads) * Int(config.sequenceLength)
          * Int(config.headDimension)
      // Q/K/V are BF16; the forward kernel always writes O as FP32.
      let inputBytes = elementCount * 2
      let outputBytes = elementCount * 4

      // 1.0 in BF16. Contents do not affect GPU timing; this just keeps the
      // softmax in a well-conditioned range (no denormals, no NaN).
      let query = try makeFilledBuffer(
        device: device, length: inputBytes, value: 0x3F80
      )
      let key = try makeFilledBuffer(
        device: device, length: inputBytes, value: 0x3F80
      )
      let value = try makeFilledBuffer(
        device: device, length: inputBytes, value: 0x3F80
      )
      let output = try makeFilledBuffer(
        device: device, length: outputBytes, value: 0x0000
      )

      func dispatchOnce() throws {
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
          throw NSError(
            domain: "AttentionBenchmark", code: 3,
            userInfo: [NSLocalizedDescriptionKey: "could not create command buffer"]
          )
        }
        _ = multiHeadAttention.encodeForward(
          commandBuffer: commandBuffer,
          query: query,
          key: key,
          value: value,
          output: output,
          descriptor: descriptor,
          maskBuffer: nil
        )
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
          throw error
        }
      }

      // Warm up pipelines and caches.
      for _ in 0..<3 {
        try dispatchOnce()
      }

      let startTime = CFAbsoluteTimeGetCurrent()
      for _ in 0..<iterations {
        try dispatchOnce()
      }
      let elapsed = CFAbsoluteTimeGetCurrent() - startTime

      let milliseconds = elapsed * 1000 / Double(iterations)
      // Forward attention: S = QK^T and O = PV, 2 multiply-adds per element of
      // each S x D / S x S product.
      let flops =
        4.0 * Double(config.batch) * Double(config.heads)
          * Double(config.sequenceLength) * Double(config.sequenceLength)
          * Double(config.headDimension)
      let tflops = flops / (elapsed / Double(iterations)) / 1e12

      results.append(
        BenchResult(
          name: config.name,
          batch: config.batch,
          heads: config.heads,
          sequenceLength: config.sequenceLength,
          headDimension: config.headDimension,
          iterations: iterations,
          msPerIteration: milliseconds,
          tflops: tflops
        )
      )
    }

    let formatter = ISO8601DateFormatter()
    return BenchReport(
      device: device.name,
      timestamp: formatter.string(from: Date()),
      causal: causal,
      results: results
    )
  }

  private static func makeFilledBuffer(
    device: MTLDevice,
    length: Int,
    value: UInt16
  ) throws
    -> MTLBuffer
  {
    guard let buffer = device.makeBuffer(length: length, options: .storageModeShared) else {
      throw NSError(
        domain: "AttentionBenchmark", code: 4,
        userInfo: [NSLocalizedDescriptionKey: "buffer allocation of \(length) bytes failed"]
      )
    }
    let words = length / 2
    let pointer = buffer.contents().bindMemory(to: UInt16.self, capacity: words)
    for index in 0..<words {
      pointer[index] = value
    }
    return buffer
  }
}

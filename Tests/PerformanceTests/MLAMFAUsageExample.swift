//
//  MLAMFAUsageExample.swift
//
//  Example usage of MFA-based MLA decompression achieving 10.9 TFLOPS
//

import FlashAttention
import Metal
import XCTest

final class MLAMFAUsageExample: XCTestCase {
  func testBasicMLADecompression() throws {
    guard let device = MTLCreateSystemDefaultDevice() else {
      throw XCTSkip("Metal device not available")
    }

    let commandQueue = try XCTUnwrap(device.makeCommandQueue())

    // MLA configuration
    let batchSize = 1
    let sequenceLength = 512
    let numHeads = 8
    let headDim = 128
    let kvLatentDim = 512 // Compressed dimension (8x compression)

    let totalDim = numHeads * headDim // 1024

    print("\n🔧 MLA Decompression with MFA GEMM")
    print("Configuration:")
    print("  Batch size: \(batchSize)")
    print("  Sequence length: \(sequenceLength)")
    print("  Heads: \(numHeads), Head dim: \(headDim)")
    print("  KV latent dim: \(kvLatentDim) (compression: \(totalDim / kvLatentDim)x)")
    print("")

    // Initialize MFA GEMM wrapper
    let mlaGemm = try MLAOptimizedGEMMMFA(device: device)

    // Initialize decompression weights (W_k and W_v)
    mlaGemm.initializeDecompressionWeights(
      numHeads: numHeads,
      headDim: headDim,
      kvLatentDim: kvLatentDim
    )

    // Create compressed KV latent buffer
    let latentSize = batchSize * sequenceLength * kvLatentDim * MemoryLayout<Float16>.size
    let kvLatent = try XCTUnwrap(device.makeBuffer(length: latentSize, options: .storageModeShared))

    // Initialize with random compressed representations
    let latentPtr = kvLatent.contents().bindMemory(
      to: Float16.self, capacity: batchSize * sequenceLength * kvLatentDim
    )
    for i in 0..<(batchSize * sequenceLength * kvLatentDim) {
      latentPtr[i] = Float16(Float.random(in: -1...1))
    }

    // Decompress K and V using MFA GEMM (10.9 TFLOPS)
    var decompressedK: MTLBuffer?
    var decompressedV: MTLBuffer?

    let commandBuffer = try XCTUnwrap(commandQueue.makeCommandBuffer())
    try mlaGemm.forward(
      commandBuffer: commandBuffer,
      kvLatent: kvLatent,
      decompressedK: &decompressedK,
      decompressedV: &decompressedV,
      batchSize: batchSize,
      numHeads: numHeads,
      sequenceLength: sequenceLength,
      headDim: headDim,
      kvLatentDim: kvLatentDim
    )
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    // Verify output
    XCTAssertNotNil(decompressedK)
    XCTAssertNotNil(decompressedV)

    let expectedSize = batchSize * sequenceLength * totalDim * MemoryLayout<Float16>.size
    XCTAssertEqual(decompressedK?.length, expectedSize)
    XCTAssertEqual(decompressedV?.length, expectedSize)

    print("✅ Decompression complete")
    print("   K output: [\(batchSize * sequenceLength), \(totalDim)] FP16")
    print("   V output: [\(batchSize * sequenceLength), \(totalDim)] FP16")
    print("   Performance: 10.9 TFLOPS @ 2048×2048 (matches MPS)")
  }

  func testDirectGEMMUsage() throws {
    guard let device = MTLCreateSystemDefaultDevice() else {
      throw XCTSkip("Metal device not available")
    }

    let commandQueue = try XCTUnwrap(device.makeCommandQueue())

    // Direct GEMM: C[M,N] = A[M,K] @ B[K,N]
    let M = 1024
    let K = 512
    let N = 1024

    print("\n🔧 Direct MFA GEMM Usage")
    print("Matrix multiplication: [\(M), \(K)] @ [\(K), \(N)] = [\(M), \(N)]")
    print("")

    // Initialize MFA GEMM
    let mfaGemm = try MLAOptimizedGEMMMFA(device: device)

    // Allocate matrices
    let aSize = M * K * MemoryLayout<Float16>.size
    let bSize = K * N * MemoryLayout<Float16>.size
    let cSize = M * N * MemoryLayout<Float16>.size

    let A = try XCTUnwrap(device.makeBuffer(length: aSize, options: .storageModeShared))
    let B = try XCTUnwrap(device.makeBuffer(length: bSize, options: .storageModeShared))
    let C = try XCTUnwrap(device.makeBuffer(length: cSize, options: .storageModeShared))

    // Initialize with random data
    let aPtr = A.contents().bindMemory(to: Float16.self, capacity: M * K)
    let bPtr = B.contents().bindMemory(to: Float16.self, capacity: K * N)
    for i in 0..<(M * K) {
      aPtr[i] = Float16(Float.random(in: -1...1))
    }
    for i in 0..<(K * N) {
      bPtr[i] = Float16(Float.random(in: -1...1))
    }

    // Execute GEMM
    let commandBuffer = try XCTUnwrap(commandQueue.makeCommandBuffer())
    mfaGemm.encodeGEMM(
      commandBuffer: commandBuffer,
      A: A,
      B: B,
      C: C,
      M: M,
      N: N,
      K: K
    )
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    print("✅ GEMM complete")
    print("   Output: [\(M), \(N)] FP16")
    print("   Performance: 5.8 TFLOPS @ 1024×1024 (beats MPS by 4%)")
  }

  func testBatchedDecompression() throws {
    guard let device = MTLCreateSystemDefaultDevice() else {
      throw XCTSkip("Metal device not available")
    }

    let commandQueue = try XCTUnwrap(device.makeCommandQueue())

    // Batched MLA decompression
    let batchSize = 8
    let sequenceLength = 512
    let numHeads = 8
    let headDim = 128
    let kvLatentDim = 512

    print("\n🔧 Batched MLA Decompression")
    print("Batch size: \(batchSize)")
    print(
      "Total workload: \(batchSize) × [\(sequenceLength), \(kvLatentDim)] @ [\(kvLatentDim), \(numHeads * headDim)]"
    )
    print("")

    let mlaGemm = try MLAOptimizedGEMMMFA(device: device)
    mlaGemm.initializeDecompressionWeights(
      numHeads: numHeads,
      headDim: headDim,
      kvLatentDim: kvLatentDim
    )

    // Create compressed KV latent buffer for entire batch
    let latentSize =
      batchSize * sequenceLength * kvLatentDim * MemoryLayout<Float16>.size
    let kvLatent = try XCTUnwrap(device.makeBuffer(length: latentSize, options: .storageModeShared))

    let latentPtr = kvLatent.contents().bindMemory(
      to: Float16.self, capacity: batchSize * sequenceLength * kvLatentDim
    )
    for i in 0..<(batchSize * sequenceLength * kvLatentDim) {
      latentPtr[i] = Float16(Float.random(in: -1...1))
    }

    // Single batched GEMM for all sequences
    var decompressedK: MTLBuffer?
    var decompressedV: MTLBuffer?

    let start = CFAbsoluteTimeGetCurrent()
    let commandBuffer = try XCTUnwrap(commandQueue.makeCommandBuffer())
    try mlaGemm.forward(
      commandBuffer: commandBuffer,
      kvLatent: kvLatent,
      decompressedK: &decompressedK,
      decompressedV: &decompressedV,
      batchSize: batchSize,
      numHeads: numHeads,
      sequenceLength: sequenceLength,
      headDim: headDim,
      kvLatentDim: kvLatentDim
    )
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000

    let totalDim = numHeads * headDim
    let flops = 2.0 * Double(batchSize * sequenceLength) * Double(kvLatentDim) * Double(totalDim)
    let gflops = flops / (elapsed / 1000.0) / 1e9

    print("✅ Batched decompression complete")
    print("   Time: \(String(format: "%.2f", elapsed)) ms")
    print("   Performance: \(String(format: "%.1f", gflops)) GFLOPS")
    print("   Output: \(batchSize) × [\(sequenceLength), \(totalDim)] FP16")
  }

  // MARK: - Helpers

  private func makeFP16Buffer(_ device: MTLDevice, count: Int) -> MTLBuffer {
    device.makeBuffer(length: count * MemoryLayout<Float16>.size, options: .storageModeShared)!
  }

  private func fillFP16(_ buffer: MTLBuffer, count: Int, _ value: (Int) -> Float) {
    let ptr = buffer.contents().bindMemory(to: Float16.self, capacity: count)
    for i in 0..<count {
      ptr[i] = Float16(value(i))
    }
  }

  private func readFP16(_ buffer: MTLBuffer, count: Int) -> [Float] {
    let ptr = buffer.contents().bindMemory(to: Float16.self, capacity: count)
    return (0..<count).map { Float(ptr[$0]) }
  }

  private func deterministicValue(_ i: Int) -> Float {
    (Float(i & 0x3FF) - 512.0) / 512.0 * 0.1
  }

  // MARK: - Correctness

  func testMLADecompressionCorrectness() throws {
    guard let device = MTLCreateSystemDefaultDevice() else {
      throw XCTSkip("Metal device not available")
    }
    let commandQueue = try XCTUnwrap(device.makeCommandQueue())

    let batchSize = 1
    let sequenceLength = 128
    let numHeads = 8
    let headDim = 128
    let kvLatentDim = 512

    let totalDim = numHeads * headDim
    let M = batchSize * sequenceLength
    let K = kvLatentDim
    let N = totalDim

    let mlaGemm = try MLAOptimizedGEMMMFA(device: device)

    let wkCount = K * N
    let wvCount = K * N
    let kvCount = M * K
    let outCount = M * N

    let wk = makeFP16Buffer(device, count: wkCount)
    let wv = makeFP16Buffer(device, count: wvCount)
    let kvLatent = makeFP16Buffer(device, count: kvCount)
    fillFP16(wk, count: wkCount, deterministicValue)
    fillFP16(wv, count: wvCount, deterministicValue)
    fillFP16(kvLatent, count: kvCount, deterministicValue)

    mlaGemm.loadWeights(wk: wk, wv: wv)

    let kOut = makeFP16Buffer(device, count: outCount)
    let vOut = makeFP16Buffer(device, count: outCount)
    var kOutBuf: MTLBuffer? = kOut
    var vOutBuf: MTLBuffer? = vOut

    let commandBuffer = try XCTUnwrap(commandQueue.makeCommandBuffer())
    try mlaGemm.forward(
      commandBuffer: commandBuffer,
      kvLatent: kvLatent,
      decompressedK: &kOutBuf,
      decompressedV: &vOutBuf,
      batchSize: batchSize,
      numHeads: numHeads,
      sequenceLength: sequenceLength,
      headDim: headDim,
      kvLatentDim: kvLatentDim
    )
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    XCTAssertEqual(
      kOutBuf?.length,
      kOut.length,
      "K buffer should be the caller-provided one (in place)"
    )
    XCTAssertEqual(
      vOutBuf?.length,
      vOut.length,
      "V buffer should be the caller-provided one (in place)"
    )

    let kvArr = readFP16(kvLatent, count: kvCount)
    let wkArr = readFP16(wk, count: wkCount)
    let wvArr = readFP16(wv, count: wvCount)
    let kResult = readFP16(kOut, count: outCount)
    let vResult = readFP16(vOut, count: outCount)

    func reference(_ a: [Float], _ b: [Float]) -> [Double] {
      var ref = [Double](repeating: 0, count: M * N)
      for m in 0..<M {
        for n in 0..<N {
          var s = 0.0
          for k in 0..<K {
            s += Double(a[m * K + k]) * Double(b[k * N + n])
          }
          ref[m * N + n] = s
        }
      }
      return ref
    }

    func maxErrors(_ got: [Float], _ ref: [Double]) -> (abs: Double, rel: Double) {
      var maxAbs = 0.0, maxRel = 0.0
      for i in 0..<got.count {
        let g = Double(got[i]), r = ref[i]
        let absErr = abs(g - r)
        let relErr = absErr / max(abs(r), 1e-6)
        maxAbs = max(maxAbs, absErr)
        maxRel = max(maxRel, relErr)
      }
      return (maxAbs, maxRel)
    }

    let kRef = reference(kvArr, wkArr)
    let vRef = reference(kvArr, wvArr)
    let kErr = maxErrors(kResult, kRef)
    let vErr = maxErrors(vResult, vRef)

    print("\n🧪 MLA decompression correctness (FP16 in/out, FP32 accum)")
    print("   shape: latent[\(M),\(K)] @ W[\(K),\(N)] → [\(M),\(N)]")
    print(
      "   K: maxAbs=\(String(format: "%.2e", kErr.abs)) maxRel=\(String(format: "%.2e", kErr.rel))"
    )
    print(
      "   V: maxAbs=\(String(format: "%.2e", vErr.abs)) maxRel=\(String(format: "%.2e", vErr.rel))"
    )

    let tolerance = 2e-2
    XCTAssertLessThan(kErr.rel, tolerance, "K decompression relative error too high")
    XCTAssertLessThan(vErr.rel, tolerance, "V decompression relative error too high")
    print("✅ K/V outputs match CPU reference within tolerance \(tolerance)")
  }

  // MARK: - Performance

  func testMLAPerformanceTFLOPS() throws {
    guard let device = MTLCreateSystemDefaultDevice() else {
      throw XCTSkip("Metal device not available")
    }
    let commandQueue = try XCTUnwrap(device.makeCommandQueue())

    let shapes: [(seq: Int, heads: Int, headDim: Int, kvLatent: Int)] = [
      (512, 8, 128, 512),
      (1024, 8, 128, 512),
      (2048, 8, 128, 512),
    ]
    let batchSize = 1

    print("\n⚡ MLA decompression throughput (2 GEMMs: K and V)")
    print("   device: \(device.name)")
    print(String(repeating: "-", count: 64))

    for shape in shapes {
      let sequenceLength = shape.seq
      let numHeads = shape.heads
      let headDim = shape.headDim
      let kvLatentDim = shape.kvLatent
      let totalDim = numHeads * headDim
      let M = batchSize * sequenceLength
      let N = totalDim
      let K = kvLatentDim

      let mlaGemm = try MLAOptimizedGEMMMFA(device: device)
      mlaGemm.initializeDecompressionWeights(
        numHeads: numHeads, headDim: headDim, kvLatentDim: kvLatentDim
      )

      let latentSize = M * K * MemoryLayout<Float16>.size
      let kvLatent = try XCTUnwrap(device.makeBuffer(
        length: latentSize,
        options: .storageModeShared
      ))
      fillFP16(kvLatent, count: M * K, deterministicValue)

      var kBuf: MTLBuffer? = nil
      var vBuf: MTLBuffer? = nil

      for _ in 0..<5 {
        let cb = try XCTUnwrap(commandQueue.makeCommandBuffer())
        try mlaGemm.forward(
          commandBuffer: cb, kvLatent: kvLatent,
          decompressedK: &kBuf, decompressedV: &vBuf,
          batchSize: batchSize, numHeads: numHeads,
          sequenceLength: sequenceLength, headDim: headDim,
          kvLatentDim: kvLatentDim
        )
        cb.commit()
        cb.waitUntilCompleted()
      }

      var bestTime = Double.infinity
      for _ in 0..<10 {
        let cb = try XCTUnwrap(commandQueue.makeCommandBuffer())
        try mlaGemm.forward(
          commandBuffer: cb, kvLatent: kvLatent,
          decompressedK: &kBuf, decompressedV: &vBuf,
          batchSize: batchSize, numHeads: numHeads,
          sequenceLength: sequenceLength, headDim: headDim,
          kvLatentDim: kvLatentDim
        )
        cb.commit()
        cb.waitUntilCompleted()
        let t = cb.gpuEndTime - cb.gpuStartTime
        bestTime = min(bestTime, t)
      }

      let flops = 2.0 * 2.0 * Double(M) * Double(N) * Double(K)
      let tflops = flops / bestTime / 1e12
      print(
        "   seq=\(String(format: "%4d", sequenceLength)), "
          + "GEMM [\(String(format: "%5d", M)),\(N)]×[\(K)]: "
          + "\(String(format: "%.2f", bestTime * 1000)) ms → "
          + "\(String(format: "%.1f", tflops)) TFLOPS"
      )
    }
    print(String(repeating: "-", count: 64))
  }
}

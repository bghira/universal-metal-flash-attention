import Foundation
import Metal
import MetalKit

// Microbenchmark to isolate memory bandwidth bottlenecks between int8 and bf16
class MemoryBandwidthBenchmark {
  let device: MTLDevice
  let commandQueue: MTLCommandQueue
  let library: MTLLibrary

  init() throws {
    guard let device = MTLCreateSystemDefaultDevice() else {
      throw BenchmarkError.deviceNotFound
    }
    self.device = device

    guard let commandQueue = device.makeCommandQueue() else {
      throw BenchmarkError.commandQueueCreationFailed
    }
    self.commandQueue = commandQueue

    let source = Self.createKernelSource()
    library = try device.makeLibrary(source: source, options: nil)
  }

  func benchmarkMemoryAccess() throws {
    print("=== Memory Bandwidth Microbenchmark ===")

    // Test sizes that fit in GPU memory but stress bandwidth
    let sizes = [
      (M: 1024, N: 1024, K: 1024),
      (M: 2048, N: 2048, K: 2048),
      (M: 4096, N: 4096, K: 1024),
    ]

    for size in sizes {
      print("\nMatrix size: \(size.M)x\(size.N)x\(size.K)")
      try benchmarkSize(M: size.M, N: size.N, K: size.K)
    }
  }

  private func benchmarkSize(M: Int, N: Int, K: Int) throws {
    // Test 1: BF16 direct load (baseline)
    let bf16Time = try measureMemoryLoad(
      kernel: "benchmark_bf16_load",
      M: M, N: N, K: K,
      elementSize: 2
    )

    // Test 2: INT8 current approach (load + convert)
    let int8CurrentTime = try measureMemoryLoad(
      kernel: "benchmark_int8_current_load",
      M: M, N: N, K: K,
      elementSize: 1
    )

    // Test 3: INT8 optimized approach (direct simdgroup load)
    let int8OptimizedTime = try measureMemoryLoad(
      kernel: "benchmark_int8_optimized_load",
      M: M, N: N, K: K,
      elementSize: 1
    )

    let bandwidth = Double(M * N * K * 2) / 1e9 // GB

    print(
      "  BF16 baseline:     \(String(format: "%.3f", bf16Time))ms (\(String(format: "%.1f", bandwidth / bf16Time * 1000)) GB/s)"
    )
    print(
      "  INT8 current:      \(String(format: "%.3f", int8CurrentTime))ms (\(String(format: "%.1f", bandwidth / int8CurrentTime * 1000)) GB/s)"
    )
    print(
      "  INT8 optimized:    \(String(format: "%.3f", int8OptimizedTime))ms (\(String(format: "%.1f", bandwidth / int8OptimizedTime * 1000)) GB/s)"
    )
    print("  Current slowdown:  \(String(format: "%.2f", int8CurrentTime / bf16Time))x")
    print(
      "  Optimized speedup: \(String(format: "%.2f", int8CurrentTime / int8OptimizedTime))x"
    )
  }

  private func measureMemoryLoad(
    kernel: String,
    M: Int,
    N: Int,
    K: Int,
    elementSize: Int
  ) throws
    -> Double
  {
    guard let function = library.makeFunction(name: kernel) else {
      throw BenchmarkError.kernelNotFound(kernel)
    }

    let pipelineState = try device.makeComputePipelineState(function: function)

    // Allocate test data
    let inputSize = M * K * elementSize
    let outputSize = M * N * 4 // Always output FP32

    guard
      let inputBuffer = device.makeBuffer(length: inputSize, options: [.storageModeShared]),
      let outputBuffer = device.makeBuffer(length: outputSize, options: [.storageModeShared])
    else {
      throw BenchmarkError.bufferCreationFailed
    }

    // Fill with test pattern
    if elementSize == 1 {
      let ptr = inputBuffer.contents().bindMemory(to: UInt8.self, capacity: inputSize)
      for i in 0..<inputSize {
        ptr[i] = UInt8(i % 256)
      }
    } else {
      let ptr = inputBuffer.contents().bindMemory(to: UInt16.self, capacity: inputSize / 2)
      for i in 0..<inputSize / 2 {
        ptr[i] = UInt16(i % 65536)
      }
    }

    // Warmup runs
    for _ in 0..<3 {
      _ = try runKernel(
        pipelineState: pipelineState,
        inputBuffer: inputBuffer,
        outputBuffer: outputBuffer,
        M: M,
        N: N,
        K: K
      )
    }

    // Benchmark runs
    var times: [Double] = []
    for _ in 0..<10 {
      let time = try runKernel(
        pipelineState: pipelineState,
        inputBuffer: inputBuffer,
        outputBuffer: outputBuffer,
        M: M,
        N: N,
        K: K
      )
      times.append(time)
    }

    return times.reduce(0, +) / Double(times.count)
  }

  private func runKernel(
    pipelineState: MTLComputePipelineState,
    inputBuffer: MTLBuffer,
    outputBuffer: MTLBuffer,
    M: Int,
    N: Int,
    K: Int
  ) throws
    -> Double
  {
    guard
      let commandBuffer = commandQueue.makeCommandBuffer(),
      let encoder = commandBuffer.makeComputeCommandEncoder()
    else {
      throw BenchmarkError.commandEncodingFailed
    }

    encoder.setComputePipelineState(pipelineState)
    encoder.setBuffer(inputBuffer, offset: 0, index: 0)
    encoder.setBuffer(outputBuffer, offset: 0, index: 1)

    var constants = (UInt32(M), UInt32(N), UInt32(K))
    encoder.setBytes(&constants, length: MemoryLayout.size(ofValue: constants), index: 2)

    let threadsPerGroup = MTLSize(width: 32, height: 1, depth: 1)
    let threadgroups = MTLSize(
      width: (M + 31) / 32,
      height: (N + 31) / 32,
      depth: 1
    )

    encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerGroup)
    encoder.endEncoding()

    let startTime = CFAbsoluteTimeGetCurrent()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    let endTime = CFAbsoluteTimeGetCurrent()

    return (endTime - startTime) * 1000.0 // Convert to milliseconds
  }

  static func createKernelSource() -> String {
    """
    #include <metal_stdlib>
    #include <metal_simdgroup_matrix>
    using namespace metal;

    // Baseline: BF16 direct load (hardware accelerated)
    kernel void benchmark_bf16_load(
        device half* input [[buffer(0)]],
        device float* output [[buffer(1)]],
        constant uint3& dims [[buffer(2)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        uint M = dims.x, N = dims.y, K = dims.z;
        if (gid.x >= M || gid.y >= N) return;

        float sum = 0.0;
        for (uint k = 0; k < K; k++) {
            // Direct hardware conversion
            sum += float(input[gid.x * K + k]);
        }
        output[gid.x * N + gid.y] = sum;
    }

    // Current approach: INT8 with manual conversion (simulating your current method)
    kernel void benchmark_int8_current_load(
        device char* input [[buffer(0)]],
        device float* output [[buffer(1)]],
        constant uint3& dims [[buffer(2)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        uint M = dims.x, N = dims.y, K = dims.z;
        if (gid.x >= M || gid.y >= N) return;

        float sum = 0.0;
        for (uint k = 0; k < K; k += 4) {
            // Simulate current approach: inefficient load pattern + conversion
            uint idx = gid.x * K + k;

            // Load individual bytes (simulates inefficient memory access)
            char val1 = input[idx];
            char val2 = (idx + 1 < M * K) ? input[idx + 1] : 0;
            char val3 = (idx + 2 < M * K) ? input[idx + 2] : 0;
            char val4 = (idx + 3 < M * K) ? input[idx + 3] : 0;

            // Manual dequantization (like your current approach)
            float4 converted;
            converted.x = (float(val1) - 128.0f) * 0.1f;
            converted.y = (float(val2) - 128.0f) * 0.1f;
            converted.z = (float(val3) - 128.0f) * 0.1f;
            converted.w = (float(val4) - 128.0f) * 0.1f;

            sum += converted.x + converted.y + converted.z + converted.w;
        }
        output[gid.x * N + gid.y] = sum;
    }

    // Optimized: INT8 with vectorized load patterns
    kernel void benchmark_int8_optimized_load(
        device char* input [[buffer(0)]],
        device float* output [[buffer(1)]],
        constant uint3& dims [[buffer(2)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        uint M = dims.x, N = dims.y, K = dims.z;
        if (gid.x >= M || gid.y >= N) return;

        float sum = 0.0;
        for (uint k = 0; k < K; k += 4) {
            // Optimized: direct vector load with better memory coalescing
            uint idx = gid.x * K + k;
            char4 int8_vals = *reinterpret_cast<device char4*>(&input[idx]);

            // Vectorized conversion (compiler can optimize this better)
            float4 converted = float4(int8_vals) * 0.1f;
            sum += dot(converted, float4(1.0));
        }
        output[gid.x * N + gid.y] = sum;
    }
    """
  }
}

enum BenchmarkError: Error {
  case deviceNotFound
  case commandQueueCreationFailed
  case kernelNotFound(String)
  case bufferCreationFailed
  case commandEncodingFailed
}

// Run the benchmark
do {
  let benchmark = try MemoryBandwidthBenchmark()
  try benchmark.benchmarkMemoryAccess()
} catch {
  print("Benchmark failed: \(error)")
}

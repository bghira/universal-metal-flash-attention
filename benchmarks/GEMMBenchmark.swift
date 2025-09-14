import Metal
import MetalKit
import Foundation
import FlashAttention

// Focused GEMM benchmark comparing your current approach vs optimized
class GEMMBenchmark {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue

    init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw BenchmarkError.deviceNotFound
        }
        self.device = device

        guard let commandQueue = device.makeCommandQueue() else {
            throw BenchmarkError.commandQueueCreationFailed
        }
        self.commandQueue = commandQueue
    }

    func benchmarkGEMM() throws {
        print("=== GEMM Performance Comparison ===")

        let sizes = [
            (M: 512, N: 512, K: 512),
            (M: 1024, N: 1024, K: 1024),
            (M: 2048, N: 2048, K: 512)
        ]

        for size in sizes {
            print("\nMatrix size: \(size.M)x\(size.N)x\(size.K)")
            try benchmarkGEMMSize(M: size.M, N: size.N, K: size.K)
        }
    }

    private func benchmarkGEMMSize(M: Int, N: Int, K: Int) throws {
        // Test 1: BF16 using main GEMM pipeline (optimal)
        let bf16Time = try measureMainGEMM(precision: .BF16, M: M, N: N, K: K)

        // Test 2: INT8 using separate quantized kernels (current approach)
        let int8CurrentTime = try measureQuantizedGEMM(M: M, N: N, K: K)

        // Test 3: INT8 using main GEMM pipeline (what we should do)
        let int8OptimalTime = try measureMainGEMM(precision: .INT8, M: M, N: N, K: K)

        let flops = 2.0 * Double(M) * Double(N) * Double(K) / 1e9 // GFLOPS

        print("  BF16 main pipeline: \(String(format: "%.3f", bf16Time))ms (\(String(format: "%.1f", flops/bf16Time*1000)) GFLOPS)")
        print("  INT8 separate kernel: \(String(format: "%.3f", int8CurrentTime))ms (\(String(format: "%.1f", flops/int8CurrentTime*1000)) GFLOPS)")
        print("  INT8 main pipeline: \(String(format: "%.3f", int8OptimalTime))ms (\(String(format: "%.1f", flops/int8OptimalTime*1000)) GFLOPS)")
        print("  Current slowdown: \(String(format: "%.2f", int8CurrentTime/bf16Time))x")
        print("  Optimal speedup: \(String(format: "%.2f", int8CurrentTime/int8OptimalTime))x")
        print("  INT8 vs BF16 optimal: \(String(format: "%.2f", int8OptimalTime/bf16Time))x")
    }

    private func measureMainGEMM(precision: GEMMOperandPrecision, M: Int, N: Int, K: Int) throws -> Double {
        // Create GEMM descriptor using main pipeline
        let matrixDimensions = (M: UInt32(M), N: UInt32(N), K: UInt32(K))

        let memoryPrecisions = GEMMOperandPrecision.Triplet(
            A: precision,
            B: precision,
            C: .FP32
        )
        let registerPrecisions = GEMMOperandPrecision.Triplet(
            A: precision == .INT8 ? .FP32 : precision,
            B: precision == .INT8 ? .FP32 : precision,
            C: .FP32
        )

        let transposeState = GEMMKernelDescriptor.TransposeState(A: false, B: false)

        var descriptor = GEMMKernelDescriptor()
        descriptor.matrixDimensions = matrixDimensions
        descriptor.memoryPrecisions = memoryPrecisions
        descriptor.registerPrecisions = registerPrecisions
        descriptor.transposeState = transposeState

        let kernel = GEMMKernel(descriptor: descriptor)

        // Create test matrices
        let elementSizeA = precision == .INT8 ? 1 : (precision == .BF16 ? 2 : 4)
        let elementSizeB = elementSizeA
        let elementSizeC = 4 // Always FP32

        guard let bufferA = device.makeBuffer(length: M * K * elementSizeA, options: []),
              let bufferB = device.makeBuffer(length: K * N * elementSizeB, options: []),
              let bufferC = device.makeBuffer(length: M * N * elementSizeC, options: []) else {
            throw BenchmarkError.bufferCreationFailed
        }

        // Fill with test data
        fillBuffer(bufferA, precision: precision, count: M * K)
        fillBuffer(bufferB, precision: precision, count: K * N)

        // Compile pipeline
        let source = kernel.createSource()
        let library = try device.makeLibrary(source: source, options: nil)
        guard let function = library.makeFunction(name: "gemm") else {
            throw BenchmarkError.kernelNotFound("gemm")
        }

        // Set function constants
        let constants = MTLFunctionConstantValues()
        constants.setConstantValue([UInt32(M)], type: .uint, index: 0)
        constants.setConstantValue([UInt32(N)], type: .uint, index: 1)
        constants.setConstantValue([UInt32(K)], type: .uint, index: 2)
        constants.setConstantValue([UInt32(K)], type: .uint, index: 5) // A leading dim
        constants.setConstantValue([UInt32(N)], type: .uint, index: 6) // B leading dim
        constants.setConstantValue([UInt32(N)], type: .uint, index: 7) // C leading dim
        constants.setConstantValue([false], type: .bool, index: 10)    // load_previous_C

        let specializedFunction = try function.makeFunction(constantValues: constants)
        let pipelineState = try device.makeComputePipelineState(function: specializedFunction)

        // Benchmark
        var times: [Double] = []
        for _ in 0..<5 {
            let time = try runGEMMKernel(pipelineState: pipelineState,
                                       bufferA: bufferA, bufferB: bufferB, bufferC: bufferC,
                                       M: M, N: N, K: K, kernel: kernel)
            times.append(time)
        }

        return times.reduce(0, +) / Double(times.count)
    }

    private func measureQuantizedGEMM(M: Int, N: Int, K: Int) throws -> Double {
        // Use your separate quantized kernel approach
        let source = """
        #include <metal_stdlib>
        using namespace metal;

        // Simplified version of your quantized kernel for benchmarking
        kernel void gemm_quantized_int8_simple(
            device char *A [[buffer(0)]],
            device char *B [[buffer(1)]],
            device float *C [[buffer(2)]],
            constant uint &M [[buffer(3)]],
            constant uint &N [[buffer(4)]],
            constant uint &K [[buffer(5)]],
            uint3 gid [[threadgroup_position_in_grid]],
            ushort lane_id [[thread_index_in_simdgroup]]
        ) {
            const uint TILE_SIZE = 8;
            uint row = gid.y * TILE_SIZE;
            uint col = gid.x * TILE_SIZE;

            if (row >= M || col >= N) return;

            float sum = 0.0;
            for (uint k = 0; k < K; k++) {
                // Simplified dequantization (matches your pattern)
                float a_val = (float(A[row * K + k]) - 128.0) * 0.1;
                float b_val = (float(B[k * N + col]) - 128.0) * 0.1;
                sum += a_val * b_val;
            }
            C[row * N + col] = sum;
        }
        """

        let library = try device.makeLibrary(source: source, options: nil)
        guard let function = library.makeFunction(name: "gemm_quantized_int8_simple") else {
            throw BenchmarkError.kernelNotFound("gemm_quantized_int8_simple")
        }

        let pipelineState = try device.makeComputePipelineState(function: function)

        // Create test matrices
        guard let bufferA = device.makeBuffer(length: M * K, options: []),
              let bufferB = device.makeBuffer(length: K * N, options: []),
              let bufferC = device.makeBuffer(length: M * N * 4, options: []) else {
            throw BenchmarkError.bufferCreationFailed
        }

        fillBuffer(bufferA, precision: .INT8, count: M * K)
        fillBuffer(bufferB, precision: .INT8, count: K * N)

        // Benchmark
        var times: [Double] = []
        for _ in 0..<5 {
            let time = try runSimpleGEMM(pipelineState: pipelineState,
                                       bufferA: bufferA, bufferB: bufferB, bufferC: bufferC,
                                       M: M, N: N, K: K)
            times.append(time)
        }

        return times.reduce(0, +) / Double(times.count)
    }

    private func fillBuffer(_ buffer: MTLBuffer, precision: GEMMOperandPrecision, count: Int) {
        switch precision {
        case .INT8:
            let ptr = buffer.contents().bindMemory(to: Int8.self, capacity: count)
            for i in 0..<count {
                ptr[i] = Int8((i % 256) - 128)
            }
        case .BF16:
            let ptr = buffer.contents().bindMemory(to: UInt16.self, capacity: count)
            for i in 0..<count {
                ptr[i] = 0x3F80 // 1.0 in BF16
            }
        default:
            break
        }
    }

    private func runGEMMKernel(pipelineState: MTLComputePipelineState,
                              bufferA: MTLBuffer, bufferB: MTLBuffer, bufferC: MTLBuffer,
                              M: Int, N: Int, K: Int, kernel: GEMMKernel) throws -> Double {
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw BenchmarkError.commandEncodingFailed
        }

        encoder.setComputePipelineState(pipelineState)
        encoder.setBuffer(bufferA, offset: 0, index: 0)
        encoder.setBuffer(bufferB, offset: 0, index: 1)
        encoder.setBuffer(bufferC, offset: 0, index: 2)

        let threadgroupSize = kernel.threadgroupSize
        let gridSize = kernel.gridSize

        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()

        let startTime = CFAbsoluteTimeGetCurrent()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        let endTime = CFAbsoluteTimeGetCurrent()

        return (endTime - startTime) * 1000.0
    }

    private func runSimpleGEMM(pipelineState: MTLComputePipelineState,
                              bufferA: MTLBuffer, bufferB: MTLBuffer, bufferC: MTLBuffer,
                              M: Int, N: Int, K: Int) throws -> Double {
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw BenchmarkError.commandEncodingFailed
        }

        encoder.setComputePipelineState(pipelineState)
        encoder.setBuffer(bufferA, offset: 0, index: 0)
        encoder.setBuffer(bufferB, offset: 0, index: 1)
        encoder.setBuffer(bufferC, offset: 0, index: 2)

        var constants = (UInt32(M), UInt32(N), UInt32(K))
        encoder.setBytes(&constants, length: MemoryLayout.size(ofValue: constants), index: 3)

        let threadsPerGroup = MTLSize(width: 8, height: 8, depth: 1)
        let threadgroups = MTLSize(
            width: (N + 7) / 8,
            height: (M + 7) / 8,
            depth: 1
        )

        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()

        let startTime = CFAbsoluteTimeGetCurrent()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        let endTime = CFAbsoluteTimeGetCurrent()

        return (endTime - startTime) * 1000.0
    }
}

// Run the GEMM benchmark
do {
    let benchmark = try GEMMBenchmark()
    try benchmark.benchmarkGEMM()
} catch {
    print("GEMM benchmark failed: \(error)")
}
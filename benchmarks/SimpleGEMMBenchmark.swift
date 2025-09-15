#!/usr/bin/env swift

import Foundation
import Metal
import MetalKit

print("=== Simple GEMM Pipeline Comparison ===")

guard let device = MTLCreateSystemDefaultDevice() else {
  print("❌ Metal device not found")
  exit(1)
}

guard let commandQueue = device.makeCommandQueue() else {
  print("❌ Command queue creation failed")
  exit(1)
}

let source = """
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// BF16 with simdgroup_matrix (optimal baseline)
kernel void gemm_bf16_optimized(
    device half* A [[buffer(0)]],
    device half* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint3& dims [[buffer(3)]],
    uint3 gid [[threadgroup_position_in_grid]],
    ushort lane_id [[thread_index_in_simdgroup]]
) {
    uint M = dims.x, N = dims.y, K = dims.z;
    const uint TILE_SIZE = 8;

    uint row = gid.y * TILE_SIZE;
    uint col = gid.x * TILE_SIZE;

    if (row >= M || col >= N) return;

    simdgroup_matrix_storage<half> A_tile;
    simdgroup_matrix_storage<half> B_tile;
    simdgroup_matrix_storage<float> C_tile;

    C_tile.load(C + row * N + col, N, ushort2(0, 0), false);

    for (uint k = 0; k < K; k += TILE_SIZE) {
        A_tile.load(A + row * K + k, K, ushort2(k, row), false);
        B_tile.load(B + k * N + col, N, ushort2(col, k), false);
        C_tile.multiply(A_tile, B_tile);
    }

    C_tile.store(C + row * N + col, N, ushort2(0, 0), false);
}

// INT8 current approach (separate kernel like yours)
kernel void gemm_int8_current(
    device char* A [[buffer(0)]],
    device char* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint3& dims [[buffer(3)]],
    uint3 gid [[threadgroup_position_in_grid]]
) {
    uint M = dims.x, N = dims.y, K = dims.z;
    const uint TILE_SIZE = 8;

    uint row = gid.y * TILE_SIZE + (gid.z % TILE_SIZE);
    uint col = gid.x * TILE_SIZE + (gid.z / TILE_SIZE);

    if (row >= M || col >= N) return;

    float sum = 0.0;
    for (uint k = 0; k < K; k++) {
        // Simulate your current dequantization approach
        float a_val = (float(A[row * K + k]) - 128.0) * 0.1;
        float b_val = (float(B[k * N + col]) - 128.0) * 0.1;
        sum += a_val * b_val;
    }
    C[row * N + col] = sum;
}

// INT8 optimized (what you should do)
kernel void gemm_int8_optimized(
    device char* A [[buffer(0)]],
    device char* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint3& dims [[buffer(3)]],
    uint3 gid [[threadgroup_position_in_grid]],
    ushort lane_id [[thread_index_in_simdgroup]]
) {
    uint M = dims.x, N = dims.y, K = dims.z;
    const uint TILE_SIZE = 8;

    uint row = gid.y * TILE_SIZE;
    uint col = gid.x * TILE_SIZE;

    if (row >= M || col >= N) return;

    // Use simdgroup_matrix but load/convert efficiently
    simdgroup_matrix_storage<float> A_tile;
    simdgroup_matrix_storage<float> B_tile;
    simdgroup_matrix_storage<float> C_tile;

    // Initialize accumulator
    for (uint i = 0; i < 64; i++) {
        C_tile.thread_elements()[i] = 0.0;
    }

    for (uint k = 0; k < K; k += TILE_SIZE) {
        // Vectorized load and convert (key optimization)
        for (uint i = 0; i < TILE_SIZE; i++) {
            for (uint j = 0; j < TILE_SIZE; j += 4) {
                if ((row + i < M) && (k + j + 3 < K)) {
                    uint a_idx = (row + i) * K + (k + j);
                    char4 a_vals = *reinterpret_cast<device char4*>(&A[a_idx]);
                    float4 a_converted = (float4(a_vals) - 128.0) * 0.1;

                    A_tile.thread_elements()[i * TILE_SIZE + j + 0] = a_converted.x;
                    A_tile.thread_elements()[i * TILE_SIZE + j + 1] = a_converted.y;
                    A_tile.thread_elements()[i * TILE_SIZE + j + 2] = a_converted.z;
                    A_tile.thread_elements()[i * TILE_SIZE + j + 3] = a_converted.w;
                }
            }
        }

        for (uint i = 0; i < TILE_SIZE; i++) {
            for (uint j = 0; j < TILE_SIZE; j += 4) {
                if ((k + i < K) && (col + j + 3 < N)) {
                    uint b_idx = (k + i) * N + (col + j);
                    char4 b_vals = *reinterpret_cast<device char4*>(&B[b_idx]);
                    float4 b_converted = (float4(b_vals) - 128.0) * 0.1;

                    B_tile.thread_elements()[i * TILE_SIZE + j + 0] = b_converted.x;
                    B_tile.thread_elements()[i * TILE_SIZE + j + 1] = b_converted.y;
                    B_tile.thread_elements()[i * TILE_SIZE + j + 2] = b_converted.z;
                    B_tile.thread_elements()[i * TILE_SIZE + j + 3] = b_converted.w;
                }
            }
        }

        C_tile.multiply(A_tile, B_tile);
    }

    // Store results
    for (uint i = 0; i < TILE_SIZE; i++) {
        for (uint j = 0; j < TILE_SIZE; j++) {
            if ((row + i < M) && (col + j < N)) {
                C[(row + i) * N + (col + j)] = C_tile.thread_elements()[i * TILE_SIZE + j];
            }
        }
    }
}
"""

func benchmark(
  kernelName: String,
  M: Int,
  N: Int,
  K: Int,
  elementSizeA: Int,
  elementSizeB: Int
)
  -> Double
{
  do {
    let library = try device.makeLibrary(source: source, options: nil)
    guard let function = library.makeFunction(name: kernelName) else {
      print("❌ Function \(kernelName) not found")
      return 0.0
    }

    let pipelineState = try device.makeComputePipelineState(function: function)

    guard
      let bufferA = device.makeBuffer(length: M * K * elementSizeA, options: []),
      let bufferB = device.makeBuffer(length: K * N * elementSizeB, options: []),
      let bufferC = device.makeBuffer(length: M * N * 4, options: [])
    else {
      print("❌ Buffer creation failed")
      return 0.0
    }

    // Fill with test data
    if elementSizeA == 1 {
      let ptr = bufferA.contents().bindMemory(to: Int8.self, capacity: M * K)
      for i in 0..<(M * K) {
        ptr[i] = Int8((i % 256) - 128)
      }
    } else {
      let ptr = bufferA.contents().bindMemory(to: UInt16.self, capacity: M * K)
      for i in 0..<(M * K) {
        ptr[i] = 0x3F80
      } // 1.0 in BF16
    }

    if elementSizeB == 1 {
      let ptr = bufferB.contents().bindMemory(to: Int8.self, capacity: K * N)
      for i in 0..<(K * N) {
        ptr[i] = Int8((i % 256) - 128)
      }
    } else {
      let ptr = bufferB.contents().bindMemory(to: UInt16.self, capacity: K * N)
      for i in 0..<(K * N) {
        ptr[i] = 0x3F80
      } // 1.0 in BF16
    }

    var times: [Double] = []
    for _ in 0..<5 {
      guard
        let commandBuffer = commandQueue.makeCommandBuffer(),
        let encoder = commandBuffer.makeComputeCommandEncoder() else { return 0.0 }

      encoder.setComputePipelineState(pipelineState)
      encoder.setBuffer(bufferA, offset: 0, index: 0)
      encoder.setBuffer(bufferB, offset: 0, index: 1)
      encoder.setBuffer(bufferC, offset: 0, index: 2)

      var dims = (UInt32(M), UInt32(N), UInt32(K))
      encoder.setBytes(&dims, length: MemoryLayout.size(ofValue: dims), index: 3)

      let threadsPerGroup = MTLSize(width: 8, height: 8, depth: 1)
      let threadgroups = MTLSize(width: (N + 63) / 64, height: (M + 63) / 64, depth: 1)

      encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerGroup)
      encoder.endEncoding()

      let startTime = CFAbsoluteTimeGetCurrent()
      commandBuffer.commit()
      commandBuffer.waitUntilCompleted()
      let endTime = CFAbsoluteTimeGetCurrent()

      times.append((endTime - startTime) * 1000.0)
    }

    return times.reduce(0, +) / Double(times.count)
  } catch {
    print("❌ Benchmark failed: \(error)")
    return 0.0
  }
}

let sizes = [(512, 512, 512), (1024, 1024, 1024)]

for (M, N, K) in sizes {
  print("\\nMatrix size: \(M)x\(N)x\(K)")

  let bf16Time = benchmark(
    kernelName: "gemm_bf16_optimized",
    M: M,
    N: N,
    K: K,
    elementSizeA: 2,
    elementSizeB: 2
  )
  let int8CurrentTime = benchmark(
    kernelName: "gemm_int8_current",
    M: M,
    N: N,
    K: K,
    elementSizeA: 1,
    elementSizeB: 1
  )
  let int8OptimizedTime = benchmark(
    kernelName: "gemm_int8_optimized",
    M: M,
    N: N,
    K: K,
    elementSizeA: 1,
    elementSizeB: 1
  )

  let flops = 2.0 * Double(M) * Double(N) * Double(K) / 1e9

  print(
    "  BF16 optimized:    \(String(format: "%.3f", bf16Time))ms (\(String(format: "%.1f", flops / bf16Time * 1000)) GFLOPS)"
  )
  print(
    "  INT8 current:      \(String(format: "%.3f", int8CurrentTime))ms (\(String(format: "%.1f", flops / int8CurrentTime * 1000)) GFLOPS)"
  )
  print(
    "  INT8 optimized:    \(String(format: "%.3f", int8OptimizedTime))ms (\(String(format: "%.1f", flops / int8OptimizedTime * 1000)) GFLOPS)"
  )
  print("  Current slowdown:  \(String(format: "%.2f", int8CurrentTime / bf16Time))x")
  print("  Potential speedup: \(String(format: "%.2f", int8CurrentTime / int8OptimizedTime))x")
}

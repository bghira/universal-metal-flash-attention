# Quantized Metal SIMD Flash Attention

This document describes an implementation that extends the Universal Metal Flash Attention library with quantized precision support. It uses Apple's Metal SIMD group operations for hardware-accelerated int8 and int4 matrix operations on Apple Silicon GPUs.

## Features

- Metal SIMD acceleration for quantized matrix operations.
- INT8 quantization (8-bit signed integer precision with symmetric quantization).
- INT4 quantization (4-bit integer precision using packed storage).
- 2-4x memory reduction compared to FP16/FP32.
- Integration with the existing `AttentionKernel` framework.
- Performance at ~87-88% of FP16 GOPS with significant memory savings.

## Supported Quantization Formats

| **Precision** | **Storage** | **Memory Reduction** | **GPU Support** | **Use Case** |
|---------------|-------------|---------------------|-----------------|--------------|
| **INT8** | 8-bit signed | 2x vs FP16, 4x vs FP32 | SIMD Native | Balanced efficiency |
| **INT4** | 4-bit packed | 4x vs FP16, 8x vs FP32 | SIMD Native | Maximum compression |
| **Mixed** | INT8 + INT4 | 2-4x vs FP16/32 | Dynamic | Adaptive precision |

## Performance Characteristics

**Benchmark Results from Test Suite (Apple Silicon M-series):**

| Precision | Avg Time (ms) | GOPS | Relative Performance | Memory Usage |
|-----------|---------------|------|---------------------|--------------|
| **FP16**  | 0.96          | 34.98| 100% (baseline)     | 100%         |
| **INT8**  | 1.10          | 30.46| 87%                 | 50%          |
| **INT4**  | 1.09          | 30.90| 88%                 | 25%          |

### Findings

- **Memory Bandwidth Bound**: Performance scales with memory access patterns.
- **GPU Acceleration**: SIMD group operations are used to offset the overhead of dequantization.
- **Numerical Stability**: FP32 intermediate precision is used to maintain accuracy.
- **Trade-offs**: A ~13% performance cost results in 2-4x memory savings.

## Quick Start

### Basic Usage

```swift
import FlashAttention
import Metal

// Initialize quantized attention
guard let device = MTLCreateSystemDefaultDevice() else {
    fatalError("Metal device required")
}
let quantizedAttention = QuantizedAttention(device: device)

// Configure quantization settings
var config = QuantizedAttention.Configuration()
config.queryPrecision = .FP16    // Keep queries in higher precision
config.keyPrecision = .INT8      // Quantize keys for memory efficiency
config.valuePrecision = .INT8    // Quantize values for memory efficiency

// Create quantized tensors from float data
let (query, key, value) = quantizedAttention.createQuantizedTensors(
    queryData: queryFloats, keyData: keyFloats, valueData: valueFloats,
    queryShape: [batchSize, seqLen, headDim],
    keyShape: [batchSize, seqLen, headDim],
    valueShape: [batchSize, seqLen, headDim],
    config: config
)

// Set up attention descriptor
var baseDescriptor = AttentionDescriptor()
baseDescriptor.matrixDimensions = (
    row: UInt32(seqLen),
    column: UInt32(seqLen),
    head: UInt16(headDim)
)

let quantizedDescriptor = QuantizedAttention.QuantizedAttentionDescriptor(
    baseDescriptor: baseDescriptor,
    quantizationConfig: config
)

// Execute quantized attention
guard let outputBuffer = device.makeBuffer(
    length: batchSize * seqLen * headDim * MemoryLayout<Float>.size
) else {
    fatalError("Failed to create output buffer")
}

let commandBuffer = quantizedAttention.forward(
    query: query, key: key, value: value,
    output: outputBuffer,
    descriptor: quantizedDescriptor
)

// Submit for execution
commandBuffer?.commit()
commandBuffer?.waitUntilCompleted()
```

### Performance Benchmarking

```swift
// Run comprehensive benchmarks
let results = quantizedAttention.benchmark(
    batchSize: 1,
    sequenceLength: 512,
    headDim: 64,
    iterations: 100
)

print("Benchmark Results:")
for (precision, timeMs) in results {
    if precision.contains("avg_ms") {
        let gops = results[precision.replacingOccurrences(of: "_avg_ms", with: "_gops")] ?? 0
        print("\(precision): \(String(format: "%.2f", timeMs)) ms, \(String(format: "%.1f", gops)) GOPS")
    }
}
```

## Technical Implementation

### Metal SIMD Integration

The implementation uses Metal's SIMD group matrix operations:

```metal
// GPU-optimized quantized matrix load
simdgroup_matrix_storage<float> matrix;
matrix.load_quantized_int8(
    quantized_data_ptr,
    elements_per_row,
    matrix_origin,
    scale_parameter,
    zero_point_parameter,
    transpose_flag
);

// Standard SIMD group matrix multiply
result.multiply(quantized_matrix_a, quantized_matrix_b);
```

See the [full design document](/docs/QuantizedMetalSIMDImplementation.md) for more information.

### Automatic Parameter Management

The system handles:

- **Buffer Binding Generation**: Dynamic allocation of quantization parameter buffers.
- **Kernel Source Generation**: Automatic selection of quantized vs. standard load functions.
- **Memory Layout**: Packing of INT4 data and parameter alignment.

### Quantization Strategy

1. **Symmetric Quantization**: Zero point = 0.
2. **Per-Tensor Scaling**: Single scale factor per tensor.
3. **FP32 Intermediate**: All computations in FP32 after dequantization.
4. **Range Selection**: Automatic scale calculation based on input data range.

## Testing and Validation

### Test Coverage

- **Accuracy Tests**: RMSE validation for INT8/INT4 quantization.
- **Performance Tests**: Speed and memory efficiency benchmarks.
- **Edge Case Tests**: Zero data, constant data, extreme ranges.
- **Integration Tests**: End-to-end attention computation validation.
- **Metal Kernel Tests**: GPU shader compilation and execution.

### Quality Metrics

- **INT8 Quantization**: RMSE < 0.1 for standard ranges.
- **INT4 Quantization**: RMSE < 0.2 for narrow ranges.
- **Memory Compression**: 4x (INT8) to 8x (INT4) vs FP32.
- **Performance Retention**: 87-88% of FP16 throughput.

## Future Work

### Planned Features

1. **Per-Channel Quantization**: For improved accuracy with diverse activation ranges.
2. **Mixed Precision Training**: INT8/INT4 forward with FP32 backward pass.
3. **Dynamic Quantization**: Runtime precision selection based on data characteristics.
4. **Hardware Optimization**: M3/M4 specific SIMD optimizations.

### Research Directions

1. **Adaptive Quantization**: Machine learning-based quantization parameter tuning.
2. **Sparse Quantization**: Combining quantization with attention sparsity patterns.
3. **Multi-GPU Scaling**: Distributed quantized attention across multiple Apple Silicon devices.

## References

- [QuantizedMetalSIMDImplementation.md](QuantizedMetalSIMDImplementation.md) - Detailed technical documentation
- [Apple Metal Shading Language](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
- [SIMD Group Matrix Functions](https://developer.apple.com/documentation/metal/mtlcomputecommandencoder)

## Contributing

The quantized implementation is integrated with the existing codebase:

- All tests pass with quantized precision.
- Performance benchmarks are included in the test suite.
- Documentation matches the implementation.
- Code follows existing patterns and conventions.

For contributions, ensure:

1. Tests pass for all quantization modes.
2. Performance benchmarks show expected characteristics.
3. Documentation accurately reflects the implementation.
4. Integration with the `AttentionKernel` framework is maintained.

import FlashAttention
import XCTest

@testable import MFABridge

final class QuantizationTests: XCTestCase {

  func testInt8QuantizationAccuracy() throws {
    let testData: [Float] = Array(stride(from: -5.0, through: 5.0, by: 0.1))

    let (quantized, scale, zeroPoint) = quantizeInt8(testData)
    let reconstructed = dequantizeInt8(quantized, scale: scale, zeroPoint: zeroPoint)
    let rmse = calculateRMSE(testData, reconstructed)

    XCTAssertLessThan(rmse, 0.1, "INT8 quantization RMSE should be reasonable")
    XCTAssertEqual(zeroPoint, 0, "Symmetric quantization should have zero point = 0")
    XCTAssertGreaterThan(scale, 0, "Scale should be positive")
  }

  func testInt4QuantizationAccuracy() throws {
    let testData: [Float] = Array(stride(from: -1.0, through: 1.0, by: 0.01))

    let (packed, scale, zeroPoint) = quantizeInt4(testData)
    let reconstructed = dequantizeInt4(
      packed, count: testData.count, scale: scale, zeroPoint: zeroPoint)
    let rmse = calculateRMSE(testData, reconstructed)

    XCTAssertLessThan(rmse, 0.2, "INT4 quantization RMSE should be reasonable for small ranges")
    XCTAssertEqual(zeroPoint, 0, "Symmetric quantization should have zero point = 0")
    XCTAssertGreaterThan(scale, 0, "Scale should be positive")
  }

  func testQuantizationMemoryEfficiency() throws {
    let testData: [Float] = Array(repeating: 1.0, count: 1000)

    // Test INT8 compression
    let (quantizedInt8, _, _) = quantizeInt8(testData)
    XCTAssertEqual(quantizedInt8.count, testData.count, "INT8 should use 1 byte per element")

    // Test INT4 compression
    let (packedInt4, _, _) = quantizeInt4(testData)
    XCTAssertEqual(
      packedInt4.count, (testData.count + 1) / 2, "INT4 should use 0.5 bytes per element")
  }

  func testQuantizationWithEdgeCases() throws {
    // Test with all zeros
    let zerosData = Array(repeating: Float(0.0), count: 100)
    let (quantizedZeros, scaleZeros, _) = quantizeInt8(zerosData)
    let reconstructedZeros = dequantizeInt8(quantizedZeros, scale: scaleZeros, zeroPoint: 0)

    XCTAssertTrue(
      reconstructedZeros.allSatisfy { abs($0) < 1e-6 },
      "Zero data should remain zero after quantization")

    // Test with single value
    let constantData = Array(repeating: Float(5.0), count: 100)
    let (quantizedConstant, scaleConstant, _) = quantizeInt8(constantData)
    let reconstructedConstant = dequantizeInt8(
      quantizedConstant, scale: scaleConstant, zeroPoint: 0)

    let constantRmse = calculateRMSE(constantData, reconstructedConstant)
    XCTAssertLessThan(constantRmse, 0.1, "Constant data should quantize accurately")
  }
}

// MARK: - Helper Functions

private func quantizeInt8(_ input: [Float]) -> ([Int8], Float, Int32) {
  let maxVal = input.max() ?? 0.0
  let minVal = input.min() ?? 0.0
  let absMax = max(abs(maxVal), abs(minVal))
  let scale = absMax > 0 ? absMax / 127.0 : 1.0  // Avoid division by zero
  let zeroPoint: Int32 = 0  // Symmetric quantization

  let quantized = input.map { value in
    let quantizedValue = Int32(round(value / scale)) + zeroPoint
    return Int8(clamping: quantizedValue)
  }

  return (quantized, scale, zeroPoint)
}

private func quantizeInt4(_ input: [Float]) -> ([UInt8], Float, Int32) {
  let maxVal = input.max() ?? 0.0
  let minVal = input.min() ?? 0.0
  let absMax = max(abs(maxVal), abs(minVal))
  let scale = absMax > 0 ? absMax / 7.0 : 1.0  // Avoid division by zero
  let zeroPoint: Int32 = 0  // Symmetric quantization

  var packed = [UInt8]()
  for i in stride(from: 0, to: input.count, by: 2) {
    let val1 = Int32(round(input[i] / scale)) + zeroPoint
    let val2 = i + 1 < input.count ? Int32(round(input[i + 1] / scale)) + zeroPoint : 0

    let packed1 = UInt8(max(0, min(15, val1 + 8)))  // Clamp [-8,7] to [0,15] before UInt8 conversion
    let packed2 = UInt8(max(0, min(15, val2 + 8)))

    packed.append((packed2 << 4) | packed1)
  }

  return (packed, scale, zeroPoint)
}

private func dequantizeInt8(_ quantized: [Int8], scale: Float, zeroPoint: Int32) -> [Float] {
  return quantized.map { value in
    (Float(value) - Float(zeroPoint)) * scale
  }
}

private func dequantizeInt4(_ packed: [UInt8], count: Int, scale: Float, zeroPoint: Int32)
  -> [Float]
{
  var result = [Float]()
  for packedByte in packed {
    let val1 = Int32(packedByte & 0xF) - 8  // Convert from [0,15] to [-8,7]
    let val2 = Int32(packedByte >> 4) - 8

    result.append((Float(val1) - Float(zeroPoint)) * scale)
    if result.count < count {
      result.append((Float(val2) - Float(zeroPoint)) * scale)
    }
  }
  return Array(result.prefix(count))
}

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

// Note: Int8(clamping:) and UInt8(clamping:) are provided by Swift standard library

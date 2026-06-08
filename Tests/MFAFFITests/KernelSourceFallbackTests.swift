import XCTest
@testable import MFABridge

final class KernelSourceFallbackTests: XCTestCase {
  func testReplacingBFloatWithFloatTypes() {
    let source = "thread bfloat a; thread bfloat2 b; thread bfloat3 c; thread bfloat4 d; int my_bfloat_identifier = 1;"
    let replaced = replacingBFloatWithFloatTypes(in: source)
    XCTAssertEqual(
      replaced,
      "thread float a; thread float2 b; thread float3 c; thread float4 d; int my_bfloat_identifier = 1;"
    )
  }

  func testDetectsUnsupportedBFloatCompilerError() {
    let error = NSError(
      domain: "MTLLibraryErrorDomain",
      code: 3,
      userInfo: [NSLocalizedDescriptionKey: "program_source:327:46: error: unknown type name 'bfloat'"]
    )
    XCTAssertTrue(sourceUsesUnsupportedBFloatTypes(error: error))
  }

  func testIgnoresUnrelatedCompilerErrors() {
    let error = NSError(
      domain: "MTLLibraryErrorDomain",
      code: 3,
      userInfo: [NSLocalizedDescriptionKey: "program_source:12:3: error: expected ';' after expression"]
    )
    XCTAssertFalse(sourceUsesUnsupportedBFloatTypes(error: error))
  }

  func testIgnoresNonTypeNameBFloatMentions() {
    let error = NSError(
      domain: "MTLLibraryErrorDomain",
      code: 3,
      userInfo: [NSLocalizedDescriptionKey: "program_source:9: warning: variable named 'bfloat' is unused"]
    )
    XCTAssertFalse(sourceUsesUnsupportedBFloatTypes(error: error))
  }
}

import Foundation
import Metal

/// Environment detection utilities for conditional test behavior
enum TestEnvironment {
  /// Check if running in CI environment (GitHub Actions, etc.)
  static var isCI: Bool {
    ProcessInfo.processInfo.environment["CI"] != nil ||
      ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] != nil
  }

  /// Check if running on GitHub Actions specifically
  static var isGitHubActions: Bool {
    ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] == "true"
  }

  /// Get the Metal device for GPU capability checks
  static var metalDevice: MTLDevice? {
    MTLCreateSystemDefaultDevice()
  }

  /// Check if device supports Apple9+ (M3+) features
  static var supportsApple9: Bool {
    guard let device = metalDevice else { return false }
    return device.supportsFamily(.apple9)
  }

  /// Check if device supports Apple8+ (M2+) features
  static var supportsApple8: Bool {
    guard let device = metalDevice else { return false }
    return device.supportsFamily(.apple8)
  }

  /// Check if device supports Apple7+ (M1+) features
  static var supportsApple7: Bool {
    guard let device = metalDevice else { return false }
    return device.supportsFamily(.apple7)
  }

  /// Get a descriptive name for the current GPU
  static var gpuName: String {
    metalDevice?.name ?? "Unknown GPU"
  }

  /// Check if we should skip tests that are known to be unstable on CI
  static var shouldSkipUnstableTests: Bool {
    isCI
  }

  /// Check if we should use relaxed numerical tolerances
  static var shouldUseRelaxedTolerances: Bool {
    isCI || !supportsApple8
  }
}

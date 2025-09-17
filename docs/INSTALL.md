# Installation Guide

> This document covers cloning the repository, building the Swift package, and running the core test suite. For end-to-end samples (Python, Rust, Flux, Objective-C), see the companion [EXAMPLES.md](/docs/EXAMPLES.md).

---

## 1. Retrieve the Source

```bash
# Clone with submodules (recommended)
git clone --recursive https://github.com/bghira/universal-metal-flash-attention.git
cd universal-metal-flash-attention

# If you cloned without --recursive
git submodule update --init --recursive
```

> **Prerequisites**: macOS 14+ (or iOS/tvOS/visionOS SDK via Xcode 15+), Swift 5.10+, an Apple GPU with Metal support.

---

## 2. Build the Swift Package

Use Swift Package Manager; the build generates both the host Swift library and Metal shader bundle.

```bash
# Debug build (default)
swift build

# Release build
swift build -c release

# Optional: specify destination for cross-compilation
# swift build --destination path/to/destination.json
```

Artifacts are emitted under `.build/` (e.g., `.build/arm64-apple-macosx/release/libFlashAttention.dylib`). Examples that link against the library expect `DYLD_LIBRARY_PATH` to include these locations.

---

## 3. Run the Test Suite

```bash
# Full test suite (CPU + GPU). Requires a Metal-capable device.
swift test

# Run individual groups when iterating
swift test --filter QuantizedAttentionTest
swift test --filter MultiHeadAttentionTest
```

Some tests launch Metal command buffers and may take several seconds on first compilation. If you run into timeouts in CI, target specific suites as shown above.

---

## 4. Next Steps

- Browse [EXAMPLES.md](/docs/EXAMPLES.md) for language-specific integrations (Python, Rust, PyTorch custom op, Flux, Objective‑C).
- Each example directory contains a `README.md` with build/run instructions and caveats (supported precisions, masking support, etc.).
- For quantized workloads, read `Sources/FlashAttention/Attention/QuantizedAttention.swift` in conjunction with the example you plan to run to understand required scale/zero-point metadata.

Happy hacking! If you hit issues, open an issue/PR or reach out via the project discussion board or our [Discord server](https://discord.gg/CVzhX7ZA).

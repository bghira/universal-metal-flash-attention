# Swift Reference Examples

Two standalone Swift scripts demonstrate direct use of `QuantizedAttention` and bridge benchmarking without additional language bindings.

## Prerequisites

- Follow [INSTALL.md](/docs/INSTALL.md) to build the Swift package (`swift build -c release`).
- Execute commands from the repository root unless noted otherwise.

## `QuantizedAttentionExample.swift`

Location: [`examples/QuantizedAttentionExample.swift`](../QuantizedAttentionExample.swift)

This script generates synthetic tensors, quantizes them to INT8/INT4, and runs the forward pass entirely in Swift.

```bash
swift examples/QuantizedAttentionExample.swift
```

The script uses the built package in `.build/` and prints timing / accuracy summaries to stdout.

## `swift-bridge-benchmark.swift`

Location: [`examples/swift-bridge-benchmark.swift`](../swift-bridge-benchmark.swift)

Benchmarks the C bridge overhead versus pure Swift calls. Useful when integrating the FFI into a larger host application.

```bash
swift examples/swift-bridge-benchmark.swift
```

Both scripts expect the package to be built already; they rely on `.build/…/release` artifacts produced by `swift build`.

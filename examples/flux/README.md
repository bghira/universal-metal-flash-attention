# Flux Quantized Attention Benchmark

This folder demonstrates end-to-end image generation with the Flux diffusion model running on Universal Metal Flash Attention. It exercises the PyTorch custom op, quantized INT8/INT4 paths, and records speed/quality metrics.

## Prerequisites

- macOS 14+ with Apple Silicon GPU
- Python 3.11 or 3.12 (matching the PyTorch wheel you install)
- Xcode 15+ / Swift 5.10+
- The Swift package built at the repository root (`swift build`)

## Setup

1. **Create a Python virtual environment** (from the repository root):

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   ```

2. **Install Python dependencies** (PyTorch nightlies plus helper packages):

   ```bash
   # Inside the virtual environment
   python -m pip install -r examples/flux/flux_requirements.txt
   ```

3. **Build the Swift package** (ensures the Metal shaders and host library are available):

   ```bash
   swift build -c release
   ```

4. **Build the PyTorch custom op** (links against the Swift build products):

   ```bash
   # The Makefile handles DYLD_LIBRARY_PATH for you
   cd examples/flux
   make build          # or: make build-swift && make build-pytorch
   ```

## Running Benchmarks

- Quick sanity check (single-head attention stub):

  ```bash
  make test-single
  ```

- Flux Schnell / Flux standard benchmarks:

  ```bash
  source ../../.venv/bin/activate
  PYTHONPATH=examples/flux/pytorch-custom-op-ffi \
    python examples/flux/flux_quick_benchmark.py --precision int8
  ```

  See the scripts `flux_quick_benchmark.py` and `flux_schnell_benchmark.py` for additional flags (image resolution, batch size, precision selection).

## Notes & Limitations

- Multi-head quantized attention is routed through the Swift `MultiHeadAttention` class, but the current benchmark harness still runs several single-head probes for validation. The `show-real-mha-approach` Makefile target explains the expected parallel layout.
- Make sure the Swift build artifacts remain discoverable: the Makefile exports `DYLD_LIBRARY_PATH` to include both `.build/*/release` and `.build/*/debug`.
- Generated images are written into `examples/flux/output/`. A sample comparison is committed under `examples/flux/output/256x256/`.

For more context on the PyTorch extension, read [`examples/pytorch-custom-op-ffi/README.md`](../pytorch-custom-op-ffi/README.md).

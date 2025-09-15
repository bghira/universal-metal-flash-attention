# Universal Metal Flash Attention - Makefile
# Universal C bridge for Metal Flash Attention

.PHONY: all clean test debug release install docs rust-example objc-example

# Configuration
BUILD_DIR := .build
DEBUG_DIR := $(BUILD_DIR)/debug
RELEASE_DIR := $(BUILD_DIR)/release
RUST_EXAMPLE_DIR := examples/rust

# Default target
all: submodule release

# Initialize submodules if needed
submodule:
	@if [ ! -f metal-flash-attention/Package.swift ]; then \
		echo "🔄 Initializing submodules..."; \
		git submodule update --init --recursive; \
		echo "✅ Submodules initialized"; \
	fi

# Debug build with optimizations for development
debug: submodule
	@echo "🔨 Building MFA FFI (Debug)..."
	swift build -c debug
	@echo "✅ Debug build complete"

# Release build with full optimizations
release: submodule
	@echo "🚀 Building MFA FFI (Release)..."
	swift build -c release
	@echo "✅ Release build complete"

# Run tests
test: debug
	@echo "🧪 Running tests..."
	swift test -c debug
	@echo "✅ All tests passed"

# Build and run Rust example
rust-example: release
	@echo "🦀 Building Rust example..."
	cd examples/rust-ffi && DYLD_LIBRARY_PATH=../../$(RELEASE_DIR) cargo build --release
	@echo "🚀 Running Rust example..."
	cd examples/rust-ffi && DYLD_LIBRARY_PATH=../../$(RELEASE_DIR) cargo run --release

# Build and run Objective-C example
objc-example: debug
	@echo "🎯 Building Objective-C example..."
	cd examples/objc && make build
	@echo "🚀 Running Objective-C example..."
	cd examples/objc && make run

# Setup Python environment
python-setup:
	@echo "🐍 Setting up Python environment..."
	cd examples/python && \
	(test -d venv || python3 -m venv venv) && \
	source venv/bin/activate && \
	pip install -r requirements.txt
	@echo "✅ Python environment ready"

# Run Python example
python-example: release python-setup
	@echo "🐍 Running Python example..."
	cd examples/python && source venv/bin/activate && python example_basic.py

# Run Python benchmarks
python-benchmark: release python-setup
	@echo "📊 Running Python benchmarks..."
	cd examples/python && source venv/bin/activate && python benchmarks/benchmark_performance.py

# Run Python tests
python-test: release python-setup
	@echo "🧪 Running Python tests..."
	cd examples/python && source venv/bin/activate && python -m pytest tests/ -v

# Clean build artifacts
clean:
	@echo "🧹 Cleaning build artifacts..."
	rm -rf $(BUILD_DIR)
	cd $(RUST_EXAMPLE_DIR) && cargo clean
	@echo "✅ Clean complete"

# Performance benchmark
benchmark: release
	@echo "⚡ Running performance benchmark..."
	cd $(RUST_EXAMPLE_DIR) && DYLD_LIBRARY_PATH=../../$(RELEASE_DIR) cargo run --release

# Install to system (macOS)
install: release
	@echo "📦 Installing MFA FFI..."
	sudo cp $(RELEASE_DIR)/libMFAFFI.dylib /usr/local/lib/
	sudo cp Sources/MFAFFI/include/mfa_ffi.h /usr/local/include/
	@echo "✅ Installation complete"

# Generate documentation
docs:
	@echo "📚 Generating documentation..."
	swift package generate-documentation
	@echo "✅ Documentation generated"

# Development workflow
dev: clean debug test
	@echo "🛠️  Development build ready"

# CI workflow
ci: clean release test benchmark
	@echo "🎯 CI pipeline complete"

# Help
help:
	@echo "Universal Metal Flash Attention - Build System"
	@echo ""
	@echo "Available targets:"
	@echo "  all         - Build release version (default)"
	@echo "  debug       - Build debug version with development optimizations"
	@echo "  release     - Build release version with full optimizations"
	@echo "  test        - Run Swift test suite"
	@echo "  rust-example    - Build and run Rust integration example"
	@echo "  objc-example    - Build and run Objective-C integration example"
	@echo "  python-setup    - Set up Python virtual environment"
	@echo "  python-example  - Run Python integration example"
	@echo "  python-benchmark - Run Python performance benchmarks"
	@echo "  python-test     - Run Python test suite"
	@echo "  benchmark       - Run performance benchmark"
	@echo "  clean           - Remove all build artifacts"
	@echo "  install         - Install to system (requires sudo)"
	@echo "  docs        - Generate API documentation"
	@echo "  dev         - Clean + debug + test (development workflow)"
	@echo "  ci          - Full CI pipeline (clean + release + test + benchmark)"
	@echo "  help        - Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make release         # Build optimized FFI library"
	@echo "  make rust-example    # Test Rust integration"
	@echo "  make objc-example    # Test Objective-C integration"
	@echo "  make dev             # Development workflow"

#!/bin/bash

# Build and run MLA performance benchmarks

set -e

echo "=== Building and Running MLA Performance Benchmarks ==="
echo "Date: $(date)"
echo

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Change to project root
cd "$(dirname "$0")"
PROJECT_ROOT=$(pwd)

echo -e "${YELLOW}Step 1: Building FlashAttention library...${NC}"
echo "----------------------------------------"

# Build in release mode for performance testing
swift build -c release

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Build successful${NC}"
else
    echo -e "${RED}✗ Build failed${NC}"
    exit 1
fi

echo
echo -e "${YELLOW}Step 2: Running MLA Swift correctness + performance tests...${NC}"
echo "----------------------------------------"

# Runs MLAMFAUsageExample (testMLADecompressionCorrectness + testMLAPerformanceTFLOPS,
# which prints GPU-timed TFLOPS per shape).
swift test -c release --filter MLAMFAUsageExample

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ MLA Swift tests completed${NC}"
else
    echo -e "${RED}✗ MLA Swift tests failed${NC}"
    # Don't exit - try to run other benchmarks
fi

echo
echo -e "${YELLOW}Step 3: Running MLA Python benchmark (MFA vs MPS)...${NC}"
echo "----------------------------------------"

PYTHON=${PYTHON:-python3}
EXT_DIR="$PROJECT_ROOT/examples/pytorch-custom-op-ffi"
if [ -f "$EXT_DIR/metal_sdpa_extension.so" ] || [ -d "$EXT_DIR/pytorch_metal_sdpa_backend.egg-info" ]; then
    (cd "$EXT_DIR" && $PYTHON bench_mla_vs_mps.py) || echo -e "${RED}✗ Python MLA benchmark failed${NC}"
else
    echo -e "${YELLOW}Skipping Python MLA benchmark (extension not built in $EXT_DIR)${NC}"
fi

echo
echo -e "${YELLOW}Step 4: Running All Performance Tests...${NC}"
echo "----------------------------------------"

# Run all performance tests for comparison
swift test -c release --filter PerformanceTests || true

echo
echo -e "${YELLOW}Step 5: Generating Summary Report...${NC}"
echo "----------------------------------------"

# Create results directory
RESULTS_DIR="$PROJECT_ROOT/benchmark_results"
mkdir -p "$RESULTS_DIR"

# Generate timestamp for results file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_FILE="$RESULTS_DIR/mla_benchmark_results_$TIMESTAMP.txt"

# Capture MLA Swift + Python benchmark output to results file
{
    echo "=== MLA Benchmark Results ==="
    echo "Generated: $(date)"
    echo "Platform: $(uname -m) $(sw_vers -productVersion)"
    echo "Device: $(system_profiler SPDisplaysDataType | grep "Chipset Model" | head -1)"
    echo
    echo "--- Swift (MLAMFAUsageExample, GPU-timed) ---"
    swift test -c release --filter MLAMFAUsageExample 2>&1
    echo
    echo "--- Python (MFA vs MPS) ---"
    (cd "$EXT_DIR" && $PYTHON bench_mla_vs_mps.py 2>&1)

} | tee "$RESULTS_FILE"

echo
echo -e "${GREEN}=== Benchmark Complete ===${NC}"
echo -e "Results saved to: ${YELLOW}$RESULTS_FILE${NC}"
echo

# Quick summary extraction: capture TFLOPS lines from both runners.
echo "Quick Summary:"
echo "-------------"
grep -E "(TFLOPS|→ [0-9]+\.[0-9]+ TFLOPS|seq=)" "$RESULTS_FILE" | tail -30

echo
echo "To view full results:"
echo "  cat $RESULTS_FILE"

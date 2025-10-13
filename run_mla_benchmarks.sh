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
echo -e "${YELLOW}Step 2: Running MLA Performance Tests...${NC}"
echo "----------------------------------------"

# Run the specific MLA performance test
swift test -c release --filter MLAPerformanceTests

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ MLA tests completed${NC}"
else
    echo -e "${RED}✗ MLA tests failed${NC}"
    # Don't exit - try to run other benchmarks
fi

echo
echo -e "${YELLOW}Step 3: Running Memory Footprint Analysis...${NC}"
echo "----------------------------------------"

swift test -c release --filter testMemoryFootprint

echo
echo -e "${YELLOW}Step 4: Running All Performance Tests...${NC}"
echo "----------------------------------------"

# Run all performance tests for comparison
swift test -c release --filter PerformanceTests

echo
echo -e "${YELLOW}Step 5: Generating Summary Report...${NC}"
echo "----------------------------------------"

# Create results directory
RESULTS_DIR="$PROJECT_ROOT/benchmark_results"
mkdir -p "$RESULTS_DIR"

# Generate timestamp for results file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_FILE="$RESULTS_DIR/mla_benchmark_results_$TIMESTAMP.txt"

# Capture all test output to results file
{
    echo "=== MLA Benchmark Results ==="
    echo "Generated: $(date)"
    echo "Platform: $(uname -m) $(sw_vers -productVersion)"
    echo "Device: $(system_profiler SPDisplaysDataType | grep "Chipset Model" | head -1)"
    echo
    echo "Running comprehensive benchmarks..."
    echo

    swift test -c release --filter MLAPerformanceTests 2>&1

} | tee "$RESULTS_FILE"

echo
echo -e "${GREEN}=== Benchmark Complete ===${NC}"
echo -e "Results saved to: ${YELLOW}$RESULTS_FILE${NC}"
echo

# Quick summary extraction
echo "Quick Summary:"
echo "-------------"
grep -E "(Standard FP16:|INT8 Quantized:|MLA Compressed:|Memory savings:)" "$RESULTS_FILE" | tail -20

echo
echo "To view full results:"
echo "  cat $RESULTS_FILE"

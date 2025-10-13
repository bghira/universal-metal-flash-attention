#!/usr/bin/env python3
"""
Test runner for Metal SDPA PyTorch extension.

Usage:
    # Run all tests
    python run_tests.py

    # Run specific test file
    python run_tests.py tests/test_dtype_compatibility.py

    # Run specific test
    python run_tests.py tests/test_mps_edge_cases.py::TestMPSMatrixMultiplication::test_accumulator_dtype_mismatch_prevention

    # Run with verbose output
    python run_tests.py -v

    # Run only fast tests (skip slow ones)
    python run_tests.py -m "not slow"

    # Run only dtype tests
    python run_tests.py -m dtype

    # Run MPS error reproduction test specifically
    python run_tests.py -k "dtype_mismatch_reproduction"
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Run the test suite."""
    # Ensure we're in the right directory
    project_dir = Path(__file__).parent

    # Build the extension first if needed
    print("🔨 Building Metal SDPA extension...")
    build_result = subprocess.run(
        [sys.executable, "setup.py", "build"],
        cwd=project_dir,
        capture_output=True,
        text=True,
    )

    if build_result.returncode != 0:
        print("❌ Build failed:")
        print(build_result.stderr)
        return 1

    print("✅ Build successful")
    print()

    # Run pytest with any provided arguments
    pytest_args = ["pytest", "-v", "--tb=short"]
    pytest_args.extend(sys.argv[1:])

    print(f"🧪 Running tests: {' '.join(pytest_args)}")
    print("=" * 60)

    result = subprocess.run(pytest_args, cwd=project_dir)

    return result.returncode


def run_critical_tests():
    """Run only the critical tests for MPS dtype mismatch."""
    print("🚨 Running critical MPS dtype mismatch tests...")
    print("=" * 60)

    critical_tests = [
        "tests/test_dtype_compatibility.py::TestDtypeCompatibility::test_bfloat16_compatibility",
        "tests/test_dtype_compatibility.py::TestDtypeCompatibility::test_accumulator_dtype_consistency",
        "tests/test_mps_edge_cases.py::TestMPSMatrixMultiplication::test_accumulator_dtype_mismatch_prevention",
        "tests/test_integration_flux.py::TestFLUXErrorScenarios::test_flux_dtype_mismatch_reproduction",
    ]

    project_dir = Path(__file__).parent

    for test in critical_tests:
        print(f"\n📍 Running: {test}")
        result = subprocess.run(["pytest", "-xvs", test], cwd=project_dir)
        if result.returncode != 0:
            print(f"❌ Test failed: {test}")
            return 1

    print("\n✅ All critical tests passed!")
    return 0


if __name__ == "__main__":
    # Check if we should run critical tests only
    if len(sys.argv) > 1 and sys.argv[1] == "--critical":
        sys.exit(run_critical_tests())
    else:
        sys.exit(main())

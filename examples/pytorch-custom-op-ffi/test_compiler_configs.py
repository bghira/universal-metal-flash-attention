#!/usr/bin/env python3
"""
Test different Swift compiler configurations for correctness vs performance.
Author: bghira
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).parent.parent.parent


def backup_current_package_swift():
    """Backup current Package.swift"""
    package_path = ROOT_DIR / "Package.swift"
    backup_path = ROOT_DIR / "Package.swift.backup"

    if package_path.exists():
        subprocess.run(["cp", str(package_path), str(backup_path)], check=True)
        print(f"✅ Backed up Package.swift to {backup_path}")
    return backup_path


def restore_package_swift(backup_path):
    """Restore original Package.swift"""
    package_path = ROOT_DIR / "Package.swift"
    if backup_path.exists():
        subprocess.run(["cp", str(backup_path), str(package_path)], check=True)
        print(f"✅ Restored original Package.swift")


def apply_safe_config():
    """Apply the safe compiler configuration"""
    safe_config_path = Path(__file__).parent / "Package_swift_safe.swift"
    package_path = ROOT_DIR / "Package.swift"

    if safe_config_path.exists():
        subprocess.run(["cp", str(safe_config_path), str(package_path)], check=True)
        print(f"✅ Applied safe compiler configuration")
    else:
        raise FileNotFoundError("Safe config file not found")


def rebuild_swift_library():
    """Rebuild Swift library with current configuration"""
    print("🔨 Rebuilding Swift library...")

    # Clean previous build
    build_dir = ROOT_DIR / ".build"
    if build_dir.exists():
        subprocess.run(["rm", "-rf", str(build_dir)], check=True)

    try:
        # Build in release mode
        result = subprocess.run(
            ["swift", "build", "--configuration", "release", "--product", "MFAFFI"],
            cwd=ROOT_DIR,
            check=True,
            capture_output=True,
            text=True,
        )

        print("✅ Swift library rebuilt successfully")
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Swift build failed: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False


def rebuild_pytorch_extension():
    """Rebuild PyTorch extension"""
    print("🔨 Rebuilding PyTorch extension...")

    extension_dir = Path(__file__).parent

    try:
        # Clean and rebuild
        subprocess.run(
            [sys.executable, "setup.py", "clean", "--all"],
            cwd=extension_dir,
            check=True,
        )

        subprocess.run(
            [sys.executable, "setup.py", "build_ext", "--inplace"],
            cwd=extension_dir,
            check=True,
            capture_output=True,
        )

        print("✅ PyTorch extension rebuilt successfully")
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ PyTorch extension build failed: {e}")
        return False


def test_correctness():
    """Test correctness with current configuration"""
    print("🧪 Testing correctness...")

    try:
        # Import the rebuilt extension
        import importlib

        if "metal_sdpa_extension" in sys.modules:
            importlib.reload(sys.modules["metal_sdpa_extension"])
        import metal_sdpa_extension

        # Quick correctness test
        torch.manual_seed(42)
        q = torch.randn(32, 16, dtype=torch.float32)
        k = torch.randn(32, 16, dtype=torch.float32)
        v = torch.randn(32, 16, dtype=torch.float32)

        metal_output = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)
        torch_output = F.scaled_dot_product_attention(q, k, v)

        diff = torch.abs(metal_output - torch_output).max().item()
        rel_diff = (
            (
                torch.abs(metal_output - torch_output)
                / torch.abs(torch_output).clamp(min=1e-8)
            )
            .max()
            .item()
        )

        # Test causal masking
        metal_causal = metal_sdpa_extension.metal_scaled_dot_product_attention(
            q, k, v, is_causal=True
        )
        torch_causal = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        causal_diff = torch.abs(metal_causal - torch_causal).max().item()

        return {
            "max_diff": diff,
            "rel_diff": rel_diff,
            "causal_diff": causal_diff,
            "has_nan": torch.isnan(metal_output).any().item(),
            "has_inf": torch.isinf(metal_output).any().item(),
        }

    except Exception as e:
        print(f"❌ Correctness test failed: {e}")
        return None


def test_performance():
    """Test performance with current configuration"""
    print("⚡ Testing performance...")

    try:
        import metal_sdpa_extension

        q = torch.randn(256, 64, dtype=torch.float32)
        k = torch.randn(256, 64, dtype=torch.float32)
        v = torch.randn(256, 64, dtype=torch.float32)

        # Warmup
        for _ in range(5):
            _ = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)
            _ = F.scaled_dot_product_attention(q, k, v)

        # Time Metal
        start = time.perf_counter()
        for _ in range(10):
            _ = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)
        metal_time = (time.perf_counter() - start) / 10

        # Time PyTorch
        start = time.perf_counter()
        for _ in range(10):
            _ = F.scaled_dot_product_attention(q, k, v)
        torch_time = (time.perf_counter() - start) / 10

        speedup = torch_time / metal_time if metal_time > 0 else 0

        return {
            "metal_time": metal_time * 1000,  # Convert to ms
            "torch_time": torch_time * 1000,
            "speedup": speedup,
        }

    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return None


def main():
    """Main function to test different compiler configurations"""
    print("Swift Compiler Configuration Testing")
    print("Author: bghira")
    print("=" * 60)

    # Backup original configuration
    backup_path = backup_current_package_swift()

    try:
        # Test 1: Current (potentially unsafe) configuration
        print(f"\n{'='*20} CURRENT CONFIGURATION {'='*20}")

        if rebuild_swift_library() and rebuild_pytorch_extension():
            current_correctness = test_correctness()
            current_performance = test_performance()
        else:
            current_correctness = None
            current_performance = None

        # Test 2: Safe configuration
        print(f"\n{'='*20} SAFE CONFIGURATION {'='*23}")

        apply_safe_config()

        if rebuild_swift_library() and rebuild_pytorch_extension():
            safe_correctness = test_correctness()
            safe_performance = test_performance()
        else:
            safe_correctness = None
            safe_performance = None

        # Results comparison
        print(f"\n{'='*20} RESULTS COMPARISON {'='*23}")

        print(
            f"{'Configuration':15} | {'Max Diff':>10} | {'Causal Diff':>12} | {'Speedup':>8} | {'Status':>10}"
        )
        print("-" * 80)

        # Current config results
        if current_correctness and current_performance:
            current_status = (
                "❌ UNSAFE"
                if (
                    current_correctness["max_diff"] > 1e-5
                    or current_correctness["causal_diff"] > 1e-5
                    or current_correctness["has_nan"]
                )
                else "✅ SAFE"
            )
            print(
                f"{'Current':15} | {current_correctness['max_diff']:>8.2e} | {current_correctness['causal_diff']:>10.2e} | {current_performance['speedup']:>6.2f}x | {current_status:>10}"
            )
        else:
            print(
                f"{'Current':15} | {'ERROR':>10} | {'ERROR':>12} | {'N/A':>8} | {'❌ FAILED':>10}"
            )

        # Safe config results
        if safe_correctness and safe_performance:
            safe_status = (
                "✅ SAFE"
                if (
                    safe_correctness["max_diff"] < 1e-5
                    and safe_correctness["causal_diff"] < 1e-5
                    and not safe_correctness["has_nan"]
                )
                else "❌ UNSAFE"
            )
            print(
                f"{'Safe':15} | {safe_correctness['max_diff']:>8.2e} | {safe_correctness['causal_diff']:>10.2e} | {safe_performance['speedup']:>6.2f}x | {safe_status:>10}"
            )
        else:
            print(
                f"{'Safe':15} | {'ERROR':>10} | {'ERROR':>12} | {'N/A':>8} | {'❌ FAILED':>10}"
            )

        print("-" * 80)

        # Recommendation
        if (
            safe_correctness
            and safe_correctness["max_diff"] < 1e-5
            and safe_correctness["causal_diff"] < 1e-5
        ):
            print("🎯 RECOMMENDATION: Use SAFE configuration")
            print("   - Numerical correctness is maintained")
            print(
                "   - Performance may be slightly reduced but results are trustworthy"
            )

            # Ask user if they want to keep safe config
            response = input("\nApply safe configuration permanently? (y/N): ")
            if response.lower() == "y":
                print("✅ Safe configuration will be kept")
                return  # Don't restore backup

        else:
            print("⚠️ RECOMMENDATION: Investigate further")
            print("   - Both configurations may have issues")
            print("   - Manual flag tuning may be required")

    finally:
        # Restore original configuration unless user chose to keep safe config
        restore_package_swift(backup_path)

        # Rebuild with original config
        print("\n🔄 Restoring original configuration...")
        rebuild_swift_library()
        rebuild_pytorch_extension()


if __name__ == "__main__":
    main()

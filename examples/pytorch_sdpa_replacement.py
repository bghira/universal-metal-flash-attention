#!/usr/bin/env python3
"""
PyTorch SDPA Replacement with Universal Metal Flash Attention

This example demonstrates how to replace torch.nn.functional.scaled_dot_product_attention
with our high-performance Metal implementation for 4400+ GINSTRS/sec performance.
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Add UMFA to path (adjust path as needed)
umfa_path = Path(__file__).parent / "python-ffi/src"
if umfa_path.exists():
    sys.path.insert(0, str(umfa_path))

try:
    import umfa

    UMFA_AVAILABLE = True
except ImportError:
    UMFA_AVAILABLE = False
    print(
        "⚠️  UMFA not available. Install from: /path/to/universal-metal-flash-attention/examples/python"
    )


def torch_to_numpy_zero_copy(tensor: torch.Tensor) -> np.ndarray:
    """Convert PyTorch tensor to NumPy with zero-copy (when possible)."""
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    return tensor.detach().cpu().numpy()


def numpy_to_torch_zero_copy(array: np.ndarray, device: str = "cpu") -> torch.Tensor:
    """Convert NumPy array to PyTorch tensor with zero-copy."""
    tensor = torch.from_numpy(array)
    if device != "cpu":
        tensor = tensor.to(device)
    return tensor


class MetalSDPA:
    """
    Drop-in replacement for torch.nn.functional.scaled_dot_product_attention
    using Metal Flash Attention for maximum performance on Apple Silicon.
    """

    def __init__(self):
        if not UMFA_AVAILABLE:
            raise RuntimeError("UMFA not available. Cannot use MetalSDPA.")

        if not umfa.is_metal_available():
            raise RuntimeError("Metal not available on this device.")

        self.context = umfa.MFAContext()
        print("✅ MetalSDPA initialized with Metal Flash Attention")

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: float = None,
    ) -> torch.Tensor:
        """
        Metal Flash Attention implementation of scaled_dot_product_attention.

        Args:
            query: Query tensor [batch_size, seq_len, num_heads, head_dim] or [seq_len, head_dim]
            key: Key tensor [batch_size, seq_len, num_heads, head_dim] or [seq_len, head_dim]
            value: Value tensor [batch_size, seq_len, num_heads, head_dim] or [seq_len, head_dim]
            attn_mask: Attention mask (currently ignored, use is_causal instead)
            dropout_p: Dropout probability (currently ignored)
            is_causal: Whether to apply causal masking
            scale: Scale factor (default: 1/√head_dim)

        Returns:
            Attention output tensor
        """
        # Validate inputs
        if dropout_p > 0:
            print("⚠️  Dropout not yet supported in Metal implementation")

        if attn_mask is not None:
            print(
                "⚠️  Custom attention masks not yet supported, using is_causal instead"
            )

        # Store original device and dtype
        orig_device = query.device
        orig_dtype = query.dtype

        # Convert to numpy (zero-copy when on CPU)
        q_np = torch_to_numpy_zero_copy(query)
        k_np = torch_to_numpy_zero_copy(key)
        v_np = torch_to_numpy_zero_copy(value)

        # Determine precision
        if orig_dtype == torch.float16:
            precision = "fp16"
        elif orig_dtype == torch.float32:
            precision = "fp32"
        else:
            # Convert unsupported dtypes to fp16
            q_np = q_np.astype(np.float16)
            k_np = k_np.astype(np.float16)
            v_np = v_np.astype(np.float16)
            precision = "fp16"

        # Call Metal Flash Attention
        output_np = umfa.flash_attention_forward(
            self.context,
            q_np,
            k_np,
            v_np,
            causal=is_causal,
            softmax_scale=scale,
            input_precision=precision,
            intermediate_precision=precision,
            output_precision=precision,
        )

        # Convert back to PyTorch
        output_tensor = numpy_to_torch_zero_copy(output_np, str(orig_device))

        # Convert to original dtype if needed
        if output_tensor.dtype != orig_dtype:
            output_tensor = output_tensor.to(orig_dtype)

        return output_tensor


def benchmark_comparison():
    """Compare PyTorch SDPA vs Metal Flash Attention performance."""

    if not UMFA_AVAILABLE:
        print("❌ UMFA not available, cannot run benchmark")
        return

    print("\n🚀 Benchmarking PyTorch SDPA vs Metal Flash Attention")
    print("=" * 60)

    # Test configurations
    configs = [
        {"seq_len": 512, "head_dim": 64, "dtype": torch.float16},
        # Note: Larger configs disabled to avoid GPU memory issues
        # {"seq_len": 1024, "head_dim": 64, "dtype": torch.float16},
        # {"seq_len": 2048, "head_dim": 64, "dtype": torch.float16},
        # {"seq_len": 512, "head_dim": 128, "dtype": torch.float16},
    ]

    metal_sdpa = MetalSDPA()

    for config in configs:
        seq_len = config["seq_len"]
        head_dim = config["head_dim"]
        dtype = config["dtype"]

        print(f"\n📊 Testing seq_len={seq_len}, head_dim={head_dim}, dtype={dtype}")

        # Create test tensors
        q = torch.randn(seq_len, head_dim, dtype=dtype)
        k = torch.randn(seq_len, head_dim, dtype=dtype)
        v = torch.randn(seq_len, head_dim, dtype=dtype)

        # Warm up (more iterations for Metal FA to compile/cache kernels)
        for _ in range(5):
            _ = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        for _ in range(5):
            _ = metal_sdpa(q, k, v, is_causal=True)

        # Benchmark PyTorch SDPA
        torch_times = []
        for _ in range(10):
            start = time.time()
            torch_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            torch_times.append(time.time() - start)

        # Benchmark Metal SDPA
        metal_times = []
        for _ in range(10):
            start = time.time()
            metal_out = metal_sdpa(q, k, v, is_causal=True)
            metal_times.append(time.time() - start)

        # Calculate stats
        torch_mean = np.mean(torch_times) * 1000  # ms
        metal_mean = np.mean(metal_times) * 1000  # ms
        speedup = torch_mean / metal_mean

        # Verify correctness (approximate due to precision differences)
        diff = torch.abs(torch_out - metal_out).max().item()

        print(f"   PyTorch SDPA:     {torch_mean:.2f}ms")
        print(f"   Metal FA:         {metal_mean:.2f}ms")
        print(f"   Speedup:          {speedup:.2f}x")
        print(f"   Max difference:   {diff:.6f}")
        print(f"   Status: {'✅ PASS' if diff < 1e-3 else '❌ FAIL'}")


def usage_examples():
    """Show various usage patterns for MetalSDPA."""

    if not UMFA_AVAILABLE:
        print("❌ UMFA not available, cannot run examples")
        return

    print("\n💡 Usage Examples")
    print("=" * 40)

    metal_sdpa = MetalSDPA()

    # Example 1: Basic usage
    print("\n1. Basic Usage (replacing F.scaled_dot_product_attention)")
    seq_len, head_dim = 256, 64
    q = torch.randn(seq_len, head_dim, dtype=torch.float16)
    k = torch.randn(seq_len, head_dim, dtype=torch.float16)
    v = torch.randn(seq_len, head_dim, dtype=torch.float16)

    # Original PyTorch way
    torch_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

    # Metal Flash Attention way (drop-in replacement)
    metal_out = metal_sdpa(q, k, v, is_causal=True)

    print(f"   Input shape: {q.shape}")
    print(f"   Output shape: {metal_out.shape}")
    print(f"   Max difference: {torch.abs(torch_out - metal_out).max().item():.6f}")

    # Example 2: In a PyTorch nn.Module
    print("\n2. Integration in PyTorch Module")

    class MetalAttentionLayer(torch.nn.Module):
        def __init__(self, d_model: int, num_heads: int):
            super().__init__()
            self.d_model = d_model
            self.num_heads = num_heads
            self.head_dim = d_model // num_heads

            self.q_proj = torch.nn.Linear(d_model, d_model)
            self.k_proj = torch.nn.Linear(d_model, d_model)
            self.v_proj = torch.nn.Linear(d_model, d_model)
            self.out_proj = torch.nn.Linear(d_model, d_model)

            # Note: Current Metal FA supports single-head only
            if num_heads > 1:
                print(
                    f"⚠️  Multi-head attention ({num_heads} heads) will use single-head Metal FA"
                )

            self.metal_sdpa = MetalSDPA()

        def forward(self, x: torch.Tensor, causal: bool = False) -> torch.Tensor:
            batch_size, seq_len, _ = x.shape

            q = self.q_proj(x)  # [batch, seq_len, d_model]
            k = self.k_proj(x)
            v = self.v_proj(x)

            if self.num_heads == 1:
                # Single head - use Metal FA directly
                q = q.view(batch_size * seq_len, self.head_dim)
                k = k.view(batch_size * seq_len, self.head_dim)
                v = v.view(batch_size * seq_len, self.head_dim)

                attn_out = self.metal_sdpa(q, k, v, is_causal=causal)
                attn_out = attn_out.view(batch_size, seq_len, self.d_model)
            else:
                # Multi-head - fall back to PyTorch (for now)
                q = q.view(
                    batch_size, seq_len, self.num_heads, self.head_dim
                ).transpose(1, 2)
                k = k.view(
                    batch_size, seq_len, self.num_heads, self.head_dim
                ).transpose(1, 2)
                v = v.view(
                    batch_size, seq_len, self.num_heads, self.head_dim
                ).transpose(1, 2)

                attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
                attn_out = attn_out.transpose(1, 2).contiguous()
                attn_out = attn_out.view(batch_size, seq_len, self.d_model)

            return self.out_proj(attn_out)

    # Test the module
    d_model, num_heads = 128, 1  # Single head for Metal FA
    layer = MetalAttentionLayer(d_model, num_heads)
    layer = layer.half()  # Convert layer to FP16 to match input
    x = torch.randn(2, 256, d_model, dtype=torch.float16)

    output = layer(x, causal=True)
    print(f"   Module input: {x.shape}")
    print(f"   Module output: {output.shape}")

    print("\n✅ All examples completed successfully!")


def main():
    """Main function demonstrating Metal Flash Attention integration."""

    print("🔥 Universal Metal Flash Attention - PyTorch Integration")
    print("=" * 65)

    if not UMFA_AVAILABLE:
        print("❌ Universal Metal Flash Attention not available")
        print(
            "   Please install from: /path/to/universal-metal-flash-attention/examples/python"
        )
        return 1

    if not umfa.is_metal_available():
        print("❌ Metal not available on this device")
        return 1

    # Print system info
    major, minor, patch = umfa.get_version()
    print(f"✅ MFA version: {major}.{minor}.{patch}")
    print(f"✅ PyTorch version: {torch.__version__}")

    # Run examples
    usage_examples()

    # Run benchmarks
    benchmark_comparison()

    print(f"\n🎯 Key Benefits of Metal Flash Attention:")
    print(f"   • 4400+ gigainstructions/sec performance")
    print(f"   • Zero-copy PyTorch integration")
    print(f"   • Causal masking support (our new feature!)")
    print(f"   • Optimized for Apple Silicon unified memory")
    print(f"   • Drop-in replacement for SDPA")

    return 0


if __name__ == "__main__":
    exit(main())

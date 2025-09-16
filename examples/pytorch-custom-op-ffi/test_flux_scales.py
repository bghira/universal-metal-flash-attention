#!/usr/bin/env python3
import torch
from metal_sdpa_extension import (
    OutputPrecision,
    QuantizationConfig,
    quantized_scaled_dot_product_attention_per_block,
    quantized_scaled_dot_product_attention_with_config,
)

# FLUX dimensions
batch_size, seq_len, num_heads, head_dim = 1, 1024, 8, 64

# Test problematic scales from QUANTIZATION_FIX_PLAN.md
test_scales = [0.01, 0.1, 1.0, 10.0, 100.0]

for scale in test_scales:
    print(f"\n🧪 Testing scale {scale}")
    print("-" * 30)

    # Create scaled tensors
    q = torch.randn(batch_size, seq_len, num_heads, head_dim) * scale
    k = torch.randn(batch_size, seq_len, num_heads, head_dim) * scale
    v = torch.randn(batch_size, seq_len, num_heads, head_dim) * scale

    # Test per-tensor (should fail at higher scales)
    try:
        config = QuantizationConfig()
        config.precision = "int8"
        config.output_precision = OutputPrecision.FP32

        result_tensor = quantized_scaled_dot_product_attention_with_config(
            q, k, v, config
        )
        tensor_nan = torch.isnan(result_tensor).any()
        tensor_inf = torch.isinf(result_tensor).any()
        print(f"  Per-tensor: nan={tensor_nan}, inf={tensor_inf}")
    except Exception as e:
        print(f"  Per-tensor: FAILED - {e}")

    # Test per-block (should work at all scales)
    try:
        result_block = quantized_scaled_dot_product_attention_per_block(
            q, k, v, 128, 64, 64, "int8", False, None
        )
        block_nan = torch.isnan(result_block).any()
        block_inf = torch.isinf(result_block).any()
        print(f"  Per-block:  nan={block_nan}, inf={block_inf}")

        if not block_nan and not block_inf:
            print("  ✅ Per-block quantization STABLE!")
        else:
            print("  ❌ Per-block quantization unstable")

    except Exception as e:
        print(f"  Per-block: FAILED - {e}")

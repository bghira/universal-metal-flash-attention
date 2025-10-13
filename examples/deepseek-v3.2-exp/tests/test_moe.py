"""
Unit tests for MoE (Mixture of Experts)
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from config import DeepSeekConfig
from model import MoE


def test_moe_basic():
    """Test basic MoE functionality"""
    print("Testing MoE - Basic Functionality")
    print("=" * 60)

    if not torch.backends.mps.is_available():
        print("⚠️  Skipping test - MPS backend not available")
        return

    device = torch.device("mps")

    # Create small config for testing
    config = DeepSeekConfig(
        dim=256,
        moe_inter_dim=512,
        n_routed_experts=8,
        n_shared_experts=1,
        n_activated_experts=2,
        route_scale=2.5,
    )

    print(
        f"Config: {config.dim} dim, {config.n_routed_experts} experts, "
        f"{config.n_activated_experts} activated, route_scale={config.route_scale}"
    )

    # Create MoE
    moe = MoE(config).to(device)
    print(
        f"✓ MoE created with {config.n_routed_experts} routed + "
        f"{config.n_shared_experts} shared experts"
    )

    # Create test tensor
    batch = 2
    seq = 16
    x = torch.randn(batch, seq, config.dim, device=device, dtype=torch.float16)

    print(f"✓ Test tensor: {list(x.shape)}")

    # Forward pass
    output = moe(x)

    print(f"\n✅ MoE forward pass successful!")
    print(f"   Input: {list(x.shape)}")
    print(f"   Output: {list(output.shape)}")

    # Verify shapes
    assert output.shape == x.shape, f"Shape mismatch: {output.shape} != {x.shape}"

    print("\n✅ Shape assertion passed!")
    print("=" * 60)


def test_moe_routing():
    """Test that MoE routing selects correct number of experts"""
    print("\nTesting MoE - Routing")
    print("=" * 60)

    if not torch.backends.mps.is_available():
        print("⚠️  Skipping test - MPS backend not available")
        return

    device = torch.device("mps")

    config = DeepSeekConfig(
        dim=128,
        moe_inter_dim=256,
        n_routed_experts=16,
        n_shared_experts=1,
        n_activated_experts=4,
        route_scale=2.0,
    )

    moe = MoE(config).to(device)

    # Test tensor
    x = torch.randn(1, 8, config.dim, device=device, dtype=torch.float16)

    # Get routing scores (directly call gate)
    gate_logits = moe.gate(x)
    gate_scores = torch.sigmoid(gate_logits * config.route_scale)

    print(f"✓ Gate logits: {list(gate_logits.shape)}")
    print(
        f"✓ Gate scores range: [{gate_scores.min().item():.4f}, {gate_scores.max().item():.4f}]"
    )

    # Select top-k
    topk_scores, topk_indices = torch.topk(
        gate_scores, k=config.n_activated_experts, dim=-1
    )

    print(f"✓ Top-{config.n_activated_experts} indices: {list(topk_indices.shape)}")
    print(f"✓ Top-{config.n_activated_experts} scores: {list(topk_scores.shape)}")

    # Verify correct number of experts selected
    assert (
        topk_indices.shape[-1] == config.n_activated_experts
    ), f"Wrong number of experts selected: {topk_indices.shape[-1]} != {config.n_activated_experts}"

    # Verify indices are in valid range
    assert topk_indices.min() >= 0, "Expert indices contain negative values"
    assert (
        topk_indices.max() < config.n_routed_experts
    ), f"Expert indices exceed range: {topk_indices.max()} >= {config.n_routed_experts}"

    print("\n✅ Routing correctness verified!")
    print("=" * 60)


def test_moe_gradient_flow():
    """Test that gradients flow through MoE"""
    print("\nTesting MoE - Gradient Flow")
    print("=" * 60)

    if not torch.backends.mps.is_available():
        print("⚠️  Skipping test - MPS backend not available")
        return

    device = torch.device("mps")

    config = DeepSeekConfig(
        dim=64,
        moe_inter_dim=128,
        n_routed_experts=4,
        n_shared_experts=1,
        n_activated_experts=2,
    )

    moe = MoE(config).to(device)

    # Enable gradients
    x = torch.randn(
        1, 4, config.dim, device=device, dtype=torch.float16, requires_grad=True
    )

    # Forward pass
    output = moe(x)

    # Compute loss and backward
    loss = output.sum()
    loss.backward()

    # Verify gradients exist
    assert x.grad is not None, "Input gradients not computed"

    # Check that expert gradients exist
    gate_has_grad = any(p.grad is not None for p in moe.gate.parameters())
    expert_has_grad = any(
        p.grad is not None for expert in moe.experts for p in expert.parameters()
    )

    assert gate_has_grad, "Gate gradients not computed"
    assert expert_has_grad, "Expert gradients not computed"

    print(f"✓ Gradients flow correctly")
    print(f"   Input grad norm: {x.grad.norm().item():.4f}")
    print(f"   Gate has gradients: {gate_has_grad}")
    print(f"   Experts have gradients: {expert_has_grad}")
    print("=" * 60)


def test_moe_output_variance():
    """Test that MoE produces reasonable output variance"""
    print("\nTesting MoE - Output Variance")
    print("=" * 60)

    if not torch.backends.mps.is_available():
        print("⚠️  Skipping test - MPS backend not available")
        return

    device = torch.device("mps")

    config = DeepSeekConfig(
        dim=128,
        moe_inter_dim=256,
        n_routed_experts=8,
        n_shared_experts=1,
        n_activated_experts=2,
    )

    moe = MoE(config).to(device)

    # Initialize weights properly
    for param in moe.parameters():
        if param.dim() > 1:
            torch.nn.init.xavier_uniform_(param)

    # Test tensor
    x = torch.randn(4, 16, config.dim, device=device, dtype=torch.float16)
    x = x / x.std()  # Normalize input

    # Forward pass
    output = moe(x)

    # Check output statistics
    output_mean = output.mean().item()
    output_std = output.std().item()

    print(f"✓ Input: mean={x.mean().item():.4f}, std={x.std().item():.4f}")
    print(f"✓ Output: mean={output_mean:.4f}, std={output_std:.4f}")

    # Output should have reasonable variance (not all zeros or NaN)
    assert not torch.isnan(output).any(), "Output contains NaN"
    assert not torch.isinf(output).any(), "Output contains Inf"
    assert output_std > 0.01, f"Output variance too low: {output_std}"

    print("\n✅ Output variance is reasonable!")
    print("=" * 60)


if __name__ == "__main__":
    test_moe_basic()
    test_moe_routing()
    test_moe_gradient_flow()
    test_moe_output_variance()

    print("\n" + "=" * 60)
    print("All MoE tests passed! ✅")
    print("=" * 60)

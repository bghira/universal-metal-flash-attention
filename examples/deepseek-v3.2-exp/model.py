"""
DeepSeek-V3.2-Exp Model Implementation

Adapted for Metal/Apple Silicon with PyTorch.
Focus: MLA (Multi-Latent Attention), DSA (Sparse Attention), MoE (Mixture of Experts)

Based on: https://github.com/deepseek-ai/DeepSeek-V3.2-Exp
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import DeepSeekConfig
from kernel import SparseIndexer


# =============================================================================
# Basic Layers (PyTorch Functional)
# =============================================================================


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(
        self, x: torch.Tensor, residual: Optional[torch.Tensor] = None
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with optional residual connection

        Args:
            x: Input tensor
            residual: Optional residual to add before normalization

        Returns:
            Normalized tensor, or (normalized, residual) if residual provided
        """
        dtype = x.dtype
        if residual is None:
            x = x.float()
            var = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(var + self.eps)
            return (self.weight * x).to(dtype)
        else:
            x = residual = x.float() + residual.float()
            var = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(var + self.eps)
            return (self.weight * x).to(dtype), residual.to(dtype)


class Linear(nn.Module):
    """Linear layer with optional FP8 quantization support"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Use FP16 for Metal (FP8 not supported on Apple Silicon)
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=dtype)
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class ParallelEmbedding(nn.Module):
    """
    Embedding layer (simplified for single-device Metal)

    Original supports distributed parallelism, but we use single device.
    """

    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.weight = nn.Parameter(torch.empty(vocab_size, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard embedding lookup"""
        return F.embedding(x, self.weight)


# =============================================================================
# MLA (Multi-Latent Attention) with Sparse Indexing
# =============================================================================


class Indexer(nn.Module):
    """
    Sparse attention indexer using Metal kernels

    Selects top-k indices for sparse attention (k=2048 for 671B model)
    """

    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.topk = config.index_topk

        # Indexer projections
        self.q_proj = Linear(config.dim, self.n_heads * self.head_dim)
        self.k_proj = Linear(config.dim, self.n_heads * self.head_dim)

        # Metal/MPS kernel for sparse indexing
        self.sparse_indexer = SparseIndexer(
            topk=self.topk,
            scale=1.0 / math.sqrt(self.head_dim)
        )

    def forward(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute sparse attention indices

        Args:
            q: Query tensor [batch, seq, dim]
            k: Key tensor [batch, seq, dim]

        Returns:
            indices: Top-k indices [batch, n_heads, seq, topk]
            scores: Attention scores for selected indices
        """
        batch, seq, _ = q.shape

        # Project Q and K for indexing
        q_index = self.q_proj(q)  # [batch, seq, n_heads * head_dim]
        k_index = self.k_proj(k)

        # Reshape to [batch, n_heads, seq, head_dim]
        q_index = q_index.view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)
        k_index = k_index.view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)

        # Use Metal kernel for index scoring + top-k selection
        topk_indices, topk_scores = self.sparse_indexer(q_index, k_index)

        return topk_indices, topk_scores


class MLA(nn.Module):
    """
    Multi-Latent Attention with sparse indexing

    Uses low-rank projections for KV compression and learned decompression.
    """

    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank

        # Query low-rank projection
        self.q_down = Linear(config.dim, config.q_lora_rank)
        self.q_up = Linear(config.q_lora_rank, config.n_heads * config.head_dim)

        # KV low-rank projection
        self.kv_down = Linear(config.dim, config.kv_lora_rank)

        # Sparse attention indexer
        self.indexer = Indexer(config)

        # Output projection
        self.o_proj = Linear(config.n_heads * config.v_head_dim, config.dim)

    def forward(
        self,
        x: torch.Tensor,
        kv_latent: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        MLA forward pass

        Args:
            x: Input tensor [batch, seq, dim]
            kv_latent: Compressed KV cache [batch, seq, kv_lora_rank]
            attention_mask: Optional attention mask

        Returns:
            Output tensor [batch, seq, dim]
        """
        batch, seq, _ = x.shape

        # Compress query to low-rank
        q_latent = self.q_down(x)  # [batch, seq, q_lora_rank]

        # Decompress to full Q
        q = self.q_up(q_latent)  # [batch, seq, n_heads * head_dim]
        q = q.view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)

        # TODO: Integrate MLA FFI for KV decompression
        # Try to import and use MLA FFI, fall back to placeholder if not available
        try:
            # Import MLA FFI (will be available after building pytorch-custom-op-ffi)
            from python.metal_sdpa_ffi import MlaContext

            # This would require initializing MLA context with decompression weights
            # For now, use placeholder until we have weights loaded
            raise ImportError("MLA FFI integration pending - need weight loading")

        except ImportError:
            # Placeholder: Use identity mapping (not correct, just for testing structure)
            # In production, kv_latent would be decompressed to full K, V via MLA FFI
            pass

        # Sparse attention indexing
        indices, index_scores = self.indexer(x, x)

        # TODO: Use MLA FFI to decompress KV and sparse attention
        # For now, placeholder: just return projection of input
        # This bypasses proper attention until MLA FFI is integrated
        placeholder_output = torch.randn(
            batch, seq, self.n_heads * self.config.v_head_dim,
            device=x.device, dtype=x.dtype
        )
        output = self.o_proj(placeholder_output)

        return output


# =============================================================================
# MoE (Mixture of Experts)
# =============================================================================


class MoE(nn.Module):
    """
    Mixture of Experts layer

    256 routed experts + 1 shared expert, 8 activated per token
    """

    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.config = config
        self.n_routed_experts = config.n_routed_experts
        self.n_shared_experts = config.n_shared_experts
        self.n_activated_experts = config.n_activated_experts
        self.route_scale = config.route_scale

        # Expert gating
        self.gate = Linear(config.dim, config.n_routed_experts, bias=False)

        # Routed experts (simplified - use nn.ModuleList)
        self.experts = nn.ModuleList([
            self._make_expert(config) for _ in range(config.n_routed_experts)
        ])

        # Shared expert
        self.shared_expert = self._make_expert(config)

    def _make_expert(self, config: DeepSeekConfig) -> nn.Module:
        """Create a single expert (2-layer MLP)"""
        return nn.Sequential(
            Linear(config.dim, config.moe_inter_dim),
            nn.SiLU(),
            Linear(config.moe_inter_dim, config.dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        MoE forward pass with top-k routing

        Args:
            x: Input tensor [batch, seq, dim]

        Returns:
            Output tensor [batch, seq, dim]
        """
        batch, seq, dim = x.shape

        # Compute routing scores (sigmoid with scale)
        gate_logits = self.gate(x)  # [batch, seq, n_routed_experts]
        gate_scores = torch.sigmoid(gate_logits * self.route_scale)

        # Select top-k experts per token
        topk_scores, topk_indices = torch.topk(
            gate_scores, k=self.n_activated_experts, dim=-1
        )

        # Normalize scores
        topk_scores = topk_scores / (topk_scores.sum(dim=-1, keepdim=True) + 1e-9)

        # Compute expert outputs (simplified - should use grouped batching)
        output = torch.zeros_like(x)

        # Shared expert (always active)
        shared_out = self.shared_expert(x)
        output = output + shared_out

        # Routed experts (top-k)
        for i in range(self.n_activated_experts):
            expert_idx = topk_indices[..., i]  # [batch, seq]
            expert_score = topk_scores[..., i : i + 1]  # [batch, seq, 1]

            # For simplicity, apply all experts and select
            # (Production code should use expert parallelism)
            for expert_id in range(self.n_routed_experts):
                mask = (expert_idx == expert_id).unsqueeze(-1)  # [batch, seq, 1]
                expert_out = self.experts[expert_id](x)
                output = output + (expert_out * expert_score * mask.float())

        return output


# =============================================================================
# Transformer Block
# =============================================================================


class DecoderLayer(nn.Module):
    """Single decoder layer with MLA and optional MoE"""

    def __init__(self, config: DeepSeekConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.use_moe = layer_idx >= config.n_dense_layers

        # Pre-attention norm
        self.input_norm = RMSNorm(config.dim)

        # Multi-Latent Attention
        self.attn = MLA(config)

        # Pre-FFN norm
        self.post_attn_norm = RMSNorm(config.dim)

        # FFN: either MoE or standard
        if self.use_moe:
            self.ffn = MoE(config)
        else:
            # Standard FFN for first N layers
            self.ffn = nn.Sequential(
                Linear(config.dim, config.inter_dim),
                nn.SiLU(),
                Linear(config.inter_dim, config.dim),
            )

    def forward(
        self,
        x: torch.Tensor,
        kv_latent: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Decoder layer forward pass

        Args:
            x: Input tensor [batch, seq, dim]
            kv_latent: Compressed KV cache
            attention_mask: Optional attention mask

        Returns:
            Output tensor [batch, seq, dim]
        """
        # Attention with residual
        normed, residual = self.input_norm(x, x)
        attn_out = self.attn(normed, kv_latent, attention_mask)
        x = attn_out + residual

        # FFN with residual
        normed, residual = self.post_attn_norm(x, x)
        ffn_out = self.ffn(normed)
        x = ffn_out + residual

        return x


# =============================================================================
# Full Model
# =============================================================================


class DeepSeekModel(nn.Module):
    """DeepSeek-V3.2-Exp model (671B parameters)"""

    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.embed_tokens = ParallelEmbedding(config.vocab_size, config.dim)

        # Decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(config, layer_idx) for layer_idx in range(config.n_layers)
        ])

        # Final norm
        self.norm = RMSNorm(config.dim)

        # LM head
        self.lm_head = Linear(config.dim, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Full model forward pass

        Args:
            input_ids: Input token IDs [batch, seq]
            attention_mask: Optional attention mask

        Returns:
            Logits [batch, seq, vocab_size]
        """
        # Embed tokens
        x = self.embed_tokens(input_ids)

        # TODO: Compress to KV latent (placeholder)
        kv_latent = torch.randn(
            x.shape[0], x.shape[1], self.config.kv_lora_rank,
            device=x.device, dtype=x.dtype
        )

        # Pass through decoder layers
        for layer in self.layers:
            x = layer(x, kv_latent, attention_mask)

        # Final norm
        x = self.norm(x)

        # LM head
        logits = self.lm_head(x)

        return logits


if __name__ == "__main__":
    from config import load_config

    # Test model instantiation
    config = load_config("671B")
    print(f"Configuration:\n{config}\n")

    # Create model (will be large!)
    print("Creating DeepSeek-V3.2-Exp model...")
    model = DeepSeekModel(config)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n✅ Model created successfully")
    print(f"   Total parameters: {n_params:,} ({n_params / 1e9:.1f}B)")
    print(f"   Layers: {config.n_layers}")
    print(f"   MoE layers: {config.n_layers - config.n_dense_layers}")

"""
DeepSeek-V3.2-Exp Model Configuration

Loads and manages configuration for the 671B parameter model.
Adapted for Metal/Apple Silicon with PyTorch.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass
class DeepSeekConfig:
    """Configuration for DeepSeek-V3.2-Exp model"""

    # Vocabulary
    vocab_size: int = 129280

    # Model dimensions
    dim: int = 7168  # Hidden dimension
    inter_dim: int = 18432  # FFN intermediate dimension
    moe_inter_dim: int = 2048  # MoE expert intermediate dimension

    # Architecture
    n_layers: int = 61
    n_dense_layers: int = 3  # First N layers are dense (no MoE)
    n_heads: int = 128

    # MoE configuration
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    n_activated_experts: int = 8  # Per token
    n_expert_groups: int = 8
    n_limited_groups: int = 4
    route_scale: float = 2.5
    score_func: Literal["sigmoid", "softmax"] = "sigmoid"

    # MLA (Multi-Latent Attention) configuration
    q_lora_rank: int = 1536  # Query low-rank dimension
    kv_lora_rank: int = 512  # Key-Value low-rank dimension
    qk_nope_head_dim: int = 128  # QK head dim without RoPE
    qk_rope_head_dim: int = 64  # QK head dim with RoPE
    v_head_dim: int = 128  # Value head dimension

    # Sparse attention indexer
    index_n_heads: int = 64  # Indexer attention heads
    index_head_dim: int = 128  # Indexer head dimension
    index_topk: int = 2048  # Top-k indices to select

    # Precision (Metal uses FP16, not FP8)
    dtype: Literal["fp8", "fp16", "bf16"] = "fp16"
    scale_fmt: str = "ue8m0"

    @property
    def head_dim(self) -> int:
        """Total head dimension (RoPE + no-RoPE)"""
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def total_kv_dim(self) -> int:
        """Total KV dimension after decompression"""
        return self.n_heads * self.v_head_dim

    @property
    def total_q_dim(self) -> int:
        """Total Q dimension after decompression"""
        return self.n_heads * self.head_dim

    @property
    def compression_ratio(self) -> float:
        """KV cache compression ratio"""
        return self.total_kv_dim / self.kv_lora_rank

    @classmethod
    def from_json(cls, config_path: str | Path) -> "DeepSeekConfig":
        """Load configuration from JSON file"""
        config_path = Path(config_path)
        with open(config_path, "r") as f:
            config_dict = json.load(f)

        # Override dtype to fp16 for Metal
        if config_dict.get("dtype") == "fp8":
            print("⚠️  Converting dtype from fp8 to fp16 for Metal compatibility")
            config_dict["dtype"] = "fp16"

        return cls(**config_dict)

    def to_json(self, output_path: str | Path) -> None:
        """Save configuration to JSON file"""
        output_path = Path(output_path)
        config_dict = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        with open(output_path, "w") as f:
            json.dump(config_dict, f, indent=4)

    def __repr__(self) -> str:
        return (
            f"DeepSeekConfig(\n"
            f"  model_size=671B,\n"
            f"  layers={self.n_layers},\n"
            f"  hidden_dim={self.dim},\n"
            f"  heads={self.n_heads},\n"
            f"  head_dim={self.head_dim},\n"
            f"  q_lora_rank={self.q_lora_rank},\n"
            f"  kv_lora_rank={self.kv_lora_rank},\n"
            f"  compression={self.compression_ratio:.1f}x,\n"
            f"  moe_experts={self.n_routed_experts},\n"
            f"  activated={self.n_activated_experts},\n"
            f"  sparse_index_topk={self.index_topk},\n"
            f"  dtype={self.dtype}\n"
            f")"
        )


def load_config(config_name: str = "671B") -> DeepSeekConfig:
    """
    Load DeepSeek configuration by name

    Args:
        config_name: Configuration name (e.g., "671B")

    Returns:
        DeepSeekConfig instance
    """
    config_dir = Path(__file__).parent
    config_file = config_dir / f"config_{config_name}_v3.2.json"

    if not config_file.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_file}\n"
            f"Available configs: {list(config_dir.glob('config_*.json'))}"
        )

    return DeepSeekConfig.from_json(config_file)


if __name__ == "__main__":
    # Test configuration loading
    config = load_config("671B")
    print(config)
    print(f"\n✅ Configuration loaded successfully")
    print(f"   Head dimension: {config.head_dim}")
    print(f"   Total Q dim: {config.total_q_dim}")
    print(f"   Total KV dim: {config.total_kv_dim}")
    print(f"   KV compression ratio: {config.compression_ratio:.1f}x")

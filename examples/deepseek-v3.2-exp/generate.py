"""
DeepSeek-V3.2-Exp Text Generation

Demonstrates end-to-end inference with the 671B model on Metal/Apple Silicon.
"""

import argparse
from typing import List, Optional

import torch
import torch.nn.functional as F

from config import load_config, DeepSeekConfig
from model import DeepSeekModel


class TextGenerator:
    """Text generation with DeepSeek-V3.2-Exp"""

    def __init__(
        self,
        model: DeepSeekModel,
        config: DeepSeekConfig,
        device: torch.device,
    ):
        self.model = model
        self.config = config
        self.device = device

        # Move model to device
        self.model = self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
    ) -> torch.Tensor:
        """
        Generate text tokens autoregressively

        Args:
            input_ids: Input token IDs [batch, seq]
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k filtering (None = no filtering)
            top_p: Nucleus sampling threshold (None = no nucleus sampling)

        Returns:
            Generated token IDs [batch, seq + max_new_tokens]
        """
        batch_size, seq_len = input_ids.shape
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            # Forward pass
            logits = self.model(generated)  # [batch, seq, vocab_size]

            # Get logits for last token
            next_token_logits = logits[:, -1, :]  # [batch, vocab_size]

            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Apply top-k filtering
            if top_k is not None:
                indices_to_remove = next_token_logits < torch.topk(
                    next_token_logits, top_k
                )[0][..., -1, None]
                next_token_logits[indices_to_remove] = float("-inf")

            # Apply nucleus (top-p) sampling
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(
                    next_token_logits, descending=True
                )
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = float("-inf")

            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)

        return generated

    def generate_text(
        self,
        prompt_tokens: List[int],
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> List[int]:
        """
        Generate text from prompt tokens

        Args:
            prompt_tokens: Input prompt as list of token IDs
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold

        Returns:
            Generated token IDs (prompt + generated)
        """
        # Convert to tensor
        input_ids = torch.tensor([prompt_tokens], device=self.device)

        # Generate
        output_ids = self.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        return output_ids[0].tolist()


def demo_generation():
    """Demonstrate text generation with random weights"""

    print("DeepSeek-V3.2-Exp Generation Demo")
    print("=" * 60)

    # Check MPS availability
    if not torch.backends.mps.is_available():
        print("❌ MPS backend not available - Metal required")
        return

    device = torch.device("mps")
    print(f"✓ Using device: {device}\n")

    # Load configuration
    print("Loading configuration...")
    config = load_config("671B")
    print(f"✓ Configuration loaded\n{config}\n")

    # Create model (warning: this will allocate ~900GB for full model!)
    print("Creating model (this may take a while for 671B model)...")
    print("⚠️  Note: Full model requires ~900GB RAM. Using reduced config for demo.\n")

    # For demo, use smaller config
    demo_config = DeepSeekConfig(
        vocab_size=1000,  # Reduced from 129280
        dim=512,  # Reduced from 7168
        inter_dim=2048,  # Reduced from 18432
        moe_inter_dim=512,  # Reduced from 2048
        n_layers=4,  # Reduced from 61
        n_dense_layers=2,  # Reduced from 3
        n_heads=8,  # Reduced from 128
        n_routed_experts=16,  # Reduced from 256
        n_shared_experts=1,
        n_activated_experts=4,  # Reduced from 8
        n_expert_groups=4,  # Reduced from 8
        q_lora_rank=256,  # Reduced from 1536
        kv_lora_rank=128,  # Reduced from 512
        qk_nope_head_dim=64,  # Reduced from 128
        qk_rope_head_dim=32,  # Reduced from 64
        v_head_dim=64,  # Reduced from 128
        index_n_heads=4,  # Reduced from 64
        index_head_dim=64,  # Reduced from 128
        index_topk=64,  # Reduced from 2048
    )

    print(f"Using demo config:\n{demo_config}\n")

    model = DeepSeekModel(demo_config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created: {n_params:,} parameters ({n_params / 1e6:.1f}M)\n")

    # Initialize weights randomly (normally would load from checkpoint)
    print("Initializing random weights...")
    for param in model.parameters():
        if param.dim() > 1:
            torch.nn.init.xavier_uniform_(param)
        else:
            torch.nn.init.zeros_(param)
    print("✓ Weights initialized\n")

    # Create generator
    generator = TextGenerator(model, demo_config, device)

    # Generate text
    print("Generating text...")
    print("-" * 60)

    # Random prompt tokens
    prompt_tokens = [42, 123, 456, 789]
    print(f"Prompt tokens: {prompt_tokens}")

    # Generate
    generated_tokens = generator.generate_text(
        prompt_tokens,
        max_new_tokens=20,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
    )

    print(f"Generated tokens: {generated_tokens}")
    print(f"Generated {len(generated_tokens) - len(prompt_tokens)} new tokens")
    print("-" * 60)

    print("\n✅ Generation complete!")
    print("\n" + "=" * 60)
    print("Next steps:")
    print("  1. Load pre-trained weights from HuggingFace")
    print("  2. Add tokenizer integration")
    print("  3. Implement KV cache for efficient generation")
    print("  4. Add MLA FFI integration for compressed KV cache")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="DeepSeek-V3.2-Exp text generation"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo with random weights",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Text prompt for generation",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )

    args = parser.parse_args()

    if args.demo:
        demo_generation()
    else:
        print("Error: Either --demo or --prompt must be specified")
        parser.print_help()


if __name__ == "__main__":
    main()

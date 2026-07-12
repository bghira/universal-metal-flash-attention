"""
Pytest configuration and fixtures for Metal SDPA FFI tests.
"""

import gc
import sys
from pathlib import Path

import pytest
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import the extension
try:
    import metal_sdpa_extension

    METAL_EXTENSION_AVAILABLE = True
except ImportError:
    METAL_EXTENSION_AVAILABLE = False

# Detect native bfloat16 support once per session. The generated Metal source
# always emits bfloat paths; on toolchains without native bfloat a half-precision
# fallback lets kernels compile but would corrupt actual bfloat16 *data*, so
# tests that feed BFloat16 tensors must be skipped there.
NATIVE_BFLOAT_AVAILABLE = False
if METAL_EXTENSION_AVAILABLE and torch.backends.mps.is_available():
    try:
        NATIVE_BFLOAT_AVAILABLE = bool(metal_sdpa_extension.has_native_bfloat())
    except Exception:
        NATIVE_BFLOAT_AVAILABLE = False


@pytest.fixture(scope="session")
def metal_available():
    """Check if Metal/MPS is available."""
    return torch.backends.mps.is_available() and METAL_EXTENSION_AVAILABLE


@pytest.fixture
def metal_device():
    """Get MPS device if available, skip test otherwise."""
    if not torch.backends.mps.is_available():
        pytest.skip("MPS device not available")
    if not METAL_EXTENSION_AVAILABLE:
        pytest.skip("Metal SDPA extension not available")
    return torch.device("mps")


@pytest.fixture(autouse=True)
def cleanup_metal():
    """Clean up Metal resources before and after each test."""
    # Clear MPS cache before test to reclaim memory from prior tests.
    # On CI runners (virtualized M1, ~8 GiB), accumulated MPS allocations
    # from earlier tests can exhaust GPU memory before later tests start.
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()
    yield
    # Clean up after test
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()


@pytest.fixture(autouse=True)
def skip_bfloat_tests_without_native_bfloat(request):
    """Skip @pytest.mark.requires_bfloat tests on toolchains lacking native
    bfloat (the half fallback compiles but would corrupt bf16 data)."""
    marker = request.node.get_closest_marker("requires_bfloat")
    if marker is not None and not NATIVE_BFLOAT_AVAILABLE:
        pytest.skip("native bfloat16 unavailable on this Metal toolchain")


@pytest.fixture
def dtype_combinations():
    """Generate all valid dtype combinations for testing."""
    dtypes = []

    # Basic dtypes
    if torch.backends.mps.is_available():
        dtypes.extend(
            [
                torch.float32,
                torch.float16,
            ]
        )

        # BFloat16 support check
        try:
            test_tensor = torch.randn(1, device="mps", dtype=torch.bfloat16)
            dtypes.append(torch.bfloat16)
        except:
            pass

    return dtypes


@pytest.fixture
def flux_tensor_shapes():
    """Common FLUX model tensor shapes for testing."""
    return [
        # (batch, heads, seq_len, head_dim)
        (1, 12, 77, 64),  # FLUX text encoder typical shape
        (1, 24, 1536, 128),  # FLUX main transformer shape
        (2, 24, 1536, 128),  # Batched FLUX
        (1, 24, 4096, 128),  # Larger sequence length
        (1, 48, 2048, 64),  # Different head configuration
    ]


@pytest.fixture
def basic_tensor_shapes():
    """Basic tensor shapes for quick testing."""
    return [
        # (batch, heads, seq_len, head_dim)
        (1, 1, 64, 64),  # Simplest case
        (1, 4, 128, 64),  # Multi-head
        (2, 8, 256, 64),  # Batched multi-head
        (1, 1, 512, 128),  # Larger dimensions
    ]


@pytest.fixture
def create_test_tensors():
    """Factory fixture to create test tensors with specified properties."""

    def _create(
        batch_size=1,
        num_heads=1,
        seq_len=64,
        head_dim=64,
        dtype=torch.float32,
        device="mps",
        layout="flux",
    ):
        """
        Create Q, K, V tensors for testing.

        Args:
            layout: "flux" for [B,H,S,D] or "metal" for [B,S,H,D]
        """
        if layout == "flux":
            shape = (batch_size, num_heads, seq_len, head_dim)
        else:  # metal
            shape = (batch_size, seq_len, num_heads, head_dim)

        # Create tensors with reproducible random values
        torch.manual_seed(42)
        q = torch.randn(shape, dtype=dtype, device=device)
        k = torch.randn(shape, dtype=dtype, device=device)
        v = torch.randn(shape, dtype=dtype, device=device)

        # Scale to reasonable values to avoid numerical issues
        q = q * 0.1
        k = k * 0.1
        v = v * 0.1

        return q, k, v

    return _create


@pytest.fixture
def reference_attention():
    """Compute reference attention using PyTorch's implementation."""

    def _compute(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
        """Compute attention using PyTorch's F.scaled_dot_product_attention."""
        with torch.inference_mode():
            return torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
            )

    return _compute


@pytest.fixture
def tolerance_for_dtype():
    """Get appropriate tolerance values for different dtypes."""

    def _get_tolerance(dtype):
        if dtype == torch.float32:
            return {"rtol": 1e-5, "atol": 1e-6}
        elif dtype == torch.float16:
            return {"rtol": 1e-3, "atol": 1e-3}
        elif dtype == torch.bfloat16:
            return {"rtol": 1e-2, "atol": 1e-2}
        else:
            return {"rtol": 1e-4, "atol": 1e-4}

    return _get_tolerance


@pytest.fixture
def check_numerical_accuracy():
    """Helper to check numerical accuracy between two tensors."""

    def _check(actual, expected, dtype=None, name="Output"):
        """
        Check if actual matches expected within tolerance.

        Returns dict with statistics.
        """
        if dtype is None:
            dtype = actual.dtype

        # Get appropriate tolerances
        if dtype == torch.float32:
            rtol, atol = 1e-5, 1e-6
        elif dtype == torch.float16:
            rtol, atol = 1e-3, 1e-3
        elif dtype == torch.bfloat16:
            rtol, atol = 1e-2, 1e-2
        else:
            rtol, atol = 1e-4, 1e-4

        # Compute statistics
        diff = (actual - expected).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        # Check for NaN/Inf
        has_nan = torch.isnan(actual).any().item()
        has_inf = torch.isinf(actual).any().item()

        # Check if within tolerance
        matches = torch.allclose(actual, expected, rtol=rtol, atol=atol)

        return {
            "matches": matches,
            "max_diff": max_diff,
            "mean_diff": mean_diff,
            "has_nan": has_nan,
            "has_inf": has_inf,
            "rtol": rtol,
            "atol": atol,
            "dtype": str(dtype),
            "name": name,
        }

    return _check


# Markers for test categorization
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "metal: Tests requiring Metal/MPS device")
    config.addinivalue_line(
        "markers",
        "gpu: Tests requiring a working Metal GPU (excluded on CI via -m 'not gpu')",
    )
    config.addinivalue_line(
        "markers",
        "requires_bfloat: Tests feeding BFloat16 data; skipped when native bfloat is unavailable",
    )
    config.addinivalue_line("markers", "dtype: Tests for dtype compatibility")
    config.addinivalue_line("markers", "layout: Tests for layout conversions")
    config.addinivalue_line("markers", "quantization: Tests for quantization features")
    config.addinivalue_line("markers", "flux: Tests specific to FLUX model shapes")
    config.addinivalue_line("markers", "slow: Slow tests that can be skipped")
    config.addinivalue_line("markers", "memory: Tests for memory management")

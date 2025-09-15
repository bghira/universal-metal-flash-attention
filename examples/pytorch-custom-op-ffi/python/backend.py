"""
Metal SDPA Backend Registration and Context Management
"""

import threading
import warnings
from contextlib import contextmanager
from typing import Optional, Tuple

import torch

# Global state
_backend_registered = False
_registration_lock = threading.Lock()

try:
    # Import the compiled extension
    import metal_sdpa_extension as _ext

    _EXTENSION_AVAILABLE = True
except ImportError as e:
    _EXTENSION_AVAILABLE = False
    _import_error = e


def _ensure_extension_available():
    """Ensure the C++ extension is available."""
    if not _EXTENSION_AVAILABLE:
        raise RuntimeError(
            f"Metal SDPA extension not available. Import error: {_import_error}\n"
            "Please compile the extension using: python setup.py install"
        )


def is_metal_sdpa_available() -> bool:
    """Check if Metal SDPA backend is available."""
    if not _EXTENSION_AVAILABLE:
        return False

    try:
        return _ext.is_metal_available()
    except Exception:
        return False


def metal_sdpa_version() -> Optional[Tuple[int, int, int]]:
    """Get Metal Flash Attention version."""
    if not is_metal_sdpa_available():
        return None

    try:
        return _ext.get_version()
    except Exception:
        return None


def register_metal_sdpa_backend() -> None:
    """
    Register Metal SDPA backend with PyTorch.

    This enables PyTorch to use Metal Flash Attention for
    torch.nn.functional.scaled_dot_product_attention when using
    the PrivateUse1 backend.
    """
    global _backend_registered

    with _registration_lock:
        if _backend_registered:
            warnings.warn("Metal SDPA backend already registered", UserWarning)
            return

        _ensure_extension_available()

        if not is_metal_sdpa_available():
            raise RuntimeError("Metal is not available on this device")

        # Register PrivateUse1 backend as "metal_sdpa"
        torch.utils.rename_privateuse1_backend("metal_sdpa")

        # Generate backend-specific methods
        torch.utils.generate_methods_for_privateuse1_backend()

        # Register the C++ backend
        _ext.register_backend()

        _backend_registered = True

        # Print version info
        version = metal_sdpa_version()
        if version:
            major, minor, patch = version
            print(f"✅ Metal SDPA backend registered (MFA v{major}.{minor}.{patch})")
        else:
            print("✅ Metal SDPA backend registered")


def unregister_metal_sdpa_backend() -> None:
    """Unregister Metal SDPA backend from PyTorch."""
    global _backend_registered

    with _registration_lock:
        if not _backend_registered:
            warnings.warn("Metal SDPA backend not currently registered", UserWarning)
            return

        if _EXTENSION_AVAILABLE:
            _ext.unregister_backend()

        _backend_registered = False
        print("Metal SDPA backend unregistered")


@contextmanager
def use_metal_sdpa():
    """
    Context manager to temporarily enable Metal SDPA backend.

    Usage:
        with use_metal_sdpa():
            output = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    """
    if not _backend_registered:
        register_metal_sdpa_backend()

    # Get current device
    device = torch.device("metal_sdpa")

    # Create a context that uses metal_sdpa device
    try:
        # Move tensors to metal_sdpa device during context
        yield device
    except Exception as e:
        print(f"Error in Metal SDPA context: {e}")
        raise


class MetalSDPAContext:
    """
    Context manager for fine-grained control over Metal SDPA usage.

    Usage:
        with MetalSDPAContext() as ctx:
            q_metal = ctx.to_device(q)
            k_metal = ctx.to_device(k)
            v_metal = ctx.to_device(v)

            output = torch.nn.functional.scaled_dot_product_attention(
                q_metal, k_metal, v_metal, is_causal=True
            )

            result = ctx.to_cpu(output)
    """

    def __init__(self, auto_register: bool = True):
        self.auto_register = auto_register
        self.device = None

    def __enter__(self):
        if self.auto_register and not _backend_registered:
            register_metal_sdpa_backend()

        self.device = torch.device("metal_sdpa")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.device = None

    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to metal_sdpa device."""
        if self.device is None:
            raise RuntimeError("Context not active")
        return tensor.to(self.device)

    def to_cpu(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor back to CPU."""
        return tensor.cpu()

    def direct_call(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Direct call to Metal SDPA without device management.

        This bypasses PyTorch's dispatcher and calls Metal Flash Attention directly.
        Useful for debugging or when you want explicit control.
        """
        _ensure_extension_available()

        return _ext.metal_scaled_dot_product_attention(
            query, key, value, attn_mask, dropout_p, is_causal, scale
        )


# Backend configuration namespace (mimics torch.backends style)
class MetalSDPABackendConfig:
    """Configuration for Metal SDPA backend."""

    def __init__(self):
        self._enabled = False

    @property
    def enabled(self) -> bool:
        """Whether Metal SDPA backend is enabled globally."""
        return _backend_registered

    @enabled.setter
    def enabled(self, value: bool):
        """Enable/disable Metal SDPA backend globally."""
        if value and not _backend_registered:
            register_metal_sdpa_backend()
        elif not value and _backend_registered:
            unregister_metal_sdpa_backend()

    @property
    def available(self) -> bool:
        """Whether Metal SDPA backend is available."""
        return is_metal_sdpa_available()

    @property
    def version(self) -> Optional[Tuple[int, int, int]]:
        """Metal Flash Attention version."""
        return metal_sdpa_version()


# Add backend configuration to torch.backends namespace
def _install_backend_config():
    """Install backend configuration in torch.backends namespace."""
    if not hasattr(torch.backends, "metal_sdpa"):
        torch.backends.metal_sdpa = MetalSDPABackendConfig()


# Install on import
_install_backend_config()

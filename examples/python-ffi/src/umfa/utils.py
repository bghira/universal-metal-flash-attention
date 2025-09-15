"""
Utility functions for Universal Metal Flash Attention.

Provides convenience functions for common operations and compatibility checks.
"""

import ctypes
from typing import Tuple

from ._ffi import _check_error, _lib
from .core import MFAContext


def is_metal_available() -> bool:
    """
    Check if Metal is available on this device.

    Returns:
        True if Metal device is available, False otherwise

    Example:
        >>> if is_metal_available():
        ...     print("Metal Flash Attention is supported!")
    """
    return bool(_lib.mfa_is_device_supported())


def get_version() -> Tuple[int, int, int]:
    """
    Get the version of the MFA library.

    Returns:
        Tuple of (major, minor, patch) version numbers

    Example:
        >>> major, minor, patch = get_version()
        >>> print(f"MFA version: {major}.{minor}.{patch}")
    """
    major = ctypes.c_int()
    minor = ctypes.c_int()
    patch = ctypes.c_int()

    _lib.mfa_get_version(ctypes.byref(major), ctypes.byref(minor), ctypes.byref(patch))

    return (major.value, minor.value, patch.value)


def create_context() -> MFAContext:
    """
    Create a new MFA context.

    This is equivalent to MFAContext() but provides a more functional interface.

    Returns:
        New MFA context

    Example:
        >>> ctx = create_context()
        >>> # Use context...
        >>> ctx.close()
    """
    return MFAContext()


def check_requirements() -> bool:
    """
    Check if all requirements are met for using MFA.

    Performs comprehensive checks including:
    - Metal availability
    - Library loading
    - Basic functionality

    Returns:
        True if all requirements are met

    Raises:
        RuntimeError: If requirements are not met
    """
    # Check Metal availability
    if not is_metal_available():
        raise RuntimeError("Metal is not available on this device")

    # Try to create a context
    try:
        with create_context() as ctx:
            if not ctx:
                raise RuntimeError("Failed to create MFA context")
    except Exception as e:
        raise RuntimeError(f"MFA context creation failed: {e}")

    return True


def print_system_info():
    """
    Print system information relevant to MFA.

    Displays:
    - Metal availability
    - MFA version
    - Context creation status

    Example:
        >>> print_system_info()
        Universal Metal Flash Attention - System Info
        =============================================
        Metal Available: True
        MFA Version: 1.0.0
        Context Creation: OK
    """
    print("Universal Metal Flash Attention - System Info")
    print("=" * 45)

    # Metal availability
    metal_status = "✓ Available" if is_metal_available() else "✗ Not Available"
    print(f"Metal Support: {metal_status}")

    # Version
    try:
        major, minor, patch = get_version()
        print(f"MFA Version: {major}.{minor}.{patch}")
    except Exception as e:
        print(f"Version: Error - {e}")

    # Context test
    try:
        with create_context():
            print("Context Test: ✓ OK")
    except Exception as e:
        print(f"Context Test: ✗ Failed - {e}")

    print()


if __name__ == "__main__":
    print_system_info()

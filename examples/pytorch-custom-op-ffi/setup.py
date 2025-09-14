#!/usr/bin/env python3

import os
import sys
import subprocess
from pathlib import Path

from pybind11.setup_helpers import build_ext
from pybind11 import get_cmake_dir
import pybind11
from setuptools import setup
import torch
from torch.utils import cpp_extension

# Get the directory containing this setup.py
CURRENT_DIR = Path(__file__).parent
ROOT_DIR = CURRENT_DIR.parent.parent  # Go up to universal-metal-flash-attention root

# Define the extension module using PyTorch's CppExtension
ext_modules = [
    cpp_extension.CppExtension(
        "metal_sdpa_extension",
        sources=[
            str(CURRENT_DIR / "src" / "metal_sdpa_backend.cpp"),
            str(CURRENT_DIR / "src" / "python_bindings.cpp"),
        ],
        include_dirs=[
            str(CURRENT_DIR / "include"),
            # Add paths to your Swift FFI headers if needed
            str(ROOT_DIR / "include"),
        ],
        library_dirs=[
            # Add library search paths
            str(ROOT_DIR / ".build" / "arm64-apple-macosx" / "debug"),
            str(ROOT_DIR / ".build" / "arm64-apple-macosx" / "release"),
        ],
        language="c++",
    ),
]


class CustomBuildExt(cpp_extension.BuildExtension):
    """Custom build extension to handle Swift library linking."""

    def build_extension(self, ext):
        # Check if Swift library exists
        swift_lib_debug = (
            ROOT_DIR / ".build" / "arm64-apple-macosx" / "debug" / "libMFAFFI.dylib"
        )
        swift_lib_release = (
            ROOT_DIR / ".build" / "arm64-apple-macosx" / "release" / "libMFAFFI.dylib"
        )

        swift_lib_path = None
        if swift_lib_release.exists():
            swift_lib_path = swift_lib_release
        elif swift_lib_debug.exists():
            swift_lib_path = swift_lib_debug

        if swift_lib_path is None:
            print("⚠️  Swift FFI library not found. Building Swift package first...")
            self._build_swift_package()

            # Check again
            if swift_lib_release.exists():
                swift_lib_path = swift_lib_release
            elif swift_lib_debug.exists():
                swift_lib_path = swift_lib_debug
            else:
                raise RuntimeError("Failed to build Swift FFI library")

        # Add Swift library to linking using full path
        ext.extra_objects = getattr(ext, "extra_objects", [])
        ext.extra_objects.append(str(swift_lib_path))

        # Add Metal framework on macOS
        if sys.platform == "darwin":
            ext.extra_link_args = getattr(ext, "extra_link_args", [])
            ext.extra_link_args.extend(
                [
                    "-framework",
                    "Metal",
                    "-framework",
                    "MetalKit",
                    "-framework",
                    "Foundation",
                ]
            )
            # Fix rpath for Swift library and PyTorch libraries
            pytorch_lib_path = Path(torch.__file__).parent / "lib"
            ext.extra_link_args.extend(
                [
                    "-Wl,-rpath,"
                    + str(ROOT_DIR / ".build" / "arm64-apple-macosx" / "release"),
                    "-Wl,-rpath,"
                    + str(ROOT_DIR / ".build" / "arm64-apple-macosx" / "debug"),
                    "-Wl,-rpath," + str(pytorch_lib_path),
                ]
            )

        # Proceed with normal build
        super().build_extension(ext)

    def _build_swift_package(self):
        """Build the Swift package to generate FFI library."""
        print("Building Swift package for FFI...")

        try:
            # Build Swift package in release mode
            subprocess.run(
                ["swift", "build", "--configuration", "release", "--product", "MFAFFI"],
                cwd=ROOT_DIR,
                check=True,
            )
            print("✅ Swift package built successfully")

        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to build Swift package: {e}")

            # Try debug build as fallback
            try:
                subprocess.run(
                    [
                        "swift",
                        "build",
                        "--configuration",
                        "debug",
                        "--product",
                        "MFAFFI",
                    ],
                    cwd=ROOT_DIR,
                    check=True,
                )
                print("✅ Swift package built successfully (debug)")

            except subprocess.CalledProcessError as e2:
                raise RuntimeError(f"Failed to build Swift package: {e2}")


def main():
    """Main setup function."""

    # Check for required tools
    if sys.platform == "darwin":
        # Check for Swift compiler
        try:
            result = subprocess.run(
                ["swift", "--version"], capture_output=True, text=True
            )
            if result.returncode != 0:
                raise RuntimeError("Swift compiler not found")
        except FileNotFoundError:
            raise RuntimeError(
                "Swift compiler not found. Please install Xcode or Swift toolchain."
            )
    else:
        raise RuntimeError("This extension currently only supports macOS with Metal")

    # Setup configuration
    setup(
        name="pytorch-metal-sdpa-backend",
        version="0.1.0",
        author="bghira",
        description="PyTorch Custom SDPA Backend with Metal Flash Attention",
        long_description=(
            (CURRENT_DIR / "README.md").read_text()
            if (CURRENT_DIR / "README.md").exists()
            else ""
        ),
        long_description_content_type="text/markdown",
        url="https://github.com/bghira/universal-metal-flash-attention",
        packages=["pytorch_custom_op_ffi"],
        package_dir={"pytorch_custom_op_ffi": "python"},
        ext_modules=ext_modules,
        cmdclass={"build_ext": CustomBuildExt},
        install_requires=[
            "torch>=2.0.0",
            "numpy",
        ],
        python_requires=">=3.8",
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Developers",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: C++",
            "Programming Language :: Python :: Implementation :: CPython",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
        zip_safe=False,
    )


if __name__ == "__main__":
    main()

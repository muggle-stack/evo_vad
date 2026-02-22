"""
Evo VAD Python Module Setup

Build and install:
    pip install .

Development install:
    pip install -e .

Build wheel:
    pip wheel . -w dist/
"""

import os
import sys
import subprocess
from pathlib import Path

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    """CMake extension for pybind11 module"""

    def __init__(self, name, source_dir=""):
        super().__init__(name, sources=[])
        self.source_dir = os.path.abspath(source_dir)


class CMakeBuild(build_ext):
    """Build extension using CMake"""

    def build_extension(self, ext):
        ext_dir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        build_dir = os.path.join(ext.source_dir, "build")

        # Create build directory
        os.makedirs(build_dir, exist_ok=True)

        # Clean CMakeCache.txt if source directory changed (avoid cache conflicts)
        cache_file = os.path.join(build_dir, "CMakeCache.txt")
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                content = f.read()
                if ext.source_dir not in content:
                    import shutil
                    shutil.rmtree(build_dir)
                    os.makedirs(build_dir, exist_ok=True)

        # CMake arguments
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={ext_dir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            "-DBUILD_VAD_PYTHON=ON",
        ]

        # Build type
        build_type = os.environ.get("CMAKE_BUILD_TYPE", "Release")
        cmake_args.append(f"-DCMAKE_BUILD_TYPE={build_type}")

        # Build arguments
        build_args = ["--config", build_type, "--target", "_evo_vad"]

        # Parallel build
        if hasattr(os, "cpu_count"):
            build_args.extend(["--", f"-j{os.cpu_count()}"])

        # Configure
        subprocess.check_call(
            ["cmake", ext.source_dir] + cmake_args,
            cwd=build_dir
        )

        # Build
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args,
            cwd=build_dir
        )

        # For editable installs, also copy to source evo_vad package
        # (wheel builds don't need this since ext_dir is already correct)
        pkg_dir = os.path.join(os.path.dirname(__file__), "evo_vad")
        if os.path.exists(pkg_dir):
            import shutil
            for suffix in [".so", ".dylib", ".pyd"]:
                pattern = f"_evo_vad*{suffix}"
                for path in Path(ext_dir).glob(pattern):
                    pkg_dest = os.path.join(pkg_dir, path.name)
                    shutil.copy2(path, pkg_dest)
                    break


# Read version from C++ header
def get_version():
    return "1.0.0"


# Read long description
def get_long_description():
    readme = Path(__file__).parent / "README.md"
    if readme.exists():
        return readme.read_text()
    return "Evo VAD Python bindings for Voice Activity Detection"


setup(
    name="evo_vad",
    version=get_version(),
    author="muggle",
    author_email="promuggle@gmail.com",
    description="Evo VAD (Voice Activity Detection) Python bindings",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/muggle-stack/e2e_Voice",
    packages=find_packages(),
    ext_modules=[CMakeExtension("evo_vad._evo_vad", source_dir="..")],
    cmdclass={"build_ext": CMakeBuild},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "scipy>=1.5.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: C++",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
    ],
    keywords="vad voice activity detection silero",
)

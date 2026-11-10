"""Build OddlyTyped's native backend with PyBind11."""
import os
from pathlib import Path

from setuptools import Extension, setup
import pybind11

ROOT = Path(__file__).parent
README = (ROOT / "README.md").read_text(encoding="utf-8")

if os.name == "nt":
    compile_args = ["/O2", "/W4"]
else:
    # Optimize hot loops without relaxing IEEE floating-point semantics.
    # Keep warnings enabled, but do not make new compiler warnings fatal to users.
    compile_args = ["-O3", "-funroll-loops", "-Wall", "-Wextra", "-Wpedantic"]

extension = Extension(
    name="oddlytyped.backend",
    sources=["bindings.cpp", "src/matrix_ops.c"],
    include_dirs=[pybind11.get_include(), "src"],
    language="c++",
    extra_compile_args=compile_args,
)

setup(
    name="oddlytyped",
    version="0.2.0",
    description="OddlyTyped: an educational deep-learning framework with a native C matmul backend",
    long_description=README,
    long_description_content_type="text/markdown",
    license="MIT",
    license_files=["LICENSE"],
    packages=["oddlytyped"],
    python_requires=">=3.9",
    install_requires=["numpy>=1.23"],
    ext_modules=[extension],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    project_urls={
        "Source": "https://github.com/arkadyutibiswas/OddlyTyped",
        "Issues": "https://github.com/arkadyutibiswas/OddlyTyped/issues",
    },
    zip_safe=False,
)

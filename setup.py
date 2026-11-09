"""Build AstraTensor's native backend with PyBind11."""
import os
from setuptools import Extension, setup
import pybind11

if os.name == "nt":
    compile_args = ["/O2", "/W4", "/EHsc"]
else:
    # Optimized but IEEE-compliant; no -ffast-math or CPU-specific -march=native.
    compile_args = [
        "-O3", "-funroll-loops",
        "-Wall", "-Wextra", "-Wpedantic", "-Werror",
    ]

extension = Extension(
    name="astratensor.backend",
    sources=["bindings.cpp", "src/matrix_ops.c"],
    include_dirs=[pybind11.get_include(), "src"],
    language="c++",
    extra_compile_args=compile_args,
)

setup(
    name="astratensor",
    version="0.1.0",
    description="A from-scratch deep learning framework native backend",
    packages=["astratensor"],
    ext_modules=[extension],
    zip_safe=False,
)

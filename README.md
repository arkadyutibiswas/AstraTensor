# AstraTensor

AstraTensor is an educational deep-learning framework built from first principles. Its aim is to train a small multilayer perceptron without using PyTorch or TensorFlow in the framework's implementation.

## Architecture

1. **Native C backend** — row-major matrix multiplication in `src/matrix_ops.c`, with explicit allocation and deallocation.
2. **PyBind11 bridge** — `bindings.cpp` exposes the C kernel as `astratensor.backend`.
3. **Autograd engine** — `astratensor/engine.py` tracks operations in a dynamic DAG and performs reverse-mode gradient propagation.
4. **Neural-network API** — `astratensor/nn.py` provides dense layers, ReLU, MSE and cross-entropy losses; `astratensor/optim.py` provides SGD and Adam.

## Current scope

- Scalar and rectangular 2-D tensors.
- Addition, elementwise multiplication, matrix multiplication and ReLU.
- Scalar-to-matrix broadcasting for addition and multiplication.
- Gradient accumulation and topological backpropagation.
- Dense layers with explicit weight and bias tensors.
- MSE and numerically stable softmax cross-entropy.
- SGD and Adam.
- An XOR training example using AstraTensor for all model computation.
- Optional native backend: the Python engine remains usable before compilation, with a pure-Python matrix multiplication fallback.

NumPy is only used at the native extension boundary if supplied by the caller. PyTorch is not imported by production framework code.

## Build and run

Requires Python 3.9+, a C/C++ compiler, and pip.

    python -m pip install --upgrade pip
    python -m pip install -e .
    python -m pip install pytest
    python -m pytest -q
    python examples/train_xor.py

To run the independent PyTorch parity tests, install PyTorch separately and add corresponding tests. PyTorch is not a runtime dependency.

## Repository layout

    astratensor/
      __init__.py
      engine.py
      nn.py
      optim.py
    src/
      matrix_ops.c
      matrix_ops.h
    bindings.cpp
    setup.py
    pyproject.toml
    tests/
    examples/
      train_xor.py

## Limitations

This is an early educational implementation, not a production tensor library. It currently supports only scalars and 2-D matrices, has no GPU execution, and does not yet include batching utilities, convolution, or a dataset loader.

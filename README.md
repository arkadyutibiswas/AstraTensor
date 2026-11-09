# AstraTensor

AstraTensor is a small deep-learning framework implemented in Python with a native C matrix multiplication backend. It includes a tensor class, reverse-mode automatic differentiation, neural-network layers, loss functions, and optimizers.

The framework is intended for learning and experimentation. PyTorch and TensorFlow are not used by the runtime implementation.

## Components

- **Tensor engine** (`astratensor/engine.py`): scalar and 2-D tensor operations, operation graph, and reverse-mode backpropagation.
- **Native backend** (`src/matrix_ops.c`, `bindings.cpp`): row-major matrix multiplication exposed to Python through PyBind11.
- **Neural-network API** (`astratensor/nn.py`): linear layers, ReLU, mean squared error, and softmax cross-entropy.
- **Optimizers** (`astratensor/optim.py`): stochastic gradient descent and Adam.
- **Data and examples**: MNIST IDX loading, XOR training, and synthetic classification examples.

Matrix values and gradients are stored internally as contiguous NumPy float64 arrays. The public tensor interface returns Python values and nested lists. If the native extension has not been built, matrix multiplication uses the NumPy fallback.

## Supported operations

- Scalar and rectangular 2-D tensors
- Addition, elementwise multiplication, matrix multiplication, and ReLU
- Scalar and row-vector broadcasting
- Gradient accumulation and reverse-mode backpropagation
- Linear layers, MSE loss, and softmax cross-entropy
- SGD and Adam
- MNIST loading, mini-batch training, and evaluation

## Requirements

- Python 3.9 or later
- NumPy 1.23 or later
- A C/C++ compiler to build the native extension

## Install and test

From the repository root:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install pytest
python setup.py build_ext --inplace --force
python -m pytest -q
```

Run the examples:

```bash
python examples/train_xor.py
python examples/train_synthetic.py
python examples/train_mnist.py --smoke-test
```

The MNIST smoke test uses a small synthetic fixture. It checks the training path without downloading MNIST and does not produce MNIST accuracy results.

## MNIST training

The training script downloads the MNIST IDX files on the first run and caches them in `data/mnist/`.

```bash
python examples/train_mnist.py --epochs 5 --batch-size 128
```

Use `--data-dir`, `--hidden-size`, `--limit-train`, and `--limit-test` to change the data location or reduce the dataset for local experiments. The script reports training and test loss and accuracy after each epoch.

## PyTorch comparison tests

PyTorch is used only as an independent reference in the parity tests. It is not a runtime dependency.

```bash
python -m pip install torch
python -m pytest -q tests/test_torch_parity.py
```

CI runs these tests in a separate CPU job.

## Repository layout

```text
astratensor/
  __init__.py
  engine.py
  nn.py
  optim.py
  storage.py
src/
  matrix_ops.c
  matrix_ops.h
bindings.cpp
setup.py
pyproject.toml
tests/
  test_backend.py
  test_engine.py
  test_engine_gradients.py
  test_gradcheck.py
  test_mnist_data.py
  test_nn.py
  test_numpy_tensor_storage.py
  test_optim.py
  test_storage.py
  test_synthetic_data.py
  test_torch_parity.py
  test_torch_reference.py
examples/
  benchmark_matmul_boundary.py
  compare_mnist.py
  mnist_data.py
  train_mnist.py
  train_synthetic.py
  train_xor.py
```

## Limitations

AstraTensor currently supports CPU execution with scalar and 2-D tensors. It does not include GPU kernels, convolution operations, automatic mixed precision, or a general-purpose data-loader API. Performance measurements should be treated as specific to the tested hardware, dataset, and configuration.

## License

AstraTensor is released under the [MIT License](LICENSE).

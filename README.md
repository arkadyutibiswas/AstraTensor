# AstraTensor

AstraTensor is an educational deep-learning framework built from first principles. Its aim is to train multilayer perceptrons without using PyTorch or TensorFlow in the framework implementation.

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
- XOR and deterministic nonlinear XOR-cluster training examples.
- MNIST IDX loader with compact byte storage, mini-batch normalization, training/test evaluation and an end-to-end MLP example.
- Optional native backend: the Python engine remains usable before compilation, with a pure-Python matrix multiplication fallback.

The MNIST loader uses Python's standard library. Image pixels remain compact bytes until a batch is prepared, avoiding the memory cost of converting the entire dataset into Python float lists. PyTorch is not imported by production framework code.

## Build and test

Requires Python 3.9+, pip, and a C/C++ compiler for the native extension.

```bash
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install pytest
python setup.py build_ext --inplace --force
python -m pytest -q
python examples/train_xor.py
python examples/train_synthetic.py
python examples/train_mnist.py --smoke-test
```

The smoke test exercises the MNIST model/training path using a tiny synthetic fixture. It does not download data and its metrics are not MNIST results.

## Train on real MNIST

The first run downloads the four compressed MNIST IDX files into `data/mnist/`. Subsequent runs reuse the cached files.

```bash
python examples/train_mnist.py --epochs 5 --batch-size 128
```

Optional arguments include `--data-dir`, `--hidden-size`, `--limit-train`, and `--limit-test`. Limits are useful for quick local experiments; report full-dataset test metrics for a proper benchmark. The script prints training loss/accuracy and held-out test loss/accuracy after every epoch. All forward passes, loss calculations, backpropagation and parameter updates are performed by AstraTensor.

## Independent numerical parity

To run the independent PyTorch parity tests, install PyTorch separately and run:

```bash
python -m pip install torch
python -m pytest -q tests/test_torch_parity.py
```

CI runs these comparisons in a dedicated CPU-only job. PyTorch is only a test oracle, never a runtime dependency or part of the MNIST training implementation.

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

This is an educational CPU framework, not a production tensor library. It currently supports scalars and 2-D matrices, has no GPU execution or convolution, and does not yet provide automatic mixed precision or a general-purpose data-loader abstraction. MNIST is a first end-to-end benchmark, not evidence of production-scale performance.

## License

AstraTensor is distributed under the [MIT License](LICENSE).

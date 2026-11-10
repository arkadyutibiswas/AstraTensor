# OddlyTyped

[![CI](https://github.com/arkadyutibiswas/OddlyTyped/actions/workflows/tests.yml/badge.svg)](https://github.com/arkadyutibiswas/OddlyTyped/actions/workflows/tests.yml)

**An educational deep-learning framework with a native C matrix multiplication backend.**

OddlyTyped is a small deep-learning framework implemented in Python. It includes a custom tensor engine, reverse-mode automatic differentiation, neural-network layers, loss functions, optimizers, and a native C matrix multiplication backend exposed through PyBind11.

The framework is intended for learning and experimentation. PyTorch and TensorFlow are not used by the runtime implementation.

## Components

- **Tensor engine** (`oddlytyped/engine.py`): scalar and 2-D tensor operations, operation graph, and reverse-mode backpropagation.
- **Native backend**: row-major matrix multiplication exposed to Python through PyBind11.
- **Neural-network API**: linear layers, ReLU, mean squared error, and softmax cross-entropy.
- **Optimizers**: stochastic gradient descent and Adam.
- **Data and examples**: MNIST IDX loading, XOR training, and synthetic classification examples.

Matrix values and gradients are stored internally as contiguous NumPy float64 arrays. The public tensor interface returns Python values and nested lists. If the native extension has not been built, matrix multiplication uses the NumPy fallback and emits a warning.

## Supported operations

- Scalar and rectangular 2-D tensors
- Addition, elementwise multiplication, matrix multiplication, and ReLU
- Scalar and row-vector broadcasting
- Gradient accumulation and reverse-mode backpropagation
- Linear layers, MSE loss, and softmax cross-entropy
- SGD and Adam
- MNIST loading, mini-batch training, and evaluation

## Historical MNIST result

A full-dataset experiment recorded in [the NumPy storage migration report](docs/numpy-storage-migration.md) trained a 784 → 128 → 10 model for three epochs using all 60,000 training samples and all 10,000 test samples.

| Epoch | Train loss | Train accuracy | Test loss | Test accuracy |
|---|---:|---:|---:|---:|
| 1 | 0.3603 | 90.16% | 0.1975 | 94.23% |
| 2 | 0.1640 | 95.32% | 0.1429 | 95.53% |
| 3 | 0.1178 | 96.61% | 0.1081 | 96.76% |

The same report records a controlled PyTorch comparison initialized from the OddlyTyped weights. It reported matching train/test losses and accuracies at each epoch, with a final test-accuracy difference of **+0.00 percentage points**.

**These are historical results from the experiment recorded at commit `400ed8c9f09e643a3c4cf7044871c983447e128c`.** They predate subsequent validation-hardening changes and are not a fresh run of the current code. See the report for the configuration and caveats. No speed advantage is claimed from these results.

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

## Full MNIST training

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

## Limitations

OddlyTyped currently supports CPU execution with scalar and 2-D tensors. It does not include GPU kernels, convolution operations, automatic mixed precision, or a general-purpose data-loader API. The native C kernel is a straightforward row-major implementation; it is not advertised as SIMD-accelerated or faster than optimized NumPy/BLAS.

Performance measurements should be treated as specific to the tested hardware, dataset, and configuration.

## License

OddlyTyped is released under the [MIT License](LICENSE).

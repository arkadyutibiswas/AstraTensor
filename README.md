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

## Latest full-dataset MNIST validation

[GitHub Actions benchmark run #6](https://github.com/arkadyutibiswas/OddlyTyped/actions/runs/42807216471) completed successfully on commit `33cfff4a769d0a0ca8140fffc7aaa89888516b11`, after the correctness and native-input-alignment fixes.

- Dataset: all 60,000 training samples and all 10,000 test samples
- Model: 784 → 128 → 10; float64
- Optimizer: Adam (learning rate 0.001)
- Epochs: 3; batch size: 128
- Final test accuracy: **96.76%**
- Final test loss: **0.1081**

| Epoch | Train loss | Train accuracy | Test loss | Test accuracy |
|---|---:|---:|---:|---:|
| 1 | 0.3603 | 90.16% | 0.1975 | 94.23% |
| 2 | 0.1640 | 95.32% | 0.1429 | 95.53% |
| 3 | 0.1178 | 96.61% | 0.1081 | 96.76% |

The run also compared OddlyTyped with PyTorch using the same architecture, precision, optimizer settings, and initial weights copied from OddlyTyped. The reported train/test losses and accuracies matched at each epoch; final test-accuracy delta was **+0.00 percentage points**. The single-run CPU timings were 28.24 s for OddlyTyped and 27.73 s for PyTorch (1.02× ratio). This is one measurement, not a statistically robust performance result, and no speed advantage is claimed.

The run's uploaded artifact contains the training metrics, controlled comparison, and matrix-multiplication boundary measurements.

## Requirements

- Python 3.9 or later (CI currently tests 3.9, 3.10, and 3.11)
- NumPy 1.23 or later
- A C/C++ compiler to build the native extension

## Install and test

From the repository root:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install pybind11 pytest
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

PyTorch is used only as an independent reference in the parity tests and benchmark. It is not a runtime dependency.

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

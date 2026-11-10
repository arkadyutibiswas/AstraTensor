# NumPy-backed Tensor storage migration

## Status

Stages A–D are implemented and merged into `main`. Tensor matrices and gradients use owned NumPy arrays internally; public `data` and `grad` remain detached lists. Layers, losses, SGD, and Adam operate on internal arrays.

The migration's functional validation was recorded at commit `400ed8c9f09e643a3c4cf7044871c983447e128c`. It included passing CI jobs and a full-dataset MNIST run with matching reported metrics in the controlled PyTorch comparison. That benchmark predates the later correctness-hardening work; it is historical evidence, not a fresh full-dataset run of the current PR head. Performance measurements are recorded below as preliminary single-run observations, not as a statistically robust speed claim.

## Why change storage?

The earlier engine stored matrices as nested Python lists. Matrix multiplication crosses the PyBind11 boundary, and the bridge converts list inputs to NumPy arrays and native results back to arrays before the engine converts them to lists. The boundary microbenchmark below quantifies the conversion overhead for representative MNIST matrix shapes.

These measurements identify conversion overhead as a candidate bottleneck. They do not prove that the same ratio translates into end-to-end training speed.

## Historical benchmark

- GitHub Actions run: [MNIST benchmark #3](https://github.com/arkadyutibiswas/OddlyTyped/actions/runs/42641791922)
- Tested commit: `400ed8c9f09e643a3c4cf7044871c983447e128c`
- Dataset: all 60,000 training samples and all 10,000 test samples
- Model: 784 → 128 → 10; float64; Adam
- Epochs: 3; batch size: 128
- Run status: successful

| Epoch | Train loss | Train accuracy | Test loss | Test accuracy |
|---|---:|---:|---:|---:|
| 1 | 0.3603 | 90.16% | 0.1975 | 94.23% |
| 2 | 0.1640 | 95.32% | 0.1429 | 95.53% |
| 3 | 0.1178 | 96.61% | 0.1081 | 96.76% |

The controlled comparison initialized PyTorch from the OddlyTyped weights and reported the same train/test loss and accuracy at each epoch. Final test accuracy delta was +0.00 percentage points.

The recorded end-to-end CPU timing was a single run and is not evidence of a stable speed advantage. The matrix multiplication boundary microbenchmark measured conversion overhead, not a whole-framework speedup.

## Implementation and validation

The migration introduced owned C-contiguous NumPy storage, retained detached public list interfaces, preserved custom reverse-mode autodiff, and kept native C matrix multiplication. SGD, Adam, layers, losses, gradient accumulation, scalar broadcasting, and shape validation were covered by the existing test suite.

## Explicit non-goals

- Do not replace the custom autograd engine with PyTorch.
- Do not replace the native C matrix multiplication backend with NumPy or PyTorch.
- Do not change public APIs as part of the storage migration.
- Do not add another GitHub Actions workflow.
- Do not claim an end-to-end performance improvement based solely on a boundary microbenchmark or one timing run.

## Release-readiness

- [x] Existing CI test and PyTorch parity jobs passed on the integrated implementation.
- [x] Full-dataset MNIST training completed and reported held-out metrics.
- [x] Controlled PyTorch comparison reported matching metrics.
- [ ] Repeat end-to-end timing before publishing a stable performance claim.
- [ ] Review package metadata, installation instructions, supported Python/compiler versions, and license before creating a public release.
- [ ] Tag and publish only after remaining release checks are complete.

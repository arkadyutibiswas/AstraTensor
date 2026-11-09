# NumPy-backed Tensor storage migration

## Status

Stages A–D are implemented and merged into `main`. Tensor matrices and gradients use owned NumPy arrays internally; public `data` and `grad` remain detached lists. Layers, losses, SGD, and Adam operate on internal arrays.

Stage E's functional validation is complete for the current integration: the existing CI jobs passed, and the full MNIST benchmark completed successfully with exact reported metric parity against the independent PyTorch comparison. Performance measurements are recorded below as preliminary single-run observations; they are not a statistically robust speed claim.

## Why change storage?

The earlier engine stored matrices as nested Python lists. Matrix multiplication crosses the PyBind11 boundary, and the bridge converts list inputs to NumPy arrays and native results back to arrays before the engine converts them to lists. The boundary microbenchmark below quantifies the conversion overhead for representative MNIST matrix shapes.

These measurements identify conversion overhead as a candidate bottleneck. They do not prove that the same ratio translates into end-to-end training speed.

## Required compatibility contract

1. `Tensor(data)` continues to accept scalars and rectangular 2-D nested sequences.
2. `Tensor.data` continues to return a detached nested-list value for matrices and a float for scalars.
3. `Tensor.grad` retains the same public representation and accumulation behaviour.
4. `shape`, `is_scalar`, `item()`, `repr()`, and optimizer public APIs remain stable.
5. Mutating a list returned by `data` or `grad` must not silently mutate internal state. Assignment through `tensor.data = ...` remains supported and validated.
6. Internal matrix operations consume owned contiguous float64 arrays without repeated list conversion.
7. Backward callbacks accumulate gradients correctly; leaf gradients continue to accumulate across backward calls while intermediate gradients reset per pass.
8. Scalar broadcasting, matrix shape checks, deep iterative graph traversal, and explicit matrix-output gradients remain supported.

## Implementation sequence

### Stage A — storage adapter (completed)

- Added `MatrixStorage` with owned C-contiguous float64 storage.
- Added copy and detached list conversion.
- Tested ownership, invalid shapes, finite values, and copy isolation.

### Stage B — Tensor internal representation (completed)

- Added private matrix storage and a single normalization boundary.
- Kept public `data` and `grad` compatibility properties.
- Added internal array accessors used by engine operations.
- Preserved scalar representation and avoided converting matrices to lists inside each operation.

### Stage C — autograd operations (completed)

- Ported add, multiply, matmul, ReLU, and gradient accumulation to array operations.
- Kept topological traversal iterative.
- Added tests for gradients, shared subgraphs, repeated backward calls, scalar broadcasting, and shape errors.

### Stage D — layers and optimizers (completed; CI and MNIST validated)

- Used Tensor broadcasting for Linear bias addition and reduced broadcast gradients to the original shape.
- Vectorized MSE and cross-entropy forward/backward calculations while retaining AstraTensor's custom backward callbacks.
- Stored SGD/Adam parameter updates and Adam moment buffers in internal float64 arrays.
- Preserved the public parameter and gradient interfaces.

### Stage E — validation and performance (functional validation completed)

Benchmark record:

- GitHub Actions run: [MNIST benchmark #3](https://github.com/arkadyutibiswas/AstraTensor/actions/runs/42641791922)
- Tested commit: `400ed8c9f09e643a3c4cf7044871c983447e128c`
- Runner: GitHub-hosted Ubuntu 24.04, Python 3.11
- Dataset: all 60,000 training samples and all 10,000 test samples
- Model: 784 → 128 → 10; float64; Adam
- Epochs: 3; batch size: 128
- Run status: successful; benchmark artifact: `astratensor-mnist-results`

AstraTensor training metrics:

| Epoch | Train loss | Train accuracy | Test loss | Test accuracy |
|---|---:|---:|---:|---:|
| 1 | 0.3603 | 90.16% | 0.1975 | 94.23% |
| 2 | 0.1640 | 95.32% | 0.1429 | 95.53% |
| 3 | 0.1178 | 96.61% | 0.1081 | 96.76% |

The controlled comparison initialized PyTorch from the AstraTensor weights and reported the same train/test loss and accuracy at each epoch. Final test accuracy delta was +0.00 percentage points.

Single-run controlled CPU timings:

| Implementation | Train + evaluation |
|---|---:|
| AstraTensor | 21.47 s |
| PyTorch | 23.01 s |

The workflow reported a 0.93x AstraTensor/PyTorch time ratio. This is one CPU run only; it is not sufficient evidence for a stable speed advantage and must not be presented as a statistically established performance claim.

Native matrix multiplication boundary microbenchmark (30 repeats; median latency):

| Operation | NumPy array path | Python-list input | Engine-style list path |
|---|---:|---:|---:|
| Forward: (128, 784) × (784, 128) | 1.873 ms | 4.868 ms | 5.032 ms |
| Input gradient: (128, 128) × (128, 784) | 1.870 ms | 3.162 ms | 3.989 ms |
| Weight gradient: (128, 128) × (128, 10) | 0.046 ms | 0.252 ms | 0.260 ms |

All three paths call the same compiled C kernel; the measured difference is in input/output conversion. This is a microbenchmark, not a whole-framework speedup measurement.

### Remaining performance work

- Repeat the end-to-end comparison under the same runner and settings before making a stable runtime claim. This is optional for documenting the successful functional validation, but required for a robust performance claim.
- If repeated measurements are not collected, keep the single-run caveat attached to all timing figures.
- Do not compare the 21.47-second controlled-comparison timing directly with older training-only runs; they are different benchmark paths.

## Explicit non-goals

- Do not replace the custom autograd engine with PyTorch.
- Do not replace the native C matrix multiplication backend with NumPy or PyTorch. NumPy is used for storage, elementwise/loss calculations, and conversion-efficient array handling; custom backward callbacks and native C matmul remain part of AstraTensor.
- Do not change public APIs as part of the storage migration.
- Do not add another GitHub Actions workflow.
- Do not claim an end-to-end performance improvement based solely on a boundary microbenchmark or one timing run.

## Release-readiness checklist

- [x] Stages A–D merged into `main`.
- [x] Existing CI test and PyTorch parity jobs pass on the integrated implementation.
- [x] Full-dataset MNIST training completes and reports held-out metrics.
- [x] Controlled PyTorch comparison reports matching metrics.
- [x] Benchmark outputs are retained in the GitHub Actions artifact.
- [ ] Repeat end-to-end timing if publishing a stable performance claim.
- [ ] Review package metadata, installation instructions, supported Python/compiler versions, and license before creating a public release.
- [ ] Tag and publish a release only after the remaining release checks are complete.

## Merge gates

Each integration stage must pass both existing CI jobs. Run the full MNIST benchmark after the integrated implementation passes CI, not on every commit.

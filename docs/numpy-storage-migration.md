# NumPy-backed Tensor storage migration

## Status

Stages A–D are implemented on the feature branch. Tensor matrices and gradients use owned NumPy arrays internally; public `data` and `grad` remain detached lists. Layers, losses, SGD, and Adam now operate on internal arrays. Stage E validation and performance measurement remain.

## Why change storage?

The current engine stores matrices as nested Python lists. Matrix multiplication crosses the PyBind11 boundary, and the bridge converts list inputs to NumPy arrays and native results back to arrays before the engine converts them to lists. The profiling run showed a substantial latency difference between array and list paths for the tested MNIST matrix shapes.

The measurements identify conversion overhead as a candidate bottleneck; they do not prove that replacing all list storage will improve end-to-end training by the same ratio.

## Required compatibility contract

1. `Tensor(data)` continues to accept scalars and rectangular 2-D nested sequences.
2. `Tensor.data` continues to return a detached nested-list value for matrices and a float for scalars in the first migration stage.
3. `Tensor.grad` retains the same public representation and accumulation behaviour.
4. `shape`, `is_scalar`, `item()`, `repr()`, and optimizer public APIs remain stable.
5. Mutating a list returned by `data` or `grad` must not silently mutate internal state. Assignment through `tensor.data = ...` remains supported and validated.
6. Internal matrix operations should consume owned contiguous float64 arrays without repeated list conversion.
7. Backward callbacks must accumulate gradients exactly once per graph path; leaf gradients continue to accumulate across backward calls while intermediate gradients reset per pass.
8. Scalar broadcasting, matrix shape checks, deep iterative graph traversal, and explicit matrix-output gradients must remain unchanged.

## Proposed implementation sequence

### Stage A — storage adapter (completed)

- Add `MatrixStorage` with owned C-contiguous float64 storage.
- Add copy and detached list conversion.
- Test ownership, invalid shapes, finite values, and copy isolation.

### Stage B — Tensor internal representation (completed)

- Add private matrix storage and a single normalization boundary.
- Keep public `data` and `grad` compatibility properties.
- Add internal array accessors used only by engine operations.
- Preserve scalar representation and avoid converting matrices to lists inside each operation.

### Stage C — autograd operations (completed)

- Port add, multiply, matmul, ReLU, and gradient accumulation to array operations.
- Keep topological traversal iterative.
- Add finite-difference checks and tests for shared subgraphs, repeated backward calls, scalar broadcasting, and shape errors.

### Stage D — layers and optimizers (implemented; validation pending)

- Use Tensor broadcasting for Linear bias addition and reduce broadcast gradients to the original shape.
- Vectorize MSE and cross-entropy forward/backward calculations while retaining AstraTensor's custom backward callbacks.
- Store SGD/Adam parameter updates and Adam moment buffers in internal float64 arrays.
- Ensure optimizer state uses the same dtype and shape as parameters.
- Preserve the public parameter and gradient interfaces.

### Stage E — validation and performance

- Run the existing test and PyTorch parity jobs.
- Run the existing manual MNIST benchmark with identical settings.
- Compare accuracy, loss, and runtime against the recorded 121.38-second AstraTensor baseline.
- Repeat timing runs before claiming a stable speedup.

## Explicit non-goals

- Do not replace the custom autograd engine with PyTorch.
- Do not move core tensor arithmetic into NumPy as a shortcut; native C remains the matrix multiplication backend.
- Do not change public APIs in the same commit as the storage migration.
- Do not add another GitHub Actions workflow.
- Do not merge a performance claim without end-to-end benchmark evidence.

## Merge gates

Each integration stage must pass both existing CI jobs. The MNIST benchmark is run only after the integrated implementation passes CI, not on every commit.

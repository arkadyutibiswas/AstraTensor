"""High-level neural-network building blocks implemented on OddlyTyped."""
import math
import random
from numbers import Integral

import numpy as np

from .engine import Tensor, _is_matrix, _shape, _grad_add


class Module:
    """Base class for modules with explicitly exposed trainable parameters."""
    def parameters(self):
        return []

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.zero_grad()


class Linear(Module):
    """Fully connected layer: output = input @ weight + bias."""
    def __init__(self, in_features, out_features, seed=None):
        if (not isinstance(in_features, int) or isinstance(in_features, bool)
                or not isinstance(out_features, int) or isinstance(out_features, bool)):
            raise TypeError("Layer dimensions must be integers.")
        if in_features <= 0 or out_features <= 0:
            raise ValueError("Layer dimensions must be positive.")
        rng = random.Random(seed)
        limit = math.sqrt(6.0 / (in_features + out_features))
        self.weight = Tensor(
            [[rng.uniform(-limit, limit) for _ in range(out_features)]
             for _ in range(in_features)],
            requires_grad=True,
        )
        self.bias = Tensor([[0.0 for _ in range(out_features)]], requires_grad=True)
        self.in_features = in_features
        self.out_features = out_features

    def __call__(self, inputs):
        if not isinstance(inputs, Tensor):
            inputs = Tensor(inputs)
        if not _is_matrix(inputs._data) or inputs.shape[1] != self.in_features:
            raise ValueError(
                f"Linear expected a 2-D input with {self.in_features} features."
            )
        return inputs @ self.weight + self.bias

    def parameters(self):
        return [self.weight, self.bias]


class ReLU(Module):
    def __call__(self, inputs):
        if not isinstance(inputs, Tensor):
            inputs = Tensor(inputs)
        return inputs.relu()


class MSELoss(Module):
    """Mean squared error with gradients for both differentiable operands."""
    def __call__(self, prediction, target):
        if not isinstance(prediction, Tensor):
            prediction = Tensor(prediction)
        target_tensor = target if isinstance(target, Tensor) else None
        target_data = target_tensor._data if target_tensor is not None else target
        try:
            target_array = (
                np.asarray(target_data, dtype=np.float64)
                if _is_matrix(target_data) else float(target_data)
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Prediction and target shapes must match.") from exc
        if isinstance(prediction._data, np.ndarray) != isinstance(target_array, np.ndarray):
            raise ValueError("Prediction and target shapes must match.")
        if isinstance(prediction._data, np.ndarray):
            if prediction._data.shape != target_array.shape:
                raise ValueError("Prediction and target shapes must match.")
            difference = prediction._data - target_array
            count = difference.size
            loss_value = float(np.mean(difference * difference))
        else:
            difference = prediction._data - target_array
            count = 1
            loss_value = difference * difference

        children = (prediction, target_tensor) if target_tensor is not None else (prediction,)
        requires_grad = prediction.requires_grad or (
            target_tensor is not None and target_tensor.requires_grad
        )
        output = Tensor(loss_value, requires_grad, children, "mse")

        def backward():
            scale = (2.0 / count) * output._grad
            if prediction.requires_grad:
                prediction._grad = _grad_add(
                    prediction._grad, difference * scale
                )
            if target_tensor is not None and target_tensor.requires_grad:
                target_tensor._grad = _grad_add(
                    target_tensor._grad, -difference * scale
                )

        output._backward = backward
        return output


class CrossEntropyLoss(Module):
    """Stable mean softmax cross-entropy for integer class-index labels."""
    def __call__(self, logits, targets):
        if not isinstance(logits, Tensor):
            logits = Tensor(logits)
        if not isinstance(logits._data, np.ndarray):
            raise ValueError("CrossEntropyLoss expects (batch, classes) logits.")
        batch_size, class_count = logits._data.shape
        try:
            targets = list(targets)
        except TypeError as exc:
            raise TypeError("Class targets must be an iterable of integer indices.") from exc
        if len(targets) != batch_size:
            raise ValueError("Provide exactly one target class for each batch row.")
        if any(not isinstance(target, Integral) or isinstance(target, (bool, np.bool_))
               for target in targets):
            raise TypeError("Class targets must be integer indices.")
        if any(int(target) < 0 or int(target) >= class_count for target in targets):
            raise ValueError("Target class index is outside the logits range.")
        target_indices = np.fromiter(
            (int(target) for target in targets), dtype=np.int64, count=batch_size
        )

        shifted = logits._data - np.max(logits._data, axis=1, keepdims=True)
        exponentials = np.exp(shifted)
        probabilities = exponentials / np.sum(exponentials, axis=1, keepdims=True)
        log_normalizers = np.log(np.sum(exponentials, axis=1)) + np.max(
            logits._data, axis=1
        )
        losses = log_normalizers - logits._data[np.arange(batch_size), target_indices]
        loss_value = float(np.mean(losses))
        output = Tensor(loss_value, logits.requires_grad, (logits,), "cross_entropy")

        def backward():
            if not logits.requires_grad:
                return
            contribution = probabilities.copy()
            contribution[np.arange(batch_size), target_indices] -= 1.0
            contribution *= output._grad / batch_size
            logits._grad = _grad_add(logits._grad, contribution)

        output._backward = backward
        return output

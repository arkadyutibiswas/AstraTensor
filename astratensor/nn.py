"""High-level neural-network building blocks implemented on AstraTensor."""
import math
import random

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
        if not isinstance(in_features, int) or not isinstance(out_features, int):
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
        # Tensor addition handles bias broadcasting and its reverse reduction.
        return inputs @ self.weight + self.bias

    def parameters(self):
        return [self.weight, self.bias]


class ReLU(Module):
    def __call__(self, inputs):
        if not isinstance(inputs, Tensor):
            inputs = Tensor(inputs)
        return inputs.relu()


class MSELoss(Module):
    """Mean squared error over all elements, with a custom backward callback."""
    def __call__(self, prediction, target):
        if not isinstance(prediction, Tensor):
            prediction = Tensor(prediction)
        target_data = target._data if isinstance(target, Tensor) else target
        target_array = np.asarray(target_data, dtype=np.float64) if _is_matrix(target_data) else float(target_data)
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

        output = Tensor(loss_value, prediction.requires_grad, (prediction,), "mse")

        def backward():
            if prediction.requires_grad:
                contribution = (2.0 / count) * difference * output._grad
                prediction._grad = _grad_add(prediction._grad, contribution)

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
        targets = list(targets)
        if len(targets) != batch_size:
            raise ValueError("Provide exactly one target class for each batch row.")
        if any(not isinstance(target, int) or isinstance(target, bool)
               for target in targets):
            raise TypeError("Class targets must be integer indices.")
        target_indices = np.asarray(targets, dtype=np.int64)
        if np.any(target_indices < 0) or np.any(target_indices >= class_count):
            raise ValueError("Target class index is outside the logits range.")

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

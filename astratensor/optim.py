"""NumPy-backed parameter update algorithms for AstraTensor."""
import math

import numpy as np


def _finite_float(value, name):
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be a finite number.")
    return converted


def _positive_finite(value, name):
    converted = _finite_float(value, name)
    if converted <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return converted


def _zeros_like(value):
    return np.zeros_like(value, dtype=np.float64) if isinstance(value, np.ndarray) else 0.0


def _validate_finite_state(parameters):
    """Reject invalid state before an optimizer changes any parameter."""
    for parameter in parameters:
        if not np.isfinite(parameter._data).all():
            raise ValueError("Optimizer parameters must contain only finite values.")
        if not np.isfinite(parameter._grad).all():
            raise ValueError("Optimizer gradients must contain only finite values.")


class SGD:
    """Stochastic gradient descent over an iterable of Tensor parameters."""
    def __init__(self, parameters, lr=0.01):
        self.lr = _positive_finite(lr, "Learning rate")
        self.parameters = list(parameters)

    def step(self):
        _validate_finite_state(self.parameters)
        for parameter in self.parameters:
            parameter._data = parameter._data - self.lr * parameter._grad

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.zero_grad()


class Adam:
    """Adam with float64 moment buffers matching each parameter's storage."""
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = _positive_finite(lr, "Learning rate")
        self.eps = _positive_finite(eps, "Epsilon")
        self.beta1 = _finite_float(beta1, "beta1")
        self.beta2 = _finite_float(beta2, "beta2")
        if not 0.0 <= self.beta1 < 1.0 or not 0.0 <= self.beta2 < 1.0:
            raise ValueError("Adam beta values must be in [0, 1).")
        self.parameters = list(parameters)
        self.step_count = 0
        self.first_moment = {id(p): _zeros_like(p._data) for p in self.parameters}
        self.second_moment = {id(p): _zeros_like(p._data) for p in self.parameters}

    def step(self):
        _validate_finite_state(self.parameters)
        self.step_count += 1
        first_scale = 1.0 / (1.0 - self.beta1 ** self.step_count)
        second_scale = 1.0 / (1.0 - self.beta2 ** self.step_count)
        for parameter in self.parameters:
            key = id(parameter)
            gradient = parameter._grad
            first = self.first_moment[key]
            second = self.second_moment[key]
            first = self.beta1 * first + (1.0 - self.beta1) * gradient
            second = self.beta2 * second + (1.0 - self.beta2) * gradient * gradient
            corrected_first = first * first_scale
            corrected_second = second * second_scale
            parameter._data = parameter._data - self.lr * corrected_first / (
                np.sqrt(corrected_second) + self.eps
            )
            self.first_moment[key] = first
            self.second_moment[key] = second

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.zero_grad()

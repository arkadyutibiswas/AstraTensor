"""NumPy-backed parameter update algorithms for AstraTensor."""
import numpy as np


def _zeros_like(value):
    return np.zeros_like(value, dtype=np.float64) if isinstance(value, np.ndarray) else 0.0


class SGD:
    """Stochastic gradient descent over an iterable of Tensor parameters."""
    def __init__(self, parameters, lr=0.01):
        if lr <= 0:
            raise ValueError("Learning rate must be positive.")
        self.parameters = list(parameters)
        self.lr = float(lr)

    def step(self):
        for parameter in self.parameters:
            parameter._data = parameter._data - self.lr * parameter._grad

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.zero_grad()


class Adam:
    """Adam with float64 moment buffers matching each parameter's storage."""
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        if lr <= 0 or eps <= 0:
            raise ValueError("Learning rate and epsilon must be positive.")
        if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
            raise ValueError("Adam beta values must be in [0, 1).")
        self.parameters = list(parameters)
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.step_count = 0
        self.first_moment = {id(p): _zeros_like(p._data) for p in self.parameters}
        self.second_moment = {id(p): _zeros_like(p._data) for p in self.parameters}

    def step(self):
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

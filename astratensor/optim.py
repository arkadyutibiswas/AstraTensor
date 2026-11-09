"""Parameter update algorithms for AstraTensor."""
import math


def _zeros_like(value):
    if isinstance(value, list):
        return [[0.0 for _ in row] for row in value]
    return 0.0


def _combine(left, right, left_scale, right_scale):
    if isinstance(left, list):
        return [
            [left_scale * a + right_scale * b for a, b in zip(row_a, row_b)]
            for row_a, row_b in zip(left, right)
        ]
    return left_scale * left + right_scale * right


def _square(value):
    if isinstance(value, list):
        return [[item * item for item in row] for row in value]
    return value * value


def _update_parameter(data, gradient, learning_rate):
    if isinstance(data, list):
        return [
            [value - learning_rate * grad for value, grad in zip(row, grad_row)]
            for row, grad_row in zip(data, gradient)
        ]
    return data - learning_rate * gradient


def _adam_update(data, first, second, learning_rate, epsilon):
    if isinstance(data, list):
        return [
            [
                value - learning_rate * m / (math.sqrt(v) + epsilon)
                for value, m, v in zip(data_row, first_row, second_row)
            ]
            for data_row, first_row, second_row in zip(data, first, second)
        ]
    return data - learning_rate * first / (math.sqrt(second) + epsilon)


class SGD:
    """Stochastic gradient descent over an iterable of Tensor parameters."""
    def __init__(self, parameters, lr=0.01):
        if lr <= 0:
            raise ValueError("Learning rate must be positive.")
        self.parameters = list(parameters)
        self.lr = float(lr)

    def step(self):
        for parameter in self.parameters:
            parameter.data = _update_parameter(
                parameter.data, parameter.grad, self.lr
            )

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.zero_grad()


class Adam:
    """Adam optimizer with bias-corrected first and second moments."""
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
        self.first_moment = {id(p): _zeros_like(p.data) for p in self.parameters}
        self.second_moment = {id(p): _zeros_like(p.data) for p in self.parameters}

    def step(self):
        self.step_count += 1
        for parameter in self.parameters:
            key = id(parameter)
            self.first_moment[key] = _combine(
                self.first_moment[key], parameter.grad,
                self.beta1, 1.0 - self.beta1,
            )
            self.second_moment[key] = _combine(
                self.second_moment[key], _square(parameter.grad),
                self.beta2, 1.0 - self.beta2,
            )
            first_corrected = _combine(
                _zeros_like(self.first_moment[key]),
                self.first_moment[key],
                0.0, 1.0 / (1.0 - self.beta1 ** self.step_count),
            )
            second_corrected = _combine(
                _zeros_like(self.second_moment[key]),
                self.second_moment[key],
                0.0, 1.0 / (1.0 - self.beta2 ** self.step_count),
            )
            parameter.data = _adam_update(
                parameter.data, first_corrected, second_corrected,
                self.lr, self.eps,
            )

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.zero_grad()

"""Parameter update algorithms for AstraTensor."""


def _update_parameter(data, gradient, learning_rate):
    if isinstance(data, list):
        return [
            [value - learning_rate * grad for value, grad in zip(row, grad_row)]
            for row, grad_row in zip(data, gradient)
        ]
    return data - learning_rate * gradient


def _adam_update_in_place(data, gradient, first, second, beta1, beta2,
                          first_scale, second_scale, learning_rate, epsilon):
    """Update moments and parameters in one pass, avoiding temporary matrices."""
    if isinstance(data, list):
        for row_index, row in enumerate(data):
            grad_row = gradient[row_index]
            first_row = first[row_index]
            second_row = second[row_index]
            for column, value in enumerate(row):
                grad = grad_row[column]
                moment = beta1 * first_row[column] + (1.0 - beta1) * grad
                variance = beta2 * second_row[column] + (1.0 - beta2) * grad * grad
                first_row[column] = moment
                second_row[column] = variance
                corrected_moment = moment * first_scale
                corrected_variance = variance * second_scale
                row[column] = value - learning_rate * corrected_moment / (
                    corrected_variance ** 0.5 + epsilon
                )
        return data

    moment = beta1 * first + (1.0 - beta1) * gradient
    variance = beta2 * second + (1.0 - beta2) * gradient * gradient
    return (
        data - learning_rate * (moment * first_scale) /
        ((variance * second_scale) ** 0.5 + epsilon),
        moment,
        variance,
    )


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
        first_scale = 1.0 / (1.0 - self.beta1 ** self.step_count)
        second_scale = 1.0 / (1.0 - self.beta2 ** self.step_count)
        for parameter in self.parameters:
            key = id(parameter)
            if isinstance(parameter.data, list):
                # Tensor.data returns a detached compatibility copy. The
                # in-place helper therefore updates only that copy; assign
                # the result back so the Tensor's owned NumPy storage changes.
                updated_data = _adam_update_in_place(
                    parameter.data, parameter.grad,
                    self.first_moment[key], self.second_moment[key],
                    self.beta1, self.beta2, first_scale, second_scale,
                    self.lr, self.eps,
                )
                parameter.data = updated_data
            else:
                data, first, second = _adam_update_in_place(
                    parameter.data, parameter.grad,
                    self.first_moment[key], self.second_moment[key],
                    self.beta1, self.beta2, first_scale, second_scale,
                    self.lr, self.eps,
                )
                parameter.data = data
                self.first_moment[key] = first
                self.second_moment[key] = second

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.zero_grad()


def _zeros_like(value):
    if isinstance(value, list):
        return [[0.0 for _ in row] for row in value]
    return 0.0

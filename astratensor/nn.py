"""High-level neural-network building blocks implemented on AstraTensor."""
import math
import random

from .engine import Tensor, _is_matrix, _shape, _grad_add


class Module:
    """Base class for modules with explicitly exposed trainable parameters."""
    def parameters(self):
        return []

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.zero_grad()


class Linear(Module):
    """Fully connected layer: output = input @ weight + bias.

    Inputs are 2-D batches shaped (batch, in_features). Weights use
    (in_features, out_features), and bias uses (1, out_features).
    """
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
        if not _is_matrix(inputs.data) or inputs.shape[1] != self.in_features:
            raise ValueError(
                f"Linear expected a 2-D input with {self.in_features} features."
            )

        product = inputs @ self.weight
        output_data = [
            [value + self.bias.data[0][column]
             for column, value in enumerate(row)]
            for row in product.data
        ]
        output = Tensor(
            output_data,
            product.requires_grad or self.bias.requires_grad,
            (product, self.bias),
            "linear_bias",
        )

        def backward():
            if product.requires_grad:
                product.grad = _grad_add(product.grad, output.grad)
            if self.bias.requires_grad:
                batch_size = len(output.grad)
                bias_contribution = [[
                    sum(output.grad[row][column] for row in range(batch_size))
                    for column in range(self.out_features)
                ]]
                self.bias.grad = _grad_add(self.bias.grad, bias_contribution)

        output._backward = backward
        return output

    def parameters(self):
        return [self.weight, self.bias]


class ReLU(Module):
    def __call__(self, inputs):
        if not isinstance(inputs, Tensor):
            inputs = Tensor(inputs)
        return inputs.relu()


class MSELoss(Module):
    """Mean squared error over all elements."""
    def __call__(self, prediction, target):
        if not isinstance(prediction, Tensor):
            prediction = Tensor(prediction)
        target_data = target.data if isinstance(target, Tensor) else target
        if _is_matrix(prediction.data) != _is_matrix(target_data):
            raise ValueError("Prediction and target shapes must match.")
        if _is_matrix(prediction.data) and _shape(prediction.data) != _shape(target_data):
            raise ValueError("Prediction and target shapes must match.")

        if _is_matrix(prediction.data):
            rows, columns = _shape(prediction.data)
            count = rows * columns
            differences = [
                [prediction.data[i][j] - float(target_data[i][j])
                 for j in range(columns)]
                for i in range(rows)
            ]
            loss_value = sum(
                value * value for row in differences for value in row
            ) / count
            output = Tensor(loss_value, prediction.requires_grad, (prediction,), "mse")

            def backward():
                if prediction.requires_grad:
                    contribution = [
                        [2.0 * differences[i][j] / count * output.grad
                         for j in range(columns)]
                        for i in range(rows)
                    ]
                    prediction.grad = _grad_add(prediction.grad, contribution)

            output._backward = backward
            return output

        difference = prediction.data - float(target_data)
        output = Tensor(difference * difference, prediction.requires_grad,
                        (prediction,), "mse")

        def backward():
            if prediction.requires_grad:
                prediction.grad = _grad_add(
                    prediction.grad, 2.0 * difference * output.grad
                )

        output._backward = backward
        return output


class CrossEntropyLoss(Module):
    """Mean softmax cross-entropy for logits and integer class-index labels."""
    def __call__(self, logits, targets):
        if not isinstance(logits, Tensor):
            logits = Tensor(logits)
        if not _is_matrix(logits.data):
            raise ValueError("CrossEntropyLoss expects (batch, classes) logits.")
        batch_size, class_count = _shape(logits.data)
        targets = list(targets)
        if len(targets) != batch_size:
            raise ValueError("Provide exactly one target class for each batch row.")

        probabilities = []
        losses = []
        for row, target in zip(logits.data, targets):
            if not isinstance(target, int) or isinstance(target, bool):
                raise TypeError("Class targets must be integer indices.")
            if target < 0 or target >= class_count:
                raise ValueError("Target class index is outside the logits range.")
            maximum = max(row)
            exponentials = [math.exp(value - maximum) for value in row]
            denominator = sum(exponentials)
            probabilities.append([value / denominator for value in exponentials])
            log_denominator = math.log(denominator) + maximum
            losses.append(log_denominator - row[target])

        output = Tensor(
            sum(losses) / batch_size,
            logits.requires_grad,
            (logits,),
            "cross_entropy",
        )

        def backward():
            if not logits.requires_grad:
                return
            contribution = []
            for row_index, probability_row in enumerate(probabilities):
                row_gradient = [value / batch_size for value in probability_row]
                row_gradient[targets[row_index]] -= 1.0 / batch_size
                contribution.append([value * output.grad for value in row_gradient])
            logits.grad = _grad_add(logits.grad, contribution)

        output._backward = backward
        return output

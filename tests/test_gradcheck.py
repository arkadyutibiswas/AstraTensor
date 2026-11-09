"""Numerical edge-case checks for reverse-mode gradients and loss stability."""

import math

import pytest

from astratensor import Tensor
from astratensor.nn import CrossEntropyLoss


def _cross_entropy_value(values, labels):
    return CrossEntropyLoss()(Tensor(values), labels).item()


def test_cross_entropy_gradient_matches_central_finite_differences():
    values = [[1.2, -0.7, 0.3], [-1.1, 0.8, 0.2]]
    labels = [2, 0]
    logits = Tensor(values, requires_grad=True)
    CrossEntropyLoss()(logits, labels).backward()

    epsilon = 1e-6
    for row in range(len(values)):
        for column in range(len(values[row])):
            plus = [item[:] for item in values]
            minus = [item[:] for item in values]
            plus[row][column] += epsilon
            minus[row][column] -= epsilon
            numerical = (
                _cross_entropy_value(plus, labels)
                - _cross_entropy_value(minus, labels)
            ) / (2.0 * epsilon)
            assert logits.grad[row][column] == pytest.approx(
                numerical, rel=2e-6, abs=2e-7
            )


def test_cross_entropy_remains_finite_for_extreme_logits():
    logits = Tensor(
        [[10000.0, 9999.0, -10000.0], [-10000.0, 10000.0, 9998.0]],
        requires_grad=True,
    )
    loss = CrossEntropyLoss()(logits, [1, 2])

    assert math.isfinite(loss.item())
    loss.backward()
    assert all(
        math.isfinite(value)
        for row in logits.grad
        for value in row
    )


def test_backward_handles_deep_graph_without_python_recursion():
    value = Tensor(1.0, requires_grad=True)
    result = value
    for _ in range(2000):
        result = result * 1.0001

    result.backward()

    assert value.grad == pytest.approx(1.0001 ** 2000, rel=1e-10)


@pytest.mark.parametrize("labels", [[-1], [2], [True], [0, 1]])
def test_cross_entropy_rejects_invalid_class_targets(labels):
    logits = Tensor([[0.1, 0.2]], requires_grad=True)
    with pytest.raises((TypeError, ValueError)):
        CrossEntropyLoss()(logits, labels)

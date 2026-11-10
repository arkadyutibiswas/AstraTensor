"""Numerical edge-case checks for reverse-mode gradients and loss stability."""

import math

import numpy as np
import pytest

from oddlytyped import Tensor
from oddlytyped.nn import CrossEntropyLoss


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


def test_cross_entropy_accepts_numpy_integer_labels():
    logits = Tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True)
    labels = [np.int64(0), np.int32(1)]
    loss = CrossEntropyLoss()(logits, labels)

    assert loss.item() == pytest.approx(0.1269280110)
    loss.backward()
    assert logits.grad[0][0] < 0 < logits.grad[0][1]
    assert logits.grad[1][1] < 0 < logits.grad[1][0]


def test_cross_entropy_rejects_numpy_boolean_labels():
    logits = Tensor([[0.1, 0.2]], requires_grad=True)
    with pytest.raises(TypeError, match="integer indices"):
        CrossEntropyLoss()(logits, [np.bool_(True)])


@pytest.mark.parametrize("labels", [[-1], [2], [True], [0, 1]])
def test_cross_entropy_rejects_invalid_class_targets(labels):
    logits = Tensor([[0.1, 0.2]], requires_grad=True)
    with pytest.raises((TypeError, ValueError)):
        CrossEntropyLoss()(logits, labels)

def _central_difference(values, objective, epsilon=1e-6):
    """Estimate every elementwise derivative using central differences."""
    values = np.asarray(values, dtype=np.float64)
    numerical = np.zeros_like(values)
    for index in np.ndindex(values.shape):
        plus = values.copy()
        minus = values.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        numerical[index] = (objective(plus) - objective(minus)) / (2.0 * epsilon)
    return numerical


def _assert_gradient_matches(actual, numerical, *, rtol=2e-6, atol=2e-7):
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float64),
        numerical,
        rtol=rtol,
        atol=atol,
    )


def test_addition_gradients_match_central_finite_differences():
    left_values = np.array([[0.4, -1.2], [2.1, 0.7]])
    right_values = np.array([[1.3, 0.2], [-0.8, 1.6]])
    upstream = np.array([[0.5, -1.0], [1.2, 0.3]])

    left = Tensor(left_values.tolist(), requires_grad=True)
    right = Tensor(right_values.tolist(), requires_grad=True)
    (left + right).backward(upstream.tolist())

    left_numerical = _central_difference(
        left_values, lambda x: float(np.sum((x + right_values) * upstream))
    )
    right_numerical = _central_difference(
        right_values, lambda x: float(np.sum((left_values + x) * upstream))
    )
    _assert_gradient_matches(left.grad, left_numerical)
    _assert_gradient_matches(right.grad, right_numerical)


def test_broadcast_multiply_gradients_match_central_finite_differences():
    matrix_values = np.array([[0.4, -1.2, 2.0], [1.1, 0.7, -0.3]])
    row_values = np.array([[1.3, 0.2, -0.8]])
    upstream = np.array([[0.5, -1.0, 1.2], [0.3, 0.8, -0.4]])

    matrix = Tensor(matrix_values.tolist(), requires_grad=True)
    row = Tensor(row_values.tolist(), requires_grad=True)
    (matrix * row).backward(upstream.tolist())

    matrix_numerical = _central_difference(
        matrix_values,
        lambda x: float(np.sum((x * row_values) * upstream)),
    )
    row_numerical = _central_difference(
        row_values,
        lambda x: float(np.sum((matrix_values * x) * upstream)),
    )
    _assert_gradient_matches(matrix.grad, matrix_numerical)
    _assert_gradient_matches(row.grad, row_numerical)


def test_matmul_gradients_match_central_finite_differences():
    left_values = np.array([[0.4, -1.2, 0.7], [1.1, 0.3, -0.5]])
    right_values = np.array([[1.3, 0.2], [-0.8, 1.6], [0.5, -0.9]])
    upstream = np.array([[0.5, -1.0], [1.2, 0.3]])

    left = Tensor(left_values.tolist(), requires_grad=True)
    right = Tensor(right_values.tolist(), requires_grad=True)
    (left @ right).backward(upstream.tolist())

    left_numerical = _central_difference(
        left_values, lambda x: float(np.sum((x @ right_values) * upstream))
    )
    right_numerical = _central_difference(
        right_values, lambda x: float(np.sum((left_values @ x) * upstream))
    )
    _assert_gradient_matches(left.grad, left_numerical)
    _assert_gradient_matches(right.grad, right_numerical)


def test_relu_gradients_match_central_finite_differences_away_from_zero():
    values = np.array([[-1.4, 0.6, 2.2], [0.3, -0.8, 1.7]])
    upstream = np.array([[0.5, -1.0, 1.2], [0.3, 0.8, -0.4]])

    tensor = Tensor(values.tolist(), requires_grad=True)
    tensor.relu().backward(upstream.tolist())

    numerical = _central_difference(
        values,
        lambda x: float(np.sum(np.maximum(x, 0.0) * upstream)),
    )
    _assert_gradient_matches(tensor.grad, numerical)


def test_shared_graph_gradient_matches_central_finite_difference():
    value = 1.7
    tensor = Tensor(value, requires_grad=True)
    shared = tensor * tensor
    result = shared * shared + shared
    result.backward()

    epsilon = 1e-6
    objective = lambda x: (x * x) * (x * x) + (x * x)
    numerical = (
        objective(value + epsilon) - objective(value - epsilon)
    ) / (2.0 * epsilon)
    assert tensor.grad == pytest.approx(numerical, rel=2e-6, abs=2e-7)


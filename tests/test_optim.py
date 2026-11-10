"""Regression tests for OddlyTyped's parameter update algorithms."""

import math

import numpy as np
import pytest

from oddlytyped import Tensor
from oddlytyped.optim import Adam, SGD


def test_sgd_updates_scalar_and_matrix_parameters():
    scalar = Tensor(2.0, requires_grad=True)
    matrix = Tensor([[1.0, -2.0], [3.0, 4.0]], requires_grad=True)
    scalar.grad = 0.5
    matrix.grad = [[1.0, -1.0], [0.25, 2.0]]
    SGD([scalar, matrix], lr=0.1).step()
    assert scalar.data == pytest.approx(1.95)
    assert matrix.data == [
        [pytest.approx(0.9), pytest.approx(-1.9)],
        [pytest.approx(2.975), pytest.approx(3.8)],
    ]


def test_sgd_zero_grad_clears_all_parameters():
    parameters = [Tensor(1.0, requires_grad=True), Tensor([[2.0, 3.0]], requires_grad=True)]
    parameters[0].grad = 4.0
    parameters[1].grad = [[5.0, 6.0]]
    SGD(parameters).zero_grad()
    assert parameters[0].grad == 0.0
    assert parameters[1].grad == [[0.0, 0.0]]


def test_adam_matches_two_step_bias_corrected_reference():
    parameter = Tensor(1.0, requires_grad=True)
    optimizer = Adam([parameter], lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8)
    parameter.grad = 2.0
    optimizer.step()
    first_m, first_v = 0.2, 0.004
    expected = 1.0 - 0.1 * (first_m / 0.1) / (math.sqrt(first_v / 0.001) + 1e-8)
    assert parameter.data == pytest.approx(expected, abs=1e-12)
    parameter.grad = 1.0
    optimizer.step()
    second_m = 0.9 * first_m + 0.1
    second_v = 0.999 * first_v + 0.001
    expected -= 0.1 * (second_m / (1.0 - 0.9 ** 2)) / (
        math.sqrt(second_v / (1.0 - 0.999 ** 2)) + 1e-8
    )
    assert parameter.data == pytest.approx(expected, abs=1e-12)
    assert optimizer.step_count == 2


def test_adam_zero_grad_preserves_moments():
    parameter = Tensor([[1.0, 2.0]], requires_grad=True)
    parameter.grad = [[1.0, -1.0]]
    optimizer = Adam([parameter])
    optimizer.step()
    moments_before = optimizer.first_moment[id(parameter)].copy()
    optimizer.zero_grad()
    assert parameter.grad == [[0.0, 0.0]]
    assert np.array_equal(optimizer.first_moment[id(parameter)], moments_before)


@pytest.mark.parametrize("learning_rate", [0.0, -0.1])
def test_sgd_rejects_non_positive_learning_rate(learning_rate):
    with pytest.raises(ValueError, match="positive"):
        SGD([Tensor(1.0)], lr=learning_rate)


@pytest.mark.parametrize("learning_rate", [float("nan"), float("inf"), -float("inf")])
def test_sgd_rejects_non_finite_learning_rate(learning_rate):
    with pytest.raises(ValueError, match="finite"):
        SGD([Tensor(1.0)], lr=learning_rate)


@pytest.mark.parametrize("kwargs", [
    {"lr": 0.0}, {"lr": -0.1}, {"eps": 0.0}, {"beta1": -0.1},
    {"beta1": 1.0}, {"beta2": -0.1}, {"beta2": 1.0},
])
def test_adam_rejects_invalid_hyperparameters(kwargs):
    with pytest.raises(ValueError):
        Adam([Tensor(1.0)], **kwargs)


@pytest.mark.parametrize("kwargs", [
    {"lr": float("nan")}, {"lr": float("inf")}, {"eps": float("nan")},
    {"eps": float("inf")}, {"beta1": float("nan")}, {"beta1": float("inf")},
    {"beta2": float("nan")}, {"beta2": float("inf")},
])
def test_adam_rejects_non_finite_hyperparameters(kwargs):
    with pytest.raises(ValueError, match="finite"):
        Adam([Tensor(1.0)], **kwargs)


def test_adam_updates_matrix_parameters_and_uses_numpy_moments():
    parameter = Tensor([[1.0, -1.0], [2.0, -2.0]], requires_grad=True)
    parameter.grad = [[0.5, -0.5], [1.0, -1.0]]
    optimizer = Adam([parameter], lr=0.1)
    optimizer.step()
    assert parameter.data == [
        [pytest.approx(0.9), pytest.approx(-0.9)],
        [pytest.approx(1.9), pytest.approx(-1.9)],
    ]
    assert isinstance(optimizer.first_moment[id(parameter)], np.ndarray)
    assert optimizer.first_moment[id(parameter)].dtype == np.float64
    assert optimizer.first_moment[id(parameter)].shape == parameter.shape


@pytest.mark.parametrize("optimizer_type", [SGD, Adam])
def test_optimizers_reject_duplicate_parameter_references(optimizer_type):
    parameter = Tensor(1.0, requires_grad=True)
    with pytest.raises(ValueError, match="more than once"):
        optimizer_type([parameter, parameter])


@pytest.mark.parametrize("optimizer_type", [SGD, Adam])
@pytest.mark.parametrize("invalid_field", ["data", "grad"])
@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf")])
def test_optimizers_reject_non_finite_initial_state(
    optimizer_type, invalid_field, invalid_value
):
    valid = Tensor(2.0, requires_grad=True)
    invalid = Tensor(3.0, requires_grad=True)
    valid.grad = 0.5
    invalid.grad = 0.25
    setattr(invalid, invalid_field, invalid_value)
    optimizer = optimizer_type([valid, invalid])
    with pytest.raises(ValueError, match="finite"):
        optimizer.step()
    assert valid.data == pytest.approx(2.0)
    if optimizer_type is Adam:
        assert optimizer.step_count == 0


def test_sgd_overflow_does_not_partially_update_parameters():
    unchanged = Tensor(2.0, requires_grad=True)
    overflow = Tensor(1e308, requires_grad=True)
    unchanged.grad = 1.0
    overflow.grad = -1e308
    optimizer = SGD([unchanged, overflow], lr=2.0)
    with pytest.raises(FloatingPointError, match="no state was updated"):
        optimizer.step()
    assert unchanged.data == pytest.approx(2.0)
    assert overflow.data == pytest.approx(1e308)


def test_adam_overflow_does_not_change_parameters_moments_or_step_count():
    unchanged = Tensor(2.0, requires_grad=True)
    overflow = Tensor(3.0, requires_grad=True)
    unchanged.grad = 1.0
    overflow.grad = 1e308
    optimizer = Adam([unchanged, overflow])
    first_before = {key: value.copy() if isinstance(value, np.ndarray) else value
                    for key, value in optimizer.first_moment.items()}
    second_before = {key: value.copy() if isinstance(value, np.ndarray) else value
                     for key, value in optimizer.second_moment.items()}
    with pytest.raises(FloatingPointError, match="no state was updated"):
        optimizer.step()
    assert unchanged.data == pytest.approx(2.0)
    assert overflow.data == pytest.approx(3.0)
    assert optimizer.step_count == 0
    for key in first_before:
        assert np.array_equal(optimizer.first_moment[key], first_before[key])
        assert np.array_equal(optimizer.second_moment[key], second_before[key])

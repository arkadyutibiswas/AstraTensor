"""Regression tests for AstraTensor's parameter update algorithms."""

import math

import numpy as np
import pytest

from astratensor import Tensor
from astratensor.optim import Adam, SGD


def test_sgd_updates_scalar_and_matrix_parameters():
    scalar = Tensor(2.0, requires_grad=True)
    matrix = Tensor([[1.0, -2.0], [3.0, 4.0]], requires_grad=True)
    scalar.grad = 0.5
    matrix.grad = [[1.0, -1.0], [0.25, 2.0]]

    optimizer = SGD([scalar, matrix], lr=0.1)
    optimizer.step()

    assert scalar.data == pytest.approx(1.95)
    assert matrix.data == [
        [pytest.approx(0.9), pytest.approx(-1.9)],
        [pytest.approx(2.975), pytest.approx(3.8)],
    ]


def test_sgd_zero_grad_clears_all_parameters():
    parameters = [Tensor(1.0, requires_grad=True),
                  Tensor([[2.0, 3.0]], requires_grad=True)]
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
    first_m = 0.2
    first_v = 0.004
    first_m_hat = first_m / (1.0 - 0.9)
    first_v_hat = first_v / (1.0 - 0.999)
    expected = 1.0 - 0.1 * first_m_hat / (math.sqrt(first_v_hat) + 1e-8)
    assert parameter.data == pytest.approx(expected, abs=1e-12)

    parameter.grad = 1.0
    optimizer.step()
    second_m = 0.9 * first_m + 0.1
    second_v = 0.999 * first_v + 0.001
    second_m_hat = second_m / (1.0 - 0.9 ** 2)
    second_v_hat = second_v / (1.0 - 0.999 ** 2)
    expected -= 0.1 * second_m_hat / (math.sqrt(second_v_hat) + 1e-8)
    assert parameter.data == pytest.approx(expected, abs=1e-12)
    assert optimizer.step_count == 2


def test_adam_zero_grad_clears_parameters_without_resetting_moments():
    parameter = Tensor([[1.0, 2.0]], requires_grad=True)
    parameter.grad = [[1.0, -1.0]]
    optimizer = Adam([parameter])
    optimizer.step()
    moments_before = optimizer.first_moment[id(parameter)]

    optimizer.zero_grad()

    assert parameter.grad == [[0.0, 0.0]]
    assert np.array_equal(optimizer.first_moment[id(parameter)], moments_before)


@pytest.mark.parametrize("learning_rate", [0.0, -0.1])
def test_sgd_rejects_non_positive_learning_rate(learning_rate):
    with pytest.raises(ValueError, match="positive"):
        SGD([Tensor(1.0)], lr=learning_rate)


@pytest.mark.parametrize("kwargs", [
    {"lr": 0.0},
    {"lr": -0.1},
    {"eps": 0.0},
    {"beta1": -0.1},
    {"beta1": 1.0},
    {"beta2": -0.1},
    {"beta2": 1.0},
])
def test_adam_rejects_invalid_hyperparameters(kwargs):
    with pytest.raises(ValueError):
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

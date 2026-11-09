import pytest

from astratensor import Tensor
from astratensor.nn import CrossEntropyLoss, Linear, MSELoss
from astratensor.optim import Adam, SGD


def test_linear_forward_and_bias_gradient():
    layer = Linear(2, 1, seed=7)
    layer.weight.data = [[2.0], [-1.0]]
    layer.bias.data = [[0.5]]
    result = layer(Tensor([[1.0, 3.0], [2.0, 1.0]]))
    assert result.data == [[-0.5], [3.5]]
    result.backward([[1.0], [2.0]])
    assert layer.weight.grad == [[5.0], [5.0]]
    assert layer.bias.grad == [[3.0]]


def test_mse_loss_value_and_gradient():
    prediction = Tensor([[2.0, 4.0]], requires_grad=True)
    loss = MSELoss()(prediction, [[1.0, 2.0]])
    assert loss.item() == pytest.approx(2.5)
    loss.backward()
    assert prediction.grad == [[1.0, 2.0]]


def test_cross_entropy_value_and_gradient():
    logits = Tensor([[2.0, 0.0]], requires_grad=True)
    loss = CrossEntropyLoss()(logits, [0])
    assert loss.item() == pytest.approx(0.1269280110)
    loss.backward()
    assert sum(logits.grad[0]) == pytest.approx(0.0, abs=1e-12)
    assert logits.grad[0][0] < 0 < logits.grad[0][1]


def test_sgd_updates_matrix_parameters():
    parameter = Tensor([[1.0, -1.0]], requires_grad=True)
    parameter.grad = [[0.5, -0.5]]
    SGD([parameter], lr=0.1).step()
    assert parameter.data == [[0.95, -0.95]]


def test_adam_updates_scalar_parameter():
    parameter = Tensor(1.0, requires_grad=True)
    parameter.grad = 2.0
    optimizer = Adam([parameter], lr=0.1)
    optimizer.step()
    assert parameter.data == pytest.approx(0.9)
    optimizer.zero_grad()
    assert parameter.grad == 0.0


def test_mse_rejects_wrong_shape():
    with pytest.raises(ValueError, match="shapes must match"):
        MSELoss()(Tensor([[1.0, 2.0]]), [[1.0]])

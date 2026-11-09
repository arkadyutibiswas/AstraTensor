"""Independent PyTorch oracle tests; PyTorch is never imported by framework code."""
import pytest

torch = pytest.importorskip("torch")

from oddlytyped import Tensor
from oddlytyped.nn import CrossEntropyLoss, Linear


TOL = 1e-5


def _assert_close(actual, expected):
    torch.testing.assert_close(
        torch.tensor(actual, dtype=torch.float64),
        expected.detach().to(dtype=torch.float64),
        rtol=TOL,
        atol=TOL,
    )


def test_elementwise_add_and_multiply_forward_and_gradients():
    a_values = [[0.5, -1.0], [2.0, 3.0]]
    b_values = [[1.5, 2.0], [-0.5, 4.0]]

    a = Tensor(a_values, requires_grad=True)
    b = Tensor(b_values, requires_grad=True)
    actual = (a + b) * b
    actual.backward([[1.0, -2.0], [0.25, 3.0]])

    ta = torch.tensor(a_values, dtype=torch.float64, requires_grad=True)
    tb = torch.tensor(b_values, dtype=torch.float64, requires_grad=True)
    expected = (ta + tb) * tb
    expected.backward(torch.tensor([[1.0, -2.0], [0.25, 3.0]], dtype=torch.float64))

    _assert_close(actual.data, expected)
    _assert_close(a.grad, ta.grad)
    _assert_close(b.grad, tb.grad)


def test_matmul_forward_and_both_gradients():
    a_values = [[1.0, 2.0, -1.0], [0.5, -2.0, 3.0]]
    b_values = [[2.0, 1.0], [-1.0, 0.5], [3.0, -2.0]]
    upstream = [[0.5, -1.0], [2.0, 0.25]]

    a = Tensor(a_values, requires_grad=True)
    b = Tensor(b_values, requires_grad=True)
    actual = a @ b
    actual.backward(upstream)

    ta = torch.tensor(a_values, dtype=torch.float64, requires_grad=True)
    tb = torch.tensor(b_values, dtype=torch.float64, requires_grad=True)
    expected = ta @ tb
    expected.backward(torch.tensor(upstream, dtype=torch.float64))

    _assert_close(actual.data, expected)
    _assert_close(a.grad, ta.grad)
    _assert_close(b.grad, tb.grad)


def test_relu_forward_and_gradient():
    values = [[-2.0, 0.0, 1.5], [3.0, -0.25, 4.0]]
    upstream = [[1.0, 2.0, -1.0], [0.5, 3.0, 2.0]]

    actual_input = Tensor(values, requires_grad=True)
    actual = actual_input.relu()
    actual.backward(upstream)

    expected_input = torch.tensor(values, dtype=torch.float64, requires_grad=True)
    expected = torch.relu(expected_input)
    expected.backward(torch.tensor(upstream, dtype=torch.float64))

    _assert_close(actual.data, expected)
    _assert_close(actual_input.grad, expected_input.grad)


def test_linear_layer_forward_and_parameter_gradients():
    inputs = [[1.0, -2.0], [0.5, 3.0]]
    weights = [[0.25, -1.0], [2.0, 0.5]]
    bias = [[0.1, -0.2]]
    upstream = [[1.0, -0.5], [2.0, 3.0]]

    layer = Linear(2, 2, seed=4)
    layer.weight.data = [row[:] for row in weights]
    layer.bias.data = [row[:] for row in bias]
    actual = layer(Tensor(inputs))
    actual.backward(upstream)

    tw = torch.tensor(weights, dtype=torch.float64, requires_grad=True)
    tb = torch.tensor(bias, dtype=torch.float64, requires_grad=True)
    tx = torch.tensor(inputs, dtype=torch.float64)
    expected = tx @ tw + tb
    expected.backward(torch.tensor(upstream, dtype=torch.float64))

    _assert_close(actual.data, expected)
    _assert_close(layer.weight.grad, tw.grad)
    _assert_close(layer.bias.grad, tb.grad)


def test_cross_entropy_forward_and_gradient():
    values = [[2.0, -1.0, 0.5], [0.1, 1.2, -0.7]]
    labels = [0, 2]

    actual_input = Tensor(values, requires_grad=True)
    actual = CrossEntropyLoss()(actual_input, labels)
    actual.backward()

    expected_input = torch.tensor(values, dtype=torch.float64, requires_grad=True)
    expected = torch.nn.functional.cross_entropy(
        expected_input, torch.tensor(labels, dtype=torch.long)
    )
    expected.backward()

    assert actual.item() == pytest.approx(expected.item(), rel=TOL, abs=TOL)
    _assert_close(actual_input.grad, expected_input.grad)

"""Numerical and graph-behavior tests for AstraTensor reverse-mode autograd."""
import pytest

from astratensor import Tensor


def test_scalar_add_and_multiply_accumulate_gradients():
    x = Tensor(3.0, requires_grad=True)
    y = x * x + x
    y.backward()
    assert y.item() == pytest.approx(12.0)
    assert x.grad == pytest.approx(7.0)


def test_scalar_broadcast_gradient_reduces_all_elements():
    matrix = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    scale = Tensor(2.0, requires_grad=True)
    result = matrix * scale
    result.backward([[1.0, 1.0], [1.0, 1.0]])
    assert matrix.grad == [[2.0, 2.0], [2.0, 2.0]]
    assert scale.grad == pytest.approx(10.0)


def test_matrix_matmul_gradients():
    left = Tensor([[1.0, 2.0]], requires_grad=True)
    right = Tensor([[3.0], [4.0]], requires_grad=True)
    output = left @ right
    output.backward([[1.0]])
    assert output.data == [[11.0]]
    assert left.grad == [[3.0, 4.0]]
    assert right.grad == [[1.0], [2.0]]


def test_relu_gradient_is_zero_for_non_positive_inputs():
    values = Tensor([[-2.0, 0.0, 3.0]], requires_grad=True)
    output = values.relu()
    output.backward([[5.0, 5.0, 5.0]])
    assert output.data == [[0.0, 0.0, 3.0]]
    assert values.grad == [[0.0, 0.0, 5.0]]


def test_shared_subgraph_accumulates_both_paths():
    x = Tensor(2.0, requires_grad=True)
    shared = x * x
    result = shared + shared
    result.backward()
    assert x.grad == pytest.approx(8.0)


def test_leaf_gradients_accumulate_across_backward_calls():
    x = Tensor(4.0, requires_grad=True)
    (x * x).backward()
    assert x.grad == pytest.approx(8.0)
    (x * 3.0).backward()
    assert x.grad == pytest.approx(11.0)
    x.zero_grad()
    assert x.grad == pytest.approx(0.0)


def test_matrix_output_requires_explicit_upstream_gradient():
    x = Tensor([[1.0, 2.0]], requires_grad=True)
    with pytest.raises(ValueError, match="explicit gradient"):
        x.backward()


def test_backward_rejects_wrong_gradient_shape():
    x = Tensor([[1.0, 2.0]], requires_grad=True)
    with pytest.raises(ValueError, match="match output shape"):
        x.backward([[1.0]])

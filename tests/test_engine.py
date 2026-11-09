import pytest

from oddlytyped import Tensor


def test_scalar_add_and_multiply_accumulate_gradients():
    a = Tensor(3.0, requires_grad=True)
    b = Tensor(4.0, requires_grad=True)
    result = a * b + a
    result.backward()
    assert a.grad == pytest.approx(5.0)
    assert b.grad == pytest.approx(3.0)


def test_matrix_multiplication_forward_and_backward():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = Tensor([[2.0], [1.0]], requires_grad=True)
    result = a @ b
    assert result.data == [[4.0], [10.0]]
    result.backward([[1.0], [1.0]])
    assert a.grad == [[2.0, 1.0], [2.0, 1.0]]
    assert b.grad == [[4.0], [6.0]]


def test_scalar_broadcast_reduces_gradient():
    scalar = Tensor(2.0, requires_grad=True)
    matrix = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    (scalar * matrix).backward([[1.0, 1.0], [1.0, 1.0]])
    assert scalar.grad == pytest.approx(10.0)
    assert matrix.grad == [[2.0, 2.0], [2.0, 2.0]]


def test_relu_masks_non_positive_values():
    values = Tensor([[-1.0, 0.0, 2.0]], requires_grad=True)
    values.relu().backward([[1.0, 1.0, 1.0]])
    assert values.grad == [[0.0, 0.0, 1.0]]


def test_shared_subgraph_accumulates_both_paths():
    x = Tensor(3.0, requires_grad=True)
    shared = x * x
    (shared + shared).backward()
    assert x.grad == pytest.approx(12.0)


def test_intermediate_gradients_reset_but_leaf_gradients_accumulate():
    x = Tensor(2.0, requires_grad=True)
    y = x * x
    y.backward()
    assert x.grad == pytest.approx(4.0)
    y.backward()
    assert x.grad == pytest.approx(8.0)


def test_matrix_output_requires_explicit_gradient():
    x = Tensor([[1.0, 2.0]], requires_grad=True)
    with pytest.raises(ValueError, match="explicit gradient"):
        x.relu().backward()


def test_invalid_shapes_are_rejected():
    with pytest.raises(ValueError, match="rectangular"):
        Tensor([[1.0, 2.0], [3.0]])
    with pytest.raises(ValueError, match="Incompatible"):
        Tensor([[1.0, 2.0]]) @ Tensor([[1.0, 2.0]])


def test_non_grad_tensor_does_not_receive_gradient():
    x = Tensor(2.0, requires_grad=False)
    y = Tensor(5.0, requires_grad=True)
    (x * y).backward()
    assert x.grad == 0.0
    assert y.grad == pytest.approx(2.0)


def test_matrix_row_broadcast_reduces_gradient_to_original_shape():
    row = Tensor([[1.0, 2.0]], requires_grad=True)
    matrix = Tensor([[3.0, 4.0], [5.0, 6.0]], requires_grad=True)
    (row + matrix).backward([[1.0, 2.0], [3.0, 4.0]])
    assert row.grad == [[4.0, 6.0]]
    assert matrix.grad == [[1.0, 2.0], [3.0, 4.0]]


def test_data_and_gradient_setters_reject_shape_changes():
    tensor = Tensor([[1.0, 2.0]], requires_grad=True)
    with pytest.raises(ValueError, match="data shape cannot be changed"):
        tensor.data = [[1.0], [2.0]]
    with pytest.raises(ValueError, match="Gradient shape must match"):
        tensor.grad = [[1.0], [2.0]]

    tensor.data = [[3.0, 4.0]]
    tensor.grad = [[0.5, 0.25]]
    assert tensor.data == [[3.0, 4.0]]
    assert tensor.grad == [[0.5, 0.25]]

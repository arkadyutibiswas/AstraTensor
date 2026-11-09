import numpy as np
import pytest

from astratensor import Tensor


def test_matrix_storage_is_internal_numpy_and_public_data_is_detached():
    tensor = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    assert isinstance(tensor._data, np.ndarray)
    assert tensor._data.dtype == np.float64
    assert tensor._data.flags.c_contiguous
    exposed = tensor.data
    exposed[0][0] = 99.0
    assert tensor.data == [[1.0, 2.0], [3.0, 4.0]]


def test_matrix_gradients_are_internal_numpy_and_public_grad_is_detached():
    tensor = Tensor([[1.0, -2.0]], requires_grad=True)
    tensor.relu().backward([[2.0, 3.0]])
    assert isinstance(tensor._grad, np.ndarray)
    assert tensor.grad == [[2.0, 0.0]]
    exposed = tensor.grad
    exposed[0][0] = 99.0
    assert tensor.grad == [[2.0, 0.0]]


def test_numpy_input_is_copied_and_keeps_public_list_api():
    source = np.array([[1.0, 2.0]], dtype=np.float32)
    tensor = Tensor(source)
    source[0, 0] = 7.0
    assert tensor.data == [[1.0, 2.0]]


def test_scalar_and_matrix_broadcast_gradients_with_array_storage():
    scalar = Tensor(2.0, requires_grad=True)
    matrix = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    (scalar * matrix + 1.0).backward([[1.0, 1.0], [1.0, 1.0]])
    assert scalar.grad == pytest.approx(10.0)
    assert matrix.grad == [[2.0, 2.0], [2.0, 2.0]]

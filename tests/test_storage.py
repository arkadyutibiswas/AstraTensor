import numpy as np
import pytest

from oddlytyped.storage import MatrixStorage


def test_storage_owns_contiguous_float64_copy():
    source = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)[:, ::-1]
    storage = MatrixStorage(source)
    assert storage.shape == (2, 2)
    assert storage.array.dtype == np.float64
    assert storage.array.flags.c_contiguous
    source[0, 0] = 99.0
    assert storage.to_list() == [[2.0, 1.0], [4.0, 3.0]]


def test_list_conversion_is_detached():
    storage = MatrixStorage([[1.0, 2.0]])
    result = storage.to_list()
    result[0][0] = 99.0
    assert storage.to_list() == [[1.0, 2.0]]


def test_copy_has_independent_array():
    original = MatrixStorage([[1.0, 2.0]])
    duplicate = original.copy()
    duplicate.array[0, 0] = 5.0
    assert original.to_list() == [[1.0, 2.0]]
    assert duplicate.to_list() == [[5.0, 2.0]]


@pytest.mark.parametrize("invalid", [[], [[]], [1.0, 2.0], [[1.0], [2.0, 3.0]]])
def test_invalid_matrix_shapes_are_rejected(invalid):
    with pytest.raises(ValueError):
        MatrixStorage(invalid)


def test_non_finite_values_are_rejected():
    with pytest.raises(ValueError, match="finite"):
        MatrixStorage([[float("nan")]])

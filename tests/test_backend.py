"""Contract tests for the compiled PyBind11/C matrix backend.

These tests intentionally require the extension to be built: CI builds it before
running pytest, so a missing or broken native module must fail the job.
"""
import numpy as np
import pytest

from oddlytyped.backend import matmul


def test_matmul_accepts_python_nested_lists():
    result = matmul([[1.0, 2.0], [3.0, 4.0]], [[2.0], [1.0]])
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64
    assert result.shape == (2, 1)
    np.testing.assert_allclose(result, [[4.0], [10.0]], rtol=0.0, atol=1e-12)


def test_matmul_accepts_numpy_arrays():
    left = np.array([[1.5, -2.0, 0.5]], dtype=np.float64)
    right = np.array([[2.0], [3.0], [-4.0]], dtype=np.float64)
    np.testing.assert_allclose(matmul(left, right), [[-5.0]], atol=1e-12)


def test_matmul_handles_non_contiguous_numpy_views():
    left_storage = np.arange(12, dtype=np.float64).reshape(3, 4)
    right_storage = np.arange(12, dtype=np.float64).reshape(4, 3)
    left = left_storage[:, ::2]
    right = right_storage[::2, :]
    assert not left.flags.c_contiguous
    assert not right.flags.c_contiguous
    np.testing.assert_allclose(matmul(left, right), left @ right, atol=1e-12)


def test_matmul_copies_unaligned_contiguous_numpy_inputs():
    left_buffer = bytearray(1 + 4 * np.dtype(np.float64).itemsize)
    right_buffer = bytearray(1 + 4 * np.dtype(np.float64).itemsize)
    left = np.ndarray((2, 2), dtype=np.float64, buffer=left_buffer, offset=1)
    right = np.ndarray((2, 2), dtype=np.float64, buffer=right_buffer, offset=1)
    left[:] = [[1.0, 2.0], [3.0, 4.0]]
    right[:] = [[5.0, 6.0], [7.0, 8.0]]

    assert left.flags.c_contiguous and not left.flags.aligned
    assert right.flags.c_contiguous and not right.flags.aligned
    np.testing.assert_allclose(matmul(left, right), left @ right, atol=1e-12)


def test_matmul_rejects_incompatible_dimensions():
    with pytest.raises(ValueError, match="incompatible"):
        matmul([[1.0, 2.0]], [[1.0, 2.0]])


def test_matmul_rejects_non_matrix_inputs():
    with pytest.raises(ValueError, match="2-D"):
        matmul([1.0, 2.0], [[1.0], [2.0]])


@pytest.mark.parametrize(
    "left,right",
    [
        (np.empty((0, 2)), np.empty((2, 1))),
        (np.empty((1, 0)), np.empty((0, 2))),
        (np.empty((2, 1)), np.empty((1, 0))),
    ],
)
def test_matmul_rejects_empty_matrix_dimensions(left, right):
    with pytest.raises(ValueError, match="non-zero"):
        matmul(left, right)

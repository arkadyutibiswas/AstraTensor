"""Experimental NumPy-backed matrix storage for the next Tensor engine.

This module is an isolated migration prototype. The public Tensor engine still
uses list-backed storage until the adapter contract has been validated.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np


class MatrixStorage:
    """Own a contiguous float64 2-D array with explicit list conversion.

    Inputs are copied so callers cannot mutate storage through an aliased
    source array. to_list returns a detached compatibility representation.
    """

    __slots__ = ("_array",)

    def __init__(self, values: Iterable[Iterable[float]]):
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2:
            raise ValueError("MatrixStorage requires a 2-D matrix.")
        if array.shape[0] == 0 or array.shape[1] == 0:
            raise ValueError("A matrix must have at least one row and one column.")
        if not np.isfinite(array).all():
            raise ValueError("MatrixStorage values must be finite.")
        self._array = np.array(array, dtype=np.float64, order="C", copy=True)

    @property
    def shape(self) -> tuple[int, int]:
        return self._array.shape

    @property
    def array(self) -> np.ndarray:
        """Return the owned array for internal engine/backend operations."""
        return self._array

    def to_list(self) -> list[list[float]]:
        """Return a detached nested-list representation for public compatibility."""
        return self._array.tolist()

    def copy(self) -> "MatrixStorage":
        return MatrixStorage(self._array)

    def __repr__(self) -> str:
        return f"MatrixStorage(shape={self.shape}, dtype=float64)"

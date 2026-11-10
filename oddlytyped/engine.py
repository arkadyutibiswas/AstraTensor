"""Scalar and matrix tensors with reverse-mode automatic differentiation."""
from __future__ import annotations

from typing import Callable
import warnings

import numpy as np


def _is_matrix(value):
    return isinstance(value, (list, tuple, np.ndarray))


def _normalise(data):
    if _is_matrix(data):
        try:
            array = np.asarray(data, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("Tensor matrices must be rectangular and numeric.") from exc
        if array.ndim != 2:
            raise ValueError("Tensor matrices must be 2-D.")
        if array.shape[0] == 0 or array.shape[1] == 0:
            raise ValueError("A matrix must have at least one row and one column.")
        return np.array(array, dtype=np.float64, order="C", copy=True)
    if isinstance(data, (int, float)):
        return float(data)
    raise TypeError("Tensor data must be a number or a 2-D nested list.")


def _public(value):
    return value.tolist() if isinstance(value, np.ndarray) else value


def _shape(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        return tuple(np.shape(value))
    return ()


def _zeros_like(value):
    return np.zeros_like(value, dtype=np.float64) if isinstance(value, np.ndarray) else 0.0


def _transpose(value):
    return value.T


# Resolve the optional extension once. Repeated matrix operations should not
# repeatedly attempt an import when the native module is unavailable.
_NATIVE_MATMUL = None
_BACKEND_CHECKED = False


def _matrix_matmul(left, right):
    global _NATIVE_MATMUL, _BACKEND_CHECKED

    if not _BACKEND_CHECKED:
        try:
            from .backend import matmul as native_matmul
        except ModuleNotFoundError as exc:
            # Fall back only when the extension itself is absent. An import
            # failure inside an installed backend can indicate a broken build
            # or missing dependency and must remain visible to the caller.
            if exc.name != f"{__package__}.backend":
                raise
            warnings.warn(
                "OddlyTyped native backend is unavailable; using NumPy "
                f"matrix multiplication instead ({exc}).",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            _NATIVE_MATMUL = native_matmul

        # Mark resolution complete only after import/fallback handling finishes.
        # Unexpected import errors or warnings promoted to exceptions must not
        # leave the module in a falsely initialized state.
        _BACKEND_CHECKED = True

    if _NATIVE_MATMUL is None:
        return np.matmul(left, right)
    return _NATIVE_MATMUL(left, right)


def _elementwise(left, right, operation):
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return operation(left, right)
    return float(operation(left, right))


def _sum_all(value):
    return float(np.sum(value)) if isinstance(value, np.ndarray) else value


def _reduce_broadcast(gradient, original):
    """Sum broadcast axes so a gradient matches its input's original shape."""
    if not isinstance(original, np.ndarray):
        return _sum_all(gradient)
    result = np.asarray(gradient, dtype=np.float64)
    while result.ndim > original.ndim:
        result = result.sum(axis=0)
    for axis, size in enumerate(original.shape):
        if size == 1 and result.shape[axis] != 1:
            result = result.sum(axis=axis, keepdims=True)
    if result.shape != original.shape:
        raise ValueError("Gradient shapes do not match the original tensor.")
    return result


def _grad_add(left, right):
    # Convert matrix gradients before addition; Python lists would concatenate.
    left_is_matrix = isinstance(left, (list, tuple, np.ndarray))
    right_is_matrix = isinstance(right, (list, tuple, np.ndarray))
    if left_is_matrix or right_is_matrix:
        if not (left_is_matrix and right_is_matrix):
            raise ValueError("Gradient shapes do not match.")
        left_array = np.asarray(left, dtype=np.float64)
        right_array = np.asarray(right, dtype=np.float64)
        if left_array.shape != right_array.shape:
            raise ValueError("Gradient shapes do not match.")
        return left_array + right_array
    return left + right


class Tensor:
    """A scalar or 2-D matrix with reverse-mode automatic differentiation.

    Matrix data and gradients are stored internally as NumPy float64 arrays.
    The public data/grad properties return detached Python lists to preserve
    the original API while the engine avoids list conversions between ops.
    """

    def __init__(self, data, requires_grad=False, _children=(), _op=""):
        self._data = _normalise(data)
        self._grad = _zeros_like(self._data)
        self.requires_grad = bool(requires_grad)
        self._prev = tuple(_children)
        self._op = _op
        self._backward: Callable[[], None] = lambda: None

    @property
    def data(self):
        """Return a scalar or detached nested-list matrix."""
        return _public(self._data)

    @data.setter
    def data(self, value):
        normalised = _normalise(value)
        if _shape(normalised) != self.shape:
            raise ValueError("Tensor data shape cannot be changed.")
        self._data = normalised

    @property
    def grad(self):
        """Return a scalar or detached nested-list gradient."""
        return _public(self._grad)

    @grad.setter
    def grad(self, value):
        normalised = _normalise(value)
        if _shape(normalised) != self.shape:
            raise ValueError("Gradient shape must match tensor data shape.")
        self._grad = normalised

    @property
    def shape(self):
        """Return () for scalars or (rows, columns) for matrices."""
        return _shape(self._data)

    @property
    def is_scalar(self):
        return not isinstance(self._data, np.ndarray)

    def zero_grad(self):
        """Clear this tensor's accumulated gradient."""
        self._grad = _zeros_like(self._data)

    def _coerce(self, other):
        return other if isinstance(other, Tensor) else Tensor(other)

    def __add__(self, other):
        other = self._coerce(other)
        output = Tensor(
            _elementwise(self._data, other._data, np.add),
            self.requires_grad or other.requires_grad,
            (self, other), "add",
        )

        def backward():
            if self.requires_grad:
                contribution = _reduce_broadcast(output._grad, self._data)
                self._grad = _grad_add(self._grad, contribution)
            if other.requires_grad:
                contribution = _reduce_broadcast(output._grad, other._data)
                other._grad = _grad_add(other._grad, contribution)

        output._backward = backward
        return output

    __radd__ = __add__

    def __mul__(self, other):
        other = self._coerce(other)
        output = Tensor(
            _elementwise(self._data, other._data, np.multiply),
            self.requires_grad or other.requires_grad,
            (self, other), "multiply",
        )

        def backward():
            if self.requires_grad:
                contribution = _elementwise(output._grad, other._data, np.multiply)
                self._grad = _grad_add(
                    self._grad, _reduce_broadcast(contribution, self._data)
                )
            if other.requires_grad:
                contribution = _elementwise(output._grad, self._data, np.multiply)
                other._grad = _grad_add(
                    other._grad, _reduce_broadcast(contribution, other._data)
                )

        output._backward = backward
        return output

    __rmul__ = __mul__

    def __matmul__(self, other):
        other = self._coerce(other)
        if not isinstance(self._data, np.ndarray) or not isinstance(other._data, np.ndarray):
            raise ValueError("Matrix multiplication requires two 2-D tensors.")
        if self.shape[1] != other.shape[0]:
            raise ValueError(
                f"Incompatible matrix shapes: {self.shape} and {other.shape}."
            )

        output = Tensor(
            _matrix_matmul(self._data, other._data),
            self.requires_grad or other.requires_grad,
            (self, other), "matmul",
        )

        def backward():
            if self.requires_grad:
                contribution = _matrix_matmul(output._grad, _transpose(other._data))
                self._grad = _grad_add(self._grad, contribution)
            if other.requires_grad:
                contribution = _matrix_matmul(_transpose(self._data), output._grad)
                other._grad = _grad_add(other._grad, contribution)

        output._backward = backward
        return output

    def relu(self):
        """Apply max(0, x) elementwise."""
        if isinstance(self._data, np.ndarray):
            result = np.maximum(self._data, 0.0)
        else:
            result = max(0.0, self._data)
        output = Tensor(result, self.requires_grad, (self,), "relu")

        def backward():
            if not self.requires_grad:
                return
            if isinstance(self._data, np.ndarray):
                contribution = output._grad * (self._data > 0.0)
            else:
                contribution = output._grad if self._data > 0.0 else 0.0
            self._grad = _grad_add(self._grad, contribution)

        output._backward = backward
        return output

    def backward(self, gradient=None):
        """Backpropagate through the graph using iterative topological traversal."""
        if gradient is None:
            if not self.is_scalar:
                raise ValueError(
                    "Matrix outputs require an explicit gradient. "
                    "Backpropagate from a scalar loss where possible."
                )
            gradient = 1.0
        else:
            gradient = _normalise(gradient)
            if _shape(gradient) != self.shape:
                raise ValueError("The supplied gradient must match output shape.")

        topological = []
        visited = set()
        stack = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                topological.append(node)
                continue
            if node in visited:
                continue
            visited.add(node)
            stack.append((node, True))
            for parent in reversed(node._prev):
                if parent not in visited:
                    stack.append((parent, False))

        # Intermediate gradients reset per pass; leaf gradients accumulate.
        for node in topological:
            if node._prev:
                node._grad = _zeros_like(node._data)

        if self._prev:
            self._grad = gradient
        else:
            self._grad = _grad_add(self._grad, gradient)

        for node in reversed(topological):
            node._backward()

    def item(self):
        """Return a scalar value; reject matrices to avoid ambiguous behavior."""
        if not self.is_scalar:
            raise ValueError("item() is only available for scalar tensors.")
        return self._data

    def __repr__(self):
        return (
            f"Tensor(data={self.data!r}, grad={self.grad!r}, "
            f"requires_grad={self.requires_grad})"
        )

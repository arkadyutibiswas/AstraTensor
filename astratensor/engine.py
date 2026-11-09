"""A small reverse-mode automatic differentiation engine.

The first release deliberately supports scalars and rectangular 2-D matrices.
It uses Python lists for tensor storage; matrix multiplication dispatches to
AstraTensor's compiled C backend when that extension is available.
"""
from __future__ import annotations

from typing import Callable, Iterable


def _is_matrix(value):
    return isinstance(value, (list, tuple))


def _normalise(data):
    if _is_matrix(data):
        rows = [list(row) for row in data]
        if not rows or not rows[0]:
            raise ValueError("A matrix must have at least one row and one column.")
        width = len(rows[0])
        if any(not _is_matrix(row) or len(row) != width for row in rows):
            raise ValueError("Tensor matrices must be rectangular.")
        result = []
        for row in rows:
            converted = []
            for item in row:
                if not isinstance(item, (int, float)):
                    raise TypeError("Tensor values must be numeric.")
                converted.append(float(item))
            result.append(converted)
        return result
    if isinstance(data, (int, float)):
        return float(data)
    raise TypeError("Tensor data must be a number or a 2-D nested list.")


def _shape(value):
    return (len(value), len(value[0])) if _is_matrix(value) else ()


def _zeros_like(value):
    if _is_matrix(value):
        return [[0.0 for _ in row] for row in value]
    return 0.0


def _transpose(value):
    return [list(column) for column in zip(*value)]


def _matrix_matmul(left, right):
    try:
        from .backend import matmul as native_matmul
    except ImportError:
        # Pure-Python fallback keeps the autograd engine usable before building
        # the optional native extension. The C backend is used when installed.
        rows, inner, cols = len(left), len(left[0]), len(right[0])
        return [
            [sum(left[i][k] * right[k][j] for k in range(inner))
             for j in range(cols)]
            for i in range(rows)
        ]
    return native_matmul(left, right).tolist()


def _elementwise(left, right, operation):
    left_matrix, right_matrix = _is_matrix(left), _is_matrix(right)
    if left_matrix and right_matrix:
        if _shape(left) != _shape(right):
            raise ValueError("Matrix shapes must match for elementwise operations.")
        return [
            [operation(a, b) for a, b in zip(row_a, row_b)]
            for row_a, row_b in zip(left, right)
        ]
    if left_matrix:
        return [[operation(value, right) for value in row] for row in left]
    if right_matrix:
        return [[operation(left, value) for value in row] for row in right]
    return operation(left, right)


def _sum_all(value):
    return sum(sum(row) for row in value) if _is_matrix(value) else value


def _reduce_broadcast(gradient, original):
    """Reduce a broadcasted gradient back to the original scalar/matrix shape."""
    if not _is_matrix(original):
        return _sum_all(gradient)
    return gradient


def _grad_add(left, right):
    if _is_matrix(left):
        if not _is_matrix(right) or _shape(left) != _shape(right):
            raise ValueError("Gradient shapes do not match.")
        return [
            [a + b for a, b in zip(row_a, row_b)]
            for row_a, row_b in zip(left, right)
        ]
    if _is_matrix(right):
        raise ValueError("Gradient shapes do not match.")
    return left + right


class Tensor:
    """A scalar or 2-D matrix that records operations for reverse-mode AD.

    Args:
        data: A number or rectangular nested list.
        requires_grad: Whether to calculate and accumulate this tensor's gradient.
    """

    def __init__(self, data, requires_grad=False, _children=(), _op=""):
        self.data = _normalise(data)
        self.grad = _zeros_like(self.data)
        self.requires_grad = bool(requires_grad)
        self._prev = tuple(_children)
        self._op = _op
        self._backward: Callable[[], None] = lambda: None

    @property
    def shape(self):
        """Return () for scalars or (rows, columns) for matrices."""
        return _shape(self.data)

    @property
    def is_scalar(self):
        return not _is_matrix(self.data)

    def zero_grad(self):
        """Clear this tensor's accumulated gradient in place."""
        self.grad = _zeros_like(self.data)

    def _coerce(self, other):
        return other if isinstance(other, Tensor) else Tensor(other)

    def __add__(self, other):
        other = self._coerce(other)
        output = Tensor(
            _elementwise(self.data, other.data, lambda a, b: a + b),
            self.requires_grad or other.requires_grad,
            (self, other), "add",
        )

        def backward():
            if self.requires_grad:
                contribution = _reduce_broadcast(output.grad, self.data)
                self.grad = _grad_add(self.grad, contribution)
            if other.requires_grad:
                contribution = _reduce_broadcast(output.grad, other.data)
                other.grad = _grad_add(other.grad, contribution)

        output._backward = backward
        return output

    __radd__ = __add__

    def __mul__(self, other):
        other = self._coerce(other)
        output = Tensor(
            _elementwise(self.data, other.data, lambda a, b: a * b),
            self.requires_grad or other.requires_grad,
            (self, other), "multiply",
        )

        def backward():
            if self.requires_grad:
                contribution = _elementwise(
                    output.grad, other.data, lambda grad, value: grad * value
                )
                self.grad = _grad_add(
                    self.grad, _reduce_broadcast(contribution, self.data)
                )
            if other.requires_grad:
                contribution = _elementwise(
                    output.grad, self.data, lambda grad, value: grad * value
                )
                other.grad = _grad_add(
                    other.grad, _reduce_broadcast(contribution, other.data)
                )

        output._backward = backward
        return output

    __rmul__ = __mul__

    def __matmul__(self, other):
        other = self._coerce(other)
        if not _is_matrix(self.data) or not _is_matrix(other.data):
            raise ValueError("Matrix multiplication requires two 2-D tensors.")
        if self.shape[1] != other.shape[0]:
            raise ValueError(
                f"Incompatible matrix shapes: {self.shape} and {other.shape}."
            )

        output = Tensor(
            _matrix_matmul(self.data, other.data),
            self.requires_grad or other.requires_grad,
            (self, other), "matmul",
        )

        def backward():
            if self.requires_grad:
                contribution = _matrix_matmul(output.grad, _transpose(other.data))
                self.grad = _grad_add(self.grad, contribution)
            if other.requires_grad:
                contribution = _matrix_matmul(_transpose(self.data), output.grad)
                other.grad = _grad_add(other.grad, contribution)

        output._backward = backward
        return output

    def relu(self):
        """Apply max(0, x) elementwise."""
        if _is_matrix(self.data):
            result = [[max(0.0, value) for value in row] for row in self.data]
        else:
            result = max(0.0, self.data)
        output = Tensor(result, self.requires_grad, (self,), "relu")

        def backward():
            if not self.requires_grad:
                return
            if _is_matrix(self.data):
                contribution = [
                    [
                        output.grad[i][j] if self.data[i][j] > 0.0 else 0.0
                        for j in range(len(self.data[0]))
                    ]
                    for i in range(len(self.data))
                ]
            else:
                contribution = output.grad if self.data > 0.0 else 0.0
            self.grad = _grad_add(self.grad, contribution)

        output._backward = backward
        return output

    def backward(self, gradient=None):
        """Backpropagate from this tensor through its dynamically-built DAG.

        A scalar output defaults to an upstream gradient of 1. A matrix output
        requires an explicit gradient with exactly the same shape.
        """
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

        # Iterative DFS avoids Python recursion limits on deep computation graphs.
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

        # Intermediate gradients belong to this backward pass and must not
        # retain stale values. Leaf gradients intentionally accumulate.
        for node in topological:
            if node._prev:
                node.grad = _zeros_like(node.data)

        if self._prev:
            self.grad = gradient
        else:
            self.grad = _grad_add(self.grad, gradient)

        for node in reversed(topological):
            node._backward()

    def item(self):
        """Return a scalar value; reject matrices to avoid ambiguous behavior."""
        if not self.is_scalar:
            raise ValueError("item() is only available for scalar tensors.")
        return self.data

    def __repr__(self):
        return (
            f"Tensor(data={self.data!r}, grad={self.grad!r}, "
            f"requires_grad={self.requires_grad})"
        )

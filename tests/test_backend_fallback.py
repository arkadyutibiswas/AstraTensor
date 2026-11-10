"""Tests for native-backend discovery and NumPy fallback behaviour."""

import builtins
import warnings

import numpy as np
import pytest

import oddlytyped.engine as engine


def _block_backend_import(monkeypatch, error):
    original_import = builtins.__import__
    attempts = []

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        # A relative "from .backend import ..." reaches __import__ as
        # name="backend", level=1. Also accept its fully qualified spelling.
        if name == "backend" and level == 1 or name.endswith(".backend"):
            attempts.append((name, level))
            raise error
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(engine, "_NATIVE_MATMUL", None)
    monkeypatch.setattr(engine, "_BACKEND_CHECKED", False)
    return attempts


def test_missing_native_backend_warns_once_and_uses_numpy(monkeypatch):
    attempts = _block_backend_import(monkeypatch, ImportError("simulated missing extension"))
    left = np.array([[1.0, 2.0], [3.0, 4.0]])
    right = np.array([[2.0], [1.0]])

    with pytest.warns(RuntimeWarning, match="using NumPy"):
        first = engine._matrix_matmul(left, right)
    second = engine._matrix_matmul(left, right)

    np.testing.assert_allclose(first, left @ right)
    np.testing.assert_allclose(second, left @ right)
    assert len(attempts) == 1
    assert engine._BACKEND_CHECKED is True


def test_unexpected_backend_import_error_does_not_mark_lookup_complete(monkeypatch):
    attempts = _block_backend_import(monkeypatch, OSError("simulated loader failure"))

    with pytest.raises(OSError, match="simulated loader failure"):
        engine._matrix_matmul(np.eye(2), np.eye(2))

    assert len(attempts) == 1
    assert engine._BACKEND_CHECKED is False


def test_warning_promoted_to_error_does_not_mark_lookup_complete(monkeypatch):
    attempts = _block_backend_import(monkeypatch, ImportError("simulated missing extension"))

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(RuntimeWarning, match="using NumPy"):
            engine._matrix_matmul(np.eye(2), np.eye(2))

    assert len(attempts) == 1
    assert engine._BACKEND_CHECKED is False

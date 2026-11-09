"""Independent numerical checks; PyTorch is test-only, never a runtime dependency."""
import pytest

torch = pytest.importorskip("torch")

from oddlytyped import Tensor
from oddlytyped.nn import CrossEntropyLoss, MSELoss


def assert_matrix_close(actual, expected, tolerance=1e-5):
    assert len(actual) == len(expected)
    for actual_row, expected_row in zip(actual, expected):
        assert actual_row == pytest.approx(expected_row, abs=tolerance)


def test_matmul_forward_and_gradients_match_torch():
    left = [[0.2, -0.4], [1.1, 0.7]]
    right = [[0.3, 0.8], [-0.5, 0.9]]
    upstream = [[1.0, -0.5], [0.25, 2.0]]

    a = Tensor(left, requires_grad=True)
    b = Tensor(right, requires_grad=True)
    result = a @ b
    result.backward(upstream)

    ta = torch.tensor(left, dtype=torch.float64, requires_grad=True)
    tb = torch.tensor(right, dtype=torch.float64, requires_grad=True)
    expected = ta @ tb
    (expected * torch.tensor(upstream, dtype=torch.float64)).sum().backward()

    assert_matrix_close(result.data, expected.detach().tolist())
    assert_matrix_close(a.grad, ta.grad.tolist())
    assert_matrix_close(b.grad, tb.grad.tolist())


def test_cross_entropy_matches_torch():
    values = [[1.2, -0.3, 0.7], [0.1, 0.8, -0.2]]
    labels = [0, 2]
    actual = Tensor(values, requires_grad=True)
    actual_loss = CrossEntropyLoss()(actual, labels)
    actual_loss.backward()

    reference = torch.tensor(values, dtype=torch.float64, requires_grad=True)
    expected_loss = torch.nn.functional.cross_entropy(
        reference, torch.tensor(labels, dtype=torch.long)
    )
    expected_loss.backward()

    assert actual_loss.item() == pytest.approx(expected_loss.item(), abs=1e-5)
    assert_matrix_close(actual.grad, reference.grad.tolist())


def test_mse_matches_torch():
    values = [[1.0, 2.0], [-1.0, 3.0]]
    targets = [[0.5, 1.0], [0.0, 2.0]]
    actual = Tensor(values, requires_grad=True)
    actual_loss = MSELoss()(actual, targets)
    actual_loss.backward()

    reference = torch.tensor(values, dtype=torch.float64, requires_grad=True)
    expected_loss = torch.nn.functional.mse_loss(
        reference, torch.tensor(targets, dtype=torch.float64)
    )
    expected_loss.backward()

    assert actual_loss.item() == pytest.approx(expected_loss.item(), abs=1e-5)
    assert_matrix_close(actual.grad, reference.grad.tolist())

"""Tests for the deterministic nonlinear classification benchmark."""
from examples.train_synthetic import make_xor_clusters


def test_xor_cluster_dataset_is_deterministic_and_balanced():
    features_a, labels_a = make_xor_clusters(samples_per_cluster=8, seed=123)
    features_b, labels_b = make_xor_clusters(samples_per_cluster=8, seed=123)

    assert features_a == features_b
    assert labels_a == labels_b
    assert len(features_a) == 32
    assert len(labels_a) == 32
    assert labels_a.count(0) == 16
    assert labels_a.count(1) == 16
    assert all(len(point) == 2 for point in features_a)


def test_xor_cluster_dataset_rejects_invalid_size():
    import pytest

    with pytest.raises(ValueError, match="positive integer"):
        make_xor_clusters(samples_per_cluster=0)

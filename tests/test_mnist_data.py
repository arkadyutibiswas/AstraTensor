"""Tests for compressed MNIST IDX parsing and compact mini-batch conversion."""
import gzip
import struct

import pytest

from examples.mnist_data import load_split


def _write_images(path, images, rows=28, columns=28, magic=2051):
    with gzip.open(path, "wb") as stream:
        stream.write(struct.pack(">IIII", magic, len(images), rows, columns))
        stream.write(bytes(pixel for image in images for pixel in image))


def _write_labels(path, labels, magic=2049):
    with gzip.open(path, "wb") as stream:
        stream.write(struct.pack(">II", magic, len(labels)))
        stream.write(bytes(labels))


def test_load_split_keeps_pixels_compact_and_normalizes_only_requested_batch(tmp_path):
    first = bytearray(28 * 28)
    second = bytearray(28 * 28)
    first[0] = 255
    second[17] = 128
    image_path = tmp_path / "images.gz"
    label_path = tmp_path / "labels.gz"
    _write_images(image_path, [first, second])
    _write_labels(label_path, [3, 8])

    dataset = load_split(image_path, label_path)

    assert len(dataset) == 2
    assert isinstance(dataset.images, bytes)
    assert dataset.labels == (3, 8)
    batch = dataset.batch([1, 0])
    assert batch[0][17] == pytest.approx(128 / 255)
    assert batch[1][0] == pytest.approx(1.0)
    assert dataset.labels_for([1, 0]) == [8, 3]


def test_load_split_rejects_invalid_image_magic(tmp_path):
    image_path = tmp_path / "images.gz"
    label_path = tmp_path / "labels.gz"
    _write_images(image_path, [bytearray(28 * 28)], magic=123)
    _write_labels(label_path, [0])

    with pytest.raises(ValueError, match="magic"):
        load_split(image_path, label_path)


def test_load_split_rejects_image_label_count_mismatch(tmp_path):
    image_path = tmp_path / "images.gz"
    label_path = tmp_path / "labels.gz"
    _write_images(image_path, [bytearray(28 * 28), bytearray(28 * 28)])
    _write_labels(label_path, [0])

    with pytest.raises(ValueError, match="counts differ"):
        load_split(image_path, label_path)


def test_load_split_rejects_non_28_by_28_images(tmp_path):
    image_path = tmp_path / "images.gz"
    label_path = tmp_path / "labels.gz"
    _write_images(image_path, [bytearray(16 * 16)], rows=16, columns=16)
    _write_labels(label_path, [0])

    with pytest.raises(ValueError, match="28x28"):
        load_split(image_path, label_path)


def test_batch_rejects_out_of_range_indices(tmp_path):
    image_path = tmp_path / "images.gz"
    label_path = tmp_path / "labels.gz"
    _write_images(image_path, [bytearray(28 * 28)])
    _write_labels(label_path, [0])
    dataset = load_split(image_path, label_path)

    with pytest.raises(IndexError):
        dataset.batch([1])

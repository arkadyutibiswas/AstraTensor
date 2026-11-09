"""Memory-conscious MNIST IDX loading using only the Python standard library.

Images stay as compact uint8 bytes until a mini-batch is requested. This avoids
expanding all 60,000 images into millions of Python float objects at once.
"""
from dataclasses import dataclass
import gzip
from pathlib import Path
import struct
import urllib.request


_BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist"
_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


@dataclass(frozen=True)
class MNISTDataset:
    """Compact MNIST split with lazily normalized, flattened image batches."""

    images: bytes
    labels: tuple
    rows: int = 28
    columns: int = 28

    def __post_init__(self):
        if not isinstance(self.images, bytes):
            raise TypeError("images must be raw bytes.")
        if not isinstance(self.rows, int) or not isinstance(self.columns, int):
            raise TypeError("Image dimensions must be integers.")
        if self.rows <= 0 or self.columns <= 0:
            raise ValueError("Image dimensions must be positive.")
        if len(self.images) != len(self.labels) * self.rows * self.columns:
            raise ValueError("Image bytes and label count do not match.")
        if any(not isinstance(label, int) or isinstance(label, bool)
               or not 0 <= label <= 9 for label in self.labels):
            raise ValueError("MNIST labels must be integers from 0 through 9.")

    def __len__(self):
        return len(self.labels)

    @property
    def feature_count(self):
        return self.rows * self.columns

    def labels_for(self, indices):
        return [self.labels[index] for index in indices]

    def batch(self, indices):
        """Return selected images as flattened float rows normalized to [0, 1]."""
        width = self.feature_count
        result = []
        for index in indices:
            if not isinstance(index, int) or isinstance(index, bool):
                raise TypeError("Batch indices must be integers.")
            if index < 0 or index >= len(self):
                raise IndexError("Batch index is outside the dataset.")
            start = index * width
            pixels = self.images[start:start + width]
            result.append([pixel / 255.0 for pixel in pixels])
        return result


def _download_if_missing(directory, filename):
    destination = directory / filename
    if destination.is_file() and destination.stat().st_size:
        return destination

    temporary = destination.with_suffix(destination.suffix + ".part")
    url = f"{_BASE_URL}/{filename}"
    try:
        request = urllib.request.Request(
            url, headers={"User-Agent": "AstraTensor-MNIST-loader/1.0"}
        )
        with urllib.request.urlopen(request, timeout=45) as response:
            with temporary.open("wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
        temporary.replace(destination)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return destination


def _read_images(path):
    try:
        with gzip.open(path, "rb") as stream:
            header = stream.read(16)
            if len(header) != 16:
                raise ValueError("MNIST image file has a truncated header.")
            magic, count, rows, columns = struct.unpack(">IIII", header)
            if magic != 2051:
                raise ValueError(f"Invalid MNIST image magic number: {magic}.")
            if count <= 0 or rows <= 0 or columns <= 0:
                raise ValueError("MNIST image dimensions and count must be positive.")
            pixels = stream.read()
    except (OSError, EOFError) as error:
        raise ValueError(f"Could not read MNIST image file: {path}") from error

    expected = count * rows * columns
    if len(pixels) != expected:
        raise ValueError(
            f"MNIST image payload has {len(pixels)} bytes; expected {expected}."
        )
    return pixels, count, rows, columns


def _read_labels(path):
    try:
        with gzip.open(path, "rb") as stream:
            header = stream.read(8)
            if len(header) != 8:
                raise ValueError("MNIST label file has a truncated header.")
            magic, count = struct.unpack(">II", header)
            if magic != 2049:
                raise ValueError(f"Invalid MNIST label magic number: {magic}.")
            labels = stream.read()
    except (OSError, EOFError) as error:
        raise ValueError(f"Could not read MNIST label file: {path}") from error

    if len(labels) != count:
        raise ValueError(
            f"MNIST label payload has {len(labels)} bytes; expected {count}."
        )
    if any(label > 9 for label in labels):
        raise ValueError("MNIST contains a label outside the range 0 through 9.")
    return tuple(labels), count


def load_split(images_path, labels_path):
    """Load and validate one pair of compressed MNIST IDX files."""
    images, image_count, rows, columns = _read_images(images_path)
    labels, label_count = _read_labels(labels_path)
    if image_count != label_count:
        raise ValueError(
            f"MNIST image/label counts differ: {image_count} and {label_count}."
        )
    if rows != 28 or columns != 28:
        raise ValueError(f"Expected 28x28 MNIST images, received {rows}x{columns}.")
    return MNISTDataset(images, labels, rows, columns)


def load_mnist(data_dir="data/mnist", download=True):
    """Return ((training_images, training_labels), (test_images, test_labels)).

    Downloads the four official MNIST gzip files on first use when download=True.
    """
    directory = Path(data_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, filename in _FILES.items():
        path = directory / filename
        if download:
            path = _download_if_missing(directory, filename)
        elif not path.is_file():
            raise FileNotFoundError(
                f"Missing {path}. Enable download or place the MNIST gzip file there."
            )
        paths[key] = path

    training = load_split(paths["train_images"], paths["train_labels"])
    test = load_split(paths["test_images"], paths["test_labels"])
    return training, test

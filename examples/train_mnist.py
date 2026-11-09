"""Train an AstraTensor MLP on MNIST without PyTorch/TensorFlow.

The optional --smoke-test mode uses tiny in-memory fixtures so CI can validate
the complete training path without downloading MNIST.
"""
import argparse
import random

from astratensor import Tensor
from astratensor.nn import CrossEntropyLoss, Linear, ReLU
from astratensor.optim import Adam
from mnist_data import MNISTDataset, limit_dataset, load_mnist


class DigitMLP:
    """784 -> hidden ReLU -> 10 logits classifier."""

    def __init__(self, hidden_size=128, seed=2026):
        self.hidden = Linear(784, hidden_size, seed=seed)
        self.activation = ReLU()
        self.output = Linear(hidden_size, 10, seed=seed + 1)

    def __call__(self, inputs):
        return self.output(self.activation(self.hidden(inputs)))

    def parameters(self):
        return self.hidden.parameters() + self.output.parameters()


def _batches(dataset, indices, batch_size):
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start:start + batch_size]
        yield dataset.batch(batch_indices), dataset.labels_for(batch_indices)


def train_epoch(model, dataset, optimizer, criterion, batch_size, rng):
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    total_loss = 0.0
    total_correct = 0

    for features, labels in _batches(dataset, indices, batch_size):
        logits = model(Tensor(features))
        loss = criterion(logits, labels)
        predictions = [row.index(max(row)) for row in logits.data]
        total_correct += sum(
            int(prediction == target)
            for prediction, target in zip(predictions, labels)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)

    count = len(dataset)
    return total_loss / count, total_correct / count


def evaluate(model, dataset, batch_size=256):
    total_loss = 0.0
    total_correct = 0
    criterion = CrossEntropyLoss()
    indices = list(range(len(dataset)))

    for features, labels in _batches(dataset, indices, batch_size):
        logits = model(Tensor(features))
        loss = criterion(logits, labels)
        predictions = [row.index(max(row)) for row in logits.data]
        total_loss += loss.item() * len(labels)
        total_correct += sum(
            int(prediction == target)
            for prediction, target in zip(predictions, labels)
        )

    count = len(dataset)
    return total_loss / count, total_correct / count


def _smoke_datasets():
    """Small deterministic 28x28 fixture; not a substitute for MNIST metrics."""
    width = 28 * 28

    def build(repeats):
        pixels, labels = bytearray(), []
        for label in range(10):
            for _ in range(repeats):
                image = bytearray(width)
                image[label * 40] = 255
                image[label * 40 + 1] = 180
                pixels.extend(image)
                labels.append(label)
        return MNISTDataset(bytes(pixels), tuple(labels))

    return build(3), build(1)


def main():
    parser = argparse.ArgumentParser(description="Train AstraTensor on MNIST.")
    parser.add_argument("--data-dir", default="data/mnist")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-test", type=int)
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Run a tiny synthetic 28x28 fixture without network access.",
    )
    args = parser.parse_args()

    if args.epochs <= 0 or args.batch_size <= 0 or args.hidden_size <= 0:
        parser.error("epochs, batch-size and hidden-size must be positive.")

    if args.smoke_test:
        training, test = _smoke_datasets()
        epochs = 1
        print("AstraTensor | MNIST pipeline smoke test (synthetic fixture)")
    else:
        training, test = load_mnist(args.data_dir)
        try:
            training = limit_dataset(training, args.limit_train)
            test = limit_dataset(test, args.limit_test)
        except (TypeError, ValueError) as exc:
            parser.error(str(exc))
        epochs = args.epochs
        print("AstraTensor | MNIST handwritten-digit classification")

    print(
        f"Train: {len(training)} | Test: {len(test)} | "
        f"Batch size: {args.batch_size} | Hidden units: {args.hidden_size}"
    )
    model = DigitMLP(hidden_size=args.hidden_size, seed=args.seed)
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = CrossEntropyLoss()
    rng = random.Random(args.seed)

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = train_epoch(
            model, training, optimizer, criterion, args.batch_size, rng
        )
        test_loss, test_accuracy = evaluate(model, test, args.batch_size)
        print(
            f"epoch {epoch:02d}/{epochs:02d} | "
            f"train loss {train_loss:.4f} | train acc {train_accuracy * 100:.2f}% | "
            f"test loss {test_loss:.4f} | test acc {test_accuracy * 100:.2f}%"
        )

    if args.smoke_test:
        print("Smoke test completed; these are not MNIST benchmark results.")
    else:
        print(f"Final MNIST test accuracy: {test_accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()

"""Controlled MNIST comparison: OddlyTyped versus a CPU PyTorch reference.

Both models use identical initial parameters, data order, batches, architecture,
and Adam hyperparameters. PyTorch is an optional benchmark oracle only.
"""
import argparse
import random
import time

import torch
from torch import nn

from oddlytyped import Tensor
from oddlytyped.nn import CrossEntropyLoss, Linear, ReLU
from oddlytyped.optim import Adam
from mnist_data import limit_dataset, load_mnist


class OddlyTypedMLP:
    def __init__(self, hidden_size=128, seed=2026):
        self.hidden = Linear(784, hidden_size, seed=seed)
        self.activation = ReLU()
        self.output = Linear(hidden_size, 10, seed=seed + 1)

    def __call__(self, inputs):
        return self.output(self.activation(self.hidden(inputs)))

    def parameters(self):
        return self.hidden.parameters() + self.output.parameters()


def make_torch_model(oddlytyped_model):
    hidden_size = len(oddlytyped_model.hidden.bias.data[0])
    model = nn.Sequential(nn.Linear(784, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 10))
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor(oddlytyped_model.hidden.weight.data, dtype=torch.float64).T)
        model[0].bias.copy_(torch.tensor(oddlytyped_model.hidden.bias.data[0], dtype=torch.float64))
        model[2].weight.copy_(torch.tensor(oddlytyped_model.output.weight.data, dtype=torch.float64).T)
        model[2].bias.copy_(torch.tensor(oddlytyped_model.output.bias.data[0], dtype=torch.float64))
    return model.to(dtype=torch.float64)


def batches(dataset, indices, batch_size):
    for start in range(0, len(indices), batch_size):
        selected = indices[start:start + batch_size]
        yield dataset.batch(selected), dataset.labels_for(selected)


def train_oddlytyped(model, optimizer, criterion, dataset, indices, batch_size):
    total_loss = 0.0
    total_correct = 0
    for features, labels in batches(dataset, indices, batch_size):
        logits = model(Tensor(features))
        loss = criterion(logits, labels)
        predictions = [row.index(max(row)) for row in logits.data]
        total_loss += loss.item() * len(labels)
        total_correct += sum(int(p == y) for p, y in zip(predictions, labels))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    count = len(dataset)
    return total_loss / count, total_correct / count


def evaluate_oddlytyped(model, criterion, dataset, batch_size):
    total_loss = 0.0
    total_correct = 0
    indices = list(range(len(dataset)))
    for features, labels in batches(dataset, indices, batch_size):
        logits = model(Tensor(features))
        loss = criterion(logits, labels)
        predictions = [row.index(max(row)) for row in logits.data]
        total_loss += loss.item() * len(labels)
        total_correct += sum(int(p == y) for p, y in zip(predictions, labels))
    count = len(dataset)
    return total_loss / count, total_correct / count


def train_torch(model, optimizer, criterion, dataset, indices, batch_size):
    total_loss = 0.0
    total_correct = 0
    model.train()
    for features, labels in batches(dataset, indices, batch_size):
        x = torch.tensor(features, dtype=torch.float64)
        y = torch.tensor(labels, dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        predictions = logits.argmax(dim=1)
        total_loss += loss.item() * len(labels)
        total_correct += int((predictions == y).sum().item())
        loss.backward()
        optimizer.step()
    count = len(dataset)
    return total_loss / count, total_correct / count


def evaluate_torch(model, criterion, dataset, batch_size):
    total_loss = 0.0
    total_correct = 0
    model.eval()
    with torch.no_grad():
        for features, labels in batches(dataset, range(len(dataset)), batch_size):
            x = torch.tensor(features, dtype=torch.float64)
            y = torch.tensor(labels, dtype=torch.long)
            logits = model(x)
            total_loss += criterion(logits, y).item() * len(labels)
            total_correct += int((logits.argmax(dim=1) == y).sum().item())
    count = len(dataset)
    return total_loss / count, total_correct / count


def main():
    parser = argparse.ArgumentParser(description="Compare OddlyTyped and PyTorch on MNIST.")
    parser.add_argument("--data-dir", default="data/mnist")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-test", type=int)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.hidden_size) <= 0:
        parser.error("epochs, batch-size and hidden-size must be positive.")

    torch.set_num_threads(1)
    training, test = load_mnist(args.data_dir)
    try:
        training = limit_dataset(training, args.limit_train)
        test = limit_dataset(test, args.limit_test)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))

    oddlytyped_model = OddlyTypedMLP(args.hidden_size, args.seed)
    reference = make_torch_model(oddlytyped_model)
    oddlytyped_optimizer = Adam(oddlytyped_model.parameters(), lr=0.001)
    torch_optimizer = torch.optim.Adam(
        reference.parameters(), lr=0.001, betas=(0.9, 0.999),
        eps=1e-8, foreach=False, fused=False,
    )
    oddlytyped_loss = CrossEntropyLoss()
    torch_loss = nn.CrossEntropyLoss()
    rng = random.Random(args.seed)

    print("OddlyTyped vs PyTorch | controlled MNIST comparison")
    print(f"Train: {len(training)} | Test: {len(test)} | Epochs: {args.epochs}")
    print(f"Architecture: 784 -> {args.hidden_size} -> 10 | Batch: {args.batch_size}")
    print("Precision: float64 | Optimizer: Adam(lr=0.001, betas=(0.9, 0.999), eps=1e-8)")
    print("Initial weights: copied from OddlyTyped to PyTorch")
    print()
    oddlytyped_elapsed = 0.0
    torch_elapsed = 0.0

    for epoch in range(1, args.epochs + 1):
        indices = list(range(len(training)))
        rng.shuffle(indices)
        start = time.perf_counter()
        a_train = train_oddlytyped(oddlytyped_model, oddlytyped_optimizer, oddlytyped_loss,
                              training, indices, args.batch_size)
        a_test = evaluate_oddlytyped(oddlytyped_model, oddlytyped_loss, test, args.batch_size)
        oddlytyped_elapsed += time.perf_counter() - start

        start = time.perf_counter()
        t_train = train_torch(reference, torch_optimizer, torch_loss,
                              training, indices, args.batch_size)
        t_test = evaluate_torch(reference, torch_loss, test, args.batch_size)
        torch_elapsed += time.perf_counter() - start

        print(f"Epoch {epoch}/{args.epochs}")
        print(f"  OddlyTyped | train loss {a_train[0]:.4f} | train acc {a_train[1]*100:.2f}%"
              f" | test loss {a_test[0]:.4f} | test acc {a_test[1]*100:.2f}%")
        print(f"  PyTorch     | train loss {t_train[0]:.4f} | train acc {t_train[1]*100:.2f}%"
              f" | test loss {t_test[0]:.4f} | test acc {t_test[1]*100:.2f}%")

    print()
    print(f"OddlyTyped elapsed (train + evaluation): {oddlytyped_elapsed:.2f}s")
    print(f"PyTorch elapsed (train + evaluation): {torch_elapsed:.2f}s")
    print(f"Final test accuracy delta (OddlyTyped - PyTorch): "
          f"{(a_test[1] - t_test[1]) * 100:+.2f} percentage points")
    if torch_elapsed > 0:
        print(f"Observed speed ratio (OddlyTyped / PyTorch): {oddlytyped_elapsed / torch_elapsed:.2f}x")
    print("Note: timings are single-run CPU measurements, not statistically robust claims.")


if __name__ == "__main__":
    main()

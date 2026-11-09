"""Controlled MNIST comparison: AstraTensor versus a CPU PyTorch reference.

Both models use identical initial parameters, data order, batches, architecture,
and Adam hyperparameters. PyTorch is an optional benchmark oracle only.
"""
import argparse
import random
import time

import torch
from torch import nn

from astratensor import Tensor
from astratensor.nn import CrossEntropyLoss, Linear, ReLU
from astratensor.optim import Adam
from mnist_data import load_mnist


class AstraMLP:
    def __init__(self, hidden_size=128, seed=2026):
        self.hidden = Linear(784, hidden_size, seed=seed)
        self.activation = ReLU()
        self.output = Linear(hidden_size, 10, seed=seed + 1)

    def __call__(self, inputs):
        return self.output(self.activation(self.hidden(inputs)))

    def parameters(self):
        return self.hidden.parameters() + self.output.parameters()


def make_torch_model(astra_model):
    hidden_size = len(astra_model.hidden.bias.data[0])
    model = nn.Sequential(nn.Linear(784, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 10))
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor(astra_model.hidden.weight.data, dtype=torch.float64).T)
        model[0].bias.copy_(torch.tensor(astra_model.hidden.bias.data[0], dtype=torch.float64))
        model[2].weight.copy_(torch.tensor(astra_model.output.weight.data, dtype=torch.float64).T)
        model[2].bias.copy_(torch.tensor(astra_model.output.bias.data[0], dtype=torch.float64))
    return model.to(dtype=torch.float64)


def batches(dataset, indices, batch_size):
    for start in range(0, len(indices), batch_size):
        selected = indices[start:start + batch_size]
        yield dataset.batch(selected), dataset.labels_for(selected)


def train_astra(model, optimizer, criterion, dataset, indices, batch_size):
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


def evaluate_astra(model, criterion, dataset, batch_size):
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
    parser = argparse.ArgumentParser(description="Compare AstraTensor and PyTorch on MNIST.")
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
    if args.limit_train:
        training = type(training)(training.images[:args.limit_train * 784],
                                  training.labels[:args.limit_train])
    if args.limit_test:
        test = type(test)(test.images[:args.limit_test * 784],
                          test.labels[:args.limit_test])

    astra = AstraMLP(args.hidden_size, args.seed)
    reference = make_torch_model(astra)
    astra_optimizer = Adam(astra.parameters(), lr=0.001)
    torch_optimizer = torch.optim.Adam(
        reference.parameters(), lr=0.001, betas=(0.9, 0.999),
        eps=1e-8, foreach=False, fused=False,
    )
    astra_loss = CrossEntropyLoss()
    torch_loss = nn.CrossEntropyLoss()
    rng = random.Random(args.seed)

    print("AstraTensor vs PyTorch | controlled MNIST comparison")
    print(f"Train: {len(training)} | Test: {len(test)} | Epochs: {args.epochs}")
    print(f"Architecture: 784 -> {args.hidden_size} -> 10 | Batch: {args.batch_size}")
    print("Precision: float64 | Optimizer: Adam(lr=0.001, betas=(0.9, 0.999), eps=1e-8)")
    print("Initial weights: copied from AstraTensor to PyTorch")
    print()
    astra_elapsed = 0.0
    torch_elapsed = 0.0

    for epoch in range(1, args.epochs + 1):
        indices = list(range(len(training)))
        rng.shuffle(indices)
        start = time.perf_counter()
        a_train = train_astra(astra, astra_optimizer, astra_loss,
                              training, indices, args.batch_size)
        a_test = evaluate_astra(astra, astra_loss, test, args.batch_size)
        astra_elapsed += time.perf_counter() - start

        start = time.perf_counter()
        t_train = train_torch(reference, torch_optimizer, torch_loss,
                              training, indices, args.batch_size)
        t_test = evaluate_torch(reference, torch_loss, test, args.batch_size)
        torch_elapsed += time.perf_counter() - start

        print(f"Epoch {epoch}/{args.epochs}")
        print(f"  AstraTensor | train loss {a_train[0]:.4f} | train acc {a_train[1]*100:.2f}%"
              f" | test loss {a_test[0]:.4f} | test acc {a_test[1]*100:.2f}%")
        print(f"  PyTorch     | train loss {t_train[0]:.4f} | train acc {t_train[1]*100:.2f}%"
              f" | test loss {t_test[0]:.4f} | test acc {t_test[1]*100:.2f}%")

    print()
    print(f"AstraTensor elapsed (train + evaluation): {astra_elapsed:.2f}s")
    print(f"PyTorch elapsed (train + evaluation): {torch_elapsed:.2f}s")
    print(f"Final test accuracy delta (AstraTensor - PyTorch): "
          f"{(a_test[1] - t_test[1]) * 100:+.2f} percentage points")
    if torch_elapsed > 0:
        print(f"Observed speed ratio (AstraTensor / PyTorch): {astra_elapsed / torch_elapsed:.2f}x")
    print("Note: timings are single-run CPU measurements, not statistically robust claims.")


if __name__ == "__main__":
    main()

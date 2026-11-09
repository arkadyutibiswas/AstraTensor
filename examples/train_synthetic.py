"""Train AstraTensor on a deterministic nonlinear XOR-cluster benchmark.

The dataset is generated with Python's standard library. All model operations,
loss calculation, gradient propagation, and parameter updates use AstraTensor.
"""
import random

from astratensor import Tensor
from astratensor.nn import CrossEntropyLoss, Linear, ReLU
from astratensor.optim import Adam


def make_xor_clusters(samples_per_cluster=64, seed=2026, spread=0.22):
    """Return four Gaussian clusters with XOR labels (positive product -> 0)."""
    if not isinstance(samples_per_cluster, int) or samples_per_cluster <= 0:
        raise ValueError("samples_per_cluster must be a positive integer.")
    if spread <= 0:
        raise ValueError("spread must be positive.")

    rng = random.Random(seed)
    clusters = (
        ((-0.75, -0.75), 0),
        ((-0.75, 0.75), 1),
        ((0.75, -0.75), 1),
        ((0.75, 0.75), 0),
    )
    features, labels = [], []
    for (center_x, center_y), label in clusters:
        for _ in range(samples_per_cluster):
            features.append([
                rng.gauss(center_x, spread),
                rng.gauss(center_y, spread),
            ])
            labels.append(label)

    order = list(range(len(labels)))
    rng.shuffle(order)
    return [features[index] for index in order], [labels[index] for index in order]


class XORClusterClassifier:
    def __init__(self, seed=17):
        self.hidden = Linear(2, 16, seed=seed)
        self.activation = ReLU()
        self.output = Linear(16, 2, seed=seed + 1)

    def __call__(self, inputs):
        return self.output(self.activation(self.hidden(inputs)))

    def parameters(self):
        return self.hidden.parameters() + self.output.parameters()


def evaluate(model, features, labels):
    logits = model(Tensor(features)).data
    predictions = [row.index(max(row)) for row in logits]
    correct = sum(int(pred == target) for pred, target in zip(predictions, labels))
    return correct / len(labels)


def main():
    features, labels = make_xor_clusters()
    model = XORClusterClassifier()
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.02)

    print("AstraTensor | Nonlinear XOR-cluster classification")
    print(f"Samples: {len(features)} | Features: 2 | Classes: 2")
    print("Training: full-batch Adam | Hidden layer: 16 ReLU units")

    final_loss = None
    for epoch in range(1, 601):
        logits = model(Tensor(features))
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        final_loss = loss.item()

        if epoch == 1 or epoch % 100 == 0:
            accuracy = evaluate(model, features, labels)
            print(
                f"epoch {epoch:03d} | loss {final_loss:.6f} | "
                f"accuracy {accuracy * 100:.2f}%"
            )

    final_accuracy = evaluate(model, features, labels)
    print("\nFinal training metrics")
    print(f"loss: {final_loss:.6f}")
    print(f"accuracy: {final_accuracy * 100:.2f}%")

    # This is a deterministic CI benchmark, not a claim of generalization.
    if final_accuracy < 0.95 or final_loss > 0.15:
        raise SystemExit(
            "Training acceptance threshold not met: "
            f"loss={final_loss:.6f}, accuracy={final_accuracy:.4f}"
        )


if __name__ == "__main__":
    main()

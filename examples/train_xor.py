"""Train a small XOR classifier using AstraTensor only."""
from astratensor import Tensor
from astratensor.nn import CrossEntropyLoss, Linear, ReLU
from astratensor.optim import Adam


class XORModel:
    def __init__(self):
        self.hidden = Linear(2, 8, seed=11)
        self.activation = ReLU()
        self.output = Linear(8, 2, seed=23)

    def __call__(self, inputs):
        return self.output(self.activation(self.hidden(inputs)))

    def parameters(self):
        return self.hidden.parameters() + self.output.parameters()


def accuracy(model, inputs, labels):
    correct = 0
    for sample, label in zip(inputs, labels):
        scores = model(Tensor([sample])).data[0]
        correct += int(scores.index(max(scores)) == label)
    return correct / len(labels)


def main():
    inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    labels = [0, 1, 1, 0]
    model = XORModel()
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.03)

    print("AstraTensor | XOR classification")
    for epoch in range(1, 501):
        logits = model(Tensor(inputs))
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 50 == 0:
            score = accuracy(model, inputs, labels) * 100.0
            print(f"epoch {epoch:03d} | loss {loss.item():.6f} | accuracy {score:.1f}%")

    print("\nPredictions:")
    for sample in inputs:
        scores = model(Tensor([sample])).data[0]
        predicted = scores.index(max(scores))
        print(f"{sample} -> class {predicted} | logits={scores}")


if __name__ == "__main__":
    main()

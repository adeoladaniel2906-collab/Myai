from linear import Linear
from activation import relu


class FeedForward:
    def __init__(self, embedding_size, hidden_size):
        self.first = Linear(
            embedding_size,
            hidden_size
        )

        self.second = Linear(
            hidden_size,
            embedding_size
        )

    def forward(self, inputs):
        hidden = self.first.forward(inputs)

        activated = []

        for value in hidden:
            activated.append(relu(value))

        output = self.second.forward(activated)

        return output

    def parameters(self):
        return (
            self.first.parameters()
            + self.second.parameters()
        )


if __name__ == "__main__":

    from tensor import Tensor

    network = FeedForward(
        embedding_size=4,
        hidden_size=8
    )

    inputs = [
        Tensor(0.1),
        Tensor(0.2),
        Tensor(0.3),
        Tensor(0.4)
    ]

    output = network.forward(inputs)

    print("Input:")
    print([x.data for x in inputs])

    print("\nOutput:")
    print([x.data for x in output])

    print("\nOutput size:")
    print(len(output))

    print("\nTrainable parameters:")
    print(len(network.parameters()))

    loss = output[0]

    for parameter in network.parameters():
        parameter.grad = 0.0

    loss.backward()

    print("\nGradient test:")
    print([x.grad for x in inputs])

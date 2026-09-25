from tensor import Tensor
import math


class LayerNorm:

    def __init__(self, size, epsilon=1e-5):

        self.size = size
        self.epsilon = epsilon

        self.gamma = [
            Tensor(1.0)
            for _ in range(size)
        ]

        self.beta = [
            Tensor(0.0)
            for _ in range(size)
        ]

    def forward(self, inputs):

        mean = Tensor(0.0)

        for value in inputs:
            mean = mean + value

        mean = mean / len(inputs)

        variance = Tensor(0.0)

        for value in inputs:
            difference = value - mean
            variance = variance + difference ** 2

        variance = variance / len(inputs)

        denominator = (
            variance + self.epsilon
        ) ** 0.5

        outputs = []

        for i, value in enumerate(inputs):

            normalized = (
                (value - mean)
                / denominator
            )

            output = (
                self.gamma[i] * normalized
                + self.beta[i]
            )

            outputs.append(output)

        return outputs

    def parameters(self):

        return self.gamma + self.beta


if __name__ == "__main__":

    inputs = [
        Tensor(1.0),
        Tensor(2.0),
        Tensor(3.0),
        Tensor(4.0)
    ]

    layer_norm = LayerNorm(4)

    outputs = layer_norm.forward(inputs)

    print("Inputs:")
    print([value.data for value in inputs])

    print("\nNormalized outputs:")
    print([value.data for value in outputs])

    print("\nParameters:")
    print(len(layer_norm.parameters()))

    print("\nOutput mean:")

    mean = sum(
        value.data
        for value in outputs
    ) / len(outputs)

    print(mean)

    print("\nOutput variance:")

    variance = sum(
        (value.data - mean) ** 2
        for value in outputs
    ) / len(outputs)

    print(variance)

    loss = outputs[0]

    for parameter in layer_norm.parameters():
        parameter.grad = 0.0

    loss.backward()

    print("\nInput gradients:")
    print([value.grad for value in inputs])

    print("\nGamma gradients:")
    print([value.grad for value in layer_norm.gamma])

    print("\nBeta gradients:")
    print([value.grad for value in layer_norm.beta])

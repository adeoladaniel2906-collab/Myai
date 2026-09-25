import random
from tensor import Tensor


class Neuron:
    def __init__(self, input_count):
        self.weights = [
            Tensor(random.uniform(-1, 1))
            for _ in range(input_count)
        ]

        self.bias = Tensor(0.0)

    def forward(self, inputs):
        output = self.bias

        for weight, input_value in zip(self.weights, inputs):
            output = output + weight * input_value

        return output

    def parameters(self):
        return self.weights + [self.bias]


if __name__ == "__main__":
    neuron = Neuron(2)

    x1 = Tensor(2.0)
    x2 = Tensor(3.0)

    output = neuron.forward([x1, x2])

    print("Input 1:", x1)
    print("Input 2:", x2)
    print("Neuron output:", output)

    output.backward()

    print("\nGradients:")

    for i, weight in enumerate(neuron.weights):
        print(f"Weight {i}: {weight.grad}")

    print("Bias:", neuron.bias.grad)
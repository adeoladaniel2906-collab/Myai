from nn import Neuron
from tensor import Tensor
from activation import relu


class Layer:
    def __init__(self, input_count, neuron_count):
        self.neurons = [
            Neuron(input_count)
            for _ in range(neuron_count)
        ]

    def forward(self, inputs):
        outputs = []

        for neuron in self.neurons:
            output = neuron.forward(inputs)
            output = relu(output)
            outputs.append(output)

        return outputs

    def parameters(self):
        params = []

        for neuron in self.neurons:
            params.extend(neuron.parameters())

        return params


if __name__ == "__main__":
    layer = Layer(2, 3)

    x1 = Tensor(2.0)
    x2 = Tensor(3.0)

    outputs = layer.forward([x1, x2])

    print("Layer outputs:")

    for i, output in enumerate(outputs):
        print(f"Neuron {i}: {output}")
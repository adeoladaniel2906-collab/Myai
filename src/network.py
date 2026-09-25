from layer import Layer
from tensor import Tensor


class Network:
    def __init__(self, input_count, layer_sizes):
        self.layers = []

        current_inputs = input_count

        for neuron_count in layer_sizes:
            layer = Layer(current_inputs, neuron_count)
            self.layers.append(layer)
            current_inputs = neuron_count

    def forward(self, inputs):
        outputs = inputs

        for layer in self.layers:
            outputs = layer.forward(outputs)

        return outputs

    def parameters(self):
        params = []

        for layer in self.layers:
            params.extend(layer.parameters())

        return params


if __name__ == "__main__":
    # 2 inputs
    # First layer: 3 neurons
    # Second layer: 2 neurons
    network = Network(2, [3, 2])

    x1 = Tensor(2.0)
    x2 = Tensor(3.0)

    outputs = network.forward([x1, x2])

    print("Network outputs:")

    for i, output in enumerate(outputs):
        print(f"Output {i}: {output}")

    print("\nNumber of parameters:", len(network.parameters()))
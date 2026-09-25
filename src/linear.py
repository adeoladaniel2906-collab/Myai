import random
from tensor import Tensor


class Linear:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

        self.weights = [
            [
                Tensor(random.uniform(-0.1, 0.1))
                for _ in range(input_size)
            ]
            for _ in range(output_size)
        ]

        self.biases = [
            Tensor(0.0)
            for _ in range(output_size)
        ]

    def forward(self, inputs):
        outputs = []

        for output_index in range(self.output_size):

            result = self.biases[output_index]

            for input_index in range(self.input_size):
                result = result + (
                    self.weights[output_index][input_index]
                    * inputs[input_index]
                )

            outputs.append(result)

        return outputs

    def parameters(self):
        parameters = []

        for row in self.weights:
            parameters.extend(row)

        parameters.extend(self.biases)

        return parameters


if __name__ == "__main__":

    layer = Linear(3, 2)

    inputs = [
        Tensor(1.0),
        Tensor(2.0),
        Tensor(3.0)
    ]

    outputs = layer.forward(inputs)

    print("Inputs:")
    print([x.data for x in inputs])

    print("\nOutputs:")
    print([x.data for x in outputs])

    print("\nTrainable parameters:")
    print(len(layer.parameters()))

    loss = outputs[0]

    loss.backward()

    print("\nInput gradients:")
    print([x.grad for x in inputs])

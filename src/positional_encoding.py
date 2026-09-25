from tensor import Tensor
import math


class PositionalEncoding:

    def __init__(self, max_length, embedding_size):

        self.max_length = max_length
        self.embedding_size = embedding_size

        self.encoding = []

        for position in range(max_length):

            vector = []

            for dimension in range(embedding_size):

                angle = (
                    position
                    / (
                        10000
                        ** (
                            dimension
                            / embedding_size
                        )
                    )
                )

                if dimension % 2 == 0:
                    value = math.sin(angle)
                else:
                    value = math.cos(angle)

                vector.append(value)

            self.encoding.append(vector)

    def forward(self, inputs):

        if len(inputs) > self.max_length:
            raise ValueError(
                "Input sequence is longer than max_length."
            )

        outputs = []

        for position, token in enumerate(inputs):

            if len(token) != self.embedding_size:
                raise ValueError(
                    "Token embedding size does not match."
                )

            output = []

            for dimension, value in enumerate(token):

                position_value = Tensor(
                    self.encoding[position][dimension]
                )

                output.append(
                    value + position_value
                )

            outputs.append(output)

        return outputs

    def parameters(self):
        return []


if __name__ == "__main__":

    from tensor import Tensor

    positional = PositionalEncoding(
        max_length=4,
        embedding_size=4
    )

    inputs = [
        [
            Tensor(0.1),
            Tensor(0.2),
            Tensor(0.3),
            Tensor(0.4)
        ],
        [
            Tensor(0.5),
            Tensor(0.6),
            Tensor(0.7),
            Tensor(0.8)
        ]
    ]

    outputs = positional.forward(inputs)

    print("Positional encodings:")

    for position in range(2):

        print(
            f"Position {position}:",
            positional.encoding[position]
        )

    print("\nEncoded outputs:")

    for position, output in enumerate(outputs):

        print(
            f"Token {position}:",
            [value.data for value in output]
        )

    print("\nTrainable parameters:")
    print(len(positional.parameters()))

    loss = outputs[0][0]

    for token in inputs:
        for value in token:
            value.grad = 0.0

    loss.backward()

    print("\nGradient test:")

    print(
        "First input gradients:",
        [value.grad for value in inputs[0]]
    )

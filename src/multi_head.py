from attention_head import AttentionHead
from linear import Linear


class MultiHeadAttention:
    def __init__(self, embedding_size, number_of_heads):
        if embedding_size % number_of_heads != 0:
            raise ValueError(
                "Embedding size must be divisible by number of heads."
            )

        self.embedding_size = embedding_size
        self.number_of_heads = number_of_heads
        self.head_size = embedding_size // number_of_heads

        self.heads = [
            AttentionHead(
                embedding_size,
                self.head_size
            )
            for _ in range(number_of_heads)
        ]

        self.output_projection = Linear(
            embedding_size,
            embedding_size
        )

    def forward(self, inputs):
        head_outputs = []

        for head in self.heads:
            outputs, _ = head.forward(inputs)
            head_outputs.append(outputs)

        combined = []

        for token_index in range(len(inputs)):
            vector = []

            for head_index in range(self.number_of_heads):
                vector.extend(
                    head_outputs[head_index][token_index]
                )

            combined.append(vector)

        outputs = []

        for vector in combined:
            outputs.append(
                self.output_projection.forward(vector)
            )

        return outputs

    def parameters(self):
        parameters = []

        for head in self.heads:
            parameters.extend(head.parameters())

        parameters.extend(
            self.output_projection.parameters()
        )

        return parameters


if __name__ == "__main__":

    from tensor import Tensor

    model = MultiHeadAttention(
        embedding_size=4,
        number_of_heads=2
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

    outputs = model.forward(inputs)

    print("Number of tokens:")
    print(len(outputs))

    print("\nOutput size per token:")
    print(len(outputs[0]))

    print("\nOutputs:")

    for i, output in enumerate(outputs):
        print(
            f"Token {i}:",
            [value.data for value in output]
        )

    print("\nNumber of heads:")
    print(model.number_of_heads)

    print("\nHead size:")
    print(model.head_size)

    print("\nTrainable parameters:")
    print(len(model.parameters()))

    loss = outputs[0][0]

    for parameter in model.parameters():
        parameter.grad = 0.0

    loss.backward()

    print("\nGradient test:")
    print(
        "First input gradients:",
        [value.grad for value in inputs[0]]
    )

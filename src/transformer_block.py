from multi_head import MultiHeadAttention
from feed_forward import FeedForward
from layer_norm import LayerNorm


class TransformerBlock:

    def __init__(
        self,
        embedding_size,
        number_of_heads,
        hidden_size
    ):

        self.norm1 = LayerNorm(
            embedding_size
        )

        self.attention = MultiHeadAttention(
            embedding_size,
            number_of_heads
        )

        self.norm2 = LayerNorm(
            embedding_size
        )

        self.feed_forward = FeedForward(
            embedding_size,
            hidden_size
        )

    def residual_add(
        self,
        original,
        update
    ):

        return [
            original_value + update_value
            for original_value, update_value
            in zip(
                original,
                update
            )
        ]

    def forward(self, inputs):

        normalized_inputs = []

        for token in inputs:

            normalized_inputs.append(
                self.norm1.forward(token)
            )

        attention_output = (
            self.attention.forward(
                normalized_inputs
            )
        )

        after_attention = []

        for original, update in zip(
            inputs,
            attention_output
        ):

            after_attention.append(
                self.residual_add(
                    original,
                    update
                )
            )

        normalized_after_attention = []

        for token in after_attention:

            normalized_after_attention.append(
                self.norm2.forward(token)
            )

        final_outputs = []

        for original, token in zip(
            after_attention,
            normalized_after_attention
        ):

            feed_forward_output = (
                self.feed_forward.forward(token)
            )

            final_output = self.residual_add(
                original,
                feed_forward_output
            )

            final_outputs.append(
                final_output
            )

        return final_outputs

    def parameters(self):

        return (
            self.norm1.parameters()
            + self.attention.parameters()
            + self.norm2.parameters()
            + self.feed_forward.parameters()
        )


if __name__ == "__main__":

    from tensor import Tensor

    block = TransformerBlock(
        embedding_size=4,
        number_of_heads=2,
        hidden_size=8
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

    outputs = block.forward(
        inputs
    )

    print("Number of tokens:")
    print(len(outputs))

    print("\nOutput size:")
    print(len(outputs[0]))

    print("\nOutputs:")

    for i, output in enumerate(outputs):

        print(
            f"Token {i}:",
            [value.data for value in output]
        )

    print("\nTrainable parameters:")
    print(len(block.parameters()))

    loss = outputs[0][0]

    for parameter in block.parameters():
        parameter.grad = 0.0

    loss.backward()

    print("\nGradient test:")

    print(
        "First input gradients:",
        [value.grad for value in inputs[0]]
    )

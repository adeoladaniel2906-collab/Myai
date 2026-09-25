from qkv import QKVProjection
from attention import scaled_dot_product_attention


class AttentionHead:
    def __init__(self, embedding_size, attention_size):
        self.projection = QKVProjection(
            embedding_size,
            attention_size
        )

    def forward(self, inputs):
        queries, keys, values = self.projection.forward(inputs)

        outputs = []
        all_weights = []

        for query_index, query in enumerate(queries):

            allowed_keys = keys[:query_index + 1]
            allowed_values = values[:query_index + 1]

            output, weights = scaled_dot_product_attention(
                query,
                allowed_keys,
                allowed_values
            )

            outputs.append(output)
            all_weights.append(weights)

        return outputs, all_weights

    def parameters(self):
        return self.projection.parameters()


if __name__ == "__main__":

    head = AttentionHead(
        embedding_size=4,
        attention_size=4
    )

    inputs = [
        [
            __import__("tensor").Tensor(0.1),
            __import__("tensor").Tensor(0.2),
            __import__("tensor").Tensor(0.3),
            __import__("tensor").Tensor(0.4)
        ],
        [
            __import__("tensor").Tensor(0.5),
            __import__("tensor").Tensor(0.6),
            __import__("tensor").Tensor(0.7),
            __import__("tensor").Tensor(0.8)
        ]
    ]

    outputs, weights = head.forward(inputs)

    print("Number of tokens:")
    print(len(outputs))

    print("\nCausal attention weights:")

    for i, token_weights in enumerate(weights):
        print(
            f"Token {i}:",
            [weight.data for weight in token_weights]
        )

    print("\nAttention outputs:")

    for i, output in enumerate(outputs):
        print(
            f"Token {i}:",
            [value.data for value in output]
        )

    print("\nTrainable parameters:")
    print(len(head.parameters()))

    loss = outputs[0][0]

    for parameter in head.parameters():
        parameter.grad = 0.0

    loss.backward()

    print("\nGradient test:")
    print(
        "First input gradients:",
        [value.grad for value in inputs[0]]
    )

from tensor import Tensor
from linear import Linear


class QKVProjection:
    def __init__(self, embedding_size, attention_size):
        self.query = Linear(embedding_size, attention_size)
        self.key = Linear(embedding_size, attention_size)
        self.value = Linear(embedding_size, attention_size)

    def forward(self, inputs):
        queries = []
        keys = []
        values = []

        for vector in inputs:
            queries.append(self.query.forward(vector))
            keys.append(self.key.forward(vector))
            values.append(self.value.forward(vector))

        return queries, keys, values

    def parameters(self):
        return (
            self.query.parameters()
            + self.key.parameters()
            + self.value.parameters()
        )


if __name__ == "__main__":

    projection = QKVProjection(
        embedding_size=4,
        attention_size=4
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

    queries, keys, values = projection.forward(inputs)

    print("Number of input tokens:")
    print(len(inputs))

    print("\nQuery vectors:")
    for vector in queries:
        print([x.data for x in vector])

    print("\nKey vectors:")
    for vector in keys:
        print([x.data for x in vector])

    print("\nValue vectors:")
    for vector in values:
        print([x.data for x in vector])

    print("\nTrainable parameters:")
    print(len(projection.parameters()))

    loss = queries[0][0]

    for parameter in projection.parameters():
        parameter.grad = 0.0

    loss.backward()

    print("\nGradient test:")
    print("First input gradients:")
    print([x.grad for x in inputs[0]])

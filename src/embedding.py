import tensor
import random
from tensor import Tensor 

class Embedding:
    def __init__(self, vocabulary_size, embedding_size):
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size

        self.weights = [
            [
                Tensor(random.uniform(-0.1, 0.1))
                for _ in range(embedding_size)
            ]
            for _ in range(vocabulary_size)
        ]

    def forward(self, token_ids):
        return [
            self.weights[token_id]
            for token_id in token_ids
        ]

    def parameters(self):
        parameters = []

        for vector in self.weights:
            parameters.extend(vector)

        return parameters


if __name__ == "__main__":

    embedding = Embedding(
        vocabulary_size=10,
        embedding_size=4
    )

    tokens = [2, 5, 7]

    vectors = embedding.forward(tokens)

    print("Token IDs:")
    print(tokens)

    print("\nEmbedding vectors:")

    for token, vector in zip(tokens, vectors):
        values = [value.data for value in vector]
        print(f"Token {token}: {values}")

    print("\nTotal trainable parameters:")
    print(len(embedding.parameters()))

    loss = vectors[0][0] ** 2

    for parameter in embedding.parameters():
        parameter.grad = 0.0

    loss.backward()

    print("\nGradient test:")
    print("Selected value:", vectors[0][0])

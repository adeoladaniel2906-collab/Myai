from linear import Linear


class LanguageHead:

    def __init__(self, hidden_size, vocabulary_size):

        self.projection = Linear(
            hidden_size,
            vocabulary_size
        )

    def forward(self, hidden_states):

        logits = []

        for hidden in hidden_states:

            logits.append(
                self.projection.forward(hidden)
            )

        return logits

    def parameters(self):

        return self.projection.parameters()


if __name__ == "__main__":

    from tensor import Tensor

    head = LanguageHead(
        hidden_size=8,
        vocabulary_size=10
    )

    hidden_states = [
        [
            Tensor(0.1),
            Tensor(0.2),
            Tensor(0.3),
            Tensor(0.4),
            Tensor(0.5),
            Tensor(0.6),
            Tensor(0.7),
            Tensor(0.8)
        ]
    ]

    logits = head.forward(
        hidden_states
    )

    print("Number of tokens:")
    print(len(logits))

    print("\nVocabulary size:")
    print(len(logits[0]))

    print("\nLogits:")
    print(
        [value.data for value in logits[0]]
    )

    print("\nTrainable parameters:")
    print(len(head.parameters()))

    loss = logits[0][0]

    for parameter in head.parameters():
        parameter.grad = 0.0

    loss.backward()

    print("\nGradient test:")

    print(
        [value.grad for value in hidden_states[0]]
    )

from tokenizer import Tokenizer
from embedding import Embedding
from positional_encoding import PositionalEncoding
from transformer_block import TransformerBlock


class TransformerModel:

    def __init__(
        self,
        training_text,
        embedding_size=8,
        number_of_heads=2,
        hidden_size=16,
        max_length=32
    ):

        self.tokenizer = Tokenizer(
            training_text
        )

        self.embedding = Embedding(
            len(self.tokenizer.vocab),
            embedding_size
        )

        self.position = PositionalEncoding(
            max_length,
            embedding_size
        )

        self.block = TransformerBlock(
            embedding_size,
            number_of_heads,
            hidden_size
        )

    def forward(self, text):

        token_ids = self.tokenizer.encode(
            text
        )

        embeddings = self.embedding.forward(
            token_ids
        )

        positioned = self.position.forward(
            embeddings
        )

        output = self.block.forward(
            positioned
        )

        return token_ids, output

    def parameters(self):

        return (
            self.embedding.parameters()
            + self.block.parameters()
        )


if __name__ == "__main__":

    training_text = (
        "hello world "
        "hello ai "
        "hello world"
    )

    model = TransformerModel(
        training_text,
        embedding_size=8,
        number_of_heads=2,
        hidden_size=16,
        max_length=32
    )

    text = "hello"

    token_ids, outputs = model.forward(
        text
    )

    print("Input text:")
    print(text)

    print("\nToken IDs:")
    print(token_ids)

    print("\nNumber of tokens:")
    print(len(outputs))

    print("\nHidden size:")
    print(len(outputs[0]))

    print("\nFirst token representation:")
    print(
        [value.data for value in outputs[0]]
    )

    print("\nTrainable parameters:")
    print(len(model.parameters()))

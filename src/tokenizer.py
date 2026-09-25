class Tokenizer:
    def __init__(self, text):
        # Get every unique character
        characters = sorted(set(text))

        # Special token for unknown characters
        self.unknown_token = "<UNK>"

        self.vocab = [self.unknown_token] + characters

        # Character → number
        self.token_to_id = {
            token: i
            for i, token in enumerate(self.vocab)
        }

        # Number → character
        self.id_to_token = {
            i: token
            for token, i in self.token_to_id.items()
        }

    def encode(self, text):
        return [
            self.token_to_id.get(
                character,
                self.token_to_id[self.unknown_token]
            )
            for character in text
        ]

    def decode(self, tokens):
        return "".join(
            self.id_to_token[token]
            for token in tokens
        )


if __name__ == "__main__":

    training_text = "hello world"

    tokenizer = Tokenizer(training_text)

    text = "hello"

    tokens = tokenizer.encode(text)

    decoded = tokenizer.decode(tokens)

    print("Vocabulary:")
    print(tokenizer.vocab)

    print("\nText:")
    print(text)

    print("\nTokens:")
    print(tokens)

    print("\nDecoded:")
    print(decoded)

    print("\nVocabulary size:")
    print(len(tokenizer.vocab))
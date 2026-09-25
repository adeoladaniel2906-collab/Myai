from tokenizer import Tokenizer


class LanguageDataset:
    def __init__(self, text, sequence_length=32):
        if not text:
            raise ValueError("Training text cannot be empty.")

        if sequence_length < 2:
            raise ValueError("sequence_length must be at least 2.")

        self.text = text
        self.sequence_length = sequence_length
        self.tokenizer = Tokenizer(text)
        self.tokens = self.tokenizer.encode(text)

    def create_examples(self):
        examples = []

        maximum_start = len(self.tokens) - self.sequence_length

        if maximum_start < 0:
            raise ValueError(
                "Training text is shorter than sequence_length."
            )

        for start in range(maximum_start + 1):
            input_tokens = self.tokens[
                start:start + self.sequence_length - 1
            ]

            target_tokens = self.tokens[
                start + 1:start + self.sequence_length
            ]

            examples.append(
                (input_tokens, target_tokens)
            )

        return examples

    def vocabulary_size(self):
        return len(self.tokenizer.vocab)


if __name__ == "__main__":

    text = (
        "hello world hello ai "
        "hello world hello ai "
        "this is a small language model"
    )

    dataset = LanguageDataset(
        text,
        sequence_length=8
    )

    examples = dataset.create_examples()

    print("Vocabulary size:", dataset.vocabulary_size())
    print("Token count:", len(dataset.tokens))
    print("Number of training examples:", len(examples))

    print("\nFirst input:", examples[0][0])
    print("First target:", examples[0][1])

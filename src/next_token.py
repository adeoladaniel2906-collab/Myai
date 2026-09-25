from tokenizer import Tokenizer

class NextTokenDataset:
    def __init__(self, text):
        self.tokenizer = Tokenizer(text)
        self.tokens = self.tokenizer.encode(text)

    def create_examples(self):
        examples = []

        for index in range(len(self.tokens) - 1):
            input_token = self.tokens[index]
            target_token = self.tokens[index + 1]
            examples.append((input_token, target_token))

        return examples


if __name__ == '__main__':
    text = 'hello world'

    dataset = NextTokenDataset(text)
    examples = dataset.create_examples()

    print('Vocabulary:', dataset.tokenizer.vocab)
    print('Tokens:', dataset.tokens)
    print('Training examples:')

    for input_token, target_token in examples:
        print(input_token, '->', target_token)

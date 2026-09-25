from transformer_model import TransformerModel
from language_head import LanguageHead
from softmax import softmax
from loss import cross_entropy
from checkpoint import load_checkpoint

text = "hello world hello ai hello world hello ai"

model = TransformerModel(
    text,
    embedding_size=8,
    number_of_heads=2,
    hidden_size=16,
    max_length=32
)

language_head = LanguageHead(
    8,
    len(model.tokenizer.vocab)
)

parameters = (
    model.parameters()
    + language_head.parameters()
)


def calculate_loss():

    training_text = text[:32]
    token_ids, hidden_states = model.forward(training_text)

    total_loss = None
    example_count = len(token_ids) - 1

    for index in range(example_count):

        hidden = hidden_states[index]

        logits = language_head.forward(
            [[value for value in hidden]]
        )[0]

        probabilities = softmax(logits)

        target_token = token_ids[index + 1]

        loss = cross_entropy(
            probabilities,
            target_token
        )

        if total_loss is None:
            total_loss = loss
        else:
            total_loss = total_loss + loss

    return total_loss / example_count


print("Loss with random weights:")
print(calculate_loss().data)

load_checkpoint(
    parameters,
    "fast_checkpoint_final.json"
)

print("\nLoss after loading trained checkpoint:")
print(calculate_loss().data)

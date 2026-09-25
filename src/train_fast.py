from transformer_model import TransformerModel
from language_head import LanguageHead
from softmax import softmax
from loss import cross_entropy
from optimizer import SGD
from checkpoint import save_checkpoint

# Small training text
text = "hello world hello ai hello world hello ai"

# Smaller context for faster phone training
sequence_length = 32

# Build model
model = TransformerModel(
    text,
    embedding_size=8,
    number_of_heads=2,
    hidden_size=16,
    max_length=sequence_length
)

language_head = LanguageHead(
    8,
    len(model.tokenizer.vocab)
)

parameters = (
    model.parameters()
    + language_head.parameters()
)

optimizer = SGD(
    parameters,
    learning_rate=0.001
)


def calculate_loss():

    training_text = text[:sequence_length]
    token_ids, hidden_states = model.forward(training_text)

    example_count = min(
        len(token_ids) - 1,
        sequence_length - 1
    )

    total_loss = None

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


print("Fast training started.", flush=True)

initial_loss = calculate_loss()

print(
    f"Initial loss: {initial_loss.data:.6f}",
    flush=True
)


for epoch in range(100):

    loss = calculate_loss()

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    print(
        f"Epoch {epoch + 1}: loss={loss.data:.6f}",
        flush=True
    )

    if (epoch + 1) % 10 == 0:

        checkpoint_file = (
            f"fast_checkpoint_epoch_{epoch + 1}.json"
        )

        save_checkpoint(
            parameters,
            checkpoint_file
        )

        print(
            f"Checkpoint saved: {checkpoint_file}",
            flush=True
        )


final_loss = calculate_loss()

print(
    f"Final loss: {final_loss.data:.6f}",
    flush=True
)

save_checkpoint(
    parameters,
    "fast_checkpoint_final.json"
)

print(
    "Fast training completed.",
    flush=True
)

print(
    "Final checkpoint saved.",
    flush=True
)

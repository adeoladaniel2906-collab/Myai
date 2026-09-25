from transformer_model import TransformerModel
from language_head import LanguageHead
from softmax import softmax
from loss import cross_entropy
from optimizer import SGD
from checkpoint import save_checkpoint


text = "hello world hello ai hello world hello ai"

model = TransformerModel(
    text,
    embedding_size=8,
    number_of_heads=2,
    hidden_size=16,
    max_length=100
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
    token_ids, hidden_states = model.forward(text)

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


print("Initial loss:", calculate_loss().data)

for epoch in range(100):
    loss = calculate_loss()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(
            f"Epoch {epoch}: "
            f"loss={loss.data:.6f}",
            flush=True
        )

    if (epoch + 1) % 10 == 0:
        checkpoint_file = (
            f"model_checkpoint_epoch_{epoch + 1}.json"
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

print("Final loss:", final_loss.data)

save_checkpoint(
    parameters,
    "model_checkpoint_final.json"
)

print("Final checkpoint saved.")

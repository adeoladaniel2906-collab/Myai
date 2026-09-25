from transformer_model import TransformerModel
from language_head import LanguageHead
from softmax import softmax


def generate(model, language_head, prompt, max_new_tokens=20):
    generated = prompt

    for _ in range(max_new_tokens):
        token_ids, hidden_states = model.forward(generated)

        hidden = hidden_states[-1]

        logits = language_head.forward(
            [[value for value in hidden]]
        )[0]

        probabilities = softmax(logits)

        best_index = 0
        best_probability = probabilities[0].data

        for index, probability in enumerate(probabilities):
            if probability.data > best_probability:
                best_index = index
                best_probability = probability.data

        next_token = model.tokenizer.id_to_token[best_index]

        generated += next_token

    return generated


if __name__ == "__main__":
    text = "hello world hello ai"

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

    result = generate(
        model,
        language_head,
        "hello ",
        max_new_tokens=10
    )

    print("Generated:", result)

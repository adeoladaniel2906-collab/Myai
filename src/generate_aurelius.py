import argparse
import sys
import torch

sys.path.insert(0, "/content/Myai/src")

from aurelius_inference import load_cached_model
from tokenizers import Tokenizer


def sample_next_token(logits, temperature=0.8, top_k=50):
    if temperature <= 0:
        return torch.argmax(logits, dim=-1)

    logits = logits / temperature

    if top_k > 0:
        k = min(top_k, logits.size(-1))
        values, indices = torch.topk(logits, k)

        filtered = torch.full_like(logits, float("-inf"))
        filtered.scatter_(1, indices, values)
        logits = filtered

    probabilities = torch.softmax(logits, dim=-1)

    return torch.multinomial(probabilities, 1)


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens=50,
    temperature=0.8,
    top_k=50,
):
    token_ids = tokenizer.encode(prompt).ids

    if not token_ids:
        raise ValueError("Prompt produced no tokens.")

    context_length = model.model.context_length

    if len(token_ids) >= context_length:
        token_ids = token_ids[-context_length + 1:]

    tokens = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device="cuda",
    )

    logits = model.prefill(tokens)

    generated_ids = list(token_ids)

    for _ in range(max_new_tokens):
        next_token = sample_next_token(
            logits[:, -1, :],
            temperature,
            top_k,
        )

        generated_ids.append(next_token.item())

        logits = model.decode(next_token)

    return tokenizer.decode(generated_ids)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        default="/content/drive2/MyDrive/Aurelius/checkpoints/aurelius_102m/aurelius_102m_best.pt",
    )

    parser.add_argument(
        "--tokenizer",
        default="/content/drive2/MyDrive/Aurelius/data/tokenizer/aurelius_bpe.json",
    )

    parser.add_argument(
        "--prompt",
        default="Hello, my name is",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
    )

    args = parser.parse_args()

    print("Loading Aurelius...")

    model = load_cached_model(
        args.checkpoint,
        device="cuda",
    )

    tokenizer = Tokenizer.from_file(args.tokenizer)

    print("🧠 Aurelius ready.")
    print()

    result = generate(
        model,
        tokenizer,
        args.prompt,
        args.max_new_tokens,
        args.temperature,
        args.top_k,
    )

    print("Generated:")
    print(result)


if __name__ == "__main__":
    main()

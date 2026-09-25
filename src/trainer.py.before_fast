import argparse
import json
import os
import random

from transformer_model import TransformerModel
from language_head import LanguageHead
from dataset import LanguageDataset
from softmax import softmax
from loss import cross_entropy
from optimizer import SGD
from checkpoint import save_checkpoint, load_checkpoint

DEFAULT_TEXT = (
    "hello world hello ai "
    "hello world hello ai "
    "this is a small language model "
    "this model learns from text "
)

def calculate_loss(model, language_head, input_tokens, target_tokens):
    input_text = model.tokenizer.decode(input_tokens)
    _, hidden_states = model.forward(input_text)

    total_loss = None

    for index, hidden in enumerate(hidden_states):
        logits = language_head.forward([hidden])[0]
        probabilities = softmax(logits)

        token_loss = cross_entropy(
            probabilities,
            target_tokens[index]
        )

        if total_loss is None:
            total_loss = token_loss
        else:
            total_loss = total_loss + token_loss

    return total_loss / len(target_tokens)

def save_training_state(filename, epoch):
    with open(filename, "w") as file:
        json.dump({"epoch": epoch}, file)

def load_training_state(filename):
    if not os.path.exists(filename):
        return 0

    with open(filename, "r") as file:
        state = json.load(file)

    return int(state.get("epoch", 0))

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--checkpoint", type=str, default="trainer_checkpoint.json")
    parser.add_argument("--state", type=str, default="trainer_state.json")
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()

    dataset = LanguageDataset(
        DEFAULT_TEXT,
        sequence_length=args.sequence_length
    )

    model = TransformerModel(
        DEFAULT_TEXT,
        embedding_size=8,
        number_of_heads=2,
        hidden_size=16,
        max_length=args.sequence_length
    )

    language_head = LanguageHead(
        8,
        dataset.vocabulary_size()
    )

    parameters = model.parameters() + language_head.parameters()

    optimizer = SGD(
        parameters,
        learning_rate=args.learning_rate
    )

    start_epoch = 0

    if args.resume:
        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError(
                f"Checkpoint not found: {args.checkpoint}"
            )

        load_checkpoint(parameters, args.checkpoint)
        start_epoch = load_training_state(args.state)

        print(
            f"Resuming from epoch {start_epoch}",
            flush=True
        )

    examples = dataset.create_examples()

    print("Trainer started.", flush=True)
    print(
        f"Training examples: {len(examples)}",
        flush=True
    )
    print(
        f"Sequence length: {args.sequence_length}",
        flush=True
    )
    print(
        f"Parameters: {len(parameters)}",
        flush=True
    )

    for epoch in range(start_epoch, args.epochs):

        random.shuffle(examples)

        epoch_loss = 0.0
        examples_used = 0

        for input_tokens, target_tokens in examples:

            loss = calculate_loss(
                model,
                language_head,
                input_tokens,
                target_tokens
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            epoch_loss += loss.data
            examples_used += 1

        average_loss = epoch_loss / examples_used

        current_epoch = epoch + 1

        print(
            f"Epoch {current_epoch}: "
            f"average_loss={average_loss:.6f} "
            f"examples={examples_used}",
            flush=True
        )

        if current_epoch % 10 == 0:

            save_checkpoint(
                parameters,
                args.checkpoint
            )

            save_training_state(
                args.state,
                current_epoch
            )

            print(
                f"Checkpoint saved at epoch {current_epoch}",
                flush=True
            )

    save_checkpoint(
        parameters,
        args.checkpoint
    )

    save_training_state(
        args.state,
        args.epochs
    )

    print("Training completed.", flush=True)
    print(
        f"Final checkpoint: {args.checkpoint}",
        flush=True
    )

if __name__ == "__main__":
    main()

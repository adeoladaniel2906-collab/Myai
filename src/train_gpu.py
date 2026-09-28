
import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from gpu_model import GPUTransformerModel
from dataset import LanguageDataset


DEFAULT_TEXT = (
    "hello world hello ai "
    "hello world hello ai "
    "this is a small language model "
    "this model learns from text "
)


def save_checkpoint(
    filename,
    model,
    optimizer,
    epoch,
    step,
    config
):
    checkpoint = {
        "epoch": epoch,
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
    }

    os.makedirs(
        os.path.dirname(filename) or ".",
        exist_ok=True
    )

    torch.save(checkpoint, filename)


def load_checkpoint(
    filename,
    model,
    optimizer,
    device
):
    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"Checkpoint not found: {filename}"
        )

    checkpoint = torch.load(
        filename,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    epoch = int(checkpoint.get("epoch", 0))
    step = int(checkpoint.get("step", 0))

    return epoch, step, checkpoint.get("config", {})


def build_dataset(text, sequence_length):
    dataset = LanguageDataset(
        text,
        sequence_length=sequence_length
    )

    examples = dataset.create_examples()

    inputs = torch.tensor(
        [example[0] for example in examples],
        dtype=torch.long
    )

    targets = torch.tensor(
        [example[1] for example in examples],
        dtype=torch.long
    )

    return dataset, inputs, targets


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=10
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8
    )

    parser.add_argument(
        "--sequence-length",
        type=int,
        default=16
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.0003
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/gpu_checkpoint.pt"
    )

    parser.add_argument(
        "--resume",
        action="store_true"
    )

    args = parser.parse_args()

    if args.sequence_length < 2:
        raise ValueError(
            "sequence-length must be at least 2."
        )

    if args.batch_size < 1:
        raise ValueError(
            "batch-size must be at least 1."
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}", flush=True)

    if device.type == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(0)}",
            flush=True
        )

    dataset, inputs, targets = build_dataset(
        DEFAULT_TEXT,
        args.sequence_length
    )

    print(
        f"Vocabulary size: {dataset.vocabulary_size()}",
        flush=True
    )

    print(
        f"Token count: {len(dataset.tokens)}",
        flush=True
    )

    print(
        f"Training examples: {len(inputs)}",
        flush=True
    )

    print(
        f"Input shape: {tuple(inputs.shape)}",
        flush=True
    )

    print(
        f"Target shape: {tuple(targets.shape)}",
        flush=True
    )

    model = GPUTransformerModel(
        vocab_size=dataset.vocabulary_size(),
        d_model=512,
        n_heads=8,
        n_layers=13,
        d_ff=5376,
        context_length=256
    ).to(device)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        f"Parameters: {parameter_count:,}",
        flush=True
    )

    expected_parameters = 85_478_163

    if parameter_count != expected_parameters:
        raise RuntimeError(
            "Parameter count mismatch: "
            f"expected {expected_parameters:,}, "
            f"got {parameter_count:,}"
        )

    print(
        "Parameter count verified.",
        flush=True
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate
    )

    loss_function = nn.CrossEntropyLoss()

    train_dataset = TensorDataset(
        inputs,
        targets
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda")
    )

    start_epoch = 0
    global_step = 0

    if args.resume:
        start_epoch, global_step, _ = load_checkpoint(
            args.checkpoint,
            model,
            optimizer,
            device
        )

        print(
            f"Resuming from epoch {start_epoch}",
            flush=True
        )

        print(
            f"Resuming from step {global_step}",
            flush=True
        )

    print(
        "GPU trainer started.",
        flush=True
    )

    model.train()

    for epoch in range(start_epoch, args.epochs):

        epoch_loss = 0.0
        batches_used = 0

        for batch_inputs, batch_targets in train_loader:

            batch_inputs = batch_inputs.to(
                device,
                non_blocking=True
            )

            batch_targets = batch_targets.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(batch_inputs)

            loss = loss_function(
                logits.reshape(
                    -1,
                    dataset.vocabulary_size()
                ),
                batch_targets.reshape(-1)
            )

            loss.backward()

            optimizer.step()

            epoch_loss += loss.detach().item()
            batches_used += 1
            global_step += 1

        average_loss = epoch_loss / batches_used

        current_epoch = epoch + 1

        print(
            f"Epoch {current_epoch}: "
            f"average_loss={average_loss:.6f} "
            f"batches={batches_used} "
            f"step={global_step}",
            flush=True
        )

        if current_epoch % 5 == 0:

            checkpoint_path = os.path.join(
                os.path.dirname(args.checkpoint),
                f"gpu_85m_epoch{current_epoch}.pt"
            )

            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                current_epoch,
                global_step,
                vars(args)
            )

            print(
                f"Checkpoint saved: {checkpoint_path}",
                flush=True
            )

    save_checkpoint(
        args.checkpoint,
        model,
        optimizer,
        args.epochs,
        global_step,
        vars(args)
    )

    print(
        "Training completed.",
        flush=True
    )

    print(
        f"Final checkpoint: {args.checkpoint}",
        flush=True
    )


if __name__ == "__main__":
    main()

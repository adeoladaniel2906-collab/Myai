
import os
import numpy as np
import torch
from torch.utils.data import Dataset


class AureliusTokenDataset(Dataset):
    """
    Memory-efficient dataset for Aurelius tokenized .bin files.

    Each example contains:
        input  = context_length tokens
        target = the next context_length tokens

    Examples are separated by stride tokens.
    """

    def __init__(
        self,
        token_file,
        context_length=256,
        start_token=0,
        end_token=None,
        stride=None
    ):

        if not os.path.exists(token_file):
            raise FileNotFoundError(
                f"Token file not found: {token_file}"
            )

        if context_length < 2:
            raise ValueError(
                "context_length must be at least 2."
            )

        if stride is None:
            stride = context_length

        if stride < 1:
            raise ValueError(
                "stride must be at least 1."
            )

        self.token_file = token_file
        self.context_length = context_length
        self.stride = stride

        self.tokens = np.memmap(
            token_file,
            dtype=np.uint16,
            mode="r"
        )

        total_tokens = len(self.tokens)

        if end_token is None:
            end_token = total_tokens

        start_token = int(start_token)
        end_token = int(end_token)

        if start_token < 0:
            raise ValueError(
                "start_token cannot be negative."
            )

        if end_token > total_tokens:
            raise ValueError(
                "end_token exceeds token file length."
            )

        if start_token >= end_token:
            raise ValueError(
                "start_token must be smaller than end_token."
            )

        self.start_token = start_token
        self.end_token = end_token

        tokens_available = end_token - start_token
        required_tokens = context_length + 1

        if tokens_available < required_tokens:
            self.number_of_examples = 0
        else:
            self.number_of_examples = (
                (tokens_available - required_tokens)
                // stride
            ) + 1

    def __len__(self):
        return self.number_of_examples

    def __getitem__(self, index):

        if index < 0:
            index += len(self)

        if index < 0 or index >= len(self):
            raise IndexError(
                "Dataset index out of range."
            )

        start = (
            self.start_token
            + index * self.stride
        )

        end = start + self.context_length + 1

        sequence = np.asarray(
            self.tokens[start:end],
            dtype=np.int64
        )

        input_tokens = torch.from_numpy(
            sequence[:-1].copy()
        )

        target_tokens = torch.from_numpy(
            sequence[1:].copy()
        )

        return input_tokens, target_tokens


def inspect_token_file(token_file):

    if not os.path.exists(token_file):
        raise FileNotFoundError(
            f"Token file not found: {token_file}"
        )

    tokens = np.memmap(
        token_file,
        dtype=np.uint16,
        mode="r"
    )

    print(f"Token file: {token_file}")
    print(f"Token count: {len(tokens):,}")
    print(
        f"File size: {os.path.getsize(token_file):,} bytes"
    )
    print(f"Minimum token ID: {int(tokens.min())}")
    print(f"Maximum token ID: {int(tokens.max())}")

    del tokens

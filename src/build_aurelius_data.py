
import os
import re
import json
import argparse
from pathlib import Path

import numpy as np


# ============================================================
# Aurelius data configuration
# ============================================================

DEFAULT_DRIVE_ROOT = "/content/drive/MyDrive/Aurelius"

GUTENBERG_URLS = {
    "frankenstein": "https://www.gutenberg.org/files/84/84-0.txt",
    "pride_and_prejudice": "https://www.gutenberg.org/files/1342/1342-0.txt",
}

WIKIPEDIA_DATASET = "wikimedia"
WIKIPEDIA_CONFIG = "20231101.en"
WIKIPEDIA_SPLIT = "train"

WIKIPEDIA_ARTICLES = 50_000

VOCAB_SIZE = 16_384


def clean_text(text):
    """
    Basic corpus normalization.

    Keeps normal English punctuation and whitespace while
    removing control characters that should not enter the
    training corpus.
    """

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    text = re.sub(
        r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]",
        " ",
        text
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    return text.strip()


def build_directories(root):
    root = Path(root)

    directories = [
        root / "data" / "raw",
        root / "data" / "corpus",
        root / "data" / "tokenizer",
        root / "logs",
        root / "backups",
    ]

    for directory in directories:
        directory.mkdir(
            parents=True,
            exist_ok=True
        )

    return root


def save_json(path, data):

    path = Path(path)

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=2
        )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        default=DEFAULT_DRIVE_ROOT
    )

    parser.add_argument(
        "--wikipedia-articles",
        type=int,
        default=WIKIPEDIA_ARTICLES
    )

    parser.add_argument(
        "--vocab-size",
        type=int,
        default=VOCAB_SIZE
    )

    args = parser.parse_args()

    root = build_directories(args.root)

    print("=" * 60)
    print("Aurelius corpus/tokenizer configuration")
    print("=" * 60)

    print(f"Root:              {root}")
    print(f"Wikipedia dataset: {WIKIPEDIA_DATASET}")
    print(f"Wikipedia config:  {WIKIPEDIA_CONFIG}")
    print(f"Wikipedia split:   {WIKIPEDIA_SPLIT}")
    print(f"Wikipedia articles:{args.wikipedia_articles:,}")
    print(f"BPE vocabulary:    {args.vocab_size:,}")

    print("\n✅ Configuration validated.")
    print("No corpus has been downloaded yet.")


if __name__ == "__main__":
    main()

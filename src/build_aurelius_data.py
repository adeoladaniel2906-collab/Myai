
import os
import re
import json
import argparse
import urllib.request
from pathlib import Path

import numpy as np

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


# ============================================================
# Aurelius data configuration
# ============================================================

DEFAULT_DRIVE_ROOT = "/content/drive/MyDrive/Aurelius"

GUTENBERG_URLS = {
    "frankenstein": "https://www.gutenberg.org/files/84/84-0.txt",
    "pride_and_prejudice": "https://www.gutenberg.org/files/1342/1342-0.txt",
}

WIKIPEDIA_DATASET = "wikimedia/wikipedia"
WIKIPEDIA_CONFIG = "20231101.en"
WIKIPEDIA_SPLIT = "train"

WIKIPEDIA_ARTICLES = 50_000

VOCAB_SIZE = 16_384


def clean_text(text):
    """Normalize corpus text while preserving useful language structure."""

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove control characters.
    text = re.sub(
        r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]",
        " ",
        text
    )

    # Normalize excessive whitespace.
    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
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


def download_file(url, destination):

    destination = Path(destination)

    if destination.exists() and destination.stat().st_size > 0:
        print(f"✅ Already downloaded: {destination}")
        return

    print(f"Downloading: {url}")

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "AureliusResearch/1.0"
        }
    )

    with urllib.request.urlopen(request) as response:
        data = response.read()

    destination.write_bytes(data)

    print(
        f"✅ Saved {destination} "
        f"({destination.stat().st_size:,} bytes)"
    )


def download_gutenberg(root):

    raw_directory = root / "data" / "raw"

    for name, url in GUTENBERG_URLS.items():

        destination = (
            raw_directory
            / f"{name}.txt"
        )

        download_file(
            url,
            destination
        )


def load_gutenberg_text(root):

    raw_directory = root / "data" / "raw"

    texts = []

    for name in GUTENBERG_URLS:

        path = (
            raw_directory
            / f"{name}.txt"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Missing Gutenberg file: {path}"
            )

        text = path.read_text(
            encoding="utf-8",
            errors="replace"
        )

        text = clean_text(text)

        texts.append(text)

        print(
            f"Loaded {name}: "
            f"{len(text):,} characters"
        )

    return "\n\n".join(texts)


def collect_wikipedia(root, article_limit):

    raw_directory = root / "data" / "raw"

    wikipedia_path = (
        raw_directory
        / "wikipedia_stream.txt"
    )

    metadata_path = (
        raw_directory
        / "wikipedia_stream.json"
    )

    # Resume if the completed stream already exists.
    if wikipedia_path.exists() and metadata_path.exists():

        try:
            metadata = json.loads(
                metadata_path.read_text(
                    encoding="utf-8"
                )
            )

            if metadata.get("completed") is True:
                print(
                    "✅ Existing completed Wikipedia "
                    "stream found."
                )

                return wikipedia_path

        except Exception:
            pass

    print(
        f"Starting Wikipedia stream: "
        f"{article_limit:,} articles"
    )

    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "The 'datasets' package is required "
            "for Wikipedia streaming."
        ) from error

    dataset = load_dataset(
        WIKIPEDIA_DATASET,
        WIKIPEDIA_CONFIG,
        split=WIKIPEDIA_SPLIT,
        streaming=True
    )

    temporary_path = (
        raw_directory
        / "wikipedia_stream.incomplete.txt"
    )

    count = 0
    characters = 0

    with open(
        temporary_path,
        "a",
        encoding="utf-8"
    ) as output:

        for item in dataset:

            text = item.get("text", "")

            if not text:
                continue

            text = clean_text(text)

            if not text:
                continue

            output.write(text)
            output.write("\n\n")

            count += 1
            characters += len(text)

            if count % 1000 == 0:

                output.flush()

                print(
                    f"Wikipedia: "
                    f"{count:,}/{article_limit:,} "
                    f"articles | "
                    f"{characters:,} chars"
                )

            if count >= article_limit:
                break

    os.replace(
        temporary_path,
        wikipedia_path
    )

    metadata = {
        "dataset": WIKIPEDIA_DATASET,
        "config": WIKIPEDIA_CONFIG,
        "split": WIKIPEDIA_SPLIT,
        "articles": count,
        "characters": characters,
        "completed": count >= article_limit,
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2
        ),
        encoding="utf-8"
    )

    print(
        f"✅ Wikipedia stream complete: "
        f"{count:,} articles"
    )

    return wikipedia_path


def build_combined_corpus(
    root,
    wikipedia_path
):

    corpus_directory = (
        root / "data" / "corpus"
    )

    corpus_path = (
        corpus_directory
        / "aurelius_corpus.txt"
    )

    if (
        corpus_path.exists()
        and corpus_path.stat().st_size > 0
    ):

        print(
            f"✅ Existing corpus found: "
            f"{corpus_path}"
        )

        return corpus_path

    gutenberg_text = load_gutenberg_text(root)

    wikipedia_text = wikipedia_path.read_text(
        encoding="utf-8",
        errors="replace"
    )

    with open(
        corpus_path,
        "w",
        encoding="utf-8"
    ) as output:

        output.write(gutenberg_text)
        output.write("\n\n")
        output.write(wikipedia_text)

    print(
        f"✅ Combined corpus created."
    )

    print(
        f"Corpus size: "
        f"{corpus_path.stat().st_size:,} bytes"
    )

    return corpus_path



def build_tokenizer_and_tokens(
    root,
    corpus_path,
    vocab_size
):
    """Train Byte-Level BPE and encode the corpus to uint16."""

    tokenizer_directory = (
        root / "data" / "tokenizer"
    )

    corpus_directory = (
        root / "data" / "corpus"
    )

    tokenizer_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    tokenizer_path = (
        tokenizer_directory
        / "aurelius_bpe.json"
    )

    token_path = (
        corpus_directory
        / "aurelius_tokens.bin"
    )

    metadata_path = (
        tokenizer_directory
        / "metadata.json"
    )

    # Reuse an already completed tokenizer/token file.
    if (
        tokenizer_path.exists()
        and token_path.exists()
        and metadata_path.exists()
    ):
        print(
            "✅ Existing tokenizer and token file found."
        )

        metadata = json.loads(
            metadata_path.read_text(
                encoding="utf-8"
            )
        )

        print(
            f"Token count: "
            f"{metadata.get('token_count', 0):,}"
        )

        return tokenizer_path, token_path

    print()
    print("=" * 60)
    print("TRAINING BYTE-LEVEL BPE")
    print("=" * 60)

    tokenizer = Tokenizer(
        BPE(
            unk_token="[UNK]"
        )
    )

    tokenizer.pre_tokenizer = (
        ByteLevelPreTokenizer(
            add_prefix_space=False
        )
    )

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=[
            "[UNK]"
        ]
    )

    tokenizer.train(
        files=[
            str(corpus_path)
        ],
        trainer=trainer
    )

    tokenizer.decoder = ByteLevelDecoder()

    actual_vocab_size = (
        tokenizer.get_vocab_size()
    )

    if actual_vocab_size != vocab_size:
        raise RuntimeError(
            "Vocabulary size mismatch: "
            f"expected {vocab_size:,}, "
            f"got {actual_vocab_size:,}"
        )

    tokenizer.save(
        str(tokenizer_path)
    )

    print(
        f"✅ Tokenizer saved: {tokenizer_path}"
    )

    print(
        f"Vocabulary size: "
        f"{actual_vocab_size:,}"
    )

    print()
    print("Encoding corpus to uint16...")

    temporary_token_path = (
        corpus_directory
        / "aurelius_tokens.incomplete.bin"
    )

    token_count = 0
    minimum_token_id = None
    maximum_token_id = None

    with open(
        temporary_token_path,
        "wb"
    ) as output:

        with open(
            corpus_path,
            "r",
            encoding="utf-8",
            errors="replace"
        ) as corpus:

            for line in corpus:

                if not line:
                    continue

                encoded = tokenizer.encode(
                    line
                )

                if not encoded.ids:
                    continue

                ids = np.asarray(
                    encoded.ids,
                    dtype=np.uint32
                )

                current_min = int(ids.min())
                current_max = int(ids.max())

                if current_max >= vocab_size:
                    raise RuntimeError(
                        "Token ID exceeds vocabulary."
                    )

                if current_max >= 65_536:
                    raise RuntimeError(
                        "Token ID exceeds uint16 range."
                    )

                if minimum_token_id is None:
                    minimum_token_id = current_min
                    maximum_token_id = current_max
                else:
                    minimum_token_id = min(
                        minimum_token_id,
                        current_min
                    )
                    maximum_token_id = max(
                        maximum_token_id,
                        current_max
                    )

                output.write(
                    ids.astype(
                        np.uint16
                    ).tobytes()
                )

                token_count += len(ids)

    os.replace(
        temporary_token_path,
        token_path
    )

    metadata = {
        "tokenizer": str(tokenizer_path),
        "token_file": str(token_path),
        "vocab_size": actual_vocab_size,
        "token_count": token_count,
        "minimum_token_id": minimum_token_id,
        "maximum_token_id": maximum_token_id,
        "dtype": "uint16",
        "format": "raw_uint16",
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2
        ),
        encoding="utf-8"
    )

    print(
        f"✅ Tokenization complete."
    )

    print(
        f"Token count: {token_count:,}"
    )

    print(
        f"Token file size: "
        f"{token_path.stat().st_size:,} bytes"
    )

    print(
        f"Minimum token ID: "
        f"{minimum_token_id}"
    )

    print(
        f"Maximum token ID: "
        f"{maximum_token_id}"
    )

    return tokenizer_path, token_path


def write_manifest(root, corpus_path):

    manifest_path = (
        root
        / "data"
        / "corpus"
        / "manifest.json"
    )

    corpus_size = corpus_path.stat().st_size

    manifest = {
        "corpus": str(corpus_path),
        "corpus_bytes": corpus_size,
        "wikipedia_articles": WIKIPEDIA_ARTICLES,
        "tokenizer_vocab_size": VOCAB_SIZE,
        "token_format": "uint16",
        "status": "corpus_ready",
    }

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2
        ),
        encoding="utf-8"
    )

    print(
        f"✅ Manifest saved: {manifest_path}"
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
    print("AURELIUS CORPUS BUILDER")
    print("=" * 60)

    print(f"Drive root:          {root}")
    print(f"Wikipedia articles:  {args.wikipedia_articles:,}")
    print(f"BPE vocabulary:      {args.vocab_size:,}")
    print()

    download_gutenberg(root)

    wikipedia_path = collect_wikipedia(
        root,
        args.wikipedia_articles
    )

    corpus_path = build_combined_corpus(
        root,
        wikipedia_path
    )

    tokenizer_path, token_path = (
        build_tokenizer_and_tokens(
            root,
            corpus_path,
            args.vocab_size
        )
    )

    write_manifest(
        root,
        corpus_path
    )

    print()
    print(
        f"Tokenizer: {tokenizer_path}"
    )
    print(
        f"Tokens:    {token_path}"
    )
    print()
    print("🎯 COMPLETE AURELIUS DATA PIPELINE READY.")


if __name__ == "__main__":
    main()

"""
data_prep.py — Download and tokenise datasets for HyperAttn-Nano
=================================================================

Produces flat binary files (numpy uint16) consumed by train.py via memmap.

Usage:
    python scripts/data_prep.py --dataset wikitext2
    python scripts/data_prep.py --dataset shakespeare
    python scripts/data_prep.py --dataset all
"""

import argparse
import json
import os
import sys

import numpy as np
import requests


# ---------------------------------------------------------------------------
# WikiText-2
# ---------------------------------------------------------------------------

def prepare_wikitext2():
    from datasets import load_dataset
    from transformers import AutoTokenizer

    print("Preparing WikiText-2 ...")
    os.makedirs("data/wikitext2", exist_ok=True)

    dataset   = load_dataset("wikitext", "wikitext-2-raw-v1")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    split_map = {"train": "train", "validation": "val"}
    token_counts = {}

    for hf_split, file_split in split_map.items():
        text   = "\n".join(dataset[hf_split]["text"])
        tokens = tokenizer.encode(text)
        arr    = np.array(tokens, dtype=np.uint16)
        path   = f"data/wikitext2/{file_split}.bin"
        arr.tofile(path)
        token_counts[file_split] = len(arr)
        print(f"  Saved {len(arr):,} tokens → {path}")

    meta = {
        "vocab_size":   50257,
        "train_tokens": token_counts["train"],
        "val_tokens":   token_counts["val"],
    }
    with open("data/wikitext2/meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(
        f"WikiText-2: {token_counts['train']:,} train tokens | "
        f"{token_counts['val']:,} val tokens → data/wikitext2/"
    )
    return token_counts


# ---------------------------------------------------------------------------
# Tiny Shakespeare
# ---------------------------------------------------------------------------

def prepare_shakespeare():
    print("Preparing Tiny Shakespeare ...")
    os.makedirs("data/shakespeare", exist_ok=True)

    url  = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    text = resp.text

    # Build character vocabulary
    chars      = sorted(set(text))
    vocab_size = len(chars)
    stoi       = {c: i for i, c in enumerate(chars)}
    tokens     = [stoi[c] for c in text]

    # 90/10 split
    n         = len(tokens)
    split_idx = int(0.9 * n)
    train_arr = np.array(tokens[:split_idx], dtype=np.uint16)
    val_arr   = np.array(tokens[split_idx:], dtype=np.uint16)

    train_arr.tofile("data/shakespeare/train.bin")
    val_arr.tofile("data/shakespeare/val.bin")
    print(f"  Saved {len(train_arr):,} tokens → data/shakespeare/train.bin")
    print(f"  Saved {len(val_arr):,} tokens  → data/shakespeare/val.bin")

    meta = {
        "vocab_size":   vocab_size,
        "train_tokens": len(train_arr),
        "val_tokens":   len(val_arr),
    }
    with open("data/shakespeare/meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(
        f"Shakespeare: {len(train_arr):,} train tokens | "
        f"{len(val_arr):,} val tokens → data/shakespeare/"
    )
    return {"train": len(train_arr), "val": len(val_arr)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download and tokenise datasets for HyperAttn-Nano"
    )
    parser.add_argument(
        "--dataset",
        choices=["wikitext2", "shakespeare", "all"],
        required=True,
        help="Which dataset to prepare",
    )
    args = parser.parse_args()

    if args.dataset in ("wikitext2", "all"):
        prepare_wikitext2()

    if args.dataset in ("shakespeare", "all"):
        prepare_shakespeare()


if __name__ == "__main__":
    main()

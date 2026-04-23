"""
data_prep_code.py — Download and tokenise CodeParrot for HyperAttn-Nano
========================================================================

Streams codeparrot/codeparrot-clean from HuggingFace (no full download).
Tokenises using the GPT-2 tokeniser — same as WikiText-2 — to keep the
vocabulary constant across experiments.

Target token counts match WikiText-2 for direct comparability:
    Train : 2,400,000 tokens
    Val   :   248,000 tokens

Usage:
    python scripts/data_prep_code.py
    python scripts/data_prep_code.py --train-tokens 2400000 --val-tokens 248000
    python scripts/data_prep_code.py --out-dir data/codeparrot
"""

import argparse
import json
import os
import sys

import numpy as np


TRAIN_TARGET = 2_400_000
VAL_TARGET   =   248_000
DEFAULT_SEED = 42
OUT_DIR      = "data/codeparrot"


def prepare_codeparrot(
    train_target: int = TRAIN_TARGET,
    val_target:   int = VAL_TARGET,
    seed:         int = DEFAULT_SEED,
    out_dir:      str = OUT_DIR,
) -> dict:
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package not found. Run: pip install datasets")
        sys.exit(1)

    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("ERROR: 'transformers' package not found. Run: pip install transformers")
        sys.exit(1)

    print("Loading GPT-2 tokeniser ...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    unk_id    = tokenizer.unk_token_id  # None for GPT-2 (BPE — no UNK in practice)

    print("Streaming codeparrot/codeparrot-clean (train split) ...")
    # Streaming avoids downloading the full ~50 GB dataset.
    # We use the train split for both our train and val portions.
    dataset = load_dataset(
        "codeparrot/codeparrot-clean",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    # Reproducible ordering via seed-based shuffling of the stream.
    dataset = dataset.shuffle(seed=seed, buffer_size=10_000)

    os.makedirs(out_dir, exist_ok=True)

    train_tokens: list[int] = []
    val_tokens:   list[int] = []
    files_used    = 0
    total_unk     = 0
    total_tok     = 0

    print(
        f"Accumulating tokens  (target: {train_target:,} train + {val_target:,} val) ..."
    )

    for example in dataset:
        content = example.get("content", "")
        if not content:
            continue

        ids = tokenizer.encode(content)
        if not ids:
            continue

        # Count UNK tokens (GPT-2 tokeniser rarely produces them)
        if unk_id is not None:
            total_unk += ids.count(unk_id)
        total_tok  += len(ids)
        files_used += 1

        # Fill train first, then val
        if len(train_tokens) < train_target:
            remaining = train_target - len(train_tokens)
            train_tokens.extend(ids[:remaining])
            leftover = ids[remaining:]
            if leftover and len(val_tokens) < val_target:
                take = min(len(leftover), val_target - len(val_tokens))
                val_tokens.extend(leftover[:take])
        elif len(val_tokens) < val_target:
            remaining = val_target - len(val_tokens)
            val_tokens.extend(ids[:remaining])

        if len(train_tokens) >= train_target and len(val_tokens) >= val_target:
            break

        if files_used % 500 == 0:
            print(
                f"  {files_used:,} files | "
                f"train {len(train_tokens):,}/{train_target:,} | "
                f"val {len(val_tokens):,}/{val_target:,}"
            )

    # Truncate exactly to targets in case of over-accumulation edge case
    train_tokens = train_tokens[:train_target]
    val_tokens   = val_tokens[:val_target]

    # Save binary arrays
    train_arr = np.array(train_tokens, dtype=np.uint16)
    val_arr   = np.array(val_tokens,   dtype=np.uint16)

    train_path = os.path.join(out_dir, "train.bin")
    val_path   = os.path.join(out_dir, "val.bin")
    train_arr.tofile(train_path)
    val_arr.tofile(val_path)

    unk_rate = (total_unk / total_tok * 100) if total_tok > 0 else 0.0

    meta = {
        "vocab_size":   50257,
        "train_tokens": len(train_arr),
        "val_tokens":   len(val_arr),
        "files_used":   files_used,
        "unk_rate_pct": round(unk_rate, 4),
        "seed":         seed,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print()
    print("=" * 60)
    print(f"CodeParrot data prepared → {out_dir}/")
    print(f"  Train tokens : {len(train_arr):,}  (target {train_target:,})")
    print(f"  Val   tokens : {len(val_arr):,}  (target {val_target:,})")
    print(f"  Files used   : {files_used:,}")
    print(f"  UNK rate     : {unk_rate:.4f}%  (GPT-2 BPE — should be ~0)")
    if unk_rate > 5.0:
        print("  WARNING: UNK rate > 5%. Tokeniser may be poorly suited to this data.")
    print("=" * 60)

    return meta


def main():
    parser = argparse.ArgumentParser(
        description="Download and tokenise CodeParrot for HyperAttn-Nano"
    )
    parser.add_argument(
        "--train-tokens", type=int, default=TRAIN_TARGET,
        help=f"Target number of training tokens (default: {TRAIN_TARGET:,})",
    )
    parser.add_argument(
        "--val-tokens", type=int, default=VAL_TARGET,
        help=f"Target number of validation tokens (default: {VAL_TARGET:,})",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"Random seed for file ordering (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--out-dir", type=str, default=OUT_DIR,
        help=f"Output directory (default: {OUT_DIR})",
    )
    args = parser.parse_args()

    prepare_codeparrot(
        train_target=args.train_tokens,
        val_target=args.val_tokens,
        seed=args.seed,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()

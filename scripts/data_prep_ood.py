"""
data_prep_ood.py — Prepare held-out OOD tokenised splits for the eval pack.

For WikiText-2 trained models:
    Take WikiText-103 test split, filter out lines whose content overlaps any
    WikiText-2 line, tokenise with the GPT-2 tokeniser, save the first 100K
    tokens to data/ood/wikitext103_heldout/ood.bin.

Code-trained models are out of scope for this run (spec called this optional;
user said skip if it's a friction point).
"""

import json
import os

import numpy as np


def prepare_wikitext103_heldout(max_tokens: int = 100_000):
    from datasets import load_dataset
    from transformers import AutoTokenizer

    out_dir = "data/ood/wikitext103_heldout"
    os.makedirs(out_dir, exist_ok=True)

    print("Loading WikiText-2 (for overlap filtering)…")
    wt2 = load_dataset("wikitext", "wikitext-2-raw-v1")
    wt2_lines = set()
    for split in ("train", "validation", "test"):
        for line in wt2[split]["text"]:
            line = line.strip()
            if line:
                wt2_lines.add(line)
    print(f"  WikiText-2 unique non-empty lines: {len(wt2_lines):,}")

    # WikiText-103 test is a strict subset of WikiText-2 (WT2 was sampled from
    # WT103). To get genuinely held-out material, draw from WT103 *train* and
    # skip anything that appears in WT2.
    print("Loading WikiText-103 train split (streaming)…")
    wt103 = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)

    kept_lines: list[str] = []
    skipped = 0
    scanned = 0
    # Target: enough lines to yield ≥ max_tokens tokens at GPT-2 BPE (~1.3 tok/word).
    # Cap by line count not token count to keep this script cheap; tokenise after.
    LINE_CAP = 20_000
    for row in wt103:
        scanned += 1
        line = row["text"].strip()
        if not line:
            continue
        if line in wt2_lines:
            skipped += 1
            continue
        kept_lines.append(line)
        if len(kept_lines) >= LINE_CAP:
            break
    print(f"  Scanned {scanned:,} lines; kept {len(kept_lines):,} (skipped {skipped:,} overlap)")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    text = "\n".join(kept_lines)
    tokens = tokenizer.encode(text)
    print(f"  Tokenised → {len(tokens):,} tokens")

    if len(tokens) < max_tokens:
        print(f"  WARNING: only {len(tokens)} tokens available, less than max_tokens={max_tokens}")
    capped = tokens[:max_tokens]
    arr = np.array(capped, dtype=np.uint16)
    out_path = os.path.join(out_dir, "ood.bin")
    arr.tofile(out_path)
    print(f"  Saved {len(arr):,} tokens → {out_path}")

    meta = {
        "source":         "wikitext-103 test split, lines filtered against WikiText-2",
        "max_tokens":     max_tokens,
        "tokens_saved":   int(len(arr)),
        "tokens_available_before_cap": int(len(tokens)),
        "lines_kept":     len(kept_lines),
        "lines_skipped_overlap": skipped,
        "tokenizer":      "gpt2",
        "vocab_size":     50257,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Meta → {out_dir}/meta.json")


if __name__ == "__main__":
    prepare_wikitext103_heldout()

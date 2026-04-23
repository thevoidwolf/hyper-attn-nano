"""
eval_model.py — Run the full evaluation pack on a trained checkpoint
====================================================================

Usage:
    python scripts/eval_model.py --checkpoint results/checkpoints/euclid/ckpt_step8999.pt \
                                  --config configs/nano_euclid.yaml \
                                  [--ood-data data/ood/wikitext103/ood.bin]

Output:
    results/eval/<run_id>/eval_summary.json
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from model import HyperAttnNano, GPTNano, ScoresOnlyNano  # noqa: E402
from eval.ood_eval import run_eval                          # noqa: E402


def load_model_from_checkpoint(ckpt_path: str, device: torch.device):
    """Load model from checkpoint. Uses the config embedded in the checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg  = ckpt["config"]

    model_type = cfg["model"]["type"]
    model_cfg  = {
        k: cfg["model"][k]
        for k in ["d_model", "n_layers", "n_heads", "d_ff", "max_seq_len", "vocab_size"]
    }
    curvature  = cfg["model"].get("curvature", cfg["model"].get("init_K", -1.0))

    if model_type == "euclid":
        model = GPTNano(model_cfg)
    elif model_type == "hyper-scores-only":
        model = ScoresOnlyNano(model_cfg, fixed_curvature=curvature)
    else:
        model = HyperAttnNano(model_cfg, fixed_curvature=curvature)

    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, cfg


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--config",     required=True, help="Path to YAML config file")
    parser.add_argument("--ood-data",   default=None,  help="Path to OOD .bin token file")
    parser.add_argument("--eval-batches", type=int, default=50)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on: {device}")

    model, ckpt_cfg = load_model_from_checkpoint(args.checkpoint, device)
    run_id     = ckpt_cfg.get("run_id", ckpt_cfg["model"]["type"])
    data_dir   = cfg["training"]["data_dir"]
    block_size = cfg["model"]["max_seq_len"]
    batch_size = cfg["training"]["batch_size"]

    # Load token data
    def load_tokens(path):
        return np.memmap(path, dtype=np.uint16, mode="r")

    train_tokens = load_tokens(os.path.join(data_dir, "train.bin"))
    val_tokens   = load_tokens(os.path.join(data_dir, "val.bin"))
    ood_tokens   = None
    if args.ood_data and os.path.exists(args.ood_data):
        ood_tokens = load_tokens(args.ood_data)
        print(f"OOD data: {args.ood_data} ({len(ood_tokens):,} tokens)")
    else:
        print("No OOD data provided — ood_val_ppl will be null")

    print(f"Train tokens: {len(train_tokens):,}  Val tokens: {len(val_tokens):,}")
    print(f"Block size: {block_size}  Batch size: {batch_size}")
    print("Running evaluation pack...")

    results = run_eval(
        model,
        train_tokens=train_tokens,
        val_tokens=val_tokens,
        ood_tokens=ood_tokens,
        trained_block_size=block_size,
        batch_size=batch_size,
        device=device,
        eval_batches=args.eval_batches,
    )

    # Save output
    out_dir  = os.path.join("results", "eval", run_id)
    out_path = os.path.join(out_dir, "eval_summary.json")
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved → {out_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

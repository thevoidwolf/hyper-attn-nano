"""
eval_stochastic_on_checkpoint.py — Stochastic 50-batch val PPL on a saved checkpoint
====================================================================================
 
Purpose (spec A.1.1 §17.1 follow-up):
    Run the ORIGINAL stochastic 50-batch evaluation method on a checkpoint
    trained with the new main-grid recipe (float64 manifold ops + final-20%
    cosine decay + full-split eval during training).
 
    This isolates the eval-method delta between Preflight 1 (stochastic eval,
    reported 269.28 best-val) and Preflight 1b (full-split eval, reported
    280.93 best-val). By evaluating Preflight 1b's checkpoint under the
    stochastic method, we get a same-model-same-seed comparison of the two
    eval methods.
 
    Runs the 50-batch eval 5 times with different RNG seeds and reports
    mean / min / max, so we can also see how noisy the stochastic method
    inherently is.
 
Usage:
    python scripts/eval_stochastic_on_checkpoint.py \
        --checkpoint checkpoints/experiment_a/a_s2_hyper_seed42/best.pt \
        --data-dir data/wikitext2
 
    Optional:
        --eval-batches  (default 50, matches Preflight 1 method)
        --eval-seeds    (default "42,1337,2718,31415,271828" — 5 seeds)
        --output        (default: write to logs/experiment_a/<run_id>/
                                  stochastic_eval.json)
 
Produces a single JSON file with mean / min / max / all seeds.
"""
 
import argparse
import json
import math
import os
import random
import sys
 
import numpy as np
import torch
 
# Make src/ importable whether invoked from project root or scripts/
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
 
from model import HyperAttnNano, GPTNano, ScoresOnlyNano  # noqa: E402
 
 
# ---------------------------------------------------------------------------
# Reconstruct model from checkpoint
# ---------------------------------------------------------------------------
 
def build_model_from_ckpt(ckpt: dict, device: torch.device):
    """
    Rebuild model from a checkpoint produced by the A.1.1 train.py.
    Accepts both the new format (keys: model_state_dict, run_config) and the
    legacy format (keys: model, config) so this script can eval older
    checkpoints too.
    """
    # --- config ---
    if "run_config" in ckpt:
        cfg = ckpt["run_config"]
    elif "config" in ckpt:
        cfg = ckpt["config"]
    else:
        raise ValueError("Checkpoint has no config — cannot rebuild model.")
 
    # --- state dict ---
    if "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    elif "model" in ckpt:
        sd = ckpt["model"]
    else:
        raise ValueError("Checkpoint has no model weights.")
 
    # --- model construction ---
    model_cfg_raw = cfg["model"]
    model_type    = model_cfg_raw["type"]
 
    # Pass the full model sub-config so HyperAttnNano.__init__ can read
    # manifold_float64 and any other flags. Drop "type" and "curvature_schedule"
    # which the model classes don't take as constructor kwargs.
    model_cfg = {
        k: v for k, v in model_cfg_raw.items()
        if k not in ("type", "curvature_schedule")
    }
 
    curvature = model_cfg_raw.get("curvature", model_cfg_raw.get("init_K", -1.0))
 
    if model_type == "euclid":
        model = GPTNano(model_cfg)
    elif model_type == "hyper-scores-only":
        model = ScoresOnlyNano(model_cfg, fixed_curvature=curvature)
    elif model_type in ("hyper-fixed", "hyper-perhead"):
        model = HyperAttnNano(model_cfg, fixed_curvature=curvature)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
 
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    return model, cfg
 
 
# ---------------------------------------------------------------------------
# Stochastic eval (matches train.py's `evaluate()` used by Preflight 1)
# ---------------------------------------------------------------------------
 
def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack(
        [torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [torch.from_numpy(data[i + 1 : i + block_size + 1].astype(np.int64)) for i in ix]
    )
    return x.to(device), y.to(device)
 
 
@torch.no_grad()
def evaluate_stochastic(model, data, block_size, batch_size, device, eval_batches=50):
    """Exact logic from train.py `evaluate()` — stochastic batch sampling."""
    model.eval()
    losses = []
    for _ in range(eval_batches):
        x, y = get_batch(data, block_size, batch_size, device)
        if device.type == "cuda":
            with torch.amp.autocast("cuda"):
                _, loss = model(x, y)
        else:
            _, loss = model(x, y)
        losses.append(loss.item())
    return sum(losses) / len(losses)
 
 
# ---------------------------------------------------------------------------
# Seed control — isolate each eval-seed run
# ---------------------------------------------------------------------------
 
def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
 
 
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
 
def main():
    parser = argparse.ArgumentParser(
        description="Stochastic 50-batch val PPL on a saved checkpoint."
    )
    parser.add_argument("--checkpoint",  required=True, help="Path to .pt file")
    parser.add_argument("--data-dir",    default="data/wikitext2",
                        help="Directory containing val.bin")
    parser.add_argument("--eval-batches", type=int, default=50,
                        help="Batches per eval run (Preflight 1 used 50)")
    parser.add_argument("--eval-seeds",  default="42,1337,2718,31415,271828",
                        help="Comma-separated RNG seeds for the stochastic sampler")
    parser.add_argument("--output",      default=None,
                        help="Where to write the JSON summary (auto-derived if omitted)")
    args = parser.parse_args()
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
 
    # --- load checkpoint ---
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model, cfg = build_model_from_ckpt(ckpt, device)
 
    run_id      = cfg.get("run_id", "unknown")
    model_type  = cfg["model"]["type"]
    block_size  = cfg["model"]["max_seq_len"]
    batch_size  = cfg["training"]["batch_size"]
    step        = ckpt.get("step", "?")
    is_float64  = cfg["model"].get("manifold_float64", False)
 
    print(f"Run ID:     {run_id}")
    print(f"Model type: {model_type}")
    print(f"Checkpoint step: {step}")
    print(f"manifold_float64: {is_float64}")
    print(f"Block size: {block_size}, batch size: {batch_size}")
 
    # --- load val tokens ---
    val_path = os.path.join(args.data_dir, "val.bin")
    if not os.path.exists(val_path):
        raise FileNotFoundError(
            f"Could not find {val_path}. Pass --data-dir explicitly if needed."
        )
    val_tokens = np.memmap(val_path, dtype=np.uint16, mode="r")
    print(f"Val tokens: {len(val_tokens):,}")
 
    # --- run stochastic eval under each seed ---
    seeds = [int(s.strip()) for s in args.eval_seeds.split(",") if s.strip()]
    print(f"\nRunning {args.eval_batches}-batch stochastic eval "
          f"under {len(seeds)} seeds: {seeds}")
 
    per_seed = []
    for s in seeds:
        set_all_seeds(s)
        avg_loss = evaluate_stochastic(
            model, val_tokens, block_size, batch_size, device,
            eval_batches=args.eval_batches,
        )
        ppl = math.exp(avg_loss)
        per_seed.append({"seed": s, "avg_loss": avg_loss, "ppl": ppl})
        print(f"  seed={s:>7}  avg_loss={avg_loss:.4f}  ppl={ppl:.3f}")
 
    ppls = [r["ppl"] for r in per_seed]
    mean_ppl   = sum(ppls) / len(ppls)
    min_ppl    = min(ppls)
    max_ppl    = max(ppls)
    stdev_ppl  = (sum((p - mean_ppl) ** 2 for p in ppls) / len(ppls)) ** 0.5
 
    print("\nSummary:")
    print(f"  mean PPL : {mean_ppl:.3f}")
    print(f"  min  PPL : {min_ppl:.3f}")
    print(f"  max  PPL : {max_ppl:.3f}")
    print(f"  stdev    : {stdev_ppl:.3f}  (population stdev across seeds)")
 
    # --- write JSON summary ---
    if args.output:
        out_path = args.output
    else:
        log_dir  = cfg.get("log_dir", f"logs/experiment_a/{run_id}")
        out_path = os.path.join(log_dir, "stochastic_eval.json")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
 
    summary = {
        "run_id":            run_id,
        "checkpoint_path":   os.path.abspath(args.checkpoint),
        "checkpoint_step":   step,
        "model_type":        model_type,
        "manifold_float64":  is_float64,
        "eval_method":       "stochastic",
        "eval_batches":      args.eval_batches,
        "seeds":             seeds,
        "per_seed_results":  per_seed,
        "mean_ppl":          mean_ppl,
        "min_ppl":           min_ppl,
        "max_ppl":           max_ppl,
        "stdev_ppl":         stdev_ppl,
    }
 
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved → {out_path}")
 
 
if __name__ == "__main__":
    main()
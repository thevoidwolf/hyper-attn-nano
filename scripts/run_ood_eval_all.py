"""
run_ood_eval_all.py — Batch-evaluate every available checkpoint with the OOD pack.

Walks two checkpoint trees:
  results/checkpoints/<variant>/ckpt_step*.pt     (legacy format)
  checkpoints/experiment_a/<run>/{best,final}.pt  (Experiment A format)

For each checkpoint:
  - Loads model + config (the config is embedded in the checkpoint)
  - Decides which dataset to use for train/val tokens (from config.training.data_dir)
  - Routes OOD tokens: data/ood/wikitext103_heldout/ood.bin for WikiText-2 models;
    None for CodeParrot models (no code OOD prepared in this run)
  - Runs src.eval.ood_eval.run_eval
  - Writes results/eval/<run_id>__<ckpt_basename>/eval_summary.json

Finally, aggregates everything into results/eval/AGGREGATE.json.
"""

import glob
import json
import math
import os
import sys
import traceback

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import manifolds  # noqa: E402
from model import HyperAttnNano, GPTNano, ScoresOnlyNano  # noqa: E402
from eval.ood_eval import run_eval  # noqa: E402


WT103_OOD_PATH = "data/ood/wikitext103_heldout/ood.bin"


def load_tokens(path):
    return np.memmap(path, dtype=np.uint16, mode="r")


def build_model_from_cfg(cfg: dict, device: torch.device):
    """Construct a model matching the saved config, handling both formats."""
    model_type = cfg["model"]["type"]
    model_cfg = {
        k: cfg["model"][k]
        for k in ["d_model", "n_layers", "n_heads", "d_ff", "max_seq_len", "vocab_size"]
    }
    for opt_key in ("manifold_float64", "output_head", "spherical_temperature_init", "qk_norm"):
        if opt_key in cfg["model"]:
            model_cfg[opt_key] = cfg["model"][opt_key]

    init_K = cfg["model"].get("init_K", -1.0)
    curvature = cfg["model"].get("curvature", init_K)
    curvature_init = cfg["model"].get("curvature_init", None)

    # If the saved config used a curvature_schedule (no scalar curvature key),
    # use k_end as the operating curvature for evaluation.
    schedule = cfg["model"].get("curvature_schedule")
    if schedule is not None:
        curvature = schedule.get("k_end", curvature)

    if model_type == "euclid":
        model = GPTNano(model_cfg)
    elif model_type == "hyper-scores-only":
        model = ScoresOnlyNano(model_cfg, fixed_curvature=curvature)
    else:
        model = HyperAttnNano(model_cfg, fixed_curvature=curvature, curvature_init=curvature_init)

    model = model.to(device).eval()
    return model


def load_checkpoint(path: str, device: torch.device):
    """Returns (model, cfg, step). Handles legacy and Experiment A formats."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "config" in ckpt and "model" in ckpt:  # legacy
        cfg = ckpt["config"]
        sd = ckpt["model"]
        step = ckpt.get("step", -1)
    elif "run_config" in ckpt and "model_state_dict" in ckpt:  # exp A
        cfg = ckpt["run_config"]
        sd = ckpt["model_state_dict"]
        step = ckpt.get("step", -1)
    else:
        raise ValueError(f"Unknown checkpoint format: {path}; keys={list(ckpt.keys())}")
    model = build_model_from_cfg(cfg, device)
    model.load_state_dict(sd)
    return model, cfg, step


def find_checkpoints():
    """Return list of (label, path) pairs to evaluate."""
    results = []
    # Legacy
    for p in sorted(glob.glob("results/checkpoints/*/ckpt_step*.pt")):
        # Only keep the highest-step file per variant directory (final state)
        variant = os.path.basename(os.path.dirname(p))
        results.append((f"legacy::{variant}::{os.path.basename(p)}", p))
    # Experiment A — only best.pt and final.pt
    for p in sorted(glob.glob("checkpoints/experiment_a/*/best.pt")):
        run = os.path.basename(os.path.dirname(p))
        results.append((f"expA::{run}::best", p))
    for p in sorted(glob.glob("checkpoints/experiment_a/*/final.pt")):
        run = os.path.basename(os.path.dirname(p))
        results.append((f"expA::{run}::final", p))
    return results


def eval_one(path: str, label: str, device: torch.device, out_root: str, eval_batches: int):
    print(f"\n==== {label} ====")
    print(f"  path: {path}")
    try:
        model, cfg, step = load_checkpoint(path, device)
    except Exception as e:
        print(f"  LOAD FAILED: {e}")
        return {"label": label, "path": path, "status": "load_failed", "error": str(e)}

    data_dir = cfg["training"]["data_dir"]
    dataset = cfg["training"].get("dataset", os.path.basename(data_dir))
    block_size = cfg["model"]["max_seq_len"]
    batch_size = cfg["training"]["batch_size"]
    run_id = cfg.get("run_id", cfg["model"]["type"])

    train_tokens = load_tokens(os.path.join(data_dir, "train.bin"))
    val_tokens = load_tokens(os.path.join(data_dir, "val.bin"))
    print(f"  run_id: {run_id}  type: {cfg['model']['type']}  d_model: {cfg['model']['d_model']}  step: {step}")
    print(f"  dataset: {dataset}  train_tokens: {len(train_tokens):,}  val_tokens: {len(val_tokens):,}")

    # Route OOD: WikiText-2 models get the WT103 heldout; everything else None.
    ood_tokens = None
    ood_source = None
    if dataset == "wikitext2" and os.path.exists(WT103_OOD_PATH):
        ood_tokens = load_tokens(WT103_OOD_PATH)
        ood_source = WT103_OOD_PATH

    # Guard: make sure manifold flag matches the checkpoint's config.
    # (build_model_from_cfg already sets MANIFOLD_FLOAT64 via the HyperAttnNano
    # constructor when manifold_float64=True. But for non-hyperbolic models or
    # for models trained without f64, we must explicitly reset to False to
    # avoid carryover from a previous iteration in this same process.)
    manifolds.MANIFOLD_FLOAT64 = bool(cfg["model"].get("manifold_float64", False))

    try:
        with torch.no_grad():
            results = run_eval(
                model,
                train_tokens=train_tokens,
                val_tokens=val_tokens,
                ood_tokens=ood_tokens,
                trained_block_size=block_size,
                batch_size=batch_size,
                device=device,
                eval_batches=eval_batches,
            )
    except Exception as e:
        traceback.print_exc()
        return {
            "label": label, "path": path, "run_id": run_id, "status": "eval_failed",
            "error": str(e),
        }

    summary = {
        "label":             label,
        "path":              path,
        "run_id":            run_id,
        "step":              step,
        "model_type":        cfg["model"]["type"],
        "d_model":           cfg["model"]["d_model"],
        "n_layers":          cfg["model"]["n_layers"],
        "n_heads":           cfg["model"]["n_heads"],
        "curvature":         cfg["model"].get("curvature", cfg["model"].get("init_K")),
        "curvature_schedule": cfg["model"].get("curvature_schedule"),
        "manifold_float64":  cfg["model"].get("manifold_float64", False),
        "dataset":           dataset,
        "ood_source":        ood_source,
        "status":            "ok",
        **results,
    }

    out_dir = os.path.join(out_root, label.replace("::", "__"))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "eval_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  -> id_ppl={results['id_val_ppl']:.2f}  "
          f"ood_ppl={results['ood_val_ppl'] if results['ood_val_ppl'] else 'N/A'}  "
          f"rare_ppl={results['rare_word_ppl']:.2f}  "
          f"long_ctx={results['long_ctx_ppl']}")
    return summary


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", default="results/eval")
    p.add_argument("--eval-batches", type=int, default=50)
    p.add_argument("--only", default=None,
                   help="substring filter on label; only eval matching ckpts")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cks = find_checkpoints()
    if args.only:
        cks = [(lab, p) for (lab, p) in cks if args.only in lab]
    print(f"Will evaluate {len(cks)} checkpoint(s).")
    for lab, p in cks:
        print(f"  {lab}: {p}")

    os.makedirs(args.out_root, exist_ok=True)
    aggregate = []
    for label, path in cks:
        summary = eval_one(path, label, device, args.out_root, args.eval_batches)
        aggregate.append(summary)
        # Free model memory between evals
        if device.type == "cuda":
            torch.cuda.empty_cache()

    agg_path = os.path.join(args.out_root, "AGGREGATE.json")
    with open(agg_path, "w") as f:
        json.dump(aggregate, f, indent=2)
    print(f"\nAggregate -> {agg_path}")


if __name__ == "__main__":
    main()

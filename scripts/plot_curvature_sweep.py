"""
plot_curvature_sweep.py — Plots for Experiment B curvature sweep
=================================================================

Reads JSONL logs from results/logs/exp_b/ and produces three plots
saved to results/plots/exp_b/:

  1. ppl_vs_curvature.png  — bar chart of final PPL per K value
  2. ppl_curves.png        — PPL training curves per K (log Y scale)
  3. grad_norm_curves.png  — Gradient norm over training per K

Usage:
    python scripts/plot_curvature_sweep.py
"""

import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")   # headless
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

LOG_DIR  = "results/logs/exp_b"
PLOT_DIR = "results/plots/exp_b"

EUCLID_BASELINE_PPL = 277.0   # V3 Euclidean baseline from results/logs/v3/

RUNS = [
    ("exp_b_k1",  -1.0,  "K=−1"),
    ("exp_b_k2",  -2.0,  "K=−2"),
    ("exp_b_k5",  -5.0,  "K=−5"),
    ("exp_b_k10", -10.0, "K=−10"),
    ("exp_b_k50", -50.0, "K=−50"),
]

COLORS = ["#4c72b0", "#55a868", "#c44e52", "#8172b2", "#937860"]


def load_run(run_id):
    path = os.path.join(LOG_DIR, f"{run_id}_train.jsonl")
    if not os.path.exists(path):
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def _final_ppl(records):
    for rec in reversed(records):
        ppl = rec.get("val_ppl")
        if ppl is not None and math.isfinite(ppl):
            return ppl
    return None


def _divergence_step(records):
    for rec in records:
        ppl = rec.get("val_ppl", float("inf"))
        if not math.isfinite(ppl) or ppl > 1e6:
            return rec["step"]
    return None


def plot_ppl_bar(all_data):
    fig, ax = plt.subplots(figsize=(8, 5))
    x_labels, heights, bar_colors, hatches, div_labels = [], [], [], [], []

    for (run_id, K_val, label), color in zip(RUNS, COLORS):
        records  = all_data[run_id]
        div_step = _divergence_step(records)
        final    = _final_ppl(records)
        x_labels.append(label)
        heights.append(final if final is not None else 0)
        bar_colors.append(color)
        hatches.append("///" if div_step is not None else "")
        div_labels.append(f"diverged @ {div_step}" if div_step is not None else None)

    x    = np.arange(len(x_labels))
    bars = ax.bar(x, heights, color=bar_colors, hatch=hatches,
                  edgecolor="black", linewidth=0.8, alpha=0.85)

    for bar, dl in zip(bars, div_labels):
        if dl:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                    dl, ha="center", va="bottom", fontsize=8,
                    color="darkred", rotation=15)

    ax.axhline(EUCLID_BASELINE_PPL, color="black", linestyle="--",
               linewidth=1.2, label=f"Euclid V3 baseline ({EUCLID_BASELINE_PPL:.0f})")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=11)
    ax.set_xlabel("Fixed curvature K", fontsize=12)
    ax.set_ylabel("Final validation PPL", fontsize=12)
    ax.set_title("Experiment B: Final PPL vs Curvature", fontsize=13, fontweight="bold")

    hatch_patch = mpatches.Patch(facecolor="lightgrey", hatch="///",
                                  edgecolor="black", label="Diverged run")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + [hatch_patch], labels + ["Diverged run"], fontsize=10)

    fig.tight_layout()
    out = os.path.join(PLOT_DIR, "ppl_vs_curvature.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def plot_ppl_curves(all_data):
    fig, ax = plt.subplots(figsize=(9, 5))
    for (run_id, K_val, label), color in zip(RUNS, COLORS):
        records = all_data[run_id]
        if not records:
            continue
        steps = [r["step"] for r in records if "val_ppl" in r]
        ppls  = [min(r["val_ppl"], 1e4) if math.isfinite(r["val_ppl"]) else 1e4
                 for r in records if "val_ppl" in r]
        ax.plot(steps, ppls, color=color, label=label, linewidth=1.8)

    ax.axhline(EUCLID_BASELINE_PPL, color="black", linestyle="--",
               linewidth=1.2, label=f"Euclid V3 ({EUCLID_BASELINE_PPL:.0f})")
    ax.set_yscale("log")
    ax.set_xlabel("Training step", fontsize=12)
    ax.set_ylabel("Validation PPL (log scale)", fontsize=12)
    ax.set_title("Experiment B: PPL Training Curves", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    fig.tight_layout()
    out = os.path.join(PLOT_DIR, "ppl_curves.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def plot_grad_norm_curves(all_data):
    fig, ax = plt.subplots(figsize=(9, 5))
    for (run_id, K_val, label), color in zip(RUNS, COLORS):
        records = all_data[run_id]
        if not records:
            continue
        steps  = [r["step"] for r in records if "grad_norm" in r]
        gnorms = [r["grad_norm"] for r in records if "grad_norm" in r]
        ax.plot(steps, gnorms, color=color, label=label, linewidth=1.5, alpha=0.85)

    ax.set_xlabel("Training step", fontsize=12)
    ax.set_ylabel("Gradient norm", fontsize=12)
    ax.set_title("Experiment B: Gradient Norm Over Training", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    fig.tight_layout()
    out = os.path.join(PLOT_DIR, "grad_norm_curves.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    all_data = {}
    for run_id, K_val, label in RUNS:
        records = load_run(run_id)
        all_data[run_id] = records
        if records:
            print(f"  Loaded {len(records)} records for {run_id}")
        else:
            print(f"  [WARN] No data for {run_id} — will skip in plots")

    if all(not v for v in all_data.values()):
        print("[ERROR] No log data found in", LOG_DIR)
        sys.exit(1)

    plot_ppl_bar(all_data)
    plot_ppl_curves(all_data)
    plot_grad_norm_curves(all_data)
    print(f"\nAll plots saved to {PLOT_DIR}/")


if __name__ == "__main__":
    main()

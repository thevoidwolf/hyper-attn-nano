"""
plot_b2.py — Plots for Experiment B2 (float64 high-curvature probe)
====================================================================

Reads JSONL logs from results/logs/exp_b/ and results/logs/exp_b2/ and
produces three plots saved to results/plots/exp_b2/:

  1. ppl_vs_curvature_combined.png  — B (float32) vs B2 (float64) bar chart
  2. ppl_curves_b2.png              — PPL training curves for B2 runs
  3. grad_norm_b2.png               — Grad norm: B k10 vs B2 k10/k20/k50

Usage:
    python scripts/plot_b2.py
"""

import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

LOG_DIR_B   = "results/logs/exp_b"
LOG_DIR_B2  = "results/logs/exp_b2"
PLOT_DIR    = "results/plots/exp_b2"

EUCLID_BASELINE_PPL = 277.0

# Experiment B runs (float32)
RUNS_B = [
    ("exp_b_k1",  -1.0,  "K=−1"),
    ("exp_b_k2",  -2.0,  "K=−2"),
    ("exp_b_k5",  -5.0,  "K=−5"),
    ("exp_b_k10", -10.0, "K=−10"),
    ("exp_b_k50", -50.0, "K=−50"),
]

# Experiment B2 runs (float64)
RUNS_B2 = [
    ("exp_b2_k10",  -10.0,  "K=−10\n(f64)"),
    ("exp_b2_k20",  -20.0,  "K=−20\n(f64)"),
    ("exp_b2_k50",  -50.0,  "K=−50\n(f64)"),
    ("exp_b2_k100", -100.0, "K=−100\n(f64)"),
]

COLORS_B  = ["#4c72b0", "#55a868", "#c44e52", "#8172b2", "#937860"]
COLORS_B2 = ["#1a9850", "#fc8d59", "#d73027", "#7b2d8b"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_run(run_id: str, log_dir: str) -> list:
    path = os.path.join(log_dir, f"{run_id}_train.jsonl")
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


def _final_ppl(records: list) -> float | None:
    for rec in reversed(records):
        ppl = rec.get("val_ppl")
        if ppl is not None and math.isfinite(ppl):
            return ppl
    return None


def _divergence_step(records: list) -> int | None:
    for rec in records:
        ppl = rec.get("val_ppl", float("inf"))
        if not math.isfinite(ppl) or ppl > 1e6:
            return rec["step"]
    return None


# ---------------------------------------------------------------------------
# Plot 1: Combined B + B2 bar chart
# ---------------------------------------------------------------------------

def plot_combined_bar(data_b: dict, data_b2: dict):
    fig, ax = plt.subplots(figsize=(12, 5))

    # B bars (solid)
    b_labels, b_heights, b_colors = [], [], []
    for (run_id, K_val, label), color in zip(RUNS_B, COLORS_B):
        recs  = data_b[run_id]
        final = _final_ppl(recs)
        b_labels.append(label)
        b_heights.append(final if final is not None else 0)
        b_colors.append(color)

    # B2 bars (hatched)
    b2_labels, b2_heights, b2_colors = [], [], []
    for (run_id, K_val, label), color in zip(RUNS_B2, COLORS_B2):
        recs  = data_b2[run_id]
        final = _final_ppl(recs)
        b2_labels.append(label)
        b2_heights.append(final if final is not None else 0)
        b2_colors.append(color)

    # Layout: B bars, small gap, B2 bars
    n_b  = len(b_labels)
    n_b2 = len(b2_labels)
    gap  = 0.8
    x_b  = np.arange(n_b)
    x_b2 = np.arange(n_b2) + n_b + gap

    ax.bar(x_b,  b_heights,  color=b_colors,  edgecolor="black",
           linewidth=0.8, alpha=0.85, label="B (float32)")
    ax.bar(x_b2, b2_heights, color=b2_colors, edgecolor="black",
           linewidth=0.8, alpha=0.85, hatch="///", label="B2 (float64)")

    all_x = list(x_b) + list(x_b2)
    all_labels = b_labels + b2_labels
    ax.set_xticks(all_x)
    ax.set_xticklabels(all_labels, fontsize=9)

    ax.axhline(EUCLID_BASELINE_PPL, color="black", linestyle="--",
               linewidth=1.2, label=f"Euclid baseline ({EUCLID_BASELINE_PPL:.0f})")

    ax.set_ylabel("Final validation PPL", fontsize=12)
    ax.set_title("B vs B2: Final PPL — float32 vs float64 manifold ops",
                 fontsize=13, fontweight="bold")

    solid_patch  = mpatches.Patch(facecolor="lightblue",  edgecolor="black",
                                   label="B runs (float32)")
    hatch_patch  = mpatches.Patch(facecolor="lightgreen", edgecolor="black",
                                   hatch="///", label="B2 runs (float64)")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=[solid_patch, hatch_patch] + handles[2:],
              fontsize=10, loc="upper right")

    fig.tight_layout()
    out = os.path.join(PLOT_DIR, "ppl_vs_curvature_combined.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Plot 2: B2 PPL training curves
# ---------------------------------------------------------------------------

def plot_b2_ppl_curves(data_b2: dict):
    fig, ax = plt.subplots(figsize=(9, 5))
    for (run_id, K_val, label), color in zip(RUNS_B2, COLORS_B2):
        records = data_b2[run_id]
        if not records:
            continue
        steps = [r["step"] for r in records if "val_ppl" in r]
        ppls  = [min(r["val_ppl"], 1e4) if math.isfinite(r["val_ppl"]) else 1e4
                 for r in records if "val_ppl" in r]
        label_clean = label.replace("\n", " ")
        ax.plot(steps, ppls, color=color, label=label_clean, linewidth=1.8)

    ax.axhline(EUCLID_BASELINE_PPL, color="black", linestyle="--",
               linewidth=1.2, label=f"Euclid baseline ({EUCLID_BASELINE_PPL:.0f})")
    ax.set_yscale("log")
    ax.set_xlabel("Training step", fontsize=12)
    ax.set_ylabel("Validation PPL (log scale)", fontsize=12)
    ax.set_title("Experiment B2: PPL Training Curves (float64)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    fig.tight_layout()
    out = os.path.join(PLOT_DIR, "ppl_curves_b2.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Plot 3: Grad norm — B k10 vs B2 k10/k20/k50
# ---------------------------------------------------------------------------

def plot_grad_norm_comparison(data_b: dict, data_b2: dict):
    fig, ax = plt.subplots(figsize=(9, 5))

    # B k10 reference
    b_k10 = data_b.get("exp_b_k10", [])
    if b_k10:
        steps  = [r["step"] for r in b_k10 if "grad_norm" in r]
        gnorms = [r["grad_norm"] for r in b_k10 if "grad_norm" in r]
        ax.plot(steps, gnorms, color="#8172b2", linestyle="--",
                linewidth=1.5, alpha=0.85, label="B k10 (float32)")

    # B2 runs
    b2_subset = [
        ("exp_b2_k10",  "K=−10 (f64)", COLORS_B2[0]),
        ("exp_b2_k20",  "K=−20 (f64)", COLORS_B2[1]),
        ("exp_b2_k50",  "K=−50 (f64)", COLORS_B2[2]),
    ]
    for run_id, label, color in b2_subset:
        records = data_b2.get(run_id, [])
        if not records:
            continue
        steps  = [r["step"] for r in records if "grad_norm" in r]
        gnorms = [r["grad_norm"] for r in records if "grad_norm" in r]
        ax.plot(steps, gnorms, color=color, label=label,
                linewidth=1.5, alpha=0.85)

    ax.set_xlabel("Training step", fontsize=12)
    ax.set_ylabel("Gradient norm", fontsize=12)
    ax.set_title("Grad Norm: B k10 (float32) vs B2 (float64)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    fig.tight_layout()
    out = os.path.join(PLOT_DIR, "grad_norm_b2.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    # Load B data
    data_b = {}
    for run_id, _, _ in RUNS_B:
        recs = load_run(run_id, LOG_DIR_B)
        data_b[run_id] = recs
        if recs:
            print(f"  [B]  Loaded {len(recs)} records for {run_id}")
        else:
            print(f"  [B]  [WARN] No data for {run_id}")

    # Load B2 data
    data_b2 = {}
    for run_id, _, _ in RUNS_B2:
        recs = load_run(run_id, LOG_DIR_B2)
        data_b2[run_id] = recs
        if recs:
            print(f"  [B2] Loaded {len(recs)} records for {run_id}")
        else:
            print(f"  [B2] [WARN] No data for {run_id} — will skip in plots")

    if all(not v for v in data_b2.values()):
        print("[ERROR] No B2 log data found in", LOG_DIR_B2)
        sys.exit(1)

    plot_combined_bar(data_b, data_b2)
    plot_b2_ppl_curves(data_b2)
    plot_grad_norm_comparison(data_b, data_b2)

    print(f"\nAll plots saved to {PLOT_DIR}/")


if __name__ == "__main__":
    main()

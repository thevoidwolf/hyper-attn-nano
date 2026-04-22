"""
viz_curvature.py — Visualise per-head curvature evolution for hyper-perhead
============================================================================

Usage:
    python scripts/viz_curvature.py                  # uses v3 logs by default
    python scripts/viz_curvature.py --version v2
    python scripts/viz_curvature.py --version v3 --out results/plots/

Produces two plots:
  1. curvature_evolution.png  — all heads over training steps (coloured by layer)
  2. curvature_heatmap.png    — final-step K values as a layer × head heatmap
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


LAYER_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]  # one per layer


def load_curvatures(log_path: str):
    with open(log_path) as f:
        log = json.load(f)
    entries = log["curvatures"]
    steps = [e["step"] for e in entries]
    # head_name -> list of K values aligned to steps
    heads = list(entries[0]["values"].keys())
    series = {h: [e["values"][h] for e in entries] for h in heads}
    return steps, series


def parse_head(name: str):
    """layer_X_head_Y -> (layer_idx, head_idx)"""
    parts = name.split("_")
    return int(parts[1]), int(parts[3])


def plot_evolution(steps, series, out_path: str):
    fig, ax = plt.subplots(figsize=(10, 5))

    for head_name, values in sorted(series.items()):
        layer, head = parse_head(head_name)
        color = LAYER_COLORS[layer % len(LAYER_COLORS)]
        alpha = 0.5 + 0.5 * (head / (max(parse_head(h)[1] for h in series) + 1))
        ax.plot(steps, values, color=color, alpha=alpha, linewidth=1.2, label=head_name)

    # legend: one entry per layer
    from matplotlib.lines import Line2D
    n_layers = max(parse_head(h)[0] for h in series) + 1
    legend_handles = [
        Line2D([0], [0], color=LAYER_COLORS[l], linewidth=2, label=f"Layer {l}")
        for l in range(n_layers)
    ]
    ax.legend(handles=legend_handles, loc="lower right")

    ax.axhline(-1.0, color="grey", linewidth=0.8, linestyle="--", label="K=-1 (init)")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Curvature K")
    ax.set_title("Per-head curvature evolution (hyper-perhead)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_heatmap(series, out_path: str):
    heads = sorted(series.keys())
    n_layers = max(parse_head(h)[0] for h in heads) + 1
    n_heads = max(parse_head(h)[1] for h in heads) + 1

    grid = np.zeros((n_layers, n_heads))
    for h in heads:
        l, hi = parse_head(h)
        grid[l, hi] = series[h][-1]  # final value

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(grid, cmap="coolwarm_r", aspect="auto")
    plt.colorbar(im, ax=ax, label="Curvature K")

    ax.set_xticks(range(n_heads))
    ax.set_xticklabels([f"Head {i}" for i in range(n_heads)])
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f"Layer {i}" for i in range(n_layers)])

    for l in range(n_layers):
        for hi in range(n_heads):
            ax.text(hi, l, f"{grid[l, hi]:.3f}", ha="center", va="center",
                    fontsize=8, color="black")

    ax.set_title("Final curvature K by layer × head")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v3", help="Log version folder (v1/v2/v3)")
    parser.add_argument("--out", default="results/plots",
                        help="Output directory for plots")
    args = parser.parse_args()

    log_path = os.path.join("results", "logs", args.version, "hyper-perhead_log.json")
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Log not found: {log_path}")

    os.makedirs(args.out, exist_ok=True)
    steps, series = load_curvatures(log_path)

    plot_evolution(steps, series,
                   os.path.join(args.out, f"curvature_evolution_{args.version}.png"))
    plot_heatmap(series,
                 os.path.join(args.out, f"curvature_heatmap_{args.version}.png"))


if __name__ == "__main__":
    main()

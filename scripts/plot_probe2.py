"""
plot_probe2.py — Bar chart: Euclidean vs hyper-fixed K=-10 vs hyper-scores-only
=================================================================================

Produces a bar chart comparing final validation PPL for:
  1. Euclidean baseline (GPTNano)
  2. hyper-fixed K=-10 (Experiment B k10)
  3. hyper-scores-only K=-10 (Probe 2)

Usage:
    python scripts/plot_probe2.py
    python scripts/plot_probe2.py --out results/plots/probe2/probe2_bar.png
"""

import argparse
import json
import os
import sys


def final_val_ppl_from_jsonl(path: str) -> float | None:
    """Return the final val_ppl from a JSONL training log."""
    last_ppl = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "val_ppl" in rec and "event" not in rec:
                last_ppl = rec["val_ppl"]
    return last_ppl


def final_val_ppl_from_json_log(path: str) -> float | None:
    """Return the final val PPL from a .json training log (val_loss array)."""
    with open(path) as f:
        data = json.load(f)
    val_entries = data.get("val_loss", [])
    if not val_entries:
        return None
    return val_entries[-1].get("perplexity")


def main():
    parser = argparse.ArgumentParser(description="Plot Probe 2 bar chart")
    parser.add_argument(
        "--euclid-log",
        default="results/logs/v3/euclid_log.json",
        help="JSON log for Euclidean baseline",
    )
    parser.add_argument(
        "--hyper-fixed-log",
        default="results/logs/exp_b/exp_b_k10_train.jsonl",
        help="JSONL log for hyper-fixed K=-10 (Experiment B)",
    )
    parser.add_argument(
        "--probe2-log",
        default="results/logs/probes/probe2_scores_only_train.jsonl",
        help="JSONL log for Probe 2 (hyper-scores-only)",
    )
    parser.add_argument(
        "--out",
        default="results/plots/probe2/probe2_bar.png",
        help="Output path for the plot",
    )
    args = parser.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("ERROR: matplotlib not available. Install with: pip install matplotlib")
        sys.exit(1)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # Collect PPL values
    entries = []

    if os.path.exists(args.euclid_log):
        ppl = final_val_ppl_from_json_log(args.euclid_log)
        if ppl:
            entries.append(("Euclidean\nbaseline", ppl, "steelblue"))
            print(f"Euclidean final val PPL:       {ppl:.1f}")
    else:
        print(f"[WARN] Euclidean log not found: {args.euclid_log}")

    if os.path.exists(args.hyper_fixed_log):
        ppl = final_val_ppl_from_jsonl(args.hyper_fixed_log)
        if ppl:
            entries.append(("hyper-fixed\nK=-10", ppl, "tomato"))
            print(f"hyper-fixed K=-10 final PPL:   {ppl:.1f}")
    else:
        print(f"[WARN] hyper-fixed log not found: {args.hyper_fixed_log}")

    if os.path.exists(args.probe2_log):
        ppl = final_val_ppl_from_jsonl(args.probe2_log)
        if ppl:
            entries.append(("hyper-scores-only\nK=-10", ppl, "mediumseagreen"))
            print(f"hyper-scores-only final PPL:   {ppl:.1f}")
    else:
        print(f"[INFO] Probe 2 log not found: {args.probe2_log} — run probe 2 first.")

    if not entries:
        print("No data available to plot. Run the experiments first.")
        sys.exit(0)

    labels = [e[0] for e in entries]
    ppls   = [e[1] for e in entries]
    colors = [e[2] for e in entries]

    fig, ax = plt.subplots(figsize=(8, 6))
    x = range(len(labels))
    bars = ax.bar(x, ppls, color=colors, width=0.5, edgecolor="black", linewidth=0.8)

    # Annotate bars with PPL values
    for bar, ppl in zip(bars, ppls):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 2,
                f"{ppl:.1f}", ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Final val PPL (WikiText-2, step 9000)", fontsize=12)
    ax.set_title(
        "Probe 2: Does Lorentz-score attention help?\n"
        "Euclidean vs hyper-fixed vs hyper-scores-only (all at 7.2M params)",
        fontsize=12,
    )

    # Lower is better annotation
    ax.text(0.98, 0.97, "Lower PPL = better", transform=ax.transAxes,
            ha="right", va="top", fontsize=10, color="gray", style="italic")

    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(ppls) * 1.15)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Plot saved → {args.out}")
    plt.close()


if __name__ == "__main__":
    main()

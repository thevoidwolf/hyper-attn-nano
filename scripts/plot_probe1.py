"""
plot_probe1.py — Training curve comparison: Probe 1 (curvature warmup) vs B k10
=================================================================================

Produces a plot showing val PPL over steps for:
  - Experiment B k10 (no curriculum, fixed K=-10 from step 0)
  - Probe 1 (linear warmup K: -1 → -10 over 500 steps)

The interesting region is steps 0–2000 where the warmup effect should show.

Usage:
    python scripts/plot_probe1.py
    python scripts/plot_probe1.py --out results/plots/probe1/probe1_curve.png
"""

import argparse
import json
import os
import sys


def load_jsonl(path: str) -> list[dict]:
    """Load a JSONL training log, returning only eval records (have val_ppl)."""
    records = []
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
                records.append(rec)
    return records


def main():
    parser = argparse.ArgumentParser(description="Plot Probe 1 training curves")
    parser.add_argument(
        "--baseline",
        default="results/logs/exp_b/exp_b_k10_train.jsonl",
        help="JSONL log for Experiment B k10 (no curriculum)",
    )
    parser.add_argument(
        "--probe1",
        default="results/logs/probes/probe1_curvature_warmup_train.jsonl",
        help="JSONL log for Probe 1 (curvature warmup)",
    )
    parser.add_argument(
        "--out",
        default="results/plots/probe1/probe1_curve.png",
        help="Output path for the plot",
    )
    parser.add_argument(
        "--max-step", type=int, default=2000,
        help="X-axis limit (default 2000 to highlight warmup region)",
    )
    args = parser.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("ERROR: matplotlib not available. Install with: pip install matplotlib")
        sys.exit(1)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    # --- Baseline (B k10, no curriculum) ---
    if os.path.exists(args.baseline):
        records = load_jsonl(args.baseline)
        steps   = [r["step"] for r in records if r["step"] <= args.max_step]
        ppls    = [r["val_ppl"] for r in records if r["step"] <= args.max_step]
        ax.plot(steps, ppls, color="steelblue", linewidth=2,
                label="Exp B k10 (no warmup, fixed K=-10)", zorder=3)
        if records:
            final = next((r["val_ppl"] for r in reversed(records)), None)
            print(f"B k10 final val PPL: {final:.1f}")
    else:
        print(f"[WARN] Baseline log not found: {args.baseline}")

    # --- Probe 1 (curvature warmup) ---
    if os.path.exists(args.probe1):
        records = load_jsonl(args.probe1)
        steps   = [r["step"] for r in records if r["step"] <= args.max_step]
        ppls    = [r["val_ppl"] for r in records if r["step"] <= args.max_step]
        ax.plot(steps, ppls, color="darkorange", linewidth=2,
                linestyle="--", label="Probe 1 (linear warmup K: -1→-10, 500 steps)", zorder=4)
        if records:
            final = next((r["val_ppl"] for r in reversed(records)), None)
            print(f"Probe 1 final val PPL: {final:.1f}")
    else:
        print(f"[WARN] Probe 1 log not found: {args.probe1} — run probe 1 first.")

    # Warmup region annotation
    ax.axvline(x=500, color="gray", linestyle=":", linewidth=1.5, alpha=0.7)
    ax.text(510, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 1 else 1000,
            "warmup\ncomplete\n(step 500)", fontsize=9, color="gray", va="top")

    ax.set_xlabel("Training step", fontsize=13)
    ax.set_ylabel("Val PPL", fontsize=13)
    ax.set_title("Probe 1: Curvature Warmup vs No Curriculum\n"
                 "Does starting at K=-1 then annealing to K=-10 reduce the early-training penalty?",
                 fontsize=12)
    ax.legend(fontsize=11)
    ax.set_xlim(0, args.max_step)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Plot saved → {args.out}")
    plt.close()


if __name__ == "__main__":
    main()

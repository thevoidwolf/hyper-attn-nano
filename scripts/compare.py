"""
compare.py — Cross-version, cross-variant PPL comparison
=========================================================

Usage:
    python scripts/compare.py                  # compares all available versions
    python scripts/compare.py --versions v2 v3
    python scripts/compare.py --out results/plots/

Produces two plots:
  1. ppl_curves.png   — val PPL over training steps, all variants × versions
  2. ppl_summary.png  — bar chart of final val PPL per variant × version
"""

import argparse
import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np


VARIANTS = ["euclid", "hyper-fixed", "hyper-perhead"]

VARIANT_STYLE = {
    "euclid":        {"color": "#4C72B0", "linestyle": "-"},
    "hyper-fixed":   {"color": "#DD8452", "linestyle": "--"},
    "hyper-perhead": {"color": "#55A868", "linestyle": "-."},
}

VERSION_ALPHA = {"v1": 0.4, "v2": 0.65, "v3": 1.0}


def load_val_ppl(log_path: str):
    with open(log_path) as f:
        log = json.load(f)
    entries = log["val_loss"]
    steps = [e["step"] for e in entries]
    ppls = [
        e["perplexity"] if "perplexity" in e else math.exp(e["loss"])
        for e in entries
    ]
    return steps, ppls


def final_ppl(log_path: str) -> float:
    _, ppls = load_val_ppl(log_path)
    return ppls[-1]


def available_versions(logs_root: str):
    return sorted(
        d for d in os.listdir(logs_root)
        if os.path.isdir(os.path.join(logs_root, d)) and d.startswith("v")
    )


def plot_curves(data: dict, out_path: str):
    """data: {(version, variant): (steps, ppls)}"""
    fig, ax = plt.subplots(figsize=(11, 6))

    for (version, variant), (steps, ppls) in sorted(data.items()):
        style = VARIANT_STYLE[variant]
        alpha = VERSION_ALPHA.get(version, 0.8)
        ax.plot(steps, ppls,
                color=style["color"],
                linestyle=style["linestyle"],
                alpha=alpha,
                linewidth=1.8,
                label=f"{version} / {variant}")

    ax.set_xlabel("Training step")
    ax.set_ylabel("Val perplexity")
    ax.set_title("Validation PPL across versions and variants")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_summary(summary: dict, versions: list, out_path: str):
    """summary: {(version, variant): final_ppl}"""
    x = np.arange(len(versions))
    width = 0.25
    offsets = np.linspace(-(width), width, len(VARIANTS))

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, variant in enumerate(VARIANTS):
        ppls = [summary.get((v, variant), float("nan")) for v in versions]
        bars = ax.bar(x + offsets[i], ppls, width,
                      label=variant,
                      color=VARIANT_STYLE[variant]["color"],
                      alpha=0.85)
        for bar, ppl in zip(bars, ppls):
            if not math.isnan(ppl):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 2,
                        f"{ppl:.0f}",
                        ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(versions)
    ax.set_xlabel("Version")
    ax.set_ylabel("Final val perplexity")
    ax.set_title("Final val PPL by version and variant")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--versions", nargs="+", default=None,
                        help="Which version folders to include (default: all found)")
    parser.add_argument("--out", default="results/plots",
                        help="Output directory for plots")
    args = parser.parse_args()

    logs_root = os.path.join("results", "logs")
    versions = args.versions or available_versions(logs_root)

    os.makedirs(args.out, exist_ok=True)

    curves = {}
    summary = {}

    for version in versions:
        for variant in VARIANTS:
            path = os.path.join(logs_root, version, f"{variant}_log.json")
            if not os.path.exists(path):
                print(f"  Skipping missing: {path}")
                continue
            steps, ppls = load_val_ppl(path)
            curves[(version, variant)] = (steps, ppls)
            summary[(version, variant)] = ppls[-1]

    if not curves:
        print("No log files found — check your --versions and logs directory.")
        return

    plot_curves(curves, os.path.join(args.out, "ppl_curves.png"))
    plot_summary(summary, versions, os.path.join(args.out, "ppl_summary.png"))

    print("\nFinal PPL summary:")
    print(f"  {'version':<6}  {'variant':<16}  {'PPL':>8}")
    print("  " + "-" * 34)
    for version in versions:
        for variant in VARIANTS:
            ppl = summary.get((version, variant))
            if ppl is not None:
                print(f"  {version:<6}  {variant:<16}  {ppl:>8.2f}")


if __name__ == "__main__":
    main()

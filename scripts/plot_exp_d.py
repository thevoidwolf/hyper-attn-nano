"""
plot_exp_d.py — Plots for Experiment D: Hierarchical Data Probe
================================================================

Produces four plots in results/plots/exp_d/:

  1. gap_comparison.png       — Grouped bar chart: euclid vs hyper PPL gap
                                on WikiText-2 vs CodeParrot (headline plot)
  2. training_curves.png      — Val PPL (log scale) vs step, all 4 code variants
  3. overfit_diagnostic.png   — Train loss vs val PPL for f32/f64 variants +
                                B2 WikiText-2 f64 reference
  4. curvature_heatmap_code.png — Final per-head K heatmap for exp_d_perhead_k10
                                  with V3 WikiText-2 values annotated alongside

Usage:
    python scripts/plot_exp_d.py
    python scripts/plot_exp_d.py --out-dir results/plots/exp_d
"""

import argparse
import json
import math
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


# ---------------------------------------------------------------------------
# Hard-coded WikiText-2 reference values (Experiment B results)
# ---------------------------------------------------------------------------
WT2_EUCLID    = 277.0   # V3 Euclidean on WikiText-2
WT2_HYPER_K10 = 283.7   # Experiment B K=-10 float32 on WikiText-2

LOG_DIR   = "results/logs/exp_d"
B2_LOG    = "results/logs/exp_b2/exp_b2_k10_train.jsonl"
V3_LOG    = "results/logs/v3/hyper-perhead_log.json"

VARIANT_STYLES = {
    "exp_d_euclid":      {"label": "Euclid (code)",          "color": "#4C72B0", "ls": "-"},
    "exp_d_k10_f32":     {"label": "Hyper K=−10 f32 (code)", "color": "#DD8452", "ls": "-"},
    "exp_d_k10_f64":     {"label": "Hyper K=−10 f64 (code)", "color": "#55A868", "ls": "--"},
    "exp_d_perhead_k10": {"label": "Perhead init K=−10",     "color": "#C44E52", "ls": "-."},
}

LAYER_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_jsonl(run_id: str) -> list[dict]:
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


def _get_final_ppl(run_id: str) -> float | None:
    records = _load_jsonl(run_id)
    for rec in reversed(records):
        ppl = rec.get("val_ppl")
        if ppl is not None and math.isfinite(ppl):
            return ppl
    return None


def _load_jsonl_external(path: str) -> list[dict]:
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


def _load_v3_curvatures(log_path: str) -> dict[str, float] | None:
    """Load final curvature values from a v3 JSON log (hyper-perhead)."""
    if not os.path.exists(log_path):
        return None
    with open(log_path) as f:
        log = json.load(f)
    entries = log.get("curvatures", [])
    if not entries:
        return None
    last = entries[-1]
    return last.get("values", {})


# ---------------------------------------------------------------------------
# Plot 1: Cross-dataset gap comparison (headline)
# ---------------------------------------------------------------------------

def plot_gap_comparison(out_dir: str) -> None:
    code_euclid = _get_final_ppl("exp_d_euclid")
    code_hyper  = _get_final_ppl("exp_d_k10_f32")

    fig, ax = plt.subplots(figsize=(8, 5))

    datasets = ["WikiText-2", "CodeParrot"]
    euclid_ppls = [WT2_EUCLID, code_euclid]
    hyper_ppls  = [WT2_HYPER_K10, code_hyper]

    x      = np.arange(len(datasets))
    width  = 0.35
    color_e = "#4C72B0"
    color_h = "#DD8452"

    bars_e = ax.bar(x - width / 2, euclid_ppls, width, label="Euclidean",
                    color=color_e, alpha=0.85, edgecolor="white")
    bars_h = ax.bar(x + width / 2, hyper_ppls, width, label="Hyper K=−10 f32",
                    color=color_h, alpha=0.85, edgecolor="white")

    # Annotate gap values above the hyperbolic bar
    for i, (ep, hp) in enumerate(zip(euclid_ppls, hyper_ppls)):
        if ep is None or hp is None:
            continue
        gap = hp - ep
        sign = "+" if gap >= 0 else ""
        ax.annotate(
            f"gap {sign}{gap:.1f}",
            xy=(x[i] + width / 2, hp),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center", fontsize=9, color="dimgray",
        )

    # Annotate bar heights
    for bar in bars_e:
        h = bar.get_height()
        if h is not None:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=8.5)
    for bar in bars_h:
        h = bar.get_height()
        if h is not None:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=11)
    ax.set_ylabel("Validation Perplexity (PPL)", fontsize=10)
    ax.set_title("Euclid vs Hyperbolic PPL Gap: WikiText-2 vs CodeParrot", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)

    note = ("WikiText-2 values from Experiment B (hard-coded).\n"
            "CodeParrot values from Experiment D training.")
    ax.text(0.01, 0.01, note, transform=ax.transAxes,
            fontsize=7, color="gray", va="bottom")

    fig.tight_layout()
    path = os.path.join(out_dir, "gap_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Plot 2: Training curves — all four code variants
# ---------------------------------------------------------------------------

def plot_training_curves(out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    any_data = False
    for run_id, style in VARIANT_STYLES.items():
        records = _load_jsonl(run_id)
        if not records:
            continue
        steps = [r["step"] for r in records]
        ppls  = [r.get("val_ppl") for r in records]
        valid = [(s, p) for s, p in zip(steps, ppls)
                 if p is not None and math.isfinite(p)]
        if not valid:
            continue
        steps_v, ppls_v = zip(*valid)
        ax.plot(steps_v, ppls_v,
                label=style["label"],
                color=style["color"],
                linestyle=style["ls"],
                linewidth=1.8)
        any_data = True

    if not any_data:
        ax.text(0.5, 0.5, "No training data found.\nRun training first.",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
    else:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

    ax.set_xlabel("Training step", fontsize=10)
    ax.set_ylabel("Validation PPL (log scale)", fontsize=10)
    ax.set_title("Experiment D — Training Curves (CodeParrot)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    path = os.path.join(out_dir, "training_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Plot 3: Overfitting diagnostic — train loss vs val PPL
# ---------------------------------------------------------------------------

def plot_overfit_diagnostic(out_dir: str) -> None:
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    colors = {
        "exp_d_k10_f32": ("#DD8452", "-",  "K=−10 f32 (code)"),
        "exp_d_k10_f64": ("#55A868", "--", "K=−10 f64 (code)"),
    }

    for run_id, (color, ls, label) in colors.items():
        records = _load_jsonl(run_id)
        if not records:
            continue
        steps      = [r["step"] for r in records]
        train_loss = [r.get("train_loss") for r in records]
        val_ppl    = [r.get("val_ppl") for r in records]

        tl_valid = [(s, l) for s, l in zip(steps, train_loss) if l is not None]
        vp_valid = [(s, p) for s, p in zip(steps, val_ppl)
                    if p is not None and math.isfinite(p)]

        if tl_valid:
            s, l = zip(*tl_valid)
            ax1.plot(s, l, color=color, linestyle=ls, linewidth=1.5, alpha=0.7,
                     label=f"train loss — {label}")
        if vp_valid:
            s, p = zip(*vp_valid)
            ax2.plot(s, p, color=color, linestyle=ls, linewidth=1.8,
                     label=f"val PPL — {label}")

    # B2 WikiText-2 f64 reference
    b2_records = _load_jsonl_external(B2_LOG)
    if b2_records:
        b2_steps = [r["step"] for r in b2_records]
        b2_train = [r.get("train_loss") for r in b2_records]
        b2_ppl   = [r.get("val_ppl") for r in b2_records]
        b2_tl = [(s, l) for s, l in zip(b2_steps, b2_train) if l is not None]
        b2_vp = [(s, p) for s, p in zip(b2_steps, b2_ppl)
                 if p is not None and math.isfinite(p)]
        color_b2 = "#9467BD"
        if b2_tl:
            s, l = zip(*b2_tl)
            ax1.plot(s, l, color=color_b2, linestyle=":", linewidth=1.5, alpha=0.7,
                     label="train loss — K=−10 f64 WikiText-2 (B2)")
        if b2_vp:
            s, p = zip(*b2_vp)
            ax2.plot(s, p, color=color_b2, linestyle=":", linewidth=1.8,
                     label="val PPL — K=−10 f64 WikiText-2 (B2)")

    ax1.set_xlabel("Training step", fontsize=10)
    ax1.set_ylabel("Train loss", fontsize=10, color="#333333")
    ax2.set_ylabel("Validation PPL", fontsize=10, color="#333333")
    ax1.set_title("Overfitting Diagnostic: f32 vs f64 — Code vs WikiText-2", fontsize=11)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")
    ax1.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, "overfit_diagnostic.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Plot 4: Per-head curvature heatmap — code variant vs V3 WikiText-2
# ---------------------------------------------------------------------------

def _parse_head(name: str) -> tuple[int, int]:
    parts = name.split("_")
    return int(parts[1]), int(parts[3])


def plot_curvature_heatmap(out_dir: str) -> None:
    # Load code-trained perhead log
    code_log_path = os.path.join(LOG_DIR, "exp_d_perhead_k10_log.json")
    code_curvs: dict[str, float] | None = None
    if os.path.exists(code_log_path):
        with open(code_log_path) as f:
            log = json.load(f)
        entries = log.get("curvatures", [])
        if entries:
            code_curvs = entries[-1].get("values", {})

    v3_curvs = _load_v3_curvatures(V3_LOG)

    if code_curvs is None and v3_curvs is None:
        print("[WARN] No curvature data found for heatmap. Skipping plot 4.")
        return

    # Determine grid dimensions from whichever data is available
    sample = code_curvs or v3_curvs
    n_layers = max(_parse_head(h)[0] for h in sample) + 1
    n_heads  = max(_parse_head(h)[1] for h in sample) + 1

    def _to_grid(curvs: dict | None) -> np.ndarray | None:
        if curvs is None:
            return None
        g = np.full((n_layers, n_heads), np.nan)
        for name, val in curvs.items():
            l, h = _parse_head(name)
            g[l, h] = val
        return g

    code_grid = _to_grid(code_curvs)
    v3_grid   = _to_grid(v3_curvs)

    # ── Figure: one or two subplots ─────────────────────────────────────
    n_plots  = (code_grid is not None) + (v3_grid is not None)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    plot_data = []
    if code_grid is not None:
        plot_data.append((code_grid, "CodeParrot (Exp D perhead K=−10 init)"))
    if v3_grid is not None:
        plot_data.append((v3_grid,   "WikiText-2 (V3 perhead K=−1 init)"))

    # Shared colour scale across both panels for direct visual comparison
    all_vals = [v for g, _ in plot_data for v in g.flatten() if not np.isnan(v)]
    vmin = min(all_vals) if all_vals else -12
    vmax = max(all_vals) if all_vals else -1

    for ax, (grid, title) in zip(axes, plot_data):
        im = ax.imshow(grid, aspect="auto", cmap="RdYlBu_r", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(n_heads))
        ax.set_xticklabels([f"H{h}" for h in range(n_heads)], fontsize=9)
        ax.set_yticks(range(n_layers))
        ax.set_yticklabels([f"L{l}" for l in range(n_layers)], fontsize=9)
        ax.set_xlabel("Head", fontsize=9)
        ax.set_ylabel("Layer", fontsize=9)
        ax.set_title(title, fontsize=10)

        # Annotate cells with K values
        for l in range(n_layers):
            for h in range(n_heads):
                val = grid[l, h]
                if not np.isnan(val):
                    ax.text(h, l, f"{val:.2f}", ha="center", va="center",
                            fontsize=8, color="black")

        fig.colorbar(im, ax=ax, label="Curvature K")

    fig.suptitle("Per-Head Curvature: Code vs WikiText-2", fontsize=11, y=1.01)
    fig.tight_layout()
    path = os.path.join(out_dir, "curvature_heatmap_code.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment D plots")
    parser.add_argument(
        "--out-dir", default="results/plots/exp_d",
        help="Output directory for plots",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Generating Experiment D plots ...")
    print()
    plot_gap_comparison(args.out_dir)
    plot_training_curves(args.out_dir)
    plot_overfit_diagnostic(args.out_dir)
    plot_curvature_heatmap(args.out_dir)
    print()
    print(f"All plots saved to {args.out_dir}/")


if __name__ == "__main__":
    main()

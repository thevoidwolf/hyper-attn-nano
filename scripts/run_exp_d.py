"""
run_exp_d.py — Orchestration script for Experiment D: Hierarchical Data Probe
==============================================================================

Runs all four Experiment D variants sequentially on CodeParrot data.
All runs complete regardless of intermediate results (no early stopping).
After training, automatically calls plot_exp_d.py.

Usage:
    cd ~/hyper-attn-nano
    source activate.sh
    python scripts/run_exp_d.py
    python scripts/run_exp_d.py --skip-f64    # omit float64 variant
    python scripts/run_exp_d.py --smoke       # 50-step smoke test per variant
"""

import argparse
import json
import math
import os
import subprocess
import sys

RUNS = [
    # (run_id,               config_path,                              label)
    ("exp_d_euclid",        "configs/nano_euclid_code.yaml",          "Euclid (control)"),
    ("exp_d_k10_f32",       "configs/nano_hyper_fixed_k10_code.yaml", "Hyper K=-10 float32"),
    ("exp_d_k10_f64",       "configs/nano_hyper_fixed_k10_f64_code.yaml", "Hyper K=-10 float64"),
    ("exp_d_perhead_k10",   "configs/nano_hyper_perhead_k10_code.yaml", "Hyper perhead init K=-10"),
]

LOG_DIR      = "results/logs/exp_d"
SUMMARY_PATH = "results/exp_d_summary.md"
PLOT_SCRIPT  = "scripts/plot_exp_d.py"


# ---------------------------------------------------------------------------
# Log helpers (mirrors sweep_curvature.py)
# ---------------------------------------------------------------------------

def _get_final_ppl(run_id: str) -> float | None:
    log_path = os.path.join(LOG_DIR, f"{run_id}_train.jsonl")
    if not os.path.exists(log_path):
        return None
    records = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    for rec in reversed(records):
        ppl = rec.get("val_ppl")
        if ppl is not None and math.isfinite(ppl):
            return ppl
    return None


def _detect_status(run_id: str) -> tuple[str, int | None]:
    """Returns (status, divergence_step). status ∈ STABLE | DEGRADED | DIVERGED | NO_LOG."""
    log_path = os.path.join(LOG_DIR, f"{run_id}_train.jsonl")
    if not os.path.exists(log_path):
        return "NO_LOG", None

    records = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    if not records:
        return "NO_LOG", None

    div_step = None
    for rec in records:
        ppl = rec.get("val_ppl", float("inf"))
        if not math.isfinite(ppl) or ppl > 1e6:
            if div_step is None:
                div_step = rec["step"]

    final_step = records[-1]["step"]
    if div_step is not None and div_step >= final_step:
        return "DIVERGED", div_step
    elif div_step is not None:
        return "DEGRADED", div_step
    return "STABLE", None


# ---------------------------------------------------------------------------
# Summary writer
# ---------------------------------------------------------------------------

def _write_summary(results: list[dict]) -> None:
    os.makedirs(os.path.dirname(SUMMARY_PATH) or ".", exist_ok=True)
    lines = [
        "# Experiment D — Hierarchical Data Probe Results",
        "",
        "Dataset: CodeParrot (Python source code, ~2.4M train tokens)",
        "Tokeniser: GPT-2 (same as WikiText-2 experiments — vocabulary held constant)",
        "",
        "## Final PPL Table",
        "",
        "| Run ID | Variant | Final PPL | Status |",
        "|--------|---------|-----------|--------|",
    ]
    for r in results:
        ppl_str = f"{r['final_ppl']:.1f}" if r["final_ppl"] is not None else "N/A"
        lines.append(
            f"| {r['run_id']} | {r['label']} | {ppl_str} | {r['status']} |"
        )

    lines += [
        "",
        "## Cross-Dataset Gap Reference",
        "",
        "| Dataset | Euclid PPL | Hyper K=-10 f32 PPL | Gap (hyper − euclid) |",
        "|---------|-----------|---------------------|----------------------|",
    ]

    # WikiText-2 hard-coded from Experiment B results
    wt2_euclid = 277.0
    wt2_hyper  = 283.7
    wt2_gap    = wt2_hyper - wt2_euclid
    lines.append(f"| WikiText-2 | {wt2_euclid:.1f} | {wt2_hyper:.1f} | {wt2_gap:+.1f} |")

    euclid_ppl = next((r["final_ppl"] for r in results if r["run_id"] == "exp_d_euclid"), None)
    hyper_ppl  = next((r["final_ppl"] for r in results if r["run_id"] == "exp_d_k10_f32"), None)
    if euclid_ppl is not None and hyper_ppl is not None:
        code_gap = hyper_ppl - euclid_ppl
        lines.append(f"| CodeParrot | {euclid_ppl:.1f} | {hyper_ppl:.1f} | {code_gap:+.1f} |")
    else:
        lines.append("| CodeParrot | N/A | N/A | N/A |")

    lines += [
        "",
        "## Analysis",
        "",
        "**1. Gap direction** — TBD after training completes.",
        "**2. Overfitting resolution** — TBD (compare exp_d_k10_f32 vs exp_d_k10_f64).",
        "**3. Learnable curvature from K=-10 init** — TBD (compare exp_d_perhead_k10 vs exp_d_k10_f32).",
        "**4. Absolute PPL context** — Code PPL will exceed WikiText-2 PPL due to mismatched tokeniser.",
        "  The relative gap (euclid vs hyperbolic on the same dataset) is the meaningful comparison.",
        "",
        "_Generated by run_exp_d.py_",
    ]

    with open(SUMMARY_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Summary written → {SUMMARY_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment D orchestration")
    parser.add_argument(
        "--skip-f64", action="store_true",
        help="Omit the float64 variant (exp_d_k10_f64)",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Smoke test: train each variant for only 50 steps",
    )
    args = parser.parse_args()

    runs = RUNS if not args.skip_f64 else [r for r in RUNS if r[0] != "exp_d_k10_f64"]

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs("results/plots/exp_d", exist_ok=True)

    results = []

    print("=" * 70)
    print("Experiment D — Hierarchical Data Probe")
    print("Dataset: CodeParrot (Python source code)")
    print("=" * 70)

    for run_id, config_path, label in runs:
        print(f"\n{'=' * 70}")
        print(f"[RUN] {run_id}  ({label})")
        print(f"      config = {config_path}")
        print("=" * 70)

        if not os.path.exists(config_path):
            print(f"[ERROR] Config not found: {config_path}")
            results.append({
                "run_id": run_id, "label": label,
                "final_ppl": None, "status": "CONFIG_MISSING",
            })
            continue

        cmd = [sys.executable, "scripts/train.py", "--config", config_path]

        if args.smoke:
            # Smoke mode: patch max_steps inline via a temporary config
            import yaml
            import tempfile
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            cfg["training"]["max_steps"]     = 50
            cfg["training"]["eval_interval"] = 25
            cfg["training"]["warmup_steps"]  = 5
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as tmp:
                yaml.dump(cfg, tmp)
                tmp_path = tmp.name
            cmd = [sys.executable, "scripts/train.py", "--config", tmp_path]

        proc = subprocess.run(cmd, cwd=os.getcwd())

        if args.smoke and "tmp_path" in locals():
            os.unlink(tmp_path)

        status, div_step = _detect_status(run_id)
        final_ppl = _get_final_ppl(run_id)

        print(f"\n[RESULT] {run_id}: status={status}, final_ppl={final_ppl}, "
              f"divergence_step={div_step}")

        results.append({
            "run_id":   run_id,
            "label":    label,
            "final_ppl": final_ppl,
            "status":   status,
        })

    _write_summary(results)

    if os.path.exists(PLOT_SCRIPT):
        print(f"\n[PLOT] Running {PLOT_SCRIPT} ...")
        subprocess.run([sys.executable, PLOT_SCRIPT])
    else:
        print(f"[WARN] {PLOT_SCRIPT} not found — skipping plots.")

    print("\n[DONE] Experiment D complete.")
    print()
    print(f"{'Run ID':<25} {'Label':<35} {'PPL':>8}  Status")
    print("-" * 80)
    for r in results:
        ppl_str = f"{r['final_ppl']:.1f}" if r["final_ppl"] else "N/A"
        print(f"{r['run_id']:<25} {r['label']:<35} {ppl_str:>8}  {r['status']}")


if __name__ == "__main__":
    main()

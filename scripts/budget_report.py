"""
budget_report.py — Compute budget accounting across all experiments
===================================================================

Scans all JSONL training logs under results/logs/ and prints a summary
table showing which experiments have consumed what compute, plus totals.

Usage:
    python scripts/budget_report.py
    python scripts/budget_report.py --log-dir results/logs/exp_b --budget-hours 12
"""

import argparse
import json
import os
import sys


def scan_jsonl_logs(log_dir: str) -> list[dict]:
    """
    Walk log_dir recursively and collect run_meta entries from all .jsonl files.

    Returns list of dicts with:
        path, run_id, wall_time_seconds, peak_vram_gb,
        throughput_tok_per_sec, compute_budget_used_hours,
        aborted (bool)
    """
    results = []

    for root, _, files in os.walk(log_dir):
        for fname in sorted(files):
            if not fname.endswith(".jsonl"):
                continue

            fpath = os.path.join(root, fname)
            run_id = fname.replace("_train.jsonl", "")

            run_meta      = None
            aborted       = False
            last_step     = None
            last_val_ppl  = None

            try:
                with open(fpath) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        if record.get("event") == "run_complete":
                            run_meta = record
                        elif record.get("event") == "ABORTED":
                            aborted = True
                        else:
                            step = record.get("step")
                            if step is not None:
                                last_step = step
                                last_val_ppl = record.get("val_ppl")
            except OSError:
                continue

            entry = {
                "path":       fpath,
                "run_id":     run_id,
                "aborted":    aborted,
                "last_step":  last_step,
                "last_val_ppl": last_val_ppl,
            }
            if run_meta:
                entry.update({
                    "wall_time_seconds":       run_meta.get("wall_time_seconds"),
                    "peak_vram_gb":            run_meta.get("peak_vram_gb"),
                    "throughput_tok_per_sec":  run_meta.get("throughput_tok_per_sec"),
                    "compute_budget_used_hours": run_meta.get("compute_budget_used_hours"),
                })
            else:
                entry.update({
                    "wall_time_seconds":       None,
                    "peak_vram_gb":            None,
                    "throughput_tok_per_sec":  None,
                    "compute_budget_used_hours": None,
                })

            results.append(entry)

    return results


def _fmt(val, fmt=".2f", default="—"):
    if val is None:
        return default
    return format(val, fmt)


def print_report(entries: list[dict], total_budget_hours: float | None = None):
    """Print formatted compute budget table."""
    if not entries:
        print("No JSONL logs found.")
        return

    # Table header
    header = (
        f"{'Run ID':<35} {'Steps':>6} {'Val PPL':>8} "
        f"{'Hours':>7} {'VRAM GB':>8} {'Tok/s':>8} {'Status':<10}"
    )
    sep    = "-" * len(header)

    print(sep)
    print(header)
    print(sep)

    total_hours = 0.0
    for e in sorted(entries, key=lambda x: x["run_id"]):
        h = e.get("compute_budget_used_hours") or 0.0
        total_hours += h

        status = "ABORTED" if e["aborted"] else ("done" if e.get("wall_time_seconds") else "partial")

        print(
            f"{e['run_id']:<35} "
            f"{str(e['last_step'] or '—'):>6} "
            f"{_fmt(e['last_val_ppl'], '.1f'):>8} "
            f"{_fmt(e.get('compute_budget_used_hours'), '.3f'):>7} "
            f"{_fmt(e.get('peak_vram_gb'), '.2f'):>8} "
            f"{_fmt(e.get('throughput_tok_per_sec'), '.0f'):>8} "
            f"{status:<10}"
        )

    print(sep)
    print(f"{'TOTAL COMPUTE':<35} {'':>6} {'':>8} {_fmt(total_hours, '.3f'):>7} hours")

    if total_budget_hours is not None:
        remaining = total_budget_hours - total_hours
        print(f"{'REMAINING':< 35} {'':>6} {'':>8} {_fmt(remaining, '.3f'):>7} hours  "
              f"(budget: {total_budget_hours:.1f}h)")

    print(sep)


def main():
    parser = argparse.ArgumentParser(
        description="Print compute budget report for all training runs"
    )
    parser.add_argument(
        "--log-dir", default="results/logs",
        help="Root directory to scan for .jsonl training logs"
    )
    parser.add_argument(
        "--budget-hours", type=float, default=None,
        help="Total experiment budget in hours (for remaining computation)"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.log_dir):
        print(f"ERROR: log directory not found: {args.log_dir}", file=sys.stderr)
        sys.exit(1)

    entries = scan_jsonl_logs(args.log_dir)
    print(f"\nCompute Budget Report — scanned {len(entries)} run(s) in {args.log_dir}\n")
    print_report(entries, total_budget_hours=args.budget_hours)


if __name__ == "__main__":
    main()

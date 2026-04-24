#!/usr/bin/env python3
"""
run_experiment_a.py — Orchestrator for Experiment A (spec §8.2, §17.3)

Reads a manifest YAML listing configs in execution order and runs them
sequentially, one subprocess per run. Each run is a fresh process so
MANIFOLD_FLOAT64 and CUDA state cannot bleed between runs.

Usage:
    python scripts/run_experiment_a.py --manifest scripts/manifest_experiment_a.yaml
    python scripts/run_experiment_a.py --manifest scripts/manifest_experiment_a.yaml --dry-run

Manifest format (YAML list):
    - configs/experiment_a/a_s2_euclid_seed42.yaml
    - configs/experiment_a/a_s2_hyper_seed42.yaml
    ...

Skip logic:
    - A run is SKIPPED if its log_dir/summary.json exists with status: completed.
    - A run is RESUMED (subprocess handles it) if its checkpoint_dir has a checkpoint
      but no summary.json — train.py's resume logic activates automatically.
    - A run is STARTED fresh otherwise.

Error handling:
    - If a run's subprocess exits non-zero, it is logged as failed in the manifest
      run log and the orchestrator continues to the next run.
    - The orchestrator never raises on a single run's failure.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

import yaml


TRAIN_SCRIPT = os.path.join(os.path.dirname(__file__), "train.py")
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")


def load_manifest(path: str) -> list[str]:
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, list):
        raise ValueError(f"Manifest must be a YAML list of config paths, got {type(data)}")
    return [str(p) for p in data]


def is_completed(config_path: str) -> bool:
    """Return True if the run already has a summary.json with status: completed."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    log_dir = cfg.get("log_dir")
    if not log_dir:
        return False
    summary_path = os.path.join(log_dir, "summary.json")
    if not os.path.exists(summary_path):
        return False
    try:
        with open(summary_path) as sf:
            summary = json.load(sf)
        return summary.get("status") == "completed"
    except (json.JSONDecodeError, KeyError):
        return False


def has_checkpoint(config_path: str) -> bool:
    """Return True if the run has at least one step checkpoint (will auto-resume)."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    checkpoint_dir = cfg.get("checkpoint_dir")
    if not checkpoint_dir:
        return False
    import glob
    pattern = os.path.join(checkpoint_dir, "checkpoint_step_*.pt")
    return bool(glob.glob(pattern))


def run_one(config_path: str, log_file_path: str, dry_run: bool = False) -> bool:
    """
    Launch train.py for this config in a fresh subprocess.
    Returns True on success (exit code 0), False otherwise.
    """
    cmd = [sys.executable, TRAIN_SCRIPT, "--config", config_path]
    print(f"\n{'[DRY-RUN] ' if dry_run else ''}Launching: {' '.join(cmd)}")

    if dry_run:
        return True

    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    with open(log_file_path, "a") as lf:
        lf.write(f"\n=== RUN START {datetime.now().isoformat()} ===\n")
        lf.write(f"config: {config_path}\n\n")
        lf.flush()

        result = subprocess.run(
            cmd,
            stdout=lf,
            stderr=lf,
            cwd=PROJECT_ROOT,
            check=False,
        )

        lf.write(f"\n=== RUN END exit_code={result.returncode} {datetime.now().isoformat()} ===\n")

    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Experiment A run orchestrator")
    parser.add_argument("--manifest", required=True, help="Path to manifest YAML")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would run without executing")
    args = parser.parse_args()

    configs = load_manifest(args.manifest)
    total   = len(configs)
    print(f"Experiment A orchestrator — {total} runs in manifest")
    print(f"Manifest: {args.manifest}")
    if args.dry_run:
        print("[DRY-RUN mode — no training will start]")

    # Orchestrator log: one log file per run, in the run's log_dir
    orch_log_path = os.path.join("logs", "experiment_a", "orchestrator.jsonl")
    os.makedirs(os.path.dirname(orch_log_path), exist_ok=True)

    skipped = completed_prior = failed = succeeded = 0

    for i, config_path in enumerate(configs, start=1):
        if not os.path.exists(config_path):
            print(f"[{i}/{total}] MISSING config: {config_path} — skipping")
            skipped += 1
            continue

        # Read run_id for log path
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        run_id  = cfg.get("run_id", f"run_{i}")
        log_dir = cfg.get("log_dir", os.path.join("logs", "experiment_a", run_id))
        run_log = os.path.join(log_dir, "run_stdout.log")

        # Skip completed
        if is_completed(config_path):
            print(f"[{i}/{total}] SKIP (already completed): {run_id}")
            completed_prior += 1
            continue

        # Resume or fresh
        resuming = has_checkpoint(config_path)
        action   = "RESUME" if resuming else "START"
        print(f"[{i}/{total}] {action}: {run_id}")

        t0      = time.time()
        success = run_one(config_path, run_log, dry_run=args.dry_run)
        elapsed = time.time() - t0

        status = "succeeded" if success else "failed"
        if not args.dry_run:
            if success:
                succeeded += 1
            else:
                failed += 1
                print(f"  [WARN] Run failed (exit non-zero). Log: {run_log}. Continuing.")

        # Append to orchestrator JSONL
        record = {
            "run_id":      run_id,
            "config":      config_path,
            "action":      action,
            "status":      status,
            "elapsed_sec": round(elapsed, 1),
            "timestamp":   datetime.now().isoformat(),
        }
        with open(orch_log_path, "a") as olf:
            olf.write(json.dumps(record) + "\n")

        print(f"  {status.upper()} in {elapsed/3600:.2f}h")

    print(f"\n{'='*60}")
    print(f"Experiment A complete.")
    print(f"  Prior completions skipped: {completed_prior}")
    print(f"  Succeeded this session:    {succeeded}")
    print(f"  Failed this session:       {failed}")
    print(f"  Config not found:          {skipped}")
    print(f"Orchestrator log: {orch_log_path}")


if __name__ == "__main__":
    main()

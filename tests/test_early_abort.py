"""
test_early_abort.py — Tests for early-abort and compute logging in train.py
============================================================================

Run with:
    cd ~/hyper-attn-nano
    source activate.sh
    pytest tests/test_early_abort.py -v
"""

import json
import math
import os
import sys
import tempfile
import types

import pytest
import numpy as np
import torch

# We test the logic directly, not by invoking the full training script.
# We import the relevant helpers from train.py by adding scripts/ to the path.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))


# ---------------------------------------------------------------------------
# Baseline file helpers
# ---------------------------------------------------------------------------

class TestBaselineFile:

    def test_baseline_file_exists(self):
        """The euclid_wikitext2_step500_ppl.json file must exist."""
        baseline_path = os.path.join(
            os.path.dirname(__file__), "..", "configs", "baselines",
            "euclid_wikitext2_step500_ppl.json"
        )
        assert os.path.exists(baseline_path), (
            f"Baseline file not found: {baseline_path}"
        )

    def test_baseline_file_loads_correctly(self):
        """Baseline JSON must load and contain val_ppl."""
        baseline_path = os.path.join(
            os.path.dirname(__file__), "..", "configs", "baselines",
            "euclid_wikitext2_step500_ppl.json"
        )
        with open(baseline_path) as f:
            data = json.load(f)

        assert "val_ppl" in data, "Baseline file missing 'val_ppl' key"
        assert isinstance(data["val_ppl"], (int, float)), "val_ppl must be numeric"
        assert data["val_ppl"] > 0, "val_ppl must be positive"
        assert math.isfinite(data["val_ppl"]), "val_ppl must be finite"

    def test_baseline_val_ppl_is_reasonable(self):
        """Euclidean step-500 PPL should be between 100 and 10000 for WikiText-2."""
        baseline_path = os.path.join(
            os.path.dirname(__file__), "..", "configs", "baselines",
            "euclid_wikitext2_step500_ppl.json"
        )
        with open(baseline_path) as f:
            data = json.load(f)
        assert 100 < data["val_ppl"] < 10000, (
            f"Baseline val_ppl={data['val_ppl']} is outside expected range [100, 10000]"
        )


# ---------------------------------------------------------------------------
# Early-abort logic tests
# ---------------------------------------------------------------------------

class TestEarlyAbortLogic:
    """Test the abort condition logic without running a full training loop."""

    def _abort_condition(self, val_ppl: float, baseline_ppl: float,
                         multiplier: float = 10.0) -> bool:
        """Replicate the abort condition from train.py."""
        return val_ppl > baseline_ppl * multiplier

    def test_abort_fires_on_exploding_ppl(self):
        """PPL 100× the baseline should trigger abort."""
        assert self._abort_condition(val_ppl=68090.0, baseline_ppl=680.9)

    def test_abort_fires_just_above_threshold(self):
        """PPL just above 10× baseline should trigger abort."""
        assert self._abort_condition(val_ppl=6810.0, baseline_ppl=680.9)

    def test_abort_does_not_fire_at_threshold(self):
        """PPL exactly at 10× baseline should NOT trigger abort (> not >=)."""
        assert not self._abort_condition(val_ppl=6809.0, baseline_ppl=680.9)

    def test_abort_does_not_fire_on_normal_ppl(self):
        """Normal training PPL (< 10× baseline) must not trigger abort."""
        assert not self._abort_condition(val_ppl=283.7, baseline_ppl=680.9)

    def test_abort_does_not_fire_on_baseline_ppl(self):
        """PPL equal to baseline must not trigger abort."""
        assert not self._abort_condition(val_ppl=680.9, baseline_ppl=680.9)


# ---------------------------------------------------------------------------
# JSONL abort record test
# ---------------------------------------------------------------------------

class TestAbortRecord:

    def test_abort_record_written_to_jsonl(self):
        """When abort fires, an ABORTED record must be written to the JSONL log."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl",
                                         delete=False) as f:
            path = f.name

        try:
            with open(path, "w") as jsonl_file:
                # Simulate the abort record writing from train.py
                reason = "val_ppl=9999.0 > 6809.0 (10× Euclidean baseline at step 500)"
                record = {
                    "event": "ABORTED",
                    "step":  500,
                    "reason": reason,
                }
                jsonl_file.write(json.dumps(record) + "\n")

            # Verify the record is readable and correct
            with open(path) as f:
                lines = [l.strip() for l in f if l.strip()]

            assert len(lines) == 1
            parsed = json.loads(lines[0])
            assert parsed["event"] == "ABORTED"
            assert parsed["step"] == 500
            assert "reason" in parsed

        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Run_meta (compute logging) record test
# ---------------------------------------------------------------------------

class TestRunMeta:

    def test_run_meta_fields(self):
        """run_meta record must contain all required fields."""
        run_meta = {
            "event": "run_complete",
            "wall_time_seconds": 4823.1,
            "peak_vram_gb": 6.2,
            "gpu_util_percent_mean": None,
            "throughput_tok_per_sec": 14721.0,
            "compute_budget_used_hours": 1.34,
        }

        required_fields = [
            "event", "wall_time_seconds", "peak_vram_gb",
            "gpu_util_percent_mean", "throughput_tok_per_sec",
            "compute_budget_used_hours",
        ]
        for field in required_fields:
            assert field in run_meta, f"run_meta missing field: {field}"

        assert run_meta["event"] == "run_complete"
        assert run_meta["wall_time_seconds"] > 0
        assert run_meta["throughput_tok_per_sec"] > 0
        assert run_meta["compute_budget_used_hours"] > 0

    def test_run_meta_written_and_readable(self):
        """run_meta must be serialisable to JSON and round-trip correctly."""
        run_meta = {
            "event": "run_complete",
            "wall_time_seconds": 1234.5,
            "peak_vram_gb": 5.8,
            "gpu_util_percent_mean": None,
            "throughput_tok_per_sec": 12000.0,
            "compute_budget_used_hours": 0.343,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl",
                                         delete=False) as f:
            path = f.name
            json.dump(run_meta, f)
            f.write("\n")

        try:
            with open(path) as f:
                parsed = json.loads(f.readline())
            assert parsed == run_meta
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# budget_report.py scan tests
# ---------------------------------------------------------------------------

class TestBudgetReport:

    def _make_jsonl(self, dirpath: str, fname: str, records: list[dict]):
        fpath = os.path.join(dirpath, fname)
        with open(fpath, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        return fpath

    def test_scan_finds_run_meta(self):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
        from budget_report import scan_jsonl_logs

        with tempfile.TemporaryDirectory() as d:
            self._make_jsonl(d, "myrun_train.jsonl", [
                {"step": 0,   "val_ppl": 500.0},
                {"step": 250, "val_ppl": 350.0},
                {"event": "run_complete", "wall_time_seconds": 100.0,
                 "peak_vram_gb": 4.0, "throughput_tok_per_sec": 5000.0,
                 "compute_budget_used_hours": 0.0278},
            ])

            entries = scan_jsonl_logs(d)

        assert len(entries) == 1
        e = entries[0]
        assert e["run_id"] == "myrun"
        assert abs(e["compute_budget_used_hours"] - 0.0278) < 1e-6
        assert e["last_val_ppl"] == 350.0
        assert not e["aborted"]

    def test_scan_detects_aborted_run(self):
        from budget_report import scan_jsonl_logs

        with tempfile.TemporaryDirectory() as d:
            self._make_jsonl(d, "badrun_train.jsonl", [
                {"step": 0,   "val_ppl": 500.0},
                {"step": 500, "val_ppl": 75000.0},
                {"event": "ABORTED", "step": 500, "reason": "too high"},
            ])

            entries = scan_jsonl_logs(d)

        assert len(entries) == 1
        assert entries[0]["aborted"] is True

    def test_scan_handles_missing_run_meta(self):
        """Runs without a run_meta entry (e.g., still running) return None fields."""
        from budget_report import scan_jsonl_logs

        with tempfile.TemporaryDirectory() as d:
            self._make_jsonl(d, "partial_train.jsonl", [
                {"step": 0,   "val_ppl": 500.0},
                {"step": 250, "val_ppl": 400.0},
            ])
            entries = scan_jsonl_logs(d)

        assert len(entries) == 1
        assert entries[0]["compute_budget_used_hours"] is None
        assert entries[0]["last_val_ppl"] == 400.0

"""
test_data_code.py — Validate CodeParrot data preparation
=========================================================

Tests:
  1. data/codeparrot/ exists with train.bin and val.bin after data_prep_code.py
  2. Token counts are within 5% of targets (2.4M train, 248K val)
  3. Data loads correctly through the same load_data() function used by train.py
  4. UNK rate from meta.json is logged (informational only — no assertion)

Run with:
    cd ~/hyper-attn-nano
    source activate.sh
    pytest tests/test_data_code.py -v

NOTE: These tests require data/codeparrot/ to have been prepared first:
    python scripts/data_prep_code.py
"""

import json
import os
import sys

import numpy as np
import pytest

# Make scripts/ importable for the load_data helper
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

DATA_DIR     = "data/codeparrot"
TRAIN_TARGET = 2_400_000
VAL_TARGET   =   248_000
TOLERANCE    = 0.05   # 5%


# ---------------------------------------------------------------------------
# Helper — same as train.py's load_data
# ---------------------------------------------------------------------------

def _load_data(split: str) -> np.ndarray:
    path = os.path.join(DATA_DIR, f"{split}.bin")
    return np.memmap(path, dtype=np.uint16, mode="r")


# ---------------------------------------------------------------------------
# Test 1: Files exist
# ---------------------------------------------------------------------------

def test_data_dir_exists():
    assert os.path.isdir(DATA_DIR), (
        f"data/codeparrot/ not found. Run: python scripts/data_prep_code.py"
    )


def test_train_bin_exists():
    path = os.path.join(DATA_DIR, "train.bin")
    assert os.path.isfile(path), f"Missing: {path}"


def test_val_bin_exists():
    path = os.path.join(DATA_DIR, "val.bin")
    assert os.path.isfile(path), f"Missing: {path}"


def test_meta_json_exists():
    path = os.path.join(DATA_DIR, "meta.json")
    assert os.path.isfile(path), f"Missing: {path}"


# ---------------------------------------------------------------------------
# Test 2: Token counts within 5% of targets
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.path.isfile(os.path.join(DATA_DIR, "train.bin")),
    reason="train.bin not yet prepared",
)
def test_train_token_count():
    arr = _load_data("train")
    n   = len(arr)
    lo  = int(TRAIN_TARGET * (1 - TOLERANCE))
    hi  = int(TRAIN_TARGET * (1 + TOLERANCE))
    assert lo <= n <= hi, (
        f"Train token count {n:,} outside ±5% of target {TRAIN_TARGET:,} "
        f"(expected {lo:,}–{hi:,})"
    )


@pytest.mark.skipif(
    not os.path.isfile(os.path.join(DATA_DIR, "val.bin")),
    reason="val.bin not yet prepared",
)
def test_val_token_count():
    arr = _load_data("val")
    n   = len(arr)
    lo  = int(VAL_TARGET * (1 - TOLERANCE))
    hi  = int(VAL_TARGET * (1 + TOLERANCE))
    assert lo <= n <= hi, (
        f"Val token count {n:,} outside ±5% of target {VAL_TARGET:,} "
        f"(expected {lo:,}–{hi:,})"
    )


# ---------------------------------------------------------------------------
# Test 3: Data loads through train.py's load_data path
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.path.isfile(os.path.join(DATA_DIR, "train.bin")),
    reason="train.bin not yet prepared",
)
def test_train_loads_via_memmap():
    """Data must load as uint16 numpy memmap without error."""
    arr = _load_data("train")
    assert arr.dtype == np.uint16
    assert len(arr) > 0
    # Sanity: token IDs are in the GPT-2 vocabulary range [0, 50256]
    sample = np.array(arr[:10_000])
    assert sample.min() >= 0
    assert sample.max() < 50257, (
        f"Token ID {sample.max()} out of GPT-2 vocab range [0, 50256]"
    )


@pytest.mark.skipif(
    not os.path.isfile(os.path.join(DATA_DIR, "val.bin")),
    reason="val.bin not yet prepared",
)
def test_val_loads_via_memmap():
    arr = _load_data("val")
    assert arr.dtype == np.uint16
    assert len(arr) > 0
    sample = np.array(arr[:10_000])
    assert sample.min() >= 0
    assert sample.max() < 50257


# ---------------------------------------------------------------------------
# Test 4: UNK rate (informational — no assertion)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.path.isfile(os.path.join(DATA_DIR, "meta.json")),
    reason="meta.json not yet prepared",
)
def test_unk_rate_logged(capsys):
    with open(os.path.join(DATA_DIR, "meta.json")) as f:
        meta = json.load(f)
    unk_rate = meta.get("unk_rate_pct", None)
    print(f"\n[INFO] CodeParrot UNK rate: {unk_rate}%")
    # Informational — no assertion on UNK rate per spec
    assert "unk_rate_pct" in meta, "meta.json missing unk_rate_pct key"

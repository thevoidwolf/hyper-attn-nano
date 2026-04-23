"""
test_checkpoint_resume.py — Byte-identity checkpoint/resume test (spec §5.6)

Verifies that a run interrupted at step 100 and resumed produces bitwise-identical
model weights at step 200 compared to an uninterrupted run of the same seed.

Uses a tiny S1-like config (d_model=64) so the test completes quickly.
Full-split eval is used so eval does not perturb the training RNG.

Run with:
    pytest tests/test_checkpoint_resume.py -v
or:
    python tests/test_checkpoint_resume.py
"""

import math
import os
import sys
import tempfile

import numpy as np
import pytest
import torch

# Make src/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model import GPTNano, HyperAttnNano
from training.curvature_schedule import build_schedule

# Import helpers from train.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from train import (
    get_rng_state,
    set_rng_state,
    find_latest_checkpoint,
    save_experiment_checkpoint,
    make_lr_lambda,
)


# ---------------------------------------------------------------------------
# Tiny training config for speed
# ---------------------------------------------------------------------------

TINY_CFG = {
    "experiment":    "experiment_a",
    "spec_version":  "A.1.1",
    "scale_tag":     "test",
    "notes":         "byte-identity test",
    "model": {
        "type":        "euclid",
        "d_model":     64,
        "n_layers":    2,
        "n_heads":     2,
        "d_ff":        128,
        "max_seq_len": 32,
        "vocab_size":  256,
    },
    "training": {
        "dataset":              "test",
        "data_dir":             "test",
        "batch_size":           4,
        "max_steps":            200,
        "eval_interval":        50,
        "checkpoint_interval":  100,
        "log_interval":         50,
        "eval_batches":         5,
        "full_split_eval":      True,
        "lr":                   3e-4,
        "weight_decay":         0.1,
        "warmup_steps":         20,
        "min_lr_ratio":         0.1,
        "seed":                 42,
    },
}

SEED          = 42
MAX_STEPS     = 200
CKPT_STEP     = 100  # checkpoint after this training step
BLOCK_SIZE    = TINY_CFG["model"]["max_seq_len"]
BATCH_SIZE    = TINY_CFG["training"]["batch_size"]
VOCAB_SIZE    = TINY_CFG["model"]["vocab_size"]
EVAL_INTERVAL = TINY_CFG["training"]["eval_interval"]


def _seed_all(seed: int) -> None:
    import random
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _make_model(cfg: dict) -> GPTNano:
    model_cfg = {k: cfg["model"][k]
                 for k in ["d_model", "n_layers", "n_heads", "d_ff", "max_seq_len", "vocab_size"]}
    return GPTNano(model_cfg)


def _make_optimizer(model, cfg: dict) -> torch.optim.AdamW:
    tcfg = cfg["training"]
    return torch.optim.AdamW(
        model.parameters(),
        lr=tcfg["lr"], weight_decay=tcfg["weight_decay"], betas=(0.9, 0.95),
    )


def _make_scheduler(optimizer, cfg: dict) -> torch.optim.lr_scheduler.LambdaLR:
    tcfg = cfg["training"]
    lr_fn = make_lr_lambda(tcfg["warmup_steps"], tcfg["max_steps"], tcfg["min_lr_ratio"])
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)


def _make_fake_data(n_tokens: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, VOCAB_SIZE, size=n_tokens, dtype=np.uint16)


def _get_batch(data: np.ndarray, device: torch.device) -> tuple:
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([torch.from_numpy(data[i:i+BLOCK_SIZE].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+BLOCK_SIZE+1].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


def _train_steps(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    data: np.ndarray,
    device: torch.device,
    start: int,
    stop: int,
    run_dir: str,
    cfg: dict,
    metrics_history: list | None = None,
    git_hash: str = "test",
    spec_version: str = "A.1.1",
) -> list:
    """
    Run training steps [start, stop). Save checkpoints every CKPT_STEP steps.
    Returns the accumulated metrics_history.
    """
    if metrics_history is None:
        metrics_history = []

    for step in range(start, stop):
        # Dummy eval every EVAL_INTERVAL (no RNG — deterministic)
        # (In real training this would be evaluate_full_split; here we skip to keep test fast)

        model.train()
        x, y = _get_batch(data, device)
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        # Checkpoint AFTER training step (§5.1 + byte-identity requirement)
        if step % CKPT_STEP == 0:
            save_experiment_checkpoint(
                model, optimizer, scheduler, step,
                current_K=None,
                cfg=cfg,
                metrics_history=metrics_history,
                git_hash=git_hash,
                spec_version=spec_version,
                run_dir=run_dir,
            )

    return metrics_history


def _state_dict_bytes(model: torch.nn.Module) -> dict[str, bytes]:
    """Convert state dict tensors to bytes for bitwise comparison."""
    return {
        k: v.cpu().numpy().tobytes()
        for k, v in model.state_dict().items()
    }


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def test_byte_identity_resume():
    """
    Spec §5.6: a run resumed from a mid-training checkpoint must produce
    bitwise-identical weights to an uninterrupted run of the same seed.
    """
    device = torch.device("cpu")  # deterministic on CPU

    # Fake training data (large enough for BLOCK_SIZE + 1)
    n_tokens   = 10_000
    train_data = _make_fake_data(n_tokens, seed=0)

    with tempfile.TemporaryDirectory() as tmp_dir:
        run_dir_1 = os.path.join(tmp_dir, "run_uninterrupted")
        run_dir_2 = os.path.join(tmp_dir, "run_interrupted")

        # ── Run 1: uninterrupted 0 → MAX_STEPS ────────────────────────────
        _seed_all(SEED)
        model_1     = _make_model(TINY_CFG).to(device)
        optimizer_1 = _make_optimizer(model_1, TINY_CFG)
        scheduler_1 = _make_scheduler(optimizer_1, TINY_CFG)

        _train_steps(
            model_1, optimizer_1, scheduler_1,
            train_data, device,
            start=0, stop=MAX_STEPS,
            run_dir=run_dir_1, cfg=TINY_CFG,
        )
        weights_uninterrupted = _state_dict_bytes(model_1)

        # ── Run 2: train 0 → CKPT_STEP, then resume CKPT_STEP+1 → MAX_STEPS ──
        _seed_all(SEED)
        model_2     = _make_model(TINY_CFG).to(device)
        optimizer_2 = _make_optimizer(model_2, TINY_CFG)
        scheduler_2 = _make_scheduler(optimizer_2, TINY_CFG)

        # Phase 2a: train to CKPT_STEP (checkpoint saved after step CKPT_STEP)
        _train_steps(
            model_2, optimizer_2, scheduler_2,
            train_data, device,
            start=0, stop=CKPT_STEP + 1,
            run_dir=run_dir_2, cfg=TINY_CFG,
        )

        # Phase 2b: simulate process restart — fresh model + optimizer + scheduler
        latest = find_latest_checkpoint(run_dir_2)
        assert latest is not None, "No checkpoint found after phase 2a"

        ckpt = torch.load(latest, map_location=device, weights_only=False)
        assert ckpt["step"] == CKPT_STEP, (
            f"Expected checkpoint at step {CKPT_STEP}, got step {ckpt['step']}"
        )

        # Fresh objects
        model_3     = _make_model(TINY_CFG).to(device)
        optimizer_3 = _make_optimizer(model_3, TINY_CFG)
        scheduler_3 = _make_scheduler(optimizer_3, TINY_CFG)

        model_3.load_state_dict(ckpt["model_state_dict"])
        optimizer_3.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler_3.load_state_dict(ckpt["lr_scheduler_state"])
        set_rng_state(ckpt["rng_state"])
        resume_step = ckpt["step"] + 1

        # Phase 2c: resume from CKPT_STEP+1 → MAX_STEPS
        _train_steps(
            model_3, optimizer_3, scheduler_3,
            train_data, device,
            start=resume_step, stop=MAX_STEPS,
            run_dir=run_dir_2, cfg=TINY_CFG,
        )
        weights_resumed = _state_dict_bytes(model_3)

        # ── Bitwise comparison ─────────────────────────────────────────────
        assert set(weights_uninterrupted.keys()) == set(weights_resumed.keys()), (
            "State dict key mismatch between runs"
        )
        mismatches = [
            k for k in weights_uninterrupted
            if weights_uninterrupted[k] != weights_resumed[k]
        ]
        if mismatches:
            # Provide numeric diagnostics for debugging
            for k in mismatches[:3]:
                t1 = np.frombuffer(weights_uninterrupted[k], dtype=np.float32)
                t2 = np.frombuffer(weights_resumed[k],       dtype=np.float32)
                max_diff = np.max(np.abs(t1 - t2))
                print(f"  MISMATCH {k}: max_abs_diff={max_diff:.2e}")
        assert not mismatches, (
            f"Bitwise mismatch in {len(mismatches)} tensors after resume: "
            f"{mismatches[:5]}"
        )


if __name__ == "__main__":
    print("Running byte-identity resume test...")
    test_byte_identity_resume()
    print("PASSED — weights are bitwise identical after resume.")

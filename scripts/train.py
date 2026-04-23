"""
train.py — Training script for HyperAttn-Nano variants
=======================================================

Usage:
    python scripts/train.py --config configs/nano_euclid.yaml
    python scripts/train.py --config configs/nano_hyper_fixed.yaml
    python scripts/train.py --config configs/nano_hyper_perhead.yaml
    python scripts/train.py --config configs/smoke_shakespeare.yaml
    python scripts/train.py --config configs/experiment_a/a_s2_euclid_seed42.yaml

All behaviour is controlled by the yaml config file.
Experiment A configs activate new features via config flags (spec A.1.1).
"""

import argparse
import glob
import json
import math
import os
import random
import subprocess
import sys
import time
import warnings

import numpy as np
import torch
import yaml

# Make src/ importable when running from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from model import HyperAttnNano, GPTNano, ScoresOnlyNano  # noqa: E402
from training.curvature_schedule import build_schedule  # noqa: E402


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def get_git_commit_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
            cwd=os.path.join(os.path.dirname(__file__), ".."),
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Parameter count and KV-cache helpers (§6.3)
# ---------------------------------------------------------------------------

def compute_param_counts(model, config: dict) -> tuple[int, int]:
    """
    Returns (non_embedding_body_params, total_params).

    Non-embedding body = all params minus the embedding table.
    For tied models lm_head.weight == embed.weight, so parameters() counts it once.
    Embedding size = vocab_size × d_model.
    """
    total = sum(p.numel() for p in model.parameters())
    vocab_size = config["model"]["vocab_size"]
    d_model    = config["model"]["d_model"]
    non_emb    = total - vocab_size * d_model
    return non_emb, total


def compute_kv_cache_bytes(config: dict) -> int:
    """
    KV cache bytes per token (float32 activations assumed).
    Formula: 2 × n_layers × n_heads × head_dim × bytes_per_element
    Factor 2: one K cache + one V cache.
    """
    n_layers  = config["model"]["n_layers"]
    n_heads   = config["model"]["n_heads"]
    d_model   = config["model"]["d_model"]
    head_dim  = d_model // n_heads
    return 2 * n_layers * n_heads * head_dim * 4  # 4 bytes = float32


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_data(data_dir: str, split: str) -> np.ndarray:
    path = os.path.join(data_dir, f"{split}.bin")
    return np.memmap(path, dtype=np.uint16, mode="r")


def get_batch(
    data: np.ndarray,
    block_size: int,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack(
        [torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [torch.from_numpy(data[i + 1 : i + block_size + 1].astype(np.int64)) for i in ix]
    )
    return x.to(device), y.to(device)


# ---------------------------------------------------------------------------
# Evaluation — stochastic (backward compat) and full-split (Experiment A)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, data, cfg, device, eval_batches: int = 50) -> float:
    """Stochastic evaluation — kept for Preflight 1 backward compat."""
    model.eval()
    block_size = cfg["model"]["max_seq_len"]
    batch_size = cfg["training"]["batch_size"]
    losses = []
    for _ in range(eval_batches):
        x, y = get_batch(data, block_size, batch_size, device)
        if device.type == "cuda":
            with torch.amp.autocast("cuda"):
                _, loss = model(x, y)
        else:
            _, loss = model(x, y)
        losses.append(loss.item())
    return sum(losses) / len(losses)


@torch.no_grad()
def evaluate_full_split(model, data, cfg, device) -> float:
    """
    Full ordered non-overlapping pass over val data.
    Every complete block_size window is evaluated exactly once.
    No RNG usage — deterministic regardless of RNG state.
    """
    model.eval()
    block_size = cfg["model"]["max_seq_len"]
    n_chunks   = len(data) // block_size
    if n_chunks == 0:
        # Tiny dataset (smoke tests) — fall back to stochastic
        return evaluate(model, data, cfg, device)

    total_loss = 0.0
    for i in range(n_chunks):
        start  = i * block_size
        chunk  = torch.from_numpy(
            data[start : start + block_size].astype(np.int64)
        ).unsqueeze(0).to(device)
        target = torch.from_numpy(
            data[start + 1 : start + block_size + 1].astype(np.int64)
        ).unsqueeze(0).to(device)
        if device.type == "cuda":
            with torch.amp.autocast("cuda"):
                _, loss = model(chunk, target)
        else:
            _, loss = model(chunk, target)
        total_loss += loss.item()

    return total_loss / n_chunks


# ---------------------------------------------------------------------------
# RNG state helpers
# ---------------------------------------------------------------------------

def get_rng_state() -> dict:
    return {
        "torch":         torch.get_rng_state(),
        "torch_cuda":    torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        "numpy":         np.random.get_state(),
        "python_random": random.getstate(),
    }


def set_rng_state(state: dict) -> None:
    torch.set_rng_state(state["torch"])
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(state["torch_cuda"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python_random"])


# ---------------------------------------------------------------------------
# Atomic checkpoint save / load / pruning (§5.3, §5.4)
# ---------------------------------------------------------------------------

def _atomic_save(obj: dict, path: str) -> None:
    """Write to .tmp, fsync, rename — crash-safe."""
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.rename(tmp, path)


def _prune_checkpoints(run_dir: str, keep_last_n: int = 2) -> None:
    """Delete old step checkpoints; best.pt and final.pt are never touched."""
    pattern = os.path.join(run_dir, "checkpoint_step_*.pt")
    ckpts = sorted(
        glob.glob(pattern),
        key=lambda p: int(
            os.path.basename(p).replace("checkpoint_step_", "").replace(".pt", "")
        ),
    )
    for old in ckpts[:-keep_last_n]:
        try:
            os.remove(old)
        except OSError:
            pass


def _build_checkpoint(
    model,
    optimizer,
    scheduler,
    step: int,
    current_K,
    cfg: dict,
    metrics_history: list,
    git_hash: str,
    spec_version: str,
) -> dict:
    return {
        "model_state_dict":    model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state":  scheduler.state_dict(),
        "step":                step,
        "current_K":           current_K,
        "rng_state":           get_rng_state(),
        "run_config":          cfg,
        "metrics_history":     metrics_history,
        "git_commit_hash":     git_hash,
        "spec_version":        spec_version,
    }


def save_experiment_checkpoint(
    model,
    optimizer,
    scheduler,
    step: int,
    current_K,
    cfg: dict,
    metrics_history: list,
    git_hash: str,
    spec_version: str,
    run_dir: str,
    is_best: bool = False,
    is_final: bool = False,
) -> None:
    os.makedirs(run_dir, exist_ok=True)
    state = _build_checkpoint(
        model, optimizer, scheduler, step, current_K,
        cfg, metrics_history, git_hash, spec_version,
    )
    ckpt_path = os.path.join(run_dir, f"checkpoint_step_{step}.pt")
    _atomic_save(state, ckpt_path)
    if is_best:
        _atomic_save(state, os.path.join(run_dir, "best.pt"))
    if is_final:
        _atomic_save(state, os.path.join(run_dir, "final.pt"))
    _prune_checkpoints(run_dir)


def find_latest_checkpoint(run_dir: str) -> str | None:
    pattern = os.path.join(run_dir, "checkpoint_step_*.pt")
    ckpts = sorted(
        glob.glob(pattern),
        key=lambda p: int(
            os.path.basename(p).replace("checkpoint_step_", "").replace(".pt", "")
        ),
    )
    return ckpts[-1] if ckpts else None


# ---------------------------------------------------------------------------
# JSONL logging helpers (§6, §17.2)
# ---------------------------------------------------------------------------

def _jsonl_write(f, obj: dict) -> None:
    """Write one record, flush, fsync — satisfies durability requirement."""
    f.write(json.dumps(obj) + "\n")
    f.flush()
    os.fsync(f.fileno())


def open_experiment_logs(log_dir: str) -> tuple:
    """Open train.jsonl and eval.jsonl in append mode."""
    os.makedirs(log_dir, exist_ok=True)
    train_f = open(os.path.join(log_dir, "train.jsonl"), "a")
    eval_f  = open(os.path.join(log_dir, "eval.jsonl"),  "a")
    return train_f, eval_f


def save_summary(log_dir: str, data: dict) -> None:
    path = os.path.join(log_dir, "summary.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Legacy helpers (kept for backward compat with old configs)
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, step, cfg, log):
    variant  = cfg["model"]["type"]
    path     = f"results/checkpoints/{variant}/ckpt_step{step}.pt"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {"step": step, "model": model.state_dict(),
         "optimizer": optimizer.state_dict(), "config": cfg},
        path,
    )
    print(f"Checkpoint saved → {path}")


def save_log(log, cfg):
    run_id  = cfg.get("run_id", cfg["model"]["type"])
    log_dir = cfg.get("log_dir", "results/logs/exp_b")
    path    = os.path.join(log_dir, f"{run_id}_log.json")
    os.makedirs(log_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"Log saved → {path}")


def open_jsonl_log(cfg) -> tuple[str, "IO"]:
    run_id  = cfg.get("run_id", cfg["model"]["type"])
    log_dir = cfg.get("log_dir", "results/logs/exp_b")
    path    = os.path.join(log_dir, f"{run_id}_train.jsonl")
    os.makedirs(log_dir, exist_ok=True)
    return path, open(path, "w")


# ---------------------------------------------------------------------------
# LR schedule factories (§3.2)
# ---------------------------------------------------------------------------

def make_lr_lambda(warmup_steps: int, max_steps: int, min_lr_ratio: float):
    """Standard warmup + cosine decay (Euclidean variant — unchanged)."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    return lr_lambda


def make_lr_lambda_hyper(warmup_steps: int, max_steps: int, min_lr_ratio: float):
    """
    Piecewise schedule for the hyperbolic variant (spec §3.2):
      Phase 1: linear warmup (0 → warmup_steps)
      Phase 2: cosine decay over the full post-warmup range (same as Euclidean)
      Phase 3: additional multiplicative cosine factor over the final 20% of
               total steps, driving LR toward 0 to suppress the late-training
               grad-norm rise observed in Probe 1 and Probe 3.

    The factor is 1.0 at the Phase 2/3 boundary (continuous) and 0.0 at
    the final step, so the terminal LR → 0.
    """
    decay2_start = int(0.8 * max_steps)
    phase3_len   = max(max_steps - decay2_start, 1)
    post_warmup  = max(max_steps - warmup_steps, 1)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        # Base cosine over full post-warmup range (identical to Euclidean)
        progress = (step - warmup_steps) / post_warmup
        base = min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
        # Additional multiplicative cosine in final 20%
        if step >= decay2_start:
            progress3 = (step - decay2_start) / phase3_len
            extra = 0.5 * (1 + math.cos(math.pi * progress3))
            return base * extra
        return base

    return lr_lambda


# ---------------------------------------------------------------------------
# Current-K helper (works for all model types)
# ---------------------------------------------------------------------------

def _current_k(model, model_type: str, curv_schedule, step: int):
    """Return the current curvature scalar or None for Euclidean."""
    if model_type == "euclid":
        return None
    if curv_schedule is not None:
        return curv_schedule.k_at_step(step)
    if hasattr(model, "blocks") and len(model.blocks) > 0:
        b = model.blocks[0]
        if hasattr(b, "attn") and hasattr(b.attn, "log_abs_K"):
            return -math.exp(b.attn.log_abs_K.item())
    return None


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train a HyperAttn-Nano variant")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--no-abort", action="store_true",
                        help="Disable early-abort on bad configs")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ── Experiment A feature flags ─────────────────────────────────────────
    is_exp_a      = cfg.get("experiment") == "experiment_a"
    full_split    = cfg["training"].get("full_split_eval", False)
    ckpt_interval = cfg["training"].get("checkpoint_interval", None)
    extra_cosine  = cfg["training"].get("extra_cosine_decay", False)
    spec_version  = cfg.get("spec_version", "legacy")

    # ── Seeding ────────────────────────────────────────────────────────────
    seed = cfg["training"].get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # ── Device ─────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("  WARNING: CUDA not available; training on CPU will be slow.")

    # ── Model ──────────────────────────────────────────────────────────────
    model_type = cfg["model"]["type"]
    model_cfg  = {
        k: cfg["model"][k]
        for k in ["d_model", "n_layers", "n_heads", "d_ff", "max_seq_len", "vocab_size"]
    }
    for opt_key in ("manifold_float64", "output_head", "spherical_temperature_init"):
        if opt_key in cfg["model"]:
            model_cfg[opt_key] = cfg["model"][opt_key]

    init_K         = cfg["model"].get("init_K", -1.0)
    curvature      = cfg["model"].get("curvature", init_K)
    curvature_init = cfg["model"].get("curvature_init", None)
    run_id         = cfg.get("run_id", model_type)

    if model_type == "euclid":
        model = GPTNano(model_cfg)
    elif model_type == "hyper-scores-only":
        model = ScoresOnlyNano(model_cfg, fixed_curvature=curvature)
    else:
        model = HyperAttnNano(model_cfg, fixed_curvature=curvature, curvature_init=curvature_init)

    model = model.to(device)

    # ── Curvature schedule ─────────────────────────────────────────────────
    curv_schedule = None
    schedule_mode = "frozen_schedule"
    schedule_cfg  = cfg["model"].get("curvature_schedule", None)

    if schedule_cfg is not None:
        if "curvature" in cfg["model"] or "init_K" in cfg["model"]:
            warnings.warn(
                "[CONFIG] Both 'curvature' and 'curvature_schedule' are set. "
                "'curvature_schedule' takes precedence.",
                UserWarning, stacklevel=2,
            )
        curv_schedule = build_schedule(schedule_cfg)
        schedule_mode = schedule_cfg.get("schedule_mode", "frozen_schedule")
        print(f"[CONFIG] Curvature schedule: {schedule_cfg['type']}  "
              f"k_start={schedule_cfg.get('k_start', 'N/A')}  "
              f"k_end={schedule_cfg.get('k_end', schedule_cfg.get('k_start', 'N/A'))}  "
              f"warmup={schedule_cfg.get('warmup_steps', 'N/A')}  mode={schedule_mode}")

    if model_type == "hyper-fixed" or (
        model_type in ("hyper-perhead", "hyper-scores-only")
        and curv_schedule is not None
        and schedule_mode == "frozen_schedule"
    ):
        for block in model.blocks:
            block.attn.log_abs_K.requires_grad_(False)
        if curv_schedule is None:
            print("  hyper-fixed: log_abs_K frozen")
            print(f"[CONFIG] Fixed curvature K = {curvature}")
        else:
            print("  curvature schedule active: log_abs_K frozen, overridden each step")
    elif model_type == "hyper-scores-only" and curv_schedule is None:
        for block in model.blocks:
            block.attn.log_abs_K.requires_grad_(False)
        print("  hyper-scores-only: log_abs_K frozen")
        print(f"[CONFIG] Fixed curvature K = {curvature}")

    total_params_display = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params_display:,}")

    non_emb_params, total_params = compute_param_counts(model, cfg)
    kv_cache_bytes = compute_kv_cache_bytes(cfg)
    if is_exp_a:
        print(f"  Non-embedding body params: {non_emb_params:,}")
        print(f"  KV cache bytes/token: {kv_cache_bytes:,}")

    # ── Early-abort baseline ───────────────────────────────────────────────
    _BASELINE_PATH = os.path.join(
        os.path.dirname(__file__), "..", "configs", "baselines",
        "euclid_wikitext2_step500_ppl.json",
    )
    abort_threshold: float | None = None
    if not args.no_abort:
        if os.path.exists(_BASELINE_PATH):
            with open(_BASELINE_PATH) as _f:
                abort_threshold = json.load(_f)["val_ppl"] * 10.0
            print(f"[ABORT] Early-abort enabled. Threshold at step 500: PPL > {abort_threshold:.1f}")
        else:
            print(f"[ABORT] Baseline file not found — early-abort disabled.")
    else:
        print("[ABORT] Early-abort disabled (--no-abort).")

    # ── Training config ────────────────────────────────────────────────────
    tcfg          = cfg["training"]
    data_dir      = tcfg["data_dir"]
    batch_size    = tcfg["batch_size"]
    max_steps     = tcfg["max_steps"]
    eval_interval = tcfg["eval_interval"]
    log_interval  = tcfg["log_interval"]
    eval_batches  = tcfg["eval_batches"]
    block_size    = cfg["model"]["max_seq_len"]
    warmup_steps  = tcfg["warmup_steps"]
    min_lr_ratio  = tcfg.get("min_lr_ratio", 0.1)

    # ── Data ───────────────────────────────────────────────────────────────
    train_data = load_data(data_dir, "train")
    val_data   = load_data(data_dir, "val")
    print(f"  Dataset: {tcfg['dataset']}  "
          f"({len(train_data):,} train / {len(val_data):,} val tokens)")

    # ── Paths ──────────────────────────────────────────────────────────────
    log_dir = cfg.get("log_dir", "results/logs/exp_b")
    run_dir = cfg.get("checkpoint_dir", None)
    if is_exp_a and run_dir is None:
        run_dir = os.path.join("checkpoints", "experiment_a", run_id)

    git_hash = get_git_commit_hash() if is_exp_a else "unknown"

    # ── Experiment A JSONL logs ────────────────────────────────────────────
    exp_train_f = None
    exp_eval_f  = None
    if is_exp_a:
        exp_train_f, exp_eval_f = open_experiment_logs(log_dir)
        print(f"[EXP-A] Logs        → {log_dir}")
        print(f"[EXP-A] Checkpoints → {run_dir}")
        print(f"[EXP-A] git commit  : {git_hash}")

    # Legacy JSONL (backward compat)
    jsonl_path, jsonl_file = open_jsonl_log(cfg)
    last_train_loss: float = 0.0
    last_grad_norm:  float = 0.0

    # ── Optimiser ──────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = tcfg["lr"],
        weight_decay = tcfg["weight_decay"],
        betas        = (0.9, 0.95),
    )

    # ── LR schedule ────────────────────────────────────────────────────────
    if extra_cosine:
        lr_fn = make_lr_lambda_hyper(warmup_steps, max_steps, min_lr_ratio)
        print("[CONFIG] LR: hyper piecewise schedule (extra cosine in final 20%)")
    else:
        lr_fn = make_lr_lambda(warmup_steps, max_steps, min_lr_ratio)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)

    # ── AMP scaler (CUDA only) ─────────────────────────────────────────────
    use_amp = device.type == "cuda"
    scaler  = torch.amp.GradScaler("cuda") if use_amp else None

    # ── Legacy log structure ───────────────────────────────────────────────
    log = {"config": cfg, "train_loss": [], "val_loss": [], "curvatures": []}

    # ── Experiment A state ─────────────────────────────────────────────────
    metrics_history: list[dict] = []
    best_val_ppl    = float("inf")
    best_val_step   = -1
    resume_events: list[dict] = []
    start_step      = 0
    status          = "completed"

    # ── Resume (Experiment A) ──────────────────────────────────────────────
    if is_exp_a and run_dir is not None:
        latest_ckpt = find_latest_checkpoint(run_dir)
        if latest_ckpt is not None:
            print(f"[RESUME] Loading: {latest_ckpt}")
            ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["lr_scheduler_state"])
            set_rng_state(ckpt["rng_state"])
            start_step      = ckpt["step"] + 1
            metrics_history = list(ckpt.get("metrics_history", []))
            best_val_ppl    = min(
                (r["val_ppl"] for r in metrics_history if "val_ppl" in r),
                default=float("inf"),
            )
            # Rebuild any evals that happened after the checkpoint was written
            eval_jsonl = os.path.join(log_dir, "eval.jsonl")
            if os.path.exists(eval_jsonl):
                with open(eval_jsonl) as ef:
                    for line in ef:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                            if rec.get("step", -1) > ckpt["step"] and "val_ppl" in rec:
                                metrics_history.append(rec)
                                if rec["val_ppl"] < best_val_ppl:
                                    best_val_ppl  = rec["val_ppl"]
                                    best_val_step = rec["step"]
                        except json.JSONDecodeError:
                            pass

            ts = time.strftime("%Y-%m-%dT%H:%M:%S")
            resume_events.append({"step": start_step, "timestamp": ts})
            print(f"[RESUME] RESUMED from step {ckpt['step']} at {ts}")
            if exp_train_f:
                _jsonl_write(exp_train_f, {
                    "event": "RESUMED", "from_step": ckpt["step"], "timestamp": ts
                })

    print(f"\nStarting: {model_type} | {max_steps} steps | from step {start_step}")
    print("=" * 60)

    # ── Compute budget tracking ────────────────────────────────────────────
    train_start_time = time.time()
    total_tokens     = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # Rolling tokens/sec (over last log_interval steps)
    _step_times: list[float] = []
    _step_tokens: list[int]  = []
    last_eval_time = train_start_time

    # ── Training loop ──────────────────────────────────────────────────────
    for step in range(start_step, max_steps):

        # ── Eval ──────────────────────────────────────────────────────────
        if step % eval_interval == 0:
            if full_split:
                val_loss = evaluate_full_split(model, val_data, cfg, device)
            else:
                val_loss = evaluate(model, val_data, cfg, device, eval_batches)
            perplexity = math.exp(val_loss)
            now = time.time()
            elapsed_since_eval = now - last_eval_time
            last_eval_time     = now

            log["val_loss"].append({"step": step, "loss": val_loss, "perplexity": perplexity})
            print(f"step {step:6d} | val loss {val_loss:.4f} | ppl {perplexity:.1f}")

            # Legacy JSONL eval record
            legacy_rec: dict = {
                "step":       step,
                "train_loss": last_train_loss,
                "val_ppl":    perplexity,
                "grad_norm":  last_grad_norm,
            }
            if curv_schedule is not None:
                legacy_rec["scheduled_K"] = curv_schedule.k_at_step(step)
            jsonl_file.write(json.dumps(legacy_rec) + "\n")
            jsonl_file.flush()

            # Experiment A eval JSONL (§6.2)
            if exp_eval_f is not None:
                eval_rec = {
                    "step":                     step,
                    "val_loss":                 val_loss,
                    "val_ppl":                  perplexity,
                    "time_since_last_eval_sec": round(elapsed_since_eval, 3),
                }
                _jsonl_write(exp_eval_f, eval_rec)
                metrics_history.append(eval_rec)

            # Track best and save best.pt
            if perplexity < best_val_ppl:
                best_val_ppl  = perplexity
                best_val_step = step
                if is_exp_a and run_dir is not None and ckpt_interval is not None:
                    save_experiment_checkpoint(
                        model, optimizer, scheduler, step,
                        _current_k(model, model_type, curv_schedule, step),
                        cfg, metrics_history, git_hash, spec_version,
                        run_dir, is_best=True,
                    )

            # Curvature snapshot (hyper models only)
            if hasattr(model, "get_curvatures"):
                curvs = model.get_curvatures()
                if curvs:
                    log["curvatures"].append({"step": step, "values": curvs})
                    vals = list(curvs.values())
                    print(f"  curvature: min={min(vals):.3f}  max={max(vals):.3f}  "
                          f"std={torch.tensor(vals).std().item():.3f}")

            # Early-abort check at step 500
            if step == 500 and abort_threshold is not None:
                if perplexity > abort_threshold:
                    reason = (f"val_ppl={perplexity:.1f} > {abort_threshold:.1f} "
                              f"(10× Euclidean baseline at step 500)")
                    print(f"[ABORT] Early abort at step {step}: {reason}")
                    jsonl_file.write(json.dumps({"event": "ABORTED", "step": step, "reason": reason}) + "\n")
                    jsonl_file.flush()
                    jsonl_file.close()
                    if exp_train_f: exp_train_f.close()
                    if exp_eval_f:  exp_eval_f.close()
                    save_log(log, cfg)
                    return

        # ── Apply curvature schedule ──────────────────────────────────────
        if curv_schedule is not None and hasattr(model, "blocks"):
            k_now     = curv_schedule.k_at_step(step)
            log_abs_k = math.log(abs(k_now))
            warmup_done = step >= schedule_cfg.get("warmup_steps", 0)
            with torch.no_grad():
                for block in model.blocks:
                    if hasattr(block, "attn") and hasattr(block.attn, "log_abs_K"):
                        block.attn.log_abs_K.fill_(log_abs_k)
            if schedule_mode == "init_only" and warmup_done:
                for block in model.blocks:
                    if hasattr(block, "attn") and hasattr(block.attn, "log_abs_K"):
                        block.attn.log_abs_K.requires_grad_(True)

        # ── Train step ────────────────────────────────────────────────────
        model.train()
        step_t0 = time.time()
        x, y   = get_batch(train_data, block_size, batch_size, device)

        if use_amp:
            with torch.amp.autocast("cuda"):
                logits, loss = model(x, y)
            if not torch.isfinite(loss):
                print(f"[WARN] Non-finite loss at step {step}: {loss.item()}. Stopping.")
                save_checkpoint(model, optimizer, step, cfg, log)
                status = "failed"
                break
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if grad_norm.item() > 1000.0:
                print(f"[WARN] High grad norm at step {step}: {grad_norm.item():.1f}")
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, loss = model(x, y)
            if not torch.isfinite(loss):
                print(f"[WARN] Non-finite loss at step {step}: {loss.item()}. Stopping.")
                save_checkpoint(model, optimizer, step, cfg, log)
                status = "failed"
                break
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if grad_norm.item() > 1000.0:
                print(f"[WARN] High grad norm at step {step}: {grad_norm.item():.1f}")
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        # Timing for rolling tokens/sec
        step_elapsed = time.time() - step_t0
        step_tokens  = batch_size * block_size
        _step_times.append(step_elapsed)
        _step_tokens.append(step_tokens)
        if len(_step_times) > log_interval:
            _step_times.pop(0)
            _step_tokens.pop(0)

        last_train_loss = loss.item()
        last_grad_norm  = grad_norm.item()
        total_tokens   += step_tokens
        log["train_loss"].append({"step": step, "loss": loss.item(), "grad_norm": grad_norm.item()})

        # ── Periodic step checkpoint (AFTER training step) ─────────────────
        # Checkpoint at ckpt_interval multiples. Saves RNG state after get_batch,
        # so a resume from step N continues from step N+1 with the correct RNG.
        if is_exp_a and ckpt_interval and step % ckpt_interval == 0:
            save_experiment_checkpoint(
                model, optimizer, scheduler, step,
                _current_k(model, model_type, curv_schedule, step),
                cfg, metrics_history, git_hash, spec_version,
                run_dir,
            )
            print(f"[CKPT] step {step}")

        # ── Per-step log (every log_interval) ─────────────────────────────
        if step % log_interval == 0:
            current_lr  = scheduler.get_last_lr()[0]
            rolling_tps = (
                sum(_step_tokens) / max(sum(_step_times), 1e-9)
                if _step_times else 0.0
            )
            print(f"step {step:6d} | train loss {loss.item():.4f} | "
                  f"grad_norm {grad_norm.item():.3f} | lr {current_lr:.2e}")

            if exp_train_f is not None:
                _jsonl_write(exp_train_f, {
                    "step":           step,
                    "wall_time_sec":  round(time.time() - train_start_time, 3),
                    "train_loss":     round(loss.item(), 6),
                    "lr":             current_lr,
                    "current_K":      _current_k(model, model_type, curv_schedule, step),
                    "grad_norm":      round(grad_norm.item(), 6),
                    "tokens_per_sec": round(rolling_tps, 1),
                })

    # ── Final checkpoint ───────────────────────────────────────────────────
    final_step = step
    if is_exp_a and run_dir is not None and ckpt_interval is not None:
        save_experiment_checkpoint(
            model, optimizer, scheduler, final_step,
            _current_k(model, model_type, curv_schedule, final_step),
            cfg, metrics_history, git_hash, spec_version,
            run_dir, is_final=True,
        )
        print(f"[CKPT] Final checkpoint at step {final_step}")

    # ── Legacy save ────────────────────────────────────────────────────────
    save_checkpoint(model, optimizer, final_step, cfg, log)
    save_log(log, cfg)

    # ── Compute budget metadata ────────────────────────────────────────────
    wall_time  = time.time() - train_start_time
    throughput = total_tokens / max(wall_time, 1.0)
    peak_vram_bytes = 0
    peak_vram_gb    = 0.0
    if device.type == "cuda":
        peak_vram_bytes = torch.cuda.max_memory_allocated(device)
        peak_vram_gb    = peak_vram_bytes / 1e9

    run_meta = {
        "event":                     "run_complete",
        "wall_time_seconds":         round(wall_time, 1),
        "peak_vram_gb":              round(peak_vram_gb, 2),
        "gpu_util_percent_mean":     None,
        "throughput_tok_per_sec":    round(throughput, 1),
        "compute_budget_used_hours": round(wall_time / 3600.0, 4),
    }
    jsonl_file.write(json.dumps(run_meta) + "\n")
    jsonl_file.flush()
    jsonl_file.close()

    print(f"\nRun complete: {wall_time:.1f}s | {throughput:.0f} tok/s | "
          f"peak VRAM {peak_vram_gb:.2f} GB")

    # ── Experiment A summary.json (§6.3) ───────────────────────────────────
    if is_exp_a and log_dir:
        if full_split:
            final_val_loss = evaluate_full_split(model, val_data, cfg, device)
        else:
            final_val_loss = evaluate(model, val_data, cfg, device, eval_batches)
        final_val_ppl = math.exp(final_val_loss)

        recent_ppls = [
            r["val_ppl"] for r in metrics_history
            if "val_ppl" in r and r.get("step", 0) >= max_steps - 1000
        ]
        rolling_mean_final = (sum(recent_ppls) / len(recent_ppls)) if recent_ppls else None

        total_tokens_seen = (max_steps - start_step) * batch_size * block_size
        estimated_flops   = 6 * non_emb_params * total_tokens_seen

        summary = {
            "run_name":                             run_id,
            "variant":                              model_type,
            "scale_tag":                            cfg.get("scale_tag", ""),
            "seed":                                 seed,
            "git_commit_hash":                      git_hash,
            "spec_version":                         spec_version,
            "total_tokens_seen":                    total_tokens_seen,
            "total_wallclock_sec":                  round(wall_time, 1),
            "non_embedding_body_params":            non_emb_params,
            "total_params":                         total_params,
            "estimated_flops":                      estimated_flops,
            "best_val_ppl":                         best_val_ppl,
            "best_val_step":                        best_val_step,
            "rolling_mean_val_ppl_final_1000_steps": rolling_mean_final,
            "final_val_ppl":                        final_val_ppl,
            "peak_vram_bytes":                      peak_vram_bytes,
            "kv_cache_bytes_per_token_at_seqlen_1024": kv_cache_bytes,
            "resume_events":                        resume_events,
            "status":                               status,
            "notes":                                cfg.get("notes", ""),
        }
        save_summary(log_dir, summary)
        print(f"[EXP-A] Summary → {log_dir}/summary.json")
        print(f"[EXP-A] best_val_ppl={best_val_ppl:.2f} at step {best_val_step}")

    if exp_train_f: exp_train_f.close()
    if exp_eval_f:  exp_eval_f.close()
    print("\nDone.")


if __name__ == "__main__":
    main()

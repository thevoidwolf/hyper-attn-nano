"""
train.py — Training script for HyperAttn-Nano variants
=======================================================

Usage:
    python scripts/train.py --config configs/nano_euclid.yaml
    python scripts/train.py --config configs/nano_hyper_fixed.yaml
    python scripts/train.py --config configs/nano_hyper_perhead.yaml
    python scripts/train.py --config configs/smoke_shakespeare.yaml

All behaviour is controlled by the yaml config file.
"""

import argparse
import json
import math
import os
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
# Data helpers
# ---------------------------------------------------------------------------

def load_data(data_dir: str, split: str) -> np.ndarray:
    """Return a read-only memmap over the flat token array."""
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
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, data, cfg, device, eval_batches=50) -> float:
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


# ---------------------------------------------------------------------------
# Checkpoint / log saving
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, step, cfg, log):
    variant = cfg["model"]["type"]
    path = f"results/checkpoints/{variant}/ckpt_step{step}.pt"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "step":      step,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config":    cfg,
        },
        path,
    )
    print(f"Checkpoint saved → {path}")


def save_log(log, cfg):
    run_id  = cfg.get("run_id", cfg["model"]["type"])
    log_dir = cfg.get("log_dir", "results/logs/exp_b")
    path = os.path.join(log_dir, f"{run_id}_log.json")
    os.makedirs(log_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"Log saved → {path}")


def open_jsonl_log(cfg) -> tuple[str, "IO"]:
    """Open JSONL log file. Uses log_dir from config if present."""
    run_id  = cfg.get("run_id", cfg["model"]["type"])
    log_dir = cfg.get("log_dir", "results/logs/exp_b")
    path = os.path.join(log_dir, f"{run_id}_train.jsonl")
    os.makedirs(log_dir, exist_ok=True)
    f = open(path, "w")
    return path, f


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train a HyperAttn-Nano variant")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--no-abort", action="store_true",
                        help="Disable early-abort on bad configs (for deliberate exploration)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ── Seeding ────────────────────────────────────────────────────────────
    seed = cfg["training"].get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ── Device ─────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}")
        print(
            f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )
    else:
        print("  WARNING: CUDA not available; training on CPU will be slow.")

    # ── Model ──────────────────────────────────────────────────────────────
    model_type = cfg["model"]["type"]
    model_cfg = {
        k: cfg["model"][k]
        for k in ["d_model", "n_layers", "n_heads", "d_ff", "max_seq_len", "vocab_size"]
    }
    # Propagate optional model flags — these were previously dropped here (float64 bug)
    if "manifold_float64" in cfg["model"]:
        model_cfg["manifold_float64"] = cfg["model"]["manifold_float64"]
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

    # ── Curvature schedule setup ───────────────────────────────────────────
    curv_schedule = None
    schedule_mode = "frozen_schedule"
    schedule_cfg  = cfg["model"].get("curvature_schedule", None)

    if schedule_cfg is not None:
        # Warn if both curvature and curvature_schedule are present
        if "curvature" in cfg["model"] or "init_K" in cfg["model"]:
            warnings.warn(
                "[CONFIG] Both 'curvature' and 'curvature_schedule' are set. "
                "'curvature_schedule' takes precedence.",
                UserWarning,
                stacklevel=2,
            )
        curv_schedule = build_schedule(schedule_cfg)
        schedule_mode = schedule_cfg.get("schedule_mode", "frozen_schedule")
        print(f"[CONFIG] Curvature schedule: {schedule_cfg['type']}  "
              f"k_start={schedule_cfg.get('k_start', 'N/A')}  "
              f"k_end={schedule_cfg.get('k_end', schedule_cfg.get('k_start', 'N/A'))}  "
              f"warmup={schedule_cfg.get('warmup_steps', 'N/A')}  "
              f"mode={schedule_mode}")

    # Freeze curvatures for hyper-fixed variant (or when schedule overrides)
    if model_type == "hyper-fixed" or (
        model_type in ("hyper-perhead", "hyper-scores-only")
        and curv_schedule is not None
        and schedule_mode == "frozen_schedule"
    ):
        for block in model.blocks:
            block.attn.log_abs_K.requires_grad_(False)
        if curv_schedule is None:
            print("  hyper-fixed: log_abs_K frozen (curvature will not be learned)")
            print(f"[CONFIG] Fixed curvature K = {curvature}")
        else:
            print("  curvature schedule active: log_abs_K frozen and will be overridden each step")
    elif model_type == "hyper-scores-only" and curv_schedule is None:
        # No schedule → freeze the single shared curvature
        for block in model.blocks:
            block.attn.log_abs_K.requires_grad_(False)
        print("  hyper-scores-only: log_abs_K frozen (curvature will not be learned)")
        print(f"[CONFIG] Fixed curvature K = {curvature}")

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count:,}")

    # ── Early-abort baseline ───────────────────────────────────────────────
    _BASELINE_PATH = os.path.join(
        os.path.dirname(__file__), "..", "configs", "baselines",
        "euclid_wikitext2_step500_ppl.json"
    )
    abort_threshold: float | None = None
    if not args.no_abort:
        if os.path.exists(_BASELINE_PATH):
            with open(_BASELINE_PATH) as _f:
                _baseline = json.load(_f)
            abort_threshold = _baseline["val_ppl"] * 10.0
            print(f"[ABORT] Early-abort enabled. Threshold at step 500: PPL > {abort_threshold:.1f}")
        else:
            print(f"[ABORT] Baseline file not found at {_BASELINE_PATH} — early-abort disabled.")
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
    print(
        f"  Dataset: {tcfg['dataset']}  "
        f"({len(train_data):,} train tokens, {len(val_data):,} val tokens)"
    )

    # Open JSONL log for incremental per-eval writing
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

    # ── LR schedule — cosine decay with linear warmup ──────────────────────
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── AMP scaler (CUDA only) ─────────────────────────────────────────────
    use_amp = device.type == "cuda"
    scaler  = torch.amp.GradScaler("cuda") if use_amp else None

    # ── Log structure ──────────────────────────────────────────────────────
    log = {
        "config":     cfg,
        "train_loss": [],
        "val_loss":   [],
        "curvatures": [],
    }

    print(f"\nStarting training: {model_type} | {max_steps} steps")
    print("=" * 60)

    # ── Compute budget tracking ────────────────────────────────────────────
    train_start_time = time.time()
    total_tokens_processed = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # ── Training loop ──────────────────────────────────────────────────────
    for step in range(max_steps):

        # ── Eval ──────────────────────────────────────────────────────────
        if step % eval_interval == 0:
            val_loss   = evaluate(model, val_data, cfg, device, eval_batches)
            perplexity = math.exp(val_loss)
            log["val_loss"].append(
                {"step": step, "loss": val_loss, "perplexity": perplexity}
            )
            print(f"step {step:6d} | val loss {val_loss:.4f} | ppl {perplexity:.1f}")

            # Write incremental JSONL record
            import json as _json
            jsonl_record = {
                "step":       step,
                "train_loss": last_train_loss,
                "val_ppl":    perplexity,
                "grad_norm":  last_grad_norm,
            }
            if curv_schedule is not None:
                jsonl_record["scheduled_K"] = curv_schedule.k_at_step(step)
            jsonl_file.write(_json.dumps(jsonl_record) + "\n")
            jsonl_file.flush()

            # Curvature snapshot (HyperAttnNano only)
            if hasattr(model, "get_curvatures"):
                curvs = model.get_curvatures()
                if curvs:
                    log["curvatures"].append({"step": step, "values": curvs})
                    vals = list(curvs.values())
                    print(
                        f"  curvature: min={min(vals):.3f}  max={max(vals):.3f}  "
                        f"std={torch.tensor(vals).std().item():.3f}"
                    )

            # ── Early-abort check at step 500 ─────────────────────────────
            if step == 500 and abort_threshold is not None:
                if perplexity > abort_threshold:
                    reason = (
                        f"val_ppl={perplexity:.1f} > {abort_threshold:.1f} "
                        f"(10× Euclidean baseline at step 500)"
                    )
                    print(f"[ABORT] Early abort at step {step}: {reason}")
                    import json as _json2
                    jsonl_file.write(_json2.dumps({
                        "event": "ABORTED",
                        "step":  step,
                        "reason": reason,
                    }) + "\n")
                    jsonl_file.flush()
                    jsonl_file.close()
                    save_log(log, cfg)
                    return

        # ── Apply curvature schedule ──────────────────────────────────────
        if curv_schedule is not None and hasattr(model, "blocks"):
            k_now = curv_schedule.k_at_step(step)
            import math as _math
            log_abs_k = _math.log(abs(k_now))
            warmup_done = step >= schedule_cfg.get("warmup_steps", 0)
            with torch.no_grad():
                for block in model.blocks:
                    if hasattr(block, "attn") and hasattr(block.attn, "log_abs_K"):
                        block.attn.log_abs_K.fill_(log_abs_k)
            # For init_only mode: release to optimizer after warmup
            if schedule_mode == "init_only" and warmup_done:
                for block in model.blocks:
                    if hasattr(block, "attn") and hasattr(block.attn, "log_abs_K"):
                        block.attn.log_abs_K.requires_grad_(True)

        # ── Train step ────────────────────────────────────────────────────
        model.train()
        x, y = get_batch(train_data, block_size, batch_size, device)

        if use_amp:
            with torch.amp.autocast("cuda"):
                logits, loss = model(x, y)
            if not torch.isfinite(loss):
                print(f"[WARN] Non-finite loss at step {step}: {loss.item()}. Saving checkpoint and stopping.")
                save_checkpoint(model, optimizer, step, cfg, log)
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
                print(f"[WARN] Non-finite loss at step {step}: {loss.item()}. Saving checkpoint and stopping.")
                save_checkpoint(model, optimizer, step, cfg, log)
                break
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if grad_norm.item() > 1000.0:
                print(f"[WARN] High grad norm at step {step}: {grad_norm.item():.1f}")
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        last_train_loss = loss.item()
        last_grad_norm  = grad_norm.item()
        total_tokens_processed += batch_size * block_size
        log["train_loss"].append(
            {"step": step, "loss": loss.item(), "grad_norm": grad_norm.item()}
        )

        if step % log_interval == 0:
            print(
                f"step {step:6d} | train loss {loss.item():.4f} | "
                f"grad_norm {grad_norm.item():.3f} | "
                f"lr {scheduler.get_last_lr()[0]:.2e}"
            )

    # ── Save checkpoint and log ───────────────────────────────────────────
    save_checkpoint(model, optimizer, step, cfg, log)
    save_log(log, cfg)

    # ── Compute budget run_meta ───────────────────────────────────────────
    wall_time = time.time() - train_start_time
    throughput = total_tokens_processed / max(wall_time, 1.0)

    peak_vram_gb = 0.0
    if device.type == "cuda":
        peak_vram_gb = torch.cuda.max_memory_allocated(device) / 1e9

    run_meta = {
        "event":                  "run_complete",
        "wall_time_seconds":      round(wall_time, 1),
        "peak_vram_gb":           round(peak_vram_gb, 2),
        "gpu_util_percent_mean":  None,   # requires pynvml polling — not tracked
        "throughput_tok_per_sec": round(throughput, 1),
        "compute_budget_used_hours": round(wall_time / 3600.0, 4),
    }
    import json as _json_meta
    jsonl_file.write(_json_meta.dumps(run_meta) + "\n")
    jsonl_file.flush()
    jsonl_file.close()

    print(f"\nRun complete: {wall_time:.1f}s | {throughput:.0f} tok/s | "
          f"peak VRAM {peak_vram_gb:.2f} GB")
    print("\nDone.")


if __name__ == "__main__":
    main()

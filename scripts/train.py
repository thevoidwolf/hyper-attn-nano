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

import numpy as np
import torch
import yaml

# Make src/ importable when running from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from model import HyperAttnNano, GPTNano  # noqa: E402


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
    variant = cfg["model"]["type"]
    path = f"results/logs/{variant}_log.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"Log saved → {path}")


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train a HyperAttn-Nano variant")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
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
    init_K = cfg["model"].get("init_K", -1.0)

    if model_type == "euclid":
        model = GPTNano(model_cfg)
    else:
        model = HyperAttnNano(model_cfg, init_K=init_K)

    model = model.to(device)

    # Freeze curvatures for hyper-fixed variant
    if model_type == "hyper-fixed":
        for block in model.blocks:
            block.attn.log_abs_K.requires_grad_(False)
        print("  hyper-fixed: log_abs_K frozen (curvature will not be learned)")

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count:,}")

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

        # ── Train step ────────────────────────────────────────────────────
        model.train()
        x, y = get_batch(train_data, block_size, batch_size, device)

        if use_amp:
            with torch.amp.autocast("cuda"):
                logits, loss = model(x, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, loss = model(x, y)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

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
    print("\nDone.")


if __name__ == "__main__":
    main()

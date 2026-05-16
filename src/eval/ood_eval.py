"""
ood_eval.py — Out-of-distribution and extended evaluation pack
==============================================================

Produces four metrics for any trained checkpoint:

  1. id_val_ppl        — In-distribution validation perplexity (existing val split).
  2. ood_val_ppl       — OOD text perplexity:
                           • WikiText-2 models: first 100K tokens of a held-out text
                             file at data/ood/<ood_dataset>/ood.bin
                           • Code-trained models: held-out code tokens at
                             data/ood/<ood_dataset>/ood.bin
  3. rare_word_ppl     — Average per-token PPL on bottom-decile-frequency tokens.
                         Frequency computed from the TRAINING set (not val).
  4. long_ctx_ppl      — Val PPL at 1× / 2× / 4× the trained sequence length.
                         Models with fixed positional encodings are evaluated by
                         truncating to the trained length (no error).

Output format:
  {
    "id_val_ppl": 277.0,
    "ood_val_ppl": 312.4,
    "rare_word_ppl": 1843.2,
    "long_ctx_ppl": {"256": 277.0, "512": 295.1, "1024": 342.7}
  }

IMPORTANT: rare_word_ppl uses train-set frequencies, not val frequencies.
           Using val frequencies is a common subtle bug.
"""

import math
import os
from collections import Counter
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Token-level PPL helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def _eval_ppl_on_tokens(
    model,
    tokens: np.ndarray,
    block_size: int,
    batch_size: int,
    device: torch.device,
    max_batches: Optional[int] = None,
    per_token: bool = False,
) -> tuple[float, Optional[list[tuple[int, float]]]]:
    """
    Evaluate perplexity on a flat token array.

    Args:
        model      : language model with forward(input_ids, targets) -> (logits, loss)
        tokens     : flat np.ndarray of token ids
        block_size : context window length
        batch_size : evaluation batch size
        device     : torch device
        max_batches: if set, only evaluate this many batches
        per_token  : if True, also return list of (token_id, loss) pairs

    Returns:
        (ppl, per_token_losses_or_None)
        per_token_losses: list of (token_id, cross_entropy_loss) for the last
                          token in each context window evaluated.
    """
    model.eval()
    total_loss = 0.0
    n_batches  = 0
    per_token_losses: list[tuple[int, float]] = [] if per_token else None  # type: ignore

    # Non-overlapping ordered windows: each batch covers batch_size * block_size
    # *distinct* tokens. n_full = number of complete non-overlapping batches.
    # Earlier versions advanced starts by 1 token between windows within a batch
    # and by batch_size tokens between batches, so 50 batches covered only
    # ~1856 tokens of a 248K val set — making id_val_ppl meaningless.
    stride_within_batch = block_size
    stride_between_batches = batch_size * block_size
    n_full = len(tokens) // stride_between_batches
    if max_batches is not None:
        n_full = min(n_full, max_batches)

    for batch_idx in range(n_full):
        base = batch_idx * stride_between_batches
        starts = [base + i * stride_within_batch for i in range(batch_size)]
        # Skip if any start index goes out of range
        if any(s + block_size + 1 > len(tokens) for s in starts):
            break

        x = torch.stack([
            torch.from_numpy(tokens[s : s + block_size].astype(np.int64))
            for s in starts
        ]).to(device)
        y = torch.stack([
            torch.from_numpy(tokens[s + 1 : s + block_size + 1].astype(np.int64))
            for s in starts
        ]).to(device)

        if device.type == "cuda":
            with torch.amp.autocast("cuda"):
                logits, loss = model(x, y)
        else:
            logits, loss = model(x, y)

        total_loss += loss.item()
        n_batches  += 1

        if per_token:
            # Record (token_id, per-position loss) for EVERY position in the
            # batch, not just the last one. The previous last-token-only logic
            # produced only batch_size samples per batch (~1.6K for a typical
            # eval), too few to estimate rare-word PPL: most batches hit zero
            # rare tokens and the final mean was NaN.
            with torch.no_grad():
                B, S, V = logits.shape
                per_pos_loss = F.cross_entropy(
                    logits.view(-1, V), y.view(-1), reduction="none"
                ).view(B, S)
                all_token_ids = y.cpu().numpy().reshape(-1)
                all_losses    = per_pos_loss.cpu().numpy().reshape(-1)
                for tid, l in zip(all_token_ids, all_losses):
                    per_token_losses.append((int(tid), float(l)))  # type: ignore

    if n_batches == 0:
        return float("nan"), per_token_losses

    avg_loss = total_loss / n_batches
    ppl      = math.exp(avg_loss)
    return ppl, per_token_losses


# ---------------------------------------------------------------------------
# Token frequency from training data
# ---------------------------------------------------------------------------

def compute_train_token_frequencies(
    train_tokens: np.ndarray,
    max_tokens: int = 5_000_000,
) -> Counter:
    """
    Count token frequencies in the training data.

    IMPORTANT: always use TRAIN data, not val data.
    Using val frequencies for the rare-word metric is a common bug.

    Args:
        train_tokens: flat np.ndarray of token ids
        max_tokens  : read at most this many tokens (for large datasets)

    Returns:
        Counter mapping token_id -> count
    """
    n = min(len(train_tokens), max_tokens)
    return Counter(train_tokens[:n].tolist())


def rare_word_threshold(freq: Counter, decile: float = 0.1) -> int:
    """
    Return the frequency count below which a token is considered "rare".

    Rare = bottom `decile` fraction of token types by frequency.
    Default: bottom 10% of *types* (not occurrences).

    Args:
        freq  : Counter of token_id -> count
        decile: fraction of the lower tail (0.0 to 1.0)

    Returns:
        threshold count; tokens with count <= threshold are rare
    """
    counts = sorted(freq.values())
    idx    = max(0, int(len(counts) * decile) - 1)
    return counts[idx]


# ---------------------------------------------------------------------------
# Long-context evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_long_context(
    model,
    val_tokens: np.ndarray,
    trained_block_size: int,
    batch_size: int,
    device: torch.device,
    multipliers: tuple[int, ...] = (1, 2, 4, 8, 16),
    max_batches: int = 50,
) -> dict[str, float]:
    """
    Evaluate val PPL at 1× / 2× / 4× the trained sequence length.

    For models that do not support longer contexts (fixed positional encodings),
    longer contexts are TRUNCATED to the trained length — no error is raised.

    Args:
        model             : language model
        val_tokens        : flat np.ndarray of val token ids
        trained_block_size: the block_size the model was trained with
        batch_size        : eval batch size (may be reduced for longer contexts)
        device            : torch device
        multipliers       : context length multipliers to evaluate
        max_batches       : number of batches per multiplier

    Returns:
        dict mapping str(context_length) -> ppl
    """
    results = {}
    for mult in multipliers:
        ctx = trained_block_size * mult
        # If the model has max_seq_len, clamp to it (graceful truncation)
        effective_ctx = min(ctx, getattr(model, 'max_seq_len', ctx))
        # Reduce batch size for longer contexts to avoid OOM
        eff_batch = max(1, batch_size // mult)

        ppl, _ = _eval_ppl_on_tokens(
            model, val_tokens,
            block_size=effective_ctx,
            batch_size=eff_batch,
            device=device,
            max_batches=max_batches,
        )
        results[str(ctx)] = ppl

    return results


# ---------------------------------------------------------------------------
# Main eval function
# ---------------------------------------------------------------------------

def run_eval(
    model,
    train_tokens:      np.ndarray,
    val_tokens:        np.ndarray,
    ood_tokens:        Optional[np.ndarray],
    trained_block_size: int,
    batch_size:        int,
    device:            torch.device,
    eval_batches:      int = 50,
) -> dict:
    """
    Run the full evaluation pack and return a results dict.

    Args:
        model              : trained language model
        train_tokens       : flat training token array (used for freq computation)
        val_tokens         : flat val token array
        ood_tokens         : flat OOD token array, or None to skip OOD eval
        trained_block_size : context window the model was trained with
        batch_size         : eval batch size
        device             : torch device
        eval_batches       : number of batches for standard evals

    Returns:
        dict with keys: id_val_ppl, ood_val_ppl, rare_word_ppl, long_ctx_ppl
    """
    results: dict = {}

    # 1. In-distribution val PPL
    id_ppl, _ = _eval_ppl_on_tokens(
        model, val_tokens,
        block_size=trained_block_size,
        batch_size=batch_size,
        device=device,
        max_batches=eval_batches,
    )
    results["id_val_ppl"] = id_ppl

    # 2. OOD val PPL
    if ood_tokens is not None and len(ood_tokens) > trained_block_size + 1:
        ood_ppl, _ = _eval_ppl_on_tokens(
            model, ood_tokens,
            block_size=trained_block_size,
            batch_size=batch_size,
            device=device,
            max_batches=eval_batches,
        )
        results["ood_val_ppl"] = ood_ppl
    else:
        results["ood_val_ppl"] = None

    # 3. Rare-word PPL
    #    Frequencies from TRAIN set (critical — not val)
    train_freq = compute_train_token_frequencies(train_tokens)
    rare_thresh = rare_word_threshold(train_freq, decile=0.1)
    rare_token_ids = {tid for tid, cnt in train_freq.items() if cnt <= rare_thresh}

    _, per_tok = _eval_ppl_on_tokens(
        model, val_tokens,
        block_size=trained_block_size,
        batch_size=batch_size,
        device=device,
        max_batches=eval_batches,
        per_token=True,
    )

    rare_losses = [loss for tid, loss in per_tok if tid in rare_token_ids]  # type: ignore
    if rare_losses:
        avg_rare_loss = sum(rare_losses) / len(rare_losses)
        results["rare_word_ppl"] = math.exp(avg_rare_loss)
    else:
        results["rare_word_ppl"] = float("nan")

    # 4. Long-context PPL at 1× / 2× / 4×
    results["long_ctx_ppl"] = eval_long_context(
        model, val_tokens,
        trained_block_size=trained_block_size,
        batch_size=batch_size,
        device=device,
        max_batches=eval_batches,
    )

    return results

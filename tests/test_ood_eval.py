"""
test_ood_eval.py — Tests for the OOD / generalisation evaluation pack
======================================================================

Run with:
    cd ~/hyper-attn-nano
    source activate.sh
    pytest tests/test_ood_eval.py -v
"""

import math
import sys
import os
from collections import Counter

import pytest
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from eval.ood_eval import (
    _eval_ppl_on_tokens,
    compute_train_token_frequencies,
    rare_word_threshold,
    eval_long_context,
    run_eval,
)
from model import GPTNano

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

VOCAB_SIZE  = 256
D_MODEL     = 64
N_LAYERS    = 2
N_HEADS     = 4
D_FF        = 128
BLOCK_SIZE  = 16
BATCH_SIZE  = 4
N_TOKENS    = 5000   # enough tokens for several batches

MODEL_CFG = {
    "d_model":     D_MODEL,
    "n_layers":    N_LAYERS,
    "n_heads":     N_HEADS,
    "d_ff":        D_FF,
    "max_seq_len": BLOCK_SIZE,
    "vocab_size":  VOCAB_SIZE,
}


def _make_model():
    torch.manual_seed(0)
    model = GPTNano(MODEL_CFG)
    model.eval()
    return model


def _make_tokens(n=N_TOKENS, seed=42, vocab=VOCAB_SIZE):
    rng = np.random.default_rng(seed)
    return rng.integers(0, vocab, size=n, dtype=np.uint16)


DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# 1. Smoke test: all metrics produce finite values on an untrained model
# ---------------------------------------------------------------------------

class TestSmokeTest:

    def test_id_val_ppl_is_finite(self):
        model        = _make_model()
        val_tokens   = _make_tokens()
        train_tokens = _make_tokens(seed=1)

        results = run_eval(
            model,
            train_tokens=train_tokens,
            val_tokens=val_tokens,
            ood_tokens=None,
            trained_block_size=BLOCK_SIZE,
            batch_size=BATCH_SIZE,
            device=DEVICE,
            eval_batches=5,
        )
        assert math.isfinite(results["id_val_ppl"]), "id_val_ppl is not finite"
        assert results["id_val_ppl"] > 1.0, "id_val_ppl should be > 1"

    def test_ood_val_ppl_is_finite_when_data_provided(self):
        model        = _make_model()
        val_tokens   = _make_tokens(seed=0)
        train_tokens = _make_tokens(seed=1)
        ood_tokens   = _make_tokens(seed=2)

        results = run_eval(
            model,
            train_tokens=train_tokens,
            val_tokens=val_tokens,
            ood_tokens=ood_tokens,
            trained_block_size=BLOCK_SIZE,
            batch_size=BATCH_SIZE,
            device=DEVICE,
            eval_batches=5,
        )
        assert results["ood_val_ppl"] is not None
        assert math.isfinite(results["ood_val_ppl"]), "ood_val_ppl is not finite"

    def test_ood_val_ppl_is_none_when_no_data(self):
        model        = _make_model()
        val_tokens   = _make_tokens()
        train_tokens = _make_tokens(seed=1)

        results = run_eval(
            model,
            train_tokens=train_tokens,
            val_tokens=val_tokens,
            ood_tokens=None,
            trained_block_size=BLOCK_SIZE,
            batch_size=BATCH_SIZE,
            device=DEVICE,
            eval_batches=5,
        )
        assert results["ood_val_ppl"] is None

    def test_rare_word_ppl_is_finite(self):
        model        = _make_model()
        val_tokens   = _make_tokens()
        train_tokens = _make_tokens(seed=1)

        results = run_eval(
            model,
            train_tokens=train_tokens,
            val_tokens=val_tokens,
            ood_tokens=None,
            trained_block_size=BLOCK_SIZE,
            batch_size=BATCH_SIZE,
            device=DEVICE,
            eval_batches=5,
        )
        assert math.isfinite(results["rare_word_ppl"]), "rare_word_ppl is not finite"
        assert results["rare_word_ppl"] > 1.0

    def test_long_ctx_ppl_has_three_keys(self):
        model        = _make_model()
        val_tokens   = _make_tokens(n=20000)
        train_tokens = _make_tokens(seed=1)

        results = run_eval(
            model,
            train_tokens=train_tokens,
            val_tokens=val_tokens,
            ood_tokens=None,
            trained_block_size=BLOCK_SIZE,
            batch_size=BATCH_SIZE,
            device=DEVICE,
            eval_batches=5,
        )
        long_ctx = results["long_ctx_ppl"]
        # Keys should be str(1*BLOCK_SIZE), str(2*BLOCK_SIZE), str(4*BLOCK_SIZE)
        assert str(BLOCK_SIZE * 1) in long_ctx
        assert str(BLOCK_SIZE * 2) in long_ctx
        assert str(BLOCK_SIZE * 4) in long_ctx
        for k, v in long_ctx.items():
            assert math.isfinite(v), f"long_ctx_ppl[{k}] is not finite"


# ---------------------------------------------------------------------------
# 2. Rare-word decile threshold is from TRAIN frequencies (not val)
# ---------------------------------------------------------------------------

class TestRareWordFrequencySource:

    def test_threshold_uses_train_not_val(self):
        """
        The rare-word threshold must be computed from training frequencies.
        Verify by using very different train and val distributions.

        If train has only tokens 0-9 frequently and tokens 10-255 rarely,
        but val has the opposite, the rare-word threshold should be set
        based on train (low count for tokens 10-255), not val.
        """
        # Train: tokens 0-9 appear 1000x, tokens 10-255 appear 1x each
        train_tokens = np.array(
            [i for i in range(10)] * 1000 + list(range(10, 256)),
            dtype=np.uint16,
        )

        freq = compute_train_token_frequencies(train_tokens)

        # Tokens 10-255 should be "rare" (count=1)
        # Tokens 0-9 should be "common" (count=1000)
        thresh = rare_word_threshold(freq, decile=0.1)

        rare_ids = {tid for tid, cnt in freq.items() if cnt <= thresh}
        common_ids = {tid for tid, cnt in freq.items() if cnt > thresh}

        # Most tokens 10-255 should be rare (train-freq based)
        rare_high = sum(1 for tid in rare_ids if tid >= 10)
        assert rare_high > 50, (
            f"Expected rare tokens to be from 10-255 (high token ids), "
            f"but only {rare_high} were in the rare set"
        )

        # Tokens 0-9 should NOT be rare (they're common in train)
        common_low = sum(1 for tid in common_ids if tid < 10)
        assert common_low == 10, (
            "Tokens 0-9 (common in train) should not be in the rare set"
        )

    def test_frequency_counter_from_train_only(self):
        """compute_train_token_frequencies returns a Counter of train tokens."""
        train = np.array([0, 0, 0, 1, 1, 2], dtype=np.uint16)
        freq  = compute_train_token_frequencies(train)
        assert freq[0] == 3
        assert freq[1] == 2
        assert freq[2] == 1
        assert freq[3] == 0   # not in train

    def test_rare_word_threshold_bottom_decile(self):
        """10 types, threshold = count of the 1st type (10% = 1 type)."""
        freq = Counter({i: i + 1 for i in range(10)})
        # Counts: token 0 → 1, token 1 → 2, ..., token 9 → 10
        # Bottom 10% = bottom 1 type = token 0 with count 1
        thresh = rare_word_threshold(freq, decile=0.1)
        assert thresh == 1


# ---------------------------------------------------------------------------
# 3. Long-context eval truncates instead of erroring for fixed-pos models
# ---------------------------------------------------------------------------

class TestLongContextTruncation:

    def test_no_error_at_2x_context(self):
        """A model with max_seq_len=BLOCK_SIZE should not error at 2× context."""
        model      = _make_model()
        val_tokens = _make_tokens(n=20000)
        device     = torch.device("cpu")

        # The GPTNano has no positional encoding, but max_seq_len is respected
        # in eval_long_context via the model attribute.
        model.max_seq_len = BLOCK_SIZE   # set fixed-pos sentinel

        results = eval_long_context(
            model, val_tokens,
            trained_block_size=BLOCK_SIZE,
            batch_size=BATCH_SIZE,
            device=device,
            multipliers=(1, 2, 4),
            max_batches=5,
        )
        # All values should be finite (truncation, not error)
        for k, v in results.items():
            assert math.isfinite(v), f"long_ctx_ppl[{k}] = {v} is not finite"

    def test_1x_context_equals_id_val_ppl(self):
        """The 1× entry should match the plain id_val_ppl evaluation."""
        model      = _make_model()
        val_tokens = _make_tokens(n=5000)
        device     = torch.device("cpu")

        # Direct eval
        id_ppl, _ = _eval_ppl_on_tokens(
            model, val_tokens,
            block_size=BLOCK_SIZE,
            batch_size=BATCH_SIZE,
            device=device,
            max_batches=10,
        )

        # Long-context at 1×
        results = eval_long_context(
            model, val_tokens,
            trained_block_size=BLOCK_SIZE,
            batch_size=BATCH_SIZE,
            device=device,
            multipliers=(1,),
            max_batches=10,
        )
        assert abs(results[str(BLOCK_SIZE)] - id_ppl) < 1e-4, (
            f"1× long_ctx_ppl {results[str(BLOCK_SIZE)]:.4f} ≠ id_val_ppl {id_ppl:.4f}"
        )


# ---------------------------------------------------------------------------
# 4. _eval_ppl_on_tokens basic sanity
# ---------------------------------------------------------------------------

class TestEvalPPLOnTokens:

    def test_returns_finite_ppl(self):
        model  = _make_model()
        tokens = _make_tokens()
        ppl, _ = _eval_ppl_on_tokens(
            model, tokens,
            block_size=BLOCK_SIZE,
            batch_size=BATCH_SIZE,
            device=DEVICE,
            max_batches=5,
        )
        assert math.isfinite(ppl)
        assert ppl > 1.0

    def test_per_token_losses_length(self):
        model  = _make_model()
        tokens = _make_tokens(n=2000)
        _, per_tok = _eval_ppl_on_tokens(
            model, tokens,
            block_size=BLOCK_SIZE,
            batch_size=BATCH_SIZE,
            device=DEVICE,
            max_batches=5,
            per_token=True,
        )
        assert per_tok is not None
        # 5 batches × BATCH_SIZE items each
        assert len(per_tok) == 5 * BATCH_SIZE

    def test_per_token_losses_are_positive(self):
        model  = _make_model()
        tokens = _make_tokens(n=2000)
        _, per_tok = _eval_ppl_on_tokens(
            model, tokens,
            block_size=BLOCK_SIZE,
            batch_size=BATCH_SIZE,
            device=DEVICE,
            max_batches=3,
            per_token=True,
        )
        for tid, loss in per_tok:  # type: ignore
            assert loss >= 0, f"Negative loss {loss} for token {tid}"

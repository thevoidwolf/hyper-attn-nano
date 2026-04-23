"""
test_spherical_head.py — Tests for SphericalOutputHead
=======================================================

Run with:
    cd ~/hyper-attn-nano
    source activate.sh
    pytest tests/test_spherical_head.py -v
"""

import math
import sys
import os

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from output_head import SphericalOutputHead

# ---------------------------------------------------------------------------
# Shared config — small for fast CPU tests
# ---------------------------------------------------------------------------
D_MODEL    = 64
VOCAB_SIZE = 128
B, T       = 2, 16


def _hidden(seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(B, T, D_MODEL)


# ---------------------------------------------------------------------------
# 1. Output shape
# ---------------------------------------------------------------------------

class TestOutputShape:

    def test_shape_matches_standard_head(self):
        """(B, T, D) in → (B, T, V) out — same shape as nn.Linear lm_head."""
        head   = SphericalOutputHead(D_MODEL, VOCAB_SIZE)
        logits = head(_hidden())
        assert logits.shape == (B, T, VOCAB_SIZE)

    def test_shape_single_token(self):
        head   = SphericalOutputHead(D_MODEL, VOCAB_SIZE)
        logits = head(torch.randn(B, 1, D_MODEL))
        assert logits.shape == (B, 1, VOCAB_SIZE)


# ---------------------------------------------------------------------------
# 2. Temperature behaviour
# ---------------------------------------------------------------------------

class TestTemperatureBehaviour:

    def test_high_temperature_gives_sharp_distribution(self):
        """At τ=100, peak probability should dominate."""
        head   = SphericalOutputHead(D_MODEL, VOCAB_SIZE, temperature_init=100.0)
        probs  = F.softmax(head(_hidden()), dim=-1)
        max_p  = probs.max(dim=-1).values
        # With very high temperature the max class should hold most of the mass
        assert max_p.mean().item() > 0.5

    def test_low_temperature_gives_more_uniform_distribution(self):
        """τ≈1 should give higher entropy than τ=100 for the same weights."""
        head_hot  = SphericalOutputHead(D_MODEL, VOCAB_SIZE, temperature_init=100.0)
        head_cold = SphericalOutputHead(D_MODEL, VOCAB_SIZE, temperature_init=1.0)
        # Give both heads identical embedding weights for a fair comparison
        with torch.no_grad():
            head_cold.embedding.copy_(head_hot.embedding)
        x = _hidden()
        probs_hot  = F.softmax(head_hot(x),  dim=-1)
        probs_cold = F.softmax(head_cold(x), dim=-1)
        entropy_hot  = -(probs_hot  * probs_hot.clamp(min=1e-9).log()).sum(-1).mean()
        entropy_cold = -(probs_cold * probs_cold.clamp(min=1e-9).log()).sum(-1).mean()
        assert entropy_cold.item() > entropy_hot.item(), (
            f"Expected cold head (τ=1) to have higher entropy than hot head (τ=100), "
            f"got entropy_cold={entropy_cold:.4f}, entropy_hot={entropy_hot:.4f}"
        )


# ---------------------------------------------------------------------------
# 3. Unit-norm invariance
# ---------------------------------------------------------------------------

class TestUnitNormInvariance:

    def test_output_unchanged_when_h_is_rescaled(self):
        """Multiplying h by any positive scalar should not change logits."""
        head    = SphericalOutputHead(D_MODEL, VOCAB_SIZE)
        x       = _hidden()
        logits1 = head(x)
        logits2 = head(x * 3.7)
        assert torch.allclose(logits1, logits2, atol=1e-5), (
            "Logits changed when h was rescaled — unit-norm invariance broken"
        )

    def test_negative_scale_flips_direction_changes_logits(self):
        """Negating h changes direction; logits should differ (sanity check)."""
        head    = SphericalOutputHead(D_MODEL, VOCAB_SIZE)
        x       = _hidden()
        logits1 = head(x)
        logits2 = head(-x)
        assert not torch.allclose(logits1, logits2, atol=1e-3), (
            "Logits were identical for h and -h — something is wrong with normalisation"
        )


# ---------------------------------------------------------------------------
# 4. Gradient flow
# ---------------------------------------------------------------------------

class TestGradientFlow:

    def test_gradients_flow_to_embedding_and_temperature(self):
        head   = SphericalOutputHead(D_MODEL, VOCAB_SIZE)
        logits = head(_hidden())
        loss   = logits.mean()
        loss.backward()
        assert head.embedding.grad is not None
        assert head.log_temperature.grad is not None
        assert not torch.isnan(head.embedding.grad).any()
        assert not torch.isnan(head.log_temperature.grad).any()


# ---------------------------------------------------------------------------
# 5. Numerical stability
# ---------------------------------------------------------------------------

class TestNumericalStability:

    def test_no_nan_at_high_temperature(self):
        head   = SphericalOutputHead(D_MODEL, VOCAB_SIZE, temperature_init=1000.0)
        logits = head(_hidden())
        assert not torch.isnan(logits).any(), "NaN in logits at high temperature"
        assert not torch.isinf(logits).any(), "Inf in logits at high temperature"

    def test_no_nan_with_all_zero_input(self):
        """F.normalize returns zero for a zero vector; logits should still be finite."""
        head   = SphericalOutputHead(D_MODEL, VOCAB_SIZE)
        x      = torch.zeros(B, T, D_MODEL)
        logits = head(x)
        assert not torch.isnan(logits).any(), "NaN in logits for all-zero h"

    def test_invalid_temperature_raises(self):
        with pytest.raises(ValueError):
            SphericalOutputHead(D_MODEL, VOCAB_SIZE, temperature_init=0.0)
        with pytest.raises(ValueError):
            SphericalOutputHead(D_MODEL, VOCAB_SIZE, temperature_init=-5.0)


# ---------------------------------------------------------------------------
# 6. Model integration — SphericalOutputHead wired into HyperAttnNano
# ---------------------------------------------------------------------------

class TestModelIntegration:

    def _model_cfg(self) -> dict:
        return {
            "d_model":     64,
            "n_layers":    2,
            "n_heads":     2,
            "d_ff":        128,
            "max_seq_len": 16,
            "vocab_size":  256,
            "output_head": "spherical",
            "spherical_temperature_init": 10.0,
        }

    def test_hyper_attn_nano_spherical_forward_shape(self):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        from model import HyperAttnNano
        cfg    = self._model_cfg()
        model  = HyperAttnNano(cfg, fixed_curvature=-10.0)
        ids    = torch.randint(0, cfg["vocab_size"], (2, 16))
        logits, loss = model(ids, ids)
        assert logits.shape == (2, 16, cfg["vocab_size"])
        assert loss is not None
        assert torch.isfinite(loss)

    def test_hyper_attn_nano_spherical_no_weight_tying(self):
        """SphericalOutputHead has its own embedding — not tied to token embed."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        from model import HyperAttnNano
        cfg   = self._model_cfg()
        model = HyperAttnNano(cfg, fixed_curvature=-10.0)
        assert isinstance(model.lm_head, SphericalOutputHead)
        # The spherical head's embedding is a separate Parameter, not the token embed
        assert model.lm_head.embedding is not model.embed.weight

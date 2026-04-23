"""
test_config_curvature.py — Tests for fixed_curvature parameter wiring
======================================================================

Verifies that:
  1. HyperAttnNano(cfg, fixed_curvature=-5.0) initialises all attention heads
     with K == -5.0 (checked via block.attn.K property).
  2. HyperAttnNano(cfg) without fixed_curvature defaults to K == -1.0.
  3. init_K still works for backward compatibility.
  4. fixed_curvature does NOT freeze gradients — train.py handles that.

Run with:
    cd ~/hyper-attn-nano
    source activate.sh
    pytest tests/test_config_curvature.py -v
"""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from model import HyperAttnNano, NANO_CONFIG


# ---------------------------------------------------------------------------
# Test 1: fixed_curvature is stored and reflected in all attention heads
# ---------------------------------------------------------------------------

def test_fixed_curvature_stored_on_model():
    model = HyperAttnNano(NANO_CONFIG, fixed_curvature=-5.0)
    assert hasattr(model, "fixed_curvature")
    assert model.fixed_curvature == -5.0


@pytest.mark.parametrize("K_val", [-1.0, -2.0, -5.0, -10.0, -50.0])
def test_fixed_curvature_reflected_in_attention(K_val):
    """Each attention head starts at the requested curvature K_val."""
    model = HyperAttnNano(NANO_CONFIG, fixed_curvature=K_val)
    for i, block in enumerate(model.blocks):
        K = block.attn.K   # shape: (n_heads,)
        expected = torch.full_like(K, K_val)
        assert torch.allclose(K, expected, atol=1e-5), (
            f"Layer {i}: expected K={K_val}, got {K.tolist()}"
        )


# ---------------------------------------------------------------------------
# Test 2: Default K is -1.0 when fixed_curvature is not specified
# ---------------------------------------------------------------------------

def test_default_curvature_is_minus_one():
    model = HyperAttnNano(NANO_CONFIG)
    for i, block in enumerate(model.blocks):
        K = block.attn.K
        expected = torch.full_like(K, -1.0)
        assert torch.allclose(K, expected, atol=1e-5), (
            f"Layer {i}: expected default K=-1.0, got {K.tolist()}"
        )


def test_init_K_backward_compat():
    """init_K keyword still works (used by existing configs and tests)."""
    model = HyperAttnNano(NANO_CONFIG, init_K=-2.0)
    for block in model.blocks:
        K = block.attn.K
        assert torch.allclose(K, torch.full_like(K, -2.0), atol=1e-5)


# ---------------------------------------------------------------------------
# Test 3: fixed_curvature does NOT freeze gradients — train.py's job
# ---------------------------------------------------------------------------

def test_curvatures_are_learnable_after_construction():
    """
    After construction, log_abs_K must still have requires_grad=True.
    The hyper-perhead variant relies on this — curvatures stay learnable.
    train.py explicitly freezes them for hyper-fixed at runtime.
    """
    model = HyperAttnNano(NANO_CONFIG, fixed_curvature=-5.0)
    for i, block in enumerate(model.blocks):
        assert block.attn.log_abs_K.requires_grad, (
            f"Layer {i}: log_abs_K.requires_grad is False after construction; "
            "fixed_curvature must not freeze parameters"
        )


def test_gradient_flows_to_log_abs_K_at_high_curvature():
    """After forward+backward at K=-5.0, log_abs_K receives a finite gradient."""
    model = HyperAttnNano(NANO_CONFIG, fixed_curvature=-5.0)
    torch.manual_seed(0)
    ids = torch.randint(0, NANO_CONFIG["vocab_size"], (2, 16))
    _, loss = model(ids, ids)
    loss.backward()
    for i, block in enumerate(model.blocks):
        g = block.attn.log_abs_K.grad
        assert g is not None, f"Layer {i}: no gradient for log_abs_K at K=-5"
        assert torch.isfinite(g).all(), (
            f"Layer {i}: non-finite gradient for log_abs_K at K=-5: {g}"
        )


# ---------------------------------------------------------------------------
# Sanity: fixed_curvature attribute roundtrip
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("K_val", [-1.0, -2.0, -10.0])
def test_fixed_curvature_attribute_matches_init(K_val):
    model = HyperAttnNano(NANO_CONFIG, fixed_curvature=K_val)
    assert model.fixed_curvature == K_val

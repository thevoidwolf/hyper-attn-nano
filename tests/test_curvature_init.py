"""
test_curvature_init.py — Validate curvature_init wiring in attention module
============================================================================

Tests:
  1. HyperAttnNano with curvature_init=-10.0 has all log_abs_K ≈ log(10) ± 1e-6
  2. Those parameters have requires_grad=True (learnable)
  3. hyper-fixed construction (fixed_curvature=-10.0 + requires_grad_(False))
     is unaffected by the curvature_init field
  4. HyperAttnNano() with no curvature_init defaults to K=-1.0 (backward compat)

Run with:
    cd ~/hyper-attn-nano
    source activate.sh
    pytest tests/test_curvature_init.py -v
"""

import math
import sys
import os

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from model import HyperAttnNano, NANO_CONFIG


# ---------------------------------------------------------------------------
# Test 1: curvature_init=-10.0 → all log_abs_K == log(10)
# ---------------------------------------------------------------------------

def test_curvature_init_k10_sets_log_abs_K():
    """All log_abs_K parameters must equal log(10) when curvature_init=-10."""
    model = HyperAttnNano(NANO_CONFIG, curvature_init=-10.0)
    expected = math.log(10.0)
    for i, block in enumerate(model.blocks):
        param = block.attn.log_abs_K
        assert param.shape == (NANO_CONFIG["n_heads"],), (
            f"Layer {i}: unexpected log_abs_K shape {param.shape}"
        )
        for j, val in enumerate(param.detach().tolist()):
            assert abs(val - expected) < 1e-6, (
                f"Layer {i} head {j}: log_abs_K={val:.8f}, "
                f"expected log(10)={expected:.8f}"
            )


def test_curvature_init_k10_gives_K_minus_10():
    """The K property (derived from log_abs_K) must return ≈ -10."""
    model = HyperAttnNano(NANO_CONFIG, curvature_init=-10.0)
    for i, block in enumerate(model.blocks):
        K = block.attn.K.detach()
        expected = torch.full_like(K, -10.0)
        assert torch.allclose(K, expected, atol=1e-5), (
            f"Layer {i}: expected K=-10.0, got {K.tolist()}"
        )


# ---------------------------------------------------------------------------
# Test 2: parameters remain learnable (requires_grad=True)
# ---------------------------------------------------------------------------

def test_curvature_init_params_are_learnable():
    """log_abs_K must have requires_grad=True after curvature_init construction."""
    model = HyperAttnNano(NANO_CONFIG, curvature_init=-10.0)
    for i, block in enumerate(model.blocks):
        assert block.attn.log_abs_K.requires_grad, (
            f"Layer {i}: log_abs_K.requires_grad is False — should be True"
        )


# ---------------------------------------------------------------------------
# Test 3: hyper-fixed (frozen) is unaffected by curvature_init semantics
# ---------------------------------------------------------------------------

def test_hyper_fixed_frozen_curvature_works():
    """
    fixed_curvature=-10 + manual requires_grad_(False) (as done in train.py)
    must produce K=-10 frozen parameters. Passing curvature_init separately
    to a hyper-fixed variant should not override fixed_curvature.
    """
    # Simulate train.py hyper-fixed path: curvature_init is None (not in config)
    model = HyperAttnNano(NANO_CONFIG, fixed_curvature=-10.0, curvature_init=None)
    # Freeze as train.py does
    for block in model.blocks:
        block.attn.log_abs_K.requires_grad_(False)

    for i, block in enumerate(model.blocks):
        K = block.attn.K.detach()
        expected = torch.full_like(K, -10.0)
        assert torch.allclose(K, expected, atol=1e-5), (
            f"Layer {i}: expected frozen K=-10.0, got {K.tolist()}"
        )
        assert not block.attn.log_abs_K.requires_grad, (
            f"Layer {i}: log_abs_K should be frozen but requires_grad=True"
        )


# ---------------------------------------------------------------------------
# Test 4: default init (no curvature_init) gives K=-1.0 (backward compat)
# ---------------------------------------------------------------------------

def test_default_curvature_is_minus_one():
    """No curvature_init → all heads should initialise at K=-1.0."""
    model = HyperAttnNano(NANO_CONFIG)
    for i, block in enumerate(model.blocks):
        K = block.attn.K.detach()
        expected = torch.full_like(K, -1.0)
        assert torch.allclose(K, expected, atol=1e-5), (
            f"Layer {i}: expected default K=-1.0, got {K.tolist()}"
        )


# ---------------------------------------------------------------------------
# Test 5: curvature_init overrides fixed_curvature for perhead variant
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("K_init", [-1.0, -5.0, -10.0, -20.0])
def test_curvature_init_parametric(K_init: float):
    """curvature_init=K_init should result in all heads at exactly K_init."""
    model = HyperAttnNano(NANO_CONFIG, curvature_init=K_init)
    for i, block in enumerate(model.blocks):
        K = block.attn.K.detach()
        expected = torch.full_like(K, K_init)
        assert torch.allclose(K, expected, atol=1e-5), (
            f"curvature_init={K_init}, layer {i}: got {K.tolist()}"
        )

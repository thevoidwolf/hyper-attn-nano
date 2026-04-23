"""
test_scores_only.py — Tests for LorentzScoreOnlyAttention
==========================================================

Run with:
    cd ~/hyper-attn-nano
    source activate.sh
    pytest tests/test_scores_only.py -v
"""

import math
import sys
import os

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from attention import LorentzScoreOnlyAttention, EuclideanAttention

# ---------------------------------------------------------------------------
# Shared config — small for fast CPU tests
# ---------------------------------------------------------------------------
D_MODEL = 64
N_HEADS = 4
D_HEAD  = D_MODEL // N_HEADS   # 16
B, S    = 2, 12


def _euclid_input(B=B, S=S, d_model=D_MODEL, seed=42):
    torch.manual_seed(seed)
    return torch.randn(B, S, d_model) * 0.1


# ---------------------------------------------------------------------------
# 1. Output shape
# ---------------------------------------------------------------------------

class TestOutputShape:

    def test_shape_matches_euclidean(self):
        """(B, S, d_model) in → (B, S, d_model) out — same as EuclideanAttention."""
        attn = LorentzScoreOnlyAttention(D_MODEL, N_HEADS, D_HEAD)
        x    = _euclid_input()
        out  = attn(x)
        assert out.shape == (B, S, D_MODEL), f"Expected {(B,S,D_MODEL)}, got {out.shape}"

    def test_shape_single_token(self):
        attn = LorentzScoreOnlyAttention(D_MODEL, N_HEADS, D_HEAD)
        x    = _euclid_input(S=1)
        out  = attn(x)
        assert out.shape == (B, 1, D_MODEL)

    def test_shape_long_sequence(self):
        attn = LorentzScoreOnlyAttention(D_MODEL, N_HEADS, D_HEAD)
        x    = _euclid_input(S=128)
        out  = attn(x)
        assert out.shape == (B, 128, D_MODEL)


# ---------------------------------------------------------------------------
# 2. Near-zero curvature → approaches Euclidean attention
# ---------------------------------------------------------------------------

class TestLimitingBehaviour:

    def test_near_zero_curvature_produces_finite_output(self):
        """At K=-1e-4 (nearly flat), output should be finite (no NaN/Inf)."""
        attn = LorentzScoreOnlyAttention(D_MODEL, N_HEADS, D_HEAD, curvature_init=-1e-4)
        x    = _euclid_input()
        out  = attn(x)
        assert torch.isfinite(out).all(), "Output contains NaN or Inf at near-zero curvature"

    def test_near_zero_curvature_output_bounded(self):
        """At K=-1e-4, output norms should be finite and similar order of magnitude."""
        attn = LorentzScoreOnlyAttention(D_MODEL, N_HEADS, D_HEAD, curvature_init=-1e-4)
        x    = _euclid_input()
        with torch.no_grad():
            out = attn(x)
        # Outputs should not explode (scale bounded)
        assert out.abs().max().item() < 1e4, "Output blew up at near-zero curvature"

    def test_curvature_k1_is_reasonable(self):
        """At K=-1.0, outputs should be finite and reasonable."""
        attn = LorentzScoreOnlyAttention(D_MODEL, N_HEADS, D_HEAD, curvature_init=-1.0)
        x    = _euclid_input()
        out  = attn(x)
        assert torch.isfinite(out).all()

    def test_curvature_k10_is_reasonable(self):
        """At K=-10.0, outputs should be finite and reasonable."""
        attn = LorentzScoreOnlyAttention(D_MODEL, N_HEADS, D_HEAD, curvature_init=-10.0)
        x    = _euclid_input()
        out  = attn(x)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# 3. Gradient flow
# ---------------------------------------------------------------------------

class TestGradientFlow:

    def test_gradients_flow_through_q_projection(self):
        """Gradient must flow through W_q (Q touches the Lorentz manifold)."""
        attn = LorentzScoreOnlyAttention(D_MODEL, N_HEADS, D_HEAD)
        x    = _euclid_input()
        out  = attn(x)
        loss = out.sum()
        loss.backward()
        assert attn.W_q.weight.grad is not None
        assert attn.W_q.weight.grad.abs().sum() > 0, "W_q grad is zero"

    def test_gradients_flow_through_k_projection(self):
        """Gradient must flow through W_k (K touches the Lorentz manifold)."""
        attn = LorentzScoreOnlyAttention(D_MODEL, N_HEADS, D_HEAD)
        x    = _euclid_input()
        out  = attn(x)
        loss = out.sum()
        loss.backward()
        assert attn.W_k.weight.grad is not None
        assert attn.W_k.weight.grad.abs().sum() > 0, "W_k grad is zero"

    def test_gradients_flow_through_v_projection(self):
        """V path must also have gradients (weighted sum uses W_v)."""
        attn = LorentzScoreOnlyAttention(D_MODEL, N_HEADS, D_HEAD)
        x    = _euclid_input()
        out  = attn(x)
        loss = out.sum()
        loss.backward()
        assert attn.W_v.weight.grad is not None
        assert attn.W_v.weight.grad.abs().sum() > 0, "W_v grad is zero"

    def test_no_manifold_ops_in_v_autograd_graph(self):
        """
        The value tensor should NOT pass through exp_map or log_map.
        We verify this by checking that the computational graph from the
        V projection does NOT contain exp_map operations.

        Strategy: hook W_v output, check it's used directly in attn output
        without going through the Lorentz projection path.
        We do this indirectly: the V path gradient should not require the
        manifold constraint (i.e., W_v grad should be large even at K=-10
        where manifold ops would cause scaling effects).
        """
        attn_lorentz = LorentzScoreOnlyAttention(D_MODEL, N_HEADS, D_HEAD, curvature_init=-10.0)
        attn_euclid  = EuclideanAttention(D_MODEL, N_HEADS, D_HEAD)

        # Copy weights so V projections are identical
        with torch.no_grad():
            attn_lorentz.W_v.weight.copy_(attn_euclid.W_v.weight)
            attn_lorentz.W_o.weight.copy_(attn_euclid.W_o.weight)

        x = _euclid_input()
        out_lorentz = attn_lorentz(x)
        out_euclid  = attn_euclid(x)

        out_lorentz.sum().backward()
        out_euclid.sum().backward()

        # W_v grads should be non-zero for both (V path is active)
        assert attn_lorentz.W_v.weight.grad is not None
        assert attn_euclid.W_v.weight.grad is not None

        # The ratio of W_v grad norms should be within 10x
        # (if V went through manifold ops, the ratio could be huge)
        lorentz_v_grad_norm = attn_lorentz.W_v.weight.grad.norm().item()
        euclid_v_grad_norm  = attn_euclid.W_v.weight.grad.norm().item()
        ratio = lorentz_v_grad_norm / (euclid_v_grad_norm + 1e-8)
        assert 0.01 < ratio < 100.0, (
            f"W_v grad norm ratio {ratio:.3f} suggests manifold ops on V path"
        )


# ---------------------------------------------------------------------------
# 4. Curvature parameter
# ---------------------------------------------------------------------------

class TestCurvatureParameter:

    def test_k_property_is_negative(self):
        attn = LorentzScoreOnlyAttention(D_MODEL, N_HEADS, D_HEAD, curvature_init=-5.0)
        assert attn.K.item() < 0

    def test_k_property_matches_init(self):
        for k_init in [-1.0, -5.0, -10.0]:
            attn = LorentzScoreOnlyAttention(D_MODEL, N_HEADS, D_HEAD, curvature_init=k_init)
            assert abs(attn.K.item() - k_init) < 1e-5, f"K={attn.K.item()} ≠ {k_init}"

    def test_init_K_alias(self):
        """init_K= alias should work for backward compat."""
        attn = LorentzScoreOnlyAttention(D_MODEL, N_HEADS, D_HEAD, init_K=-7.0)
        assert abs(attn.K.item() - (-7.0)) < 1e-5

    def test_rejects_positive_curvature(self):
        with pytest.raises(AssertionError):
            LorentzScoreOnlyAttention(D_MODEL, N_HEADS, D_HEAD, curvature_init=1.0)

    def test_log_abs_k_parameter_exists(self):
        """log_abs_K must be a parameter so train.py can update it via schedule."""
        attn = LorentzScoreOnlyAttention(D_MODEL, N_HEADS, D_HEAD)
        param_names = [name for name, _ in attn.named_parameters()]
        assert "log_abs_K" in param_names, "log_abs_K is not a registered parameter"


# ---------------------------------------------------------------------------
# 5. Causal masking (future tokens must not attend)
# ---------------------------------------------------------------------------

class TestCausalMask:

    def test_causal_attention_is_causal(self):
        """
        Changing a future token must not affect earlier token outputs.
        Test: perturb token at position S-1, check that output at position 0
        does not change (since position 0 cannot see position S-1).
        """
        attn = LorentzScoreOnlyAttention(D_MODEL, N_HEADS, D_HEAD)
        attn.eval()

        x        = _euclid_input(S=8)
        x_perturb = x.clone()
        x_perturb[:, -1, :] += 10.0   # large perturbation at last token

        with torch.no_grad():
            out_orig    = attn(x)
            out_perturb = attn(x_perturb)

        # Position 0 output must be identical
        assert torch.allclose(out_orig[:, 0, :], out_perturb[:, 0, :], atol=1e-5), (
            "Causal mask broken: position 0 is affected by a future token"
        )

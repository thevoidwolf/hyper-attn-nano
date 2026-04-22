"""
test_attention.py — Unit tests for LorentzPerHeadAttention and EuclideanAttention
==================================================================================

Run with:
    cd ~/hyper-attn-nano
    source activate.sh
    pytest tests/test_attention.py -v

These tests must all pass before moving on to blocks.py.
"""

import math
import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from manifolds import exp_map_origin
from attention import LorentzPerHeadAttention, EuclideanAttention, _causal_mask


# ---------------------------------------------------------------------------
# Shared config — kept small for fast CPU-mode test runs
# ---------------------------------------------------------------------------
D_MODEL = 64
N_HEADS = 4
D_HEAD  = D_MODEL // N_HEADS   # 16
B, S    = 2, 12                 # batch=2, seq_len=12


def _lorentz_input(B=B, S=S, d_model=D_MODEL, K=-1.0):
    """
    Create a valid batch of Lorentz manifold points.
    Shape: (B, S, d_model+1)
    """
    torch.manual_seed(42)
    v = torch.randn(B, S, d_model) * 0.1   # small norms → stable exp_map
    return exp_map_origin(v, K)


def _euclid_input(B=B, S=S, d_model=D_MODEL):
    """Standard Euclidean embeddings. Shape: (B, S, d_model)"""
    torch.manual_seed(42)
    return torch.randn(B, S, d_model) * 0.1


# ===========================================================================
# 1. OUTPUT SHAPE
# ===========================================================================

class TestOutputShape:

    def test_lorentz_output_shape(self):
        """(B, S, d_model+1) in → (B, S, d_model) out"""
        attn = LorentzPerHeadAttention(D_MODEL, N_HEADS, D_HEAD)
        x    = _lorentz_input()
        out  = attn(x)
        assert out.shape == (B, S, D_MODEL), (
            f"Expected ({B}, {S}, {D_MODEL}), got {out.shape}"
        )

    def test_euclidean_output_shape(self):
        attn = EuclideanAttention(D_MODEL, N_HEADS, D_HEAD)
        x    = _euclid_input()
        out  = attn(x)
        assert out.shape == (B, S, D_MODEL)

    def test_lorentz_single_token(self):
        """Edge case: sequence of length 1."""
        attn = LorentzPerHeadAttention(D_MODEL, N_HEADS, D_HEAD)
        x    = _lorentz_input(S=1)
        out  = attn(x)
        assert out.shape == (B, 1, D_MODEL)

    def test_lorentz_longer_sequence(self):
        attn = LorentzPerHeadAttention(D_MODEL, N_HEADS, D_HEAD)
        x    = _lorentz_input(S=64)
        out  = attn(x)
        assert out.shape == (B, 64, D_MODEL)


# ===========================================================================
# 2. K PROPERTY — always negative, right shape, right initial value
# ===========================================================================

class TestCurvatureProperty:

    @pytest.mark.parametrize("init_K", [-0.5, -1.0, -2.0, -5.0])
    def test_K_always_negative(self, init_K):
        attn = LorentzPerHeadAttention(D_MODEL, N_HEADS, D_HEAD, init_K=init_K)
        K    = attn.K
        assert (K < 0).all(), f"Some K values are non-negative: {K}"

    @pytest.mark.parametrize("init_K", [-0.5, -1.0, -2.0, -5.0])
    def test_K_initial_value(self, init_K):
        """All heads should start at init_K."""
        attn     = LorentzPerHeadAttention(D_MODEL, N_HEADS, D_HEAD, init_K=init_K)
        K        = attn.K
        expected = torch.full((N_HEADS,), init_K)
        err      = (K - expected).abs().max().item()
        assert err < 1e-5, f"Initial K mismatch: expected {init_K}, max err {err:.2e}"

    def test_K_shape(self):
        attn = LorentzPerHeadAttention(D_MODEL, N_HEADS, D_HEAD)
        assert attn.K.shape == (N_HEADS,), f"K shape wrong: {attn.K.shape}"

    def test_K_stays_negative_after_grad_step(self):
        """
        After a gradient update, K should still be negative.
        The reparameterisation (-exp(log_abs_K)) guarantees this mathematically,
        but this test confirms it survives an actual optimiser step.
        """
        attn   = LorentzPerHeadAttention(D_MODEL, N_HEADS, D_HEAD)
        opt    = torch.optim.Adam(attn.parameters(), lr=0.1)
        x      = _lorentz_input()

        for _ in range(5):
            opt.zero_grad()
            out = attn(x)
            out.sum().backward()
            opt.step()

        assert (attn.K < 0).all(), f"K became non-negative after training: {attn.K}"

    def test_euclidean_K_is_none(self):
        attn = EuclideanAttention(D_MODEL, N_HEADS, D_HEAD)
        assert attn.K is None


# ===========================================================================
# 3. CAUSAL MASKING — token i must not attend to token j > i
# ===========================================================================

class TestCausalMasking:

    def test_causal_mask_shape(self):
        mask = _causal_mask(S, torch.device('cpu'))
        assert mask.shape == (1, 1, S, S)

    def test_causal_mask_upper_triangle(self):
        """True entries must be exactly the strict upper triangle."""
        mask = _causal_mask(S, torch.device('cpu')).squeeze()  # (S, S)
        for i in range(S):
            for j in range(S):
                if j > i:
                    assert mask[i, j].item() is True,  f"[{i},{j}] should be masked"
                else:
                    assert mask[i, j].item() is False, f"[{i},{j}] should be unmasked"

    def _attention_weights(self, attn, x):
        """
        Extract the attention weight matrix by monkey-patching softmax.
        Returns the weight tensor (B, H, S, S).
        """
        captured = {}

        original_softmax = torch.nn.functional.softmax
        def capturing_softmax(input, dim=-1, **kwargs):
            out = original_softmax(input, dim=dim, **kwargs)
            captured['weights'] = out.detach()
            return out

        import torch.nn.functional as F_module
        orig = F_module.softmax
        F_module.softmax = capturing_softmax
        try:
            attn(x)
        finally:
            F_module.softmax = orig

        return captured.get('weights')

    def test_lorentz_no_future_attention(self):
        """
        Attention weights at position i must be zero for all j > i.
        (Past and present tokens only — this is a decoder.)
        """
        attn    = LorentzPerHeadAttention(D_MODEL, N_HEADS, D_HEAD)
        x       = _lorentz_input()
        weights = self._attention_weights(attn, x)   # (B, H, S, S)

        if weights is None:
            pytest.skip("Could not capture attention weights via monkey-patch")

        # Upper triangle of each (S, S) weight slice should be (near) zero
        mask = _causal_mask(S, torch.device('cpu'))   # (1,1,S,S)
        future_weights = weights[mask.expand_as(weights)]
        max_future = future_weights.abs().max().item()
        assert max_future < 1e-6, (
            f"Non-zero weight on future token: {max_future:.2e}"
        )

    def test_euclidean_no_future_attention(self):
        attn    = EuclideanAttention(D_MODEL, N_HEADS, D_HEAD)
        x       = _euclid_input()
        weights = self._attention_weights(attn, x)

        if weights is None:
            pytest.skip("Could not capture attention weights via monkey-patch")

        mask           = _causal_mask(S, torch.device('cpu'))
        future_weights = weights[mask.expand_as(weights)]
        assert future_weights.abs().max().item() < 1e-6


# ===========================================================================
# 4. GRADIENT FLOW — no NaN or Inf in any parameter gradient
# ===========================================================================

class TestGradientFlow:

    def test_lorentz_gradients_clean(self):
        attn = LorentzPerHeadAttention(D_MODEL, N_HEADS, D_HEAD)
        x    = _lorentz_input()
        out  = attn(x)
        out.sum().backward()

        for name, param in attn.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
            assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"

    def test_lorentz_log_abs_K_gradient(self):
        """
        The curvature parameter must receive a gradient — it's the whole point.
        If it's zero or detached, nothing learns.
        """
        attn = LorentzPerHeadAttention(D_MODEL, N_HEADS, D_HEAD)
        x    = _lorentz_input()
        attn(x).sum().backward()

        grad = attn.log_abs_K.grad
        assert grad is not None, "log_abs_K received no gradient"
        assert not torch.isnan(grad).any(), "NaN in log_abs_K gradient"
        # Gradient should be non-zero (if all zeros, K will never change)
        assert grad.abs().max().item() > 1e-10, (
            "log_abs_K gradient is effectively zero — curvature will not learn"
        )

    def test_euclidean_gradients_clean(self):
        attn = EuclideanAttention(D_MODEL, N_HEADS, D_HEAD)
        x    = _euclid_input().requires_grad_(True)
        attn(x).sum().backward()

        for name, param in attn.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN in {name}"


# ===========================================================================
# 5. CURVATURE DIVERGENCE — heads can specialise after gradient steps
# ===========================================================================

class TestCurvatureDivergence:
    """
    After training on a loss that actually differentiates the heads,
    the per-head K values should diverge from their shared initialisation.
    This is a necessary (not sufficient) condition for head specialisation.
    """

    def test_heads_diverge_after_training(self):
        """
        Train briefly with a loss that creates asymmetry between heads.
        The K values should not all remain identical.
        """
        torch.manual_seed(0)
        attn = LorentzPerHeadAttention(D_MODEL, N_HEADS, D_HEAD, init_K=-1.0)
        opt  = torch.optim.Adam(attn.parameters(), lr=1e-2)

        # Asymmetric target: each position in the output has a different target
        target = torch.randn(B, S, D_MODEL)

        for _ in range(20):
            opt.zero_grad()
            x   = _lorentz_input()
            out = attn(x)
            loss = (out - target).pow(2).mean()
            loss.backward()
            opt.step()

        K_vals = attn.K.detach()
        K_std  = K_vals.std().item()
        assert K_std > 1e-4, (
            f"Heads did not diverge: K std = {K_std:.2e}. "
            f"K values: {K_vals.tolist()}"
        )


# ===========================================================================
# 6. PARAMETER COUNT — lorentz vs euclidean have same non-K parameters
# ===========================================================================

class TestParameterCount:

    def test_non_K_parameter_count_matches(self):
        """
        The projection layers (W_q, W_k, W_v, W_o) should be identical
        in both modules. Only LorentzPerHeadAttention has the extra
        n_heads K parameters.
        """
        lorentz = LorentzPerHeadAttention(D_MODEL, N_HEADS, D_HEAD)
        euclid  = EuclideanAttention(D_MODEL, N_HEADS, D_HEAD)

        def proj_params(m):
            return sum(p.numel() for name, p in m.named_parameters()
                       if 'log_abs_K' not in name)

        assert proj_params(lorentz) == proj_params(euclid), (
            f"Projection param counts differ: "
            f"Lorentz={proj_params(lorentz)}, Euclid={proj_params(euclid)}"
        )

        # Lorentz has exactly n_heads extra params (the K values)
        lorentz_total = sum(p.numel() for p in lorentz.parameters())
        euclid_total  = sum(p.numel() for p in euclid.parameters())
        assert lorentz_total == euclid_total + N_HEADS, (
            f"Expected Lorentz to have exactly {N_HEADS} extra params. "
            f"Lorentz={lorentz_total}, Euclid={euclid_total}"
        )


# ===========================================================================
# 7. NO NaN IN OUTPUT
# ===========================================================================

class TestNoNaN:

    def test_lorentz_output_clean(self):
        attn = LorentzPerHeadAttention(D_MODEL, N_HEADS, D_HEAD)
        x    = _lorentz_input()
        out  = attn(x)
        assert not torch.isnan(out).any(), "NaN in Lorentz attention output"
        assert not torch.isinf(out).any(), "Inf in Lorentz attention output"

    def test_euclidean_output_clean(self):
        attn = EuclideanAttention(D_MODEL, N_HEADS, D_HEAD)
        x    = _euclid_input()
        out  = attn(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    @pytest.mark.parametrize("init_K", [-0.1, -0.5, -2.0, -10.0])
    def test_extreme_init_K(self, init_K):
        """Extreme curvatures at initialisation should not produce NaN."""
        attn = LorentzPerHeadAttention(D_MODEL, N_HEADS, D_HEAD, init_K=init_K)
        x    = _lorentz_input()
        out  = attn(x)
        assert not torch.isnan(out).any(), f"NaN for init_K={init_K}"


# ===========================================================================
# Run directly with: python tests/test_attention.py
# ===========================================================================

if __name__ == "__main__":
    import subprocess
    subprocess.run(["pytest", __file__, "-v"], check=True)

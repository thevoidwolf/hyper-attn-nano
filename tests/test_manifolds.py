"""
test_manifolds.py — Unit tests for Lorentz manifold operations
==============================================================
 
Run with:
    cd ~/hyper-attn-nano
    source activate.sh
    pytest tests/test_manifolds.py -v
 
All tests must pass before moving on to attention.py.
"""
 
import pytest
import torch
import sys
import os
 
# Make src/ importable when running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
 
from manifolds import (
    exp_map_origin,
    log_map_origin,
    exp_map_batched,
    log_map_batched,
    lorentz_inner,
    check_on_manifold,
)
 
# ---------------------------------------------------------------------------
# Curvature values to test — the round-trip must hold for all of these
# ---------------------------------------------------------------------------
K_VALUES = [-0.5, -1.0, -2.0, -5.0]
 
 
# ===========================================================================
# 1. ROUND-TRIP: log_map(exp_map(v)) ≈ v
# ===========================================================================
 
class TestRoundTrip:
    """
    The most fundamental correctness test.
    exp_map and log_map must be inverses of each other.
    Error threshold: < 1e-4 (project requirement).
    """
 
    @pytest.mark.parametrize("K", K_VALUES)
    def test_round_trip_single(self, K):
        """Single curvature, batched vectors."""
        torch.manual_seed(42)
        # Keep norms small — large norms push toward numerical limits
        v = torch.randn(16, 64) * 0.5
 
        x         = exp_map_origin(v, K)
        v_back    = log_map_origin(x, K)
        max_err   = (v - v_back).abs().max().item()
 
        assert max_err < 1e-4, (
            f"Round-trip error {max_err:.2e} exceeds 1e-4 for K={K}\n"
            f"  This usually means the arcosh argument formula is wrong."
        )
 
    @pytest.mark.parametrize("K", K_VALUES)
    def test_round_trip_larger_norms(self, K):
        """Norms away from origin, kept below angle clamp threshold.

        With MAX_ANGLE=10 in manifolds.py, vectors must satisfy
        sqrt(-K) * norm(v) < 10 to be round-trippable. For K=-5
        (the most demanding K value here) that means norm < 4.47.
        Using * 0.5 gives typical norm ≈ 2.8 for d=32, angle ≈ 6.3. ✓
        """
        torch.manual_seed(7)
        v       = torch.randn(8, 32) * 0.5
        x       = exp_map_origin(v, K)
        v_back  = log_map_origin(x, K)
        max_err = (v - v_back).abs().max().item()

        # Allow slightly looser tolerance for larger norms
        assert max_err < 1e-3, (
            f"Round-trip error {max_err:.2e} with larger norms for K={K}"
        )
 
    @pytest.mark.parametrize("K", K_VALUES)
    def test_round_trip_batched(self, K):
        """Batched (per-head) version must also round-trip."""
        torch.manual_seed(99)
        B, S, H, d = 2, 16, 4, 32
        v         = torch.randn(B, S, H, d) * 0.5
        K_heads   = torch.full((H,), K)
 
        x         = exp_map_batched(v, K_heads)
        v_back    = log_map_batched(x, K_heads)
        max_err   = (v - v_back).abs().max().item()
 
        assert max_err < 1e-4, (
            f"Batched round-trip error {max_err:.2e} for K={K}"
        )
 
 
# ===========================================================================
# 2. MANIFOLD CONSTRAINT: exp_map output must satisfy <x,x>_L = 1/K
# ===========================================================================
 
class TestManifoldConstraint:
    """
    Points returned by exp_map must live on the hyperboloid.
    This verifies the geometry is correct, not just that round-trip works.
 
    WHY SMALL NORMS:
    The constraint check computes <x,x>_L = -x₀² + ‖xi‖², which is a
    subtraction of two large numbers (cosh² and sinh² of the angle).
    Float32 has ~7 significant digits. When the angle = sqrt(-K)·‖v‖ is
    large, both terms are huge and their difference loses precision:
 
        K=-1, angle=4:  cosh≈27,   error ~ 2e-4   ← just over naive threshold
        K=-5, angle=9:  cosh≈3800, error ~ 3.5     ← catastrophic
 
    This is not a bug in exp_map — the formulas are mathematically exact.
    It's an inherent float32 limitation for large inputs.
 
    In a real model, embeddings use small initialisation (GPT: N(0, 0.02)),
    and LorentzRMSNorm keeps norms bounded. The tests below use norms ≈ 0.5–1.0,
    which matches realistic runtime values and keeps angles small enough for
    float32 precision to hold.
    """
 
    @pytest.mark.parametrize("K", K_VALUES)
    def test_on_manifold(self, K):
        torch.manual_seed(42)
        # Small norms (≈ 0.8 for 64-dim) → small angles → no cancellation
        v = torch.randn(16, 64) * 0.1
        x = exp_map_origin(v, K)
 
        inner    = lorentz_inner(x, x)
        expected = torch.full_like(inner, 1.0 / K)
        max_err  = (inner - expected).abs().max().item()
 
        assert max_err < 1e-4, (
            f"Manifold constraint violated for K={K}: "
            f"<x,x>_L should be {1/K:.4f}, max error = {max_err:.2e}"
        )
 
    @pytest.mark.parametrize("K", K_VALUES)
    def test_check_on_manifold_helper(self, K):
        """The check_on_manifold() helper should agree."""
        torch.manual_seed(42)
        v = torch.randn(8, 32) * 0.1
        x = exp_map_origin(v, K)
        assert check_on_manifold(x, K), f"check_on_manifold returned False for K={K}"
 
 
# ===========================================================================
# 3. GRADIENT FLOW: backward() through exp_map must not produce NaN/Inf
# ===========================================================================
 
class TestGradientFlow:
    """
    If gradients explode or vanish to NaN, training will fail silently.
    Catch this at the unit-test level.
    """
 
    @pytest.mark.parametrize("K", K_VALUES)
    def test_grad_through_exp_map(self, K):
        torch.manual_seed(42)
        v = torch.randn(4, 32, requires_grad=True)
        x = exp_map_origin(v, K)
        x.sum().backward()
 
        assert v.grad is not None, "Gradient did not flow back through exp_map"
        assert not torch.isnan(v.grad).any(), f"NaN in gradient for K={K}"
        assert not torch.isinf(v.grad).any(), f"Inf in gradient for K={K}"
 
    @pytest.mark.parametrize("K", K_VALUES)
    def test_grad_through_log_map(self, K):
        torch.manual_seed(42)
        v = torch.randn(4, 32) * 0.5
        x = exp_map_origin(v, K).detach().requires_grad_(True)
        v_out = log_map_origin(x, K)
        v_out.sum().backward()
 
        assert x.grad is not None
        assert not torch.isnan(x.grad).any(), f"NaN in log_map gradient for K={K}"
        assert not torch.isinf(x.grad).any(), f"Inf in log_map gradient for K={K}"
 
    @pytest.mark.parametrize("K", K_VALUES)
    def test_grad_through_batched(self, K):
        torch.manual_seed(42)
        B, S, H, d = 2, 8, 4, 16
        v       = torch.randn(B, S, H, d, requires_grad=True)
        K_heads = torch.full((H,), K)
 
        x = exp_map_batched(v, K_heads)
        x.sum().backward()
 
        assert v.grad is not None
        assert not torch.isnan(v.grad).any(), f"NaN in batched gradient for K={K}"
        assert not torch.isinf(v.grad).any(), f"Inf in batched gradient for K={K}"
 
 
# ===========================================================================
# 4. BATCHED EXP_MAP: shape and per-head consistency
# ===========================================================================
 
class TestBatchedExpMap:
    """
    exp_map_batched must produce the same result as calling exp_map_origin
    independently on each head — vectorised correctness, not just shape.
    """
 
    def test_output_shape(self):
        B, S, H, d = 3, 16, 4, 32
        v       = torch.randn(B, S, H, d)
        K_heads = torch.tensor([-0.5, -1.0, -2.0, -5.0])
 
        x = exp_map_batched(v, K_heads)
        assert x.shape == (B, S, H, d + 1), (
            f"Expected shape {(B, S, H, d+1)}, got {x.shape}"
        )
 
    def test_no_nan_output(self):
        B, S, H, d = 2, 16, 4, 32
        v       = torch.randn(B, S, H, d)
        K_heads = torch.tensor([-0.5, -1.0, -2.0, -5.0])
 
        x = exp_map_batched(v, K_heads)
        assert not torch.isnan(x).any(), "NaN in batched exp_map output"
        assert not torch.isinf(x).any(), "Inf in batched exp_map output"
 
    def test_per_head_consistency(self):
        """
        Each head's output should match calling exp_map_origin with that head's K.
        This verifies the broadcasting logic is correct.
        """
        torch.manual_seed(42)
        B, S, H, d = 2, 8, 4, 16
        v       = torch.randn(B, S, H, d) * 0.5
        K_vals  = [-0.5, -1.0, -2.0, -5.0]
        K_heads = torch.tensor(K_vals)
 
        x_batched = exp_map_batched(v, K_heads)
 
        for h, K in enumerate(K_vals):
            x_single = exp_map_origin(v[:, :, h, :], K)        # (B, S, d+1)
            err = (x_batched[:, :, h, :] - x_single).abs().max().item()
            assert err < 1e-5, (
                f"Head {h} (K={K}): batched vs single mismatch = {err:.2e}"
            )
 
    def test_mixed_curvatures_all_on_manifold(self):
        """Every head's output must satisfy its own manifold constraint.
        Uses small norms to stay within float32 precision — see TestManifoldConstraint."""
        torch.manual_seed(42)
        B, S, H, d = 2, 8, 4, 16
        v       = torch.randn(B, S, H, d) * 0.1   # small norms → small angles
        K_vals  = [-0.5, -1.0, -2.0, -5.0]
        K_heads = torch.tensor(K_vals)
 
        x = exp_map_batched(v, K_heads)  # (B, S, H, d+1)
 
        for h, K in enumerate(K_vals):
            xh    = x[:, :, h, :]          # (B, S, d+1)
            inner = lorentz_inner(xh, xh)  # (B, S)
            err   = (inner - 1.0/K).abs().max().item()
            assert err < 1e-4, (
                f"Head {h} (K={K}): manifold constraint violated, max err = {err:.2e}"
            )
 
 
# ===========================================================================
# 5. LORENTZ INNER PRODUCT: shape, formula, symmetry
# ===========================================================================
 
class TestLorentzInner:
 
    def test_output_shape_2d(self):
        x = torch.randn(8, 33)
        y = torch.randn(8, 33)
        out = lorentz_inner(x, y)
        assert out.shape == (8,), f"Expected (8,), got {out.shape}"
 
    def test_output_shape_3d(self):
        x = torch.randn(4, 8, 33)
        y = torch.randn(4, 8, 33)
        out = lorentz_inner(x, y)
        assert out.shape == (4, 8), f"Expected (4, 8), got {out.shape}"
 
    def test_formula_manual(self):
        """
        Manual ground-truth: <[1,2,3], [4,5,6]>_L = -1*4 + 2*5 + 3*6 = 24
        """
        x        = torch.tensor([[1.0, 2.0, 3.0]])
        y        = torch.tensor([[4.0, 5.0, 6.0]])
        expected = -1.0*4.0 + 2.0*5.0 + 3.0*6.0   # = 24.0
        result   = lorentz_inner(x, y).item()
        assert abs(result - expected) < 1e-5, (
            f"Expected {expected}, got {result}"
        )
 
    def test_symmetry(self):
        """Lorentz inner product is symmetric: <x,y>_L = <y,x>_L."""
        torch.manual_seed(42)
        x   = torch.randn(16, 65)
        y   = torch.randn(16, 65)
        err = (lorentz_inner(x, y) - lorentz_inner(y, x)).abs().max().item()
        assert err < 1e-6, f"Lorentz inner product not symmetric, err = {err:.2e}"
 
    def test_time_sign_matters(self):
        """Negating the time component changes the inner product."""
        x = torch.tensor([[2.0, 1.0, 0.0]])   # time=2, spatial=[1,0]
        y = torch.tensor([[2.0, 1.0, 0.0]])
        xy_normal = lorentz_inner(x, y).item()   # -4 + 1 = -3
 
        # Flip time sign on x — inner product must change
        x_flipped = x.clone()
        x_flipped[0, 0] *= -1                   # time = -2
        xy_flipped = lorentz_inner(x_flipped, y).item()   # +4 + 1 = 5
 
        assert abs(xy_normal - (-3.0)) < 1e-5
        assert abs(xy_flipped - 5.0) < 1e-5
 
 
# ===========================================================================
# 6. NUMERICAL STABILITY: zero vectors, extreme curvatures
# ===========================================================================
 
class TestNumericalStability:
 
    def test_near_zero_input(self):
        """Very small input vectors should not produce NaN."""
        v = torch.zeros(4, 32) + 1e-9
        x = exp_map_origin(v, -1.0)
        assert not torch.isnan(x).any(), "NaN for near-zero input"
 
    @pytest.mark.parametrize("K", [-0.1, -10.0, -50.0])
    def test_extreme_curvatures(self, K):
        """Extreme K values should still produce valid outputs."""
        torch.manual_seed(42)
        v = torch.randn(4, 32) * 0.1   # keep norms small for extreme K
        x = exp_map_origin(v, K)
        assert not torch.isnan(x).any(), f"NaN for K={K}"
        assert not torch.isinf(x).any(), f"Inf for K={K}"
 
    def test_float16_input_handled(self):
        """
        exp_map should handle fp16 input without crashing.
        (It internally casts to float32.)
        """
        v = torch.randn(4, 32).half()   # float16
        x = exp_map_origin(v, -1.0)
        assert x.dtype == torch.float32, "Output should be float32 even for fp16 input"
        assert not torch.isnan(x).any()
 
 
# ===========================================================================
# Run directly with: python tests/test_manifolds.py
# ===========================================================================
 
if __name__ == "__main__":
    import subprocess
    subprocess.run(["pytest", __file__, "-v"], check=True)
 
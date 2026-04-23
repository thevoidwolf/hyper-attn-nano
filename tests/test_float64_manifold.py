"""
test_float64_manifold.py — Tests for float64 precision path in manifolds.py
============================================================================

Spec: SPEC_EXPERIMENT_B2, Tests section.

Run with:
    cd ~/hyper-attn-nano
    source activate.sh
    pytest tests/test_float64_manifold.py -v
"""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import manifolds
from manifolds import (
    exp_map_origin,
    log_map_origin,
    MANIFOLD_FLOAT64,
)

HIGH_K_VALUES = [-10.0, -20.0, -50.0, -100.0]


# ===========================================================================
# 1. ROUND-TRIP ACCURACY AT HIGH K
#    float64 error < 1e-4 AND at least 10x smaller than float32 error
# ===========================================================================

class TestRoundTripFloat64:
    """
    Validates that float64 buys a 10x precision improvement at high K.
    This is the primary correctness check for Experiment B2.
    """

    @pytest.mark.parametrize("K", HIGH_K_VALUES)
    def test_round_trip_accuracy(self, K):
        """float64 round-trip error must be < 1e-4 for norms in [0.5, 5.0]."""
        torch.manual_seed(42)
        # Scale chosen so sqrt(-K)*norm spans [0.5*sqrt(-K), 5.0*sqrt(-K)]
        # — well into the regime where float32 struggles at K=-50, -100
        v = torch.randn(32, 64)
        # Normalise to norm ≈ 1.5 so angle = sqrt(-K)*1.5 is meaningful
        v = v / v.norm(dim=-1, keepdim=True) * 1.5

        x      = exp_map_origin(v, K, use_float64=True)
        v_back = log_map_origin(x, K, use_float64=True)
        err64  = (v - v_back).abs().max().item()

        assert err64 < 1e-4, (
            f"float64 round-trip error {err64:.2e} exceeds 1e-4 for K={K}"
        )

    @pytest.mark.parametrize("K", HIGH_K_VALUES)
    def test_float64_beats_float32_by_10x(self, K):
        """float64 error must be at least 10x smaller than float32 error."""
        torch.manual_seed(42)
        v = torch.randn(32, 64)
        v = v / v.norm(dim=-1, keepdim=True) * 1.5

        # float32 path
        x32      = exp_map_origin(v, K, use_float64=False)
        v_back32 = log_map_origin(x32, K, use_float64=False)
        err32    = (v - v_back32).abs().max().item()

        # float64 path
        x64      = exp_map_origin(v, K, use_float64=True)
        v_back64 = log_map_origin(x64, K, use_float64=True)
        err64    = (v - v_back64).abs().max().item()

        # Guard: if float32 error is already tiny, both precisions are fine
        if err32 < 1e-5:
            assert err64 < 1e-5, (
                f"K={K}: float32 error is already tiny ({err32:.2e}), "
                f"but float64 error is worse ({err64:.2e})"
            )
            return

        ratio = err32 / max(err64, 1e-15)
        assert ratio >= 10.0, (
            f"K={K}: float64 is only {ratio:.1f}x better than float32 "
            f"(need ≥10x). err32={err32:.2e}, err64={err64:.2e}"
        )

    @pytest.mark.parametrize("K", HIGH_K_VALUES)
    @pytest.mark.parametrize("norm_val", [0.5, 2.0, 5.0])
    def test_round_trip_various_norms(self, K, norm_val):
        """Round-trip holds across a range of vector norms below the MAX_ANGLE clamp.

        MAX_ANGLE=30 means the max safe norm is 30/sqrt(-K):
          K=-10:  norm < 9.49  (5.0 is fine)
          K=-20:  norm < 6.71  (5.0 is fine)
          K=-50:  norm < 4.24  (skip 5.0)
          K=-100: norm < 3.0   (skip 5.0)
        Norms above the safe threshold hit the clamp and are lossy by design.
        """
        import math
        max_safe_norm = manifolds.MAX_ANGLE / math.sqrt(-K)
        if norm_val > max_safe_norm * 0.95:
            pytest.skip(
                f"norm={norm_val} exceeds MAX_ANGLE safe range for K={K} "
                f"(max_safe={max_safe_norm:.2f}); clamp is lossy by design"
            )

        torch.manual_seed(7)
        v = torch.randn(16, 32)
        v = v / v.norm(dim=-1, keepdim=True) * norm_val

        x      = exp_map_origin(v, K, use_float64=True)
        v_back = log_map_origin(x, K, use_float64=True)
        err    = (v - v_back).abs().max().item()

        assert err < 1e-3, (
            f"K={K}, norm={norm_val}: float64 round-trip error {err:.2e} exceeds 1e-3"
        )


# ===========================================================================
# 2. NO NaN / Inf AT HIGH K
#    float64 must stay finite at norms where float32 collapses
# ===========================================================================

class TestNoNaNHighK:

    @pytest.mark.parametrize("K", [-50.0, -100.0])
    @pytest.mark.parametrize("norm_val", [0.5, 2.0, 5.0, 10.0])
    def test_exp_map_finite(self, K, norm_val):
        """exp_map output must be finite at high K and large norms."""
        torch.manual_seed(42)
        v = torch.randn(8, 32)
        v = v / v.norm(dim=-1, keepdim=True) * norm_val

        x = exp_map_origin(v, K, use_float64=True)
        assert torch.isfinite(x).all(), (
            f"exp_map(K={K}, norm={norm_val}) produced non-finite values"
        )

    @pytest.mark.parametrize("K", [-50.0, -100.0])
    @pytest.mark.parametrize("norm_val", [0.5, 2.0, 5.0, 10.0])
    def test_log_map_finite(self, K, norm_val):
        """log_map output must be finite at high K and large norms."""
        torch.manual_seed(42)
        v = torch.randn(8, 32)
        v = v / v.norm(dim=-1, keepdim=True) * norm_val

        x      = exp_map_origin(v, K, use_float64=True)
        v_back = log_map_origin(x, K, use_float64=True)
        assert torch.isfinite(v_back).all(), (
            f"log_map(K={K}, norm={norm_val}) produced non-finite values"
        )


# ===========================================================================
# 3. OUTPUT DTYPE IS ALWAYS FLOAT32
#    Downstream model code must never see float64 tensors
# ===========================================================================

class TestOutputDtype:

    @pytest.mark.parametrize("K", [-10.0, -50.0])
    def test_exp_map_origin_returns_float32(self, K):
        """exp_map_origin must return float32 even when use_float64=True."""
        v = torch.randn(4, 32, dtype=torch.float32)
        x = exp_map_origin(v, K, use_float64=True)
        assert x.dtype == torch.float32, (
            f"exp_map_origin returned {x.dtype}, expected torch.float32"
        )

    @pytest.mark.parametrize("K", [-10.0, -50.0])
    def test_log_map_origin_returns_float32(self, K):
        """log_map_origin must return float32 even when use_float64=True."""
        v = torch.randn(4, 32, dtype=torch.float32)
        x = exp_map_origin(v, K, use_float64=True)
        v_back = log_map_origin(x, K, use_float64=True)
        assert v_back.dtype == torch.float32, (
            f"log_map_origin returned {v_back.dtype}, expected torch.float32"
        )

    def test_float32_input_gives_float32_output_no_flag(self):
        """Default path (use_float64=False) must still return float32."""
        v = torch.randn(4, 32)
        x = exp_map_origin(v, -5.0)
        assert x.dtype == torch.float32


# ===========================================================================
# 4. MODULE FLAG TOGGLE
#    MANIFOLD_FLOAT64 = True must route through float64 path automatically
# ===========================================================================

class TestFlagToggle:

    def setup_method(self):
        """Reset flag before each test."""
        manifolds.MANIFOLD_FLOAT64 = False

    def teardown_method(self):
        """Always reset flag after each test."""
        manifolds.MANIFOLD_FLOAT64 = False

    def test_flag_true_matches_explicit_use_float64(self):
        """
        Setting manifolds.MANIFOLD_FLOAT64 = True and calling exp_map_origin
        without use_float64 keyword must match calling with use_float64=True.
        """
        torch.manual_seed(42)
        v = torch.randn(8, 32)
        v = v / v.norm(dim=-1, keepdim=True) * 2.0

        K = -50.0

        # Explicit float64
        x_explicit = exp_map_origin(v, K, use_float64=True)

        # Via module flag
        manifolds.MANIFOLD_FLOAT64 = True
        x_flag = exp_map_origin(v, K)

        assert torch.allclose(x_explicit, x_flag, atol=1e-6), (
            "Module flag path differs from explicit use_float64=True"
        )

    def test_flag_false_matches_float32_path(self):
        """With flag False (default), result must match float32 path."""
        torch.manual_seed(42)
        v = torch.randn(8, 32) * 0.3

        K = -5.0
        manifolds.MANIFOLD_FLOAT64 = False
        x_flag = exp_map_origin(v, K)
        x_fp32 = exp_map_origin(v, K, use_float64=False)

        assert torch.allclose(x_flag, x_fp32, atol=1e-6)

    def test_flag_output_dtype_is_float32(self):
        """Module flag path must still return float32."""
        manifolds.MANIFOLD_FLOAT64 = True
        v = torch.randn(4, 32)
        x = exp_map_origin(v, -20.0)
        assert x.dtype == torch.float32, (
            f"Flag path returned {x.dtype}, expected float32"
        )


# ===========================================================================
# Run directly with: python tests/test_float64_manifold.py
# ===========================================================================

if __name__ == "__main__":
    import subprocess
    subprocess.run(["pytest", __file__, "-v"], check=True)

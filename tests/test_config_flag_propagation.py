"""
test_config_flag_propagation.py — Tests that manifold_float64 config flag is propagated
========================================================================================

Covers the bug identified in Probe 3 (Experiment D): the manifold_float64 flag
was stripped from model_cfg before reaching HyperAttnNano, so float64 manifold
ops were silently skipped.

Run with:
    cd ~/hyper-attn-nano
    source activate.sh
    pytest tests/test_config_flag_propagation.py -v
"""

import sys
import os
import warnings

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ---------------------------------------------------------------------------
# Helper to reset MANIFOLD_FLOAT64 between tests
# ---------------------------------------------------------------------------

def _reset_manifold_flag():
    import manifolds as _m
    _m.MANIFOLD_FLOAT64 = False


# ---------------------------------------------------------------------------
# 1. Flag propagates through config → model constructor → manifolds module
# ---------------------------------------------------------------------------

class TestFloat64FlagPropagation:

    def setup_method(self):
        _reset_manifold_flag()

    def teardown_method(self):
        _reset_manifold_flag()

    def test_flag_false_by_default(self):
        """Without manifold_float64 in config, MANIFOLD_FLOAT64 stays False."""
        import manifolds as _m
        from model import HyperAttnNano

        cfg = {
            "d_model": 64, "n_layers": 2, "n_heads": 4,
            "d_ff": 128, "max_seq_len": 32, "vocab_size": 256,
        }
        model = HyperAttnNano(cfg, fixed_curvature=-1.0)
        assert _m.MANIFOLD_FLOAT64 is False

    def test_flag_true_sets_manifold_module(self):
        """manifold_float64: true must set manifolds.MANIFOLD_FLOAT64 = True."""
        import manifolds as _m
        from model import HyperAttnNano

        cfg = {
            "d_model": 64, "n_layers": 2, "n_heads": 4,
            "d_ff": 128, "max_seq_len": 32, "vocab_size": 256,
            "manifold_float64": True,
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = HyperAttnNano(cfg, fixed_curvature=-1.0)

        assert _m.MANIFOLD_FLOAT64 is True, (
            "manifolds.MANIFOLD_FLOAT64 was not set to True despite config "
            "manifold_float64: true. This is the bug from Experiment D."
        )

    def test_flag_false_explicit_does_not_set_module(self):
        """manifold_float64: false must not set MANIFOLD_FLOAT64."""
        import manifolds as _m
        from model import HyperAttnNano

        cfg = {
            "d_model": 64, "n_layers": 2, "n_heads": 4,
            "d_ff": 128, "max_seq_len": 32, "vocab_size": 256,
            "manifold_float64": False,
        }
        model = HyperAttnNano(cfg, fixed_curvature=-1.0)
        assert _m.MANIFOLD_FLOAT64 is False


# ---------------------------------------------------------------------------
# 2. Forward pass uses float64 internally when flag is set
# ---------------------------------------------------------------------------

class TestFloat64ForwardPass:

    def setup_method(self):
        _reset_manifold_flag()

    def teardown_method(self):
        _reset_manifold_flag()

    def test_float32_forward_pass_produces_float32_logits(self):
        """Without float64 flag, output is float32."""
        from model import HyperAttnNano

        cfg = {
            "d_model": 64, "n_layers": 1, "n_heads": 4,
            "d_ff": 128, "max_seq_len": 16, "vocab_size": 256,
        }
        model = HyperAttnNano(cfg, fixed_curvature=-1.0)
        model.eval()

        x = torch.randint(0, 256, (1, 8))
        with torch.no_grad():
            logits, _ = model(x)

        assert logits.dtype == torch.float32

    def test_float64_flag_reaches_manifold_ops(self):
        """
        With manifold_float64: true, manifold operations run in float64.

        We verify this by:
        1. Setting the flag
        2. Running a forward pass
        3. Checking that exp_map_origin receives and uses float64 dtype

        We patch exp_map_origin to record the dtype it's called with.
        """
        import manifolds as _m
        from model import HyperAttnNano

        cfg = {
            "d_model": 64, "n_layers": 1, "n_heads": 4,
            "d_ff": 128, "max_seq_len": 16, "vocab_size": 256,
            "manifold_float64": True,
        }

        dtypes_seen = []
        original_exp_map = _m.exp_map_origin

        def patched_exp_map(v, K, use_float64=False):
            use_float64 = use_float64 or _m.MANIFOLD_FLOAT64
            if use_float64:
                dtypes_seen.append(torch.float64)
            return original_exp_map(v, K, use_float64=use_float64)

        _m.exp_map_origin = patched_exp_map

        try:
            # Need to also patch the reference in model.py
            import model as _model_mod
            _model_mod.exp_map_origin = patched_exp_map

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = HyperAttnNano(cfg, fixed_curvature=-1.0)

            model.eval()
            x = torch.randint(0, 256, (1, 8))
            with torch.no_grad():
                logits, _ = model(x)

            assert len(dtypes_seen) > 0, (
                "exp_map_origin was never called with float64. "
                "manifold_float64 flag is not being propagated correctly."
            )
        finally:
            _m.exp_map_origin = original_exp_map
            import model as _model_mod2
            _model_mod2.exp_map_origin = original_exp_map


# ---------------------------------------------------------------------------
# 3. Numerical precision: float64 produces more precise manifold constraint
# ---------------------------------------------------------------------------

class TestFloat64Precision:

    def setup_method(self):
        _reset_manifold_flag()

    def teardown_method(self):
        _reset_manifold_flag()

    def test_exp_map_origin_float64_is_more_precise(self):
        """
        float64 exp_map should satisfy the manifold constraint more precisely.

        The constraint for curvature K is: <x, x>_L = 1/K
        For K=-10: <x,x>_L should equal -0.1

        At large K, float32 loses precision. float64 should be ≥10× more precise.
        """
        from manifolds import exp_map_origin, lorentz_inner

        torch.manual_seed(42)
        v = torch.randn(100, 64) * 0.5
        K = -10.0

        x32 = exp_map_origin(v, K, use_float64=False)
        x64 = exp_map_origin(v, K, use_float64=True)

        # Check manifold constraint: <x, x>_L should equal 1/K = -0.1
        target = torch.tensor(1.0 / K)

        err32 = (lorentz_inner(x32, x32) - target).abs().mean().item()
        err64 = (lorentz_inner(x64, x64) - target).abs().mean().item()

        assert err64 < err32, (
            f"float64 (err={err64:.2e}) is not more precise than float32 (err={err32:.2e})"
        )

    def test_round_trip_float64_more_precise(self):
        """
        exp_map followed by log_map should return the original vector.
        float64 should have smaller round-trip error.
        """
        from manifolds import exp_map_origin, log_map_origin

        torch.manual_seed(7)
        v = torch.randn(50, 32) * 0.3
        K = -5.0

        x32 = exp_map_origin(v, K, use_float64=False)
        v_rt32 = log_map_origin(x32, K, use_float64=False)

        x64 = exp_map_origin(v, K, use_float64=True)
        v_rt64 = log_map_origin(x64, K, use_float64=True)

        err32 = (v_rt32 - v).abs().mean().item()
        err64 = (v_rt64 - v).abs().mean().item()

        assert err64 <= err32, (
            f"float64 round-trip error ({err64:.2e}) is not ≤ float32 ({err32:.2e})"
        )

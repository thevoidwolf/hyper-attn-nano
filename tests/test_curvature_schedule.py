"""
test_curvature_schedule.py — Tests for CurvatureSchedule classes
=================================================================

Run with:
    cd ~/hyper-attn-nano
    source activate.sh
    pytest tests/test_curvature_schedule.py -v
"""

import math
import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from training.curvature_schedule import (
    ConstantK,
    LinearWarmupK,
    CosineWarmupK,
    ExponentialK,
    build_schedule,
)

TOL = 1e-6


# ---------------------------------------------------------------------------
# ConstantK
# ---------------------------------------------------------------------------

class TestConstantK:

    def test_returns_k_at_all_steps(self):
        sched = ConstantK(-5.0)
        for step in [0, 1, 100, 500, 9999]:
            assert abs(sched.k_at_step(step) - (-5.0)) < TOL

    def test_rejects_positive_k(self):
        with pytest.raises(AssertionError):
            ConstantK(1.0)

    def test_rejects_zero_k(self):
        with pytest.raises(AssertionError):
            ConstantK(0.0)


# ---------------------------------------------------------------------------
# LinearWarmupK
# ---------------------------------------------------------------------------

class TestLinearWarmupK:

    def _make(self, k_start=-1.0, k_end=-10.0, warmup=500):
        return LinearWarmupK(k_start, k_end, warmup)

    def test_starts_at_k_start(self):
        sched = self._make()
        assert abs(sched.k_at_step(0) - (-1.0)) < TOL

    def test_reaches_k_end_at_warmup(self):
        sched = self._make()
        assert abs(sched.k_at_step(500) - (-10.0)) < TOL

    def test_returns_k_end_after_warmup(self):
        sched = self._make()
        for step in [501, 1000, 9000]:
            assert abs(sched.k_at_step(step) - (-10.0)) < TOL

    def test_midpoint(self):
        """At step warmup/2, K should be halfway between k_start and k_end."""
        sched = self._make(k_start=-1.0, k_end=-10.0, warmup=500)
        mid = sched.k_at_step(250)
        expected = -5.5  # (-1 + -10) / 2
        assert abs(mid - expected) < TOL

    def test_monotone_direction(self):
        """K should move monotonically from k_start to k_end."""
        sched = self._make(k_start=-1.0, k_end=-10.0, warmup=1000)
        ks = [sched.k_at_step(t) for t in range(0, 1001, 100)]
        for a, b in zip(ks, ks[1:]):
            assert a >= b  # more negative over time

    def test_rejects_positive_k_start(self):
        with pytest.raises(AssertionError):
            LinearWarmupK(1.0, -10.0, 500)

    def test_rejects_positive_k_end(self):
        with pytest.raises(AssertionError):
            LinearWarmupK(-1.0, 10.0, 500)

    def test_rejects_zero_warmup(self):
        with pytest.raises(AssertionError):
            LinearWarmupK(-1.0, -10.0, 0)


# ---------------------------------------------------------------------------
# CosineWarmupK
# ---------------------------------------------------------------------------

class TestCosineWarmupK:

    def _make(self, k_start=-1.0, k_end=-10.0, warmup=500):
        return CosineWarmupK(k_start, k_end, warmup)

    def test_starts_at_k_start(self):
        sched = self._make()
        assert abs(sched.k_at_step(0) - (-1.0)) < TOL

    def test_reaches_k_end_at_warmup(self):
        sched = self._make()
        assert abs(sched.k_at_step(500) - (-10.0)) < TOL

    def test_returns_k_end_after_warmup(self):
        sched = self._make()
        for step in [501, 2000, 9000]:
            assert abs(sched.k_at_step(step) - (-10.0)) < TOL

    def test_midpoint_below_linear(self):
        """Cosine warmup is slower than linear early on (convex shape)."""
        linear = LinearWarmupK(-1.0, -10.0, 1000)
        cosine = CosineWarmupK(-1.0, -10.0, 1000)
        # At 25% of warmup, cosine has moved less than linear
        assert cosine.k_at_step(250) > linear.k_at_step(250)

    def test_monotone_direction(self):
        sched = self._make(k_start=-1.0, k_end=-10.0, warmup=1000)
        ks = [sched.k_at_step(t) for t in range(0, 1001, 100)]
        for a, b in zip(ks, ks[1:]):
            assert a >= b


# ---------------------------------------------------------------------------
# ExponentialK
# ---------------------------------------------------------------------------

class TestExponentialK:

    def _make(self, k_start=-1.0, k_end=-10.0, warmup=500):
        return ExponentialK(k_start, k_end, warmup)

    def test_starts_at_k_start(self):
        sched = self._make()
        assert abs(sched.k_at_step(0) - (-1.0)) < TOL

    def test_reaches_k_end_at_warmup(self):
        sched = self._make()
        assert abs(sched.k_at_step(500) - (-10.0)) < TOL

    def test_returns_k_end_after_warmup(self):
        sched = self._make()
        for step in [501, 1000, 9000]:
            assert abs(sched.k_at_step(step) - (-10.0)) < TOL

    def test_exponential_midpoint(self):
        """At step warmup/2, exponential K should be geometric mean of start/end."""
        sched = self._make(k_start=-1.0, k_end=-10.0, warmup=1000)
        mid = sched.k_at_step(500)
        # Geometric mean of abs values: sqrt(1 * 10) = sqrt(10) ≈ 3.162
        expected = -math.sqrt(10.0)
        assert abs(mid - expected) < TOL

    def test_monotone_direction(self):
        sched = self._make(k_start=-1.0, k_end=-10.0, warmup=1000)
        ks = [sched.k_at_step(t) for t in range(0, 1001, 100)]
        for a, b in zip(ks, ks[1:]):
            assert a >= b

    def test_rejects_positive_k(self):
        with pytest.raises(AssertionError):
            ExponentialK(1.0, -10.0, 500)


# ---------------------------------------------------------------------------
# Integration test: 1000-step dummy train run
# ---------------------------------------------------------------------------

class TestIntegration:

    def test_linear_warmup_k_start_minus1_k_end_minus10_warmup500(self):
        """
        Spec requirement: log intended K at steps 0, 250, 500, 750.
        Step 0:   K = -1.0
        Step 250: K = -5.5   (midpoint)
        Step 500: K = -10.0  (warmup complete)
        Step 750: K = -10.0  (plateau)
        """
        sched = LinearWarmupK(k_start=-1.0, k_end=-10.0, warmup_steps=500)
        log = {}
        for step in range(1000):
            k = sched.k_at_step(step)
            if step in (0, 250, 500, 750):
                log[step] = k

        assert abs(log[0]   - (-1.0))  < TOL
        assert abs(log[250] - (-5.5))  < TOL
        assert abs(log[500] - (-10.0)) < TOL
        assert abs(log[750] - (-10.0)) < TOL

    def test_all_schedules_finite_and_negative_for_all_steps(self):
        schedules = [
            ConstantK(-5.0),
            LinearWarmupK(-1.0, -10.0, 500),
            CosineWarmupK(-1.0, -10.0, 500),
            ExponentialK(-1.0, -10.0, 500),
        ]
        for sched in schedules:
            for step in range(1001):
                k = sched.k_at_step(step)
                assert math.isfinite(k), f"{sched.__class__.__name__} returned non-finite at step {step}"
                assert k < 0, f"{sched.__class__.__name__} returned non-negative at step {step}"


# ---------------------------------------------------------------------------
# build_schedule factory
# ---------------------------------------------------------------------------

class TestBuildSchedule:

    def test_constant(self):
        s = build_schedule({"type": "constant", "k_end": -5.0})
        assert isinstance(s, ConstantK)
        assert abs(s.k_at_step(0) - (-5.0)) < TOL

    def test_linear_warmup(self):
        s = build_schedule({"type": "linear_warmup", "k_start": -1.0,
                            "k_end": -10.0, "warmup_steps": 500})
        assert isinstance(s, LinearWarmupK)
        assert abs(s.k_at_step(500) - (-10.0)) < TOL

    def test_cosine_warmup(self):
        s = build_schedule({"type": "cosine_warmup", "k_start": -1.0,
                            "k_end": -10.0, "warmup_steps": 500})
        assert isinstance(s, CosineWarmupK)

    def test_exponential(self):
        s = build_schedule({"type": "exponential", "k_start": -1.0,
                            "k_end": -10.0, "warmup_steps": 500})
        assert isinstance(s, ExponentialK)

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown curvature schedule"):
            build_schedule({"type": "magic", "k_start": -1.0,
                            "k_end": -10.0, "warmup_steps": 500})

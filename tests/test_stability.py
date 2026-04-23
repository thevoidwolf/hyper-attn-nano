"""
test_stability.py — Numerical stability tests for exp_map / log_map
====================================================================

Verifies that manifold operations do not produce NaN or Inf for:
  - Zero vector
  - Unit vector
  - Vector with norm = 5.0  (large)
  - Vector with norm = 20.0 (stress test, relies on angle clamping)

And that the round-trip log_map(exp_map(v)) recovers v within 1e-4
for norms up to 5.0 at K in [-1, -2] and norm=1 for K in [-5, -10].
(At K=-5, norm=5 gives angle≈11.2 which exceeds MAX_ANGLE=10 — clamping
prevents exact round-trip for those inputs by design.)

Run with:
    cd ~/hyper-attn-nano
    source activate.sh
    pytest tests/test_stability.py -v
"""

import math
import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import manifolds
from manifolds import exp_map_origin, log_map_origin

# All K values in the sweep (including K=-50 for the stress tests)
K_ALL = [-1.0, -2.0, -5.0, -10.0, -50.0]

# K values for round-trip accuracy testing
K_ROUNDTRIP = [-1.0, -2.0, -5.0, -10.0]

D = 8  # spatial dimension for all tests


def _unit_vec(d: int = D) -> torch.Tensor:
    v = torch.zeros(1, d)
    v[0, 0] = 1.0
    return v


def _normed_vec(norm_val: float, d: int = D) -> torch.Tensor:
    torch.manual_seed(7)
    v = torch.randn(4, d)
    v = v / v.norm(dim=-1, keepdim=True) * norm_val
    return v


# ---------------------------------------------------------------------------
# exp_map_origin: finite output for various input norms
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("K", K_ALL)
def test_exp_map_zero_vector(K):
    v = torch.zeros(1, D)
    out = exp_map_origin(v, K)
    assert torch.isfinite(out).all(), f"NaN/Inf in exp_map(zero, K={K})"


@pytest.mark.parametrize("K", K_ALL)
def test_exp_map_unit_vector(K):
    out = exp_map_origin(_unit_vec(), K)
    assert torch.isfinite(out).all(), f"NaN/Inf in exp_map(unit, K={K})"


@pytest.mark.parametrize("K", K_ALL)
def test_exp_map_norm5(K):
    out = exp_map_origin(_normed_vec(5.0), K)
    assert torch.isfinite(out).all(), f"NaN/Inf in exp_map(norm=5, K={K})"


@pytest.mark.parametrize("K", K_ALL)
def test_exp_map_norm20(K):
    """Stress test — angle clamping must prevent overflow."""
    out = exp_map_origin(_normed_vec(20.0), K)
    assert torch.isfinite(out).all(), f"NaN/Inf in exp_map(norm=20, K={K})"


# ---------------------------------------------------------------------------
# log_map_origin: finite output after round-trip through exp_map
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("K", K_ALL)
def test_log_map_after_exp_zero(K):
    v = torch.zeros(1, D)
    x = exp_map_origin(v, K)
    out = log_map_origin(x, K)
    assert torch.isfinite(out).all(), f"NaN/Inf in log_map(exp_map(zero), K={K})"


@pytest.mark.parametrize("K", K_ALL)
def test_log_map_after_exp_unit(K):
    x = exp_map_origin(_unit_vec(), K)
    out = log_map_origin(x, K)
    assert torch.isfinite(out).all(), f"NaN/Inf in log_map(exp_map(unit), K={K})"


@pytest.mark.parametrize("K", K_ALL)
def test_log_map_after_exp_norm5(K):
    x = exp_map_origin(_normed_vec(5.0), K)
    out = log_map_origin(x, K)
    assert torch.isfinite(out).all(), f"NaN/Inf in log_map(exp_map(norm=5), K={K})"


@pytest.mark.parametrize("K", K_ALL)
def test_log_map_after_exp_norm20(K):
    x = exp_map_origin(_normed_vec(20.0), K)
    out = log_map_origin(x, K)
    assert torch.isfinite(out).all(), f"NaN/Inf in log_map(exp_map(norm=20), K={K})"


# ---------------------------------------------------------------------------
# Round-trip accuracy: log_map(exp_map(v)) ≈ v within 1e-4
#
# Only tested where angle = sqrt(-K) * norm(v) < MAX_ANGLE.
# At K=-5, norm=5: angle = sqrt(5)*5 ≈ 11.2 > MAX_ANGLE=10 → skip.
# At K=-10, norm=5: angle = sqrt(10)*5 ≈ 15.8 > MAX_ANGLE=10 → skip.
#
# This is the most diagnostic test in the suite (per spec author):
# if round-trip fails for norm=1 at K=-5, the implementation has a
# curvature-dependent bug that corrupts all training.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("K", K_ROUNDTRIP)
@pytest.mark.parametrize("norm_val", [1.0, 5.0])
def test_round_trip_recovery(K, norm_val):
    """
    Round-trip must recover v within 1e-4 when the angle stays below MAX_ANGLE.
    Cases where angle > MAX_ANGLE are skipped — clamping is intentional there.
    """
    angle = math.sqrt(-K) * norm_val
    if angle > manifolds.MAX_ANGLE:
        pytest.skip(
            f"angle={angle:.2f} > MAX_ANGLE={manifolds.MAX_ANGLE} at K={K}, "
            f"norm={norm_val} — clamping prevents exact round-trip by design"
        )
    v = _normed_vec(norm_val)
    recovered = log_map_origin(exp_map_origin(v, K), K)
    max_err = (v - recovered).abs().max().item()
    assert max_err < 1e-4, (
        f"Round-trip failed at K={K}, norm={norm_val}: max_err={max_err:.6f} "
        f"(angle={angle:.2f})"
    )


# ---------------------------------------------------------------------------
# MAX_ANGLE constant: accessible, sensible value, and actually clamps
# ---------------------------------------------------------------------------

def test_max_angle_constant_accessible():
    assert hasattr(manifolds, "MAX_ANGLE"), "MAX_ANGLE not exported from manifolds"
    assert 1.0 <= manifolds.MAX_ANGLE <= 89.0, (
        f"MAX_ANGLE={manifolds.MAX_ANGLE} is outside sensible range [1, 89]"
    )


def test_max_angle_clamp_is_active():
    """
    Verify the clamp fires: give a vector whose angle at K=-50 is >> MAX_ANGLE
    and check that the output x0 component corresponds to cosh(MAX_ANGLE), not
    cosh(huge_number). Without the clamp this would overflow to inf.
    """
    K = -50.0
    MAX = manifolds.MAX_ANGLE
    # norm=5 at K=-50: angle = sqrt(50)*5 ≈ 35.4 >> MAX_ANGLE
    v = _normed_vec(5.0, d=1)
    out = exp_map_origin(v, K)
    x0_actual = out[..., 0]
    # If clamped at MAX_ANGLE: x0 ≈ cosh(MAX) / sqrt(-K)
    x0_expected = math.cosh(MAX) / math.sqrt(-K)
    # Allow generous tolerance — test is just checking clamping fires
    assert torch.isfinite(x0_actual).all()
    assert x0_actual.max().item() < x0_expected * 1.01 + 0.1, (
        f"x0={x0_actual.max().item():.4f} exceeds clamped ceiling {x0_expected:.4f} — "
        "MAX_ANGLE clamp may not be active"
    )

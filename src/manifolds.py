"""
manifolds.py — Lorentz (Hyperboloid) manifold operations for HyperAttn-Nano
=============================================================================

The Lorentz model of hyperbolic space works like this:

  Imagine regular 3D space, but with one "time" axis that behaves
  differently from the spatial axes. The set of all points satisfying:

      -x₀² + x₁² + x₂² + ... + xₙ² = 1/K   (K negative)

  forms a curved surface called the Lorentz hyperboloid. We live on
  this surface — it's our hyperbolic space.

  The "curvature" K controls how curved it is:
    K = -1   →  standard hyperbolic space (mildly curved)
    K = -5   →  very tightly curved (trees compress much better)
    K → 0    →  flat Euclidean space (the limit)

Two key operations:
  exp_map  — "lift" a flat Euclidean vector onto the curved surface
  log_map  — "project" a curved-surface point back to flat Euclidean

Think of it like mapping between a flat city map and the actual curved
surface of the Earth. You need both directions to do arithmetic.

FLOAT32 RULE: All operations internally cast to float32.
This is critical for numerical stability inside AMP autocast blocks,
where the ambient dtype may be float16.
"""

import torch
import torch.nn.functional as F
from typing import Union

# ---------------------------------------------------------------------------
# Type alias for curvature — always a negative scalar or broadcastable tensor
# ---------------------------------------------------------------------------
Curvature = Union[float, torch.Tensor]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as_float32(t: torch.Tensor, device=None) -> torch.Tensor:
    """Cast tensor to float32, optionally moving device."""
    if device is not None:
        return t.to(device=device, dtype=torch.float32)
    return t.float()


def _safe_norm(v: torch.Tensor, dim: int = -1, keepdim: bool = True,
               eps: float = 1e-8) -> torch.Tensor:
    """L2 norm clamped away from zero to avoid division-by-zero."""
    return v.norm(dim=dim, keepdim=keepdim).clamp(min=eps)


# ---------------------------------------------------------------------------
# Core ops — single curvature K (scalar or 0-dim tensor)
# ---------------------------------------------------------------------------

def exp_map_origin(v: torch.Tensor, K: Curvature) -> torch.Tensor:
    """
    Lift a Euclidean tangent vector to the Lorentz hyperboloid.

    Analogy: you have a flat map (v), and you're wrapping it onto a
    curved surface. exp_map tells you exactly which point on the surface
    corresponds to each point on the flat map.

    The manifold constraint for the returned point x is:
        <x, x>_L = 1/K

    Args:
        v  : Euclidean tangent vector, shape (..., d)
        K  : curvature, negative scalar (e.g. -1.0)

    Returns:
        x  : point on Lorentz manifold, shape (..., d+1)
             x[..., 0]  is the "time" component (always positive)
             x[..., 1:] are the "spatial" components
    """
    v = _as_float32(v)
    K = torch.as_tensor(K, dtype=torch.float32, device=v.device)

    sqrt_neg_K = torch.sqrt(-K)          # positive because K < 0

    norm_v = _safe_norm(v, dim=-1)       # (..., 1)
    angle  = sqrt_neg_K * norm_v        # √(-K) · ‖v‖

    # Time component: cosh(angle) / √(-K)
    x0 = torch.cosh(angle) / sqrt_neg_K                         # (..., 1)

    # Spatial components: sinh(angle) · v / (‖v‖ · √(-K))
    xi = torch.sinh(angle) * v / (norm_v * sqrt_neg_K)          # (..., d)

    return torch.cat([x0, xi], dim=-1)                           # (..., d+1)


def log_map_origin(x: torch.Tensor, K: Curvature) -> torch.Tensor:
    """
    Project a point on the Lorentz hyperboloid back to Euclidean tangent space.

    Analogy: you have a point on the curved Earth surface and you want
    to find the corresponding point on the flat map. log_map is the
    inverse of exp_map.

    CRITICAL: the arcosh argument is (√(-K) · x₀), NOT (-K · x₀).
    The latter is only correct when K = -1. For all other curvatures,
    using -K instead of √(-K) gives wrong results silently.

    Args:
        x  : point on Lorentz manifold, shape (..., d+1)
        K  : curvature, negative scalar (e.g. -1.0)

    Returns:
        v  : Euclidean tangent vector, shape (..., d)
    """
    x = _as_float32(x)
    K = torch.as_tensor(K, dtype=torch.float32, device=x.device)

    sqrt_neg_K = torch.sqrt(-K)

    x0 = x[..., :1]    # time component,    (..., 1)
    xi = x[..., 1:]    # spatial components, (..., d)

    # *** THE CRITICAL FORMULA — do not change to (-K * x0) ***
    # arcosh requires its argument ≥ 1; clamp to enforce this numerically.
    arcosh_arg = (sqrt_neg_K * x0).clamp(min=1.0 + 1e-6)
    dist = torch.acosh(arcosh_arg)          # geodesic distance from origin, (..., 1)

    norm_xi = _safe_norm(xi, dim=-1)        # (..., 1)

    # v = dist · xi / (√(-K) · ‖xi‖)
    v = dist * xi / (sqrt_neg_K * norm_xi)

    return v    # (..., d)


def lorentz_inner(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Minkowski (Lorentz) inner product:
        <x, y>_L = -x₀·y₀ + x₁·y₁ + x₂·y₂ + ... + xₙ·yₙ

    This is what replaces the dot product in hyperbolic attention.
    Two tokens on the same branch of the semantic tree have a very
    negative inner product — they are "close" in hyperbolic geometry.

    Args:
        x : shape (..., d+1)
        y : shape (..., d+1)

    Returns:
        inner product, shape (...)
    """
    # Flip the sign of the time component only
    # This is equivalent to the full sum but avoids building a sign tensor
    time  = -(x[..., 0] * y[..., 0])               # scalar, shape (...)
    space = (x[..., 1:] * y[..., 1:]).sum(dim=-1)   # sum over spatial dims
    return time + space


# ---------------------------------------------------------------------------
# Vectorised op — multiple curvatures (one per attention head)
# ---------------------------------------------------------------------------

def exp_map_batched(v: torch.Tensor, K_heads: torch.Tensor) -> torch.Tensor:
    """
    Vectorised exp_map for multiple attention heads, each with its own K.

    No Python loop over heads — everything is a single tensor operation
    via broadcasting. This is what LorentzPerHeadAttention calls.

    Args:
        v       : Euclidean vectors,  shape (B, S, H, d)
        K_heads : curvatures,         shape (H,)  — all negative

    Returns:
        x       : Lorentz points,     shape (B, S, H, d+1)
    """
    v       = _as_float32(v)
    K_heads = _as_float32(K_heads)

    # Reshape K for broadcasting over (B, S, H, d): → (1, 1, H, 1)
    K          = K_heads.view(1, 1, -1, 1)
    sqrt_neg_K = torch.sqrt(-K)                  # (1, 1, H, 1)

    norm_v = _safe_norm(v, dim=-1)               # (B, S, H, 1)
    angle  = sqrt_neg_K * norm_v                 # (B, S, H, 1)

    x0 = torch.cosh(angle) / sqrt_neg_K                 # (B, S, H, 1)
    xi = torch.sinh(angle) * v / (norm_v * sqrt_neg_K)  # (B, S, H, d)

    return torch.cat([x0, xi], dim=-1)                   # (B, S, H, d+1)


def log_map_batched(x: torch.Tensor, K_heads: torch.Tensor) -> torch.Tensor:
    """
    Vectorised log_map for multiple attention heads, each with its own K.

    Args:
        x       : Lorentz points,     shape (B, S, H, d+1)
        K_heads : curvatures,         shape (H,)  — all negative

    Returns:
        v       : Euclidean vectors,  shape (B, S, H, d)
    """
    x       = _as_float32(x)
    K_heads = _as_float32(K_heads)

    K          = K_heads.view(1, 1, -1, 1)      # (1, 1, H, 1)
    sqrt_neg_K = torch.sqrt(-K)

    x0 = x[..., :1]    # (B, S, H, 1)
    xi = x[..., 1:]    # (B, S, H, d)

    arcosh_arg = (sqrt_neg_K * x0).clamp(min=1.0 + 1e-6)
    dist       = torch.acosh(arcosh_arg)         # (B, S, H, 1)

    norm_xi = _safe_norm(xi, dim=-1)             # (B, S, H, 1)

    return dist * xi / (sqrt_neg_K * norm_xi)    # (B, S, H, d)


# ---------------------------------------------------------------------------
# Manifold constraint check (useful during debugging)
# ---------------------------------------------------------------------------

def check_on_manifold(x: torch.Tensor, K: Curvature,
                      tol: float = 1e-3) -> bool:
    """
    Verify that x satisfies the Lorentz manifold constraint: <x,x>_L = 1/K.
    Useful for debugging — call after exp_map to sanity-check.

    Returns True if all points are on the manifold within tolerance.
    """
    x = _as_float32(x)
    K = torch.as_tensor(K, dtype=torch.float32, device=x.device)

    inner    = lorentz_inner(x, x)          # should equal 1/K everywhere
    expected = (1.0 / K).expand_as(inner)
    max_err  = (inner - expected).abs().max().item()

    return max_err < tol

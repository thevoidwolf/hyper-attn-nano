"""
blocks.py -- Building blocks for HyperAttn-Nano
================================================

Three modules in dependency order:

  LorentzRMSNorm    -- RMS normalisation in tangent space, back to manifold
  LorentzFFN        -- 2-layer MLP with GELU operating on the manifold
  HyperDecoderBlock -- Full pre-norm decoder block (Lorentz attn + FFN)

All ops cast internally to float32 (critical inside AMP autocast).
All ops use fixed AMBIENT_K = -1.0 at the block level.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from manifolds import exp_map_origin, log_map_origin, lorentz_inner
from attention import LorentzPerHeadAttention

AMBIENT_K: float = -1.0


def _project_lorentz(xi: torch.Tensor, K: float = -1.0) -> torch.Tensor:
    """
    Given the spatial components xi (..., d) of a Lorentz point, recompute
    the time component x0 exactly from the manifold constraint:

        -x0^2 + ||xi||^2 = 1/K   =>   x0 = sqrt(-1/K + ||xi||^2)

    This avoids float32 catastrophic cancellation that occurs in
    cosh^2 - sinh^2 when the angle is large (||v|| >= 4).
    Returns the full Lorentz point (..., d+1).
    """
    # For K = -1: x0 = sqrt(1 + ||xi||^2)
    x0 = torch.sqrt((-1.0 / K) + xi.pow(2).sum(dim=-1, keepdim=True))
    return torch.cat([x0, xi], dim=-1)


def _exp_map_stable(v: torch.Tensor, K: float = -1.0) -> torch.Tensor:
    """
    Numerically stable exp_map_origin for use inside blocks.

    Computes xi via the standard formula, then derives x0 from the exact
    manifold constraint instead of using cosh (which loses precision at
    large angles in float32).
    """
    x_approx = exp_map_origin(v, K)
    xi = x_approx[..., 1:]          # spatial components are exact
    return _project_lorentz(xi, K)  # recompute x0 without cancellation


# ---------------------------------------------------------------------------
# LorentzRMSNorm
# ---------------------------------------------------------------------------

class LorentzRMSNorm(nn.Module):
    """
    RMS normalisation applied in Euclidean tangent space, then lifted back
    to the Lorentz manifold.

    Algorithm:
        1. log_map_origin(x, K=-1.0)  ->  Euclidean vector v  (..., d_model)
        2. RMSNorm(v)                 ->  scale * v / rms(v)
        3. exp_map_origin(v, K=-1.0)  ->  Lorentz point        (..., d_model+1)

    Input / output: (B, S, d_model+1) -- both on the K=-1 manifold.
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.full((d_model,), 1.0 / math.sqrt(d_model)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, d_model+1)
        x = x.float()
        v = log_map_origin(x, AMBIENT_K)                        # (B, S, d_model)
        rms = torch.sqrt(v.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        v_norm = v / rms                                         # (B, S, d_model)
        v_scaled = self.scale * v_norm                           # broadcast over last dim
        return _exp_map_stable(v_scaled, AMBIENT_K)              # (B, S, d_model+1)


# ---------------------------------------------------------------------------
# LorentzFFN
# ---------------------------------------------------------------------------

class LorentzFFN(nn.Module):
    """
    Two-layer MLP with GELU operating on Lorentz manifold points.

    Algorithm:
        1. log_map_origin(x, K)  ->  Euclidean v  (B, S, d_model)
        2. fc1(v) -> GELU -> fc2  ->  Euclidean v'  (B, S, d_model)
        3. exp_map_origin(v', K) ->  Lorentz point (B, S, d_model+1)

    Both linear layers have no bias. K is fixed (not learnable).
    """

    def __init__(self, d_model: int, d_ff: int, K: float = -1.0):
        super().__init__()
        self.K   = K
        self.fc1 = nn.Linear(d_model, d_ff,    bias=False)
        self.fc2 = nn.Linear(d_ff,    d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, d_model+1)
        x = x.float()
        v = log_map_origin(x, self.K)                           # (B, S, d_model)
        v = self.fc2(F.gelu(self.fc1(v)))                       # (B, S, d_model)
        return _exp_map_stable(v, self.K)                       # (B, S, d_model+1)


# ---------------------------------------------------------------------------
# HyperDecoderBlock
# ---------------------------------------------------------------------------

class HyperDecoderBlock(nn.Module):
    """
    One full pre-norm transformer decoder block on the Lorentz manifold.

    Sub-layer order: norm -> attention -> residual, then norm -> FFN -> residual.

    Residuals are added in Euclidean tangent space (log -> add -> exp).
    The attention output is already Euclidean (B, S, d_model), so the
    attention residual projects x to Euclidean, adds, then lifts back.
    The FFN residual projects both x and ffn_out to Euclidean, adds, lifts.
    """

    def __init__(
        self,
        d_model:        int,
        n_heads:        int,
        d_ff:           int,
        d_head:         int | None = None,
        curvature_init: float = -1.0,
        init_K:         float | None = None,   # backward-compat alias
    ):
        super().__init__()
        if d_head is None:
            d_head = d_model // n_heads
        if init_K is not None:
            curvature_init = init_K

        self.norm1 = LorentzRMSNorm(d_model)
        self.attn  = LorentzPerHeadAttention(d_model, n_heads, d_head, curvature_init)
        self.norm2 = LorentzRMSNorm(d_model)
        self.ffn   = LorentzFFN(d_model, d_ff)

    def forward(
        self,
        x:    torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x: (B, S, d_model+1)
        x = x.float()

        # --- Attention sub-layer ---
        x_normed = self.norm1(x)                                 # (B, S, d_model+1)
        attn_out = self.attn(x_normed, mask)                     # (B, S, d_model) Euclidean
        x_eu     = log_map_origin(x, AMBIENT_K)                  # (B, S, d_model)
        x        = _exp_map_stable(x_eu + attn_out, AMBIENT_K)   # (B, S, d_model+1)

        # --- FFN sub-layer ---
        x_normed = self.norm2(x)                                 # (B, S, d_model+1)
        ffn_out  = self.ffn(x_normed)                            # (B, S, d_model+1)
        x_eu     = log_map_origin(x,       AMBIENT_K)            # (B, S, d_model)
        ffn_eu   = log_map_origin(ffn_out, AMBIENT_K)            # (B, S, d_model)
        x        = _exp_map_stable(x_eu + ffn_eu, AMBIENT_K)    # (B, S, d_model+1)

        return x

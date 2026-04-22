"""
attention.py — Causal self-attention for HyperAttn-Nano
========================================================

Contains two modules with identical interfaces so model.py can swap them
with a single flag:

  LorentzPerHeadAttention  — the novel contribution
  EuclideanAttention       — the standard GPT-style baseline

THE CORE IDEA (plain English):
  Standard attention asks "how similar are query Q and key K?"
  using a dot product — a ruler that only works in flat space.

  Lorentz attention asks the same question but uses a ruler that
  was designed for curved space: the Minkowski inner product.

      <Q, K>_L = -Q₀·K₀ + Q₁·K₁ + Q₂·K₂ + ...
                  ^^^^ this minus sign is the entire difference

  Two tokens on the same branch of a parse tree are "close" in
  hyperbolic space — meaning their Minkowski inner product is very
  negative. The negation in the score formula flips this to a large
  positive score, so they attend strongly to each other.

THE NOVEL BIT:
  Each attention head gets its own learnable curvature K_h.
  Shallow/syntactic heads can learn K ≈ 0 (nearly flat).
  Deep/semantic heads can learn K << -1 (strongly curved).
  Nobody has measured this per-head specialisation before.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from manifolds import log_map_origin, exp_map_batched


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Upper-triangular boolean mask: True = "this position is masked out".
    Shape (1, 1, S, S) broadcasts over batch and head dimensions.

    Position [i, j] is True when j > i (token j is in the future relative
    to query position i — decoder should not see it).
    """
    ones = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
    return torch.triu(ones, diagonal=1).unsqueeze(0).unsqueeze(0)


# ---------------------------------------------------------------------------
# LorentzPerHeadAttention — the novel module
# ---------------------------------------------------------------------------

class LorentzPerHeadAttention(nn.Module):
    """
    Causal self-attention where each head computes similarity in its own
    curvature of hyperbolic space.

    Input:  x of shape (B, S, d_model+1) — points on the K=-1 Lorentz manifold
    Output: shape (B, S, d_model) — Euclidean, ready for residual addition

    The +1 in the input dimension is the Lorentz "time" component (index 0).
    The output is Euclidean so the decoder block can do plain addition:
        x_new = exp_map(log_map(x) + attn_out)

    AMBIENT_K:
        The decoder block's shared manifold uses K = -1. This is the curvature
        that LorentzRMSNorm and LorentzFFN operate with. We log_map the input
        with this curvature to get Euclidean vectors, then each head exp_maps
        Q and K with its own per-head curvature before computing scores.
    """

    AMBIENT_K: float = -1.0   # curvature of the block-level manifold

    def __init__(
        self,
        d_model:  int,
        n_heads:  int,
        d_head:   int,
        init_K:   float = -1.0,
    ):
        """
        Args:
            d_model : embedding dimension (NOT including the +1 time component)
            n_heads : number of attention heads
            d_head  : dimension per head (usually d_model // n_heads)
            init_K  : starting curvature for all heads (must be negative)
        """
        super().__init__()

        assert init_K < 0, f"init_K must be negative, got {init_K}"
        assert d_model == n_heads * d_head, (
            f"d_model ({d_model}) must equal n_heads * d_head "
            f"({n_heads} × {d_head} = {n_heads * d_head})"
        )

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head  = d_head

        # ── Per-head curvature (the novel parameter) ─────────────────────────
        #
        # We store log(|K|) instead of K directly. Why?
        # K must always be negative — if gradient descent ever made K positive
        # the manifold maths breaks. Storing log(|K|) and recovering K as:
        #
        #     K_h = -exp(log_abs_K_h)
        #
        # guarantees K is always negative for any value of log_abs_K_h.
        # It's like storing the exponent of a floating-point number — the
        # parameterisation space is unconstrained while the output is bounded.
        #
        # Initialise so K_h = init_K:
        #     -exp(log_abs_K) = init_K  →  log_abs_K = log(-init_K)
        init_log = math.log(-init_K)   # log(1.0) = 0.0 for init_K = -1.0
        self.log_abs_K = nn.Parameter(torch.full((n_heads,), init_log))

        # ── Linear projections (Euclidean, standard) ─────────────────────────
        self.W_q = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_k = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_v = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_o = nn.Linear(n_heads * d_head, d_model, bias=False)

    @property
    def K(self) -> torch.Tensor:
        """
        Current per-head curvature values.
        Always negative. Shape: (n_heads,).

        This is a derived property — the actual parameter is log_abs_K.
        """
        return -torch.exp(self.log_abs_K)

    def forward(
        self,
        x:    torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x    : (B, S, d_model+1)  — Lorentz points on the K=-1 manifold
            mask : (1, 1, S, S) bool  — True = masked (causal). Auto-generated if None.

        Returns:
            out  : (B, S, d_model)    — Euclidean, for residual add
        """
        B, S, _ = x.shape

        # Float32 throughout — critical inside AMP autocast
        x = x.float()

        # ── Step 1: Lorentz → Euclidean ──────────────────────────────────────
        # x lives on the K=-1 hyperboloid. log_map projects it to flat space
        # so we can run it through standard linear layers.
        x_eu = log_map_origin(x, self.AMBIENT_K)       # (B, S, d_model)

        # ── Step 2: Q / K / V projections ────────────────────────────────────
        Q  = self.W_q(x_eu).view(B, S, self.n_heads, self.d_head)   # (B,S,H,Dh)
        Kp = self.W_k(x_eu).view(B, S, self.n_heads, self.d_head)   # (B,S,H,Dh)
        V  = self.W_v(x_eu).view(B, S, self.n_heads, self.d_head)   # (B,S,H,Dh)

        # ── Step 3: Lift Q and K to per-head hyperbolic manifolds ─────────────
        # Each head h gets exp_map with its own curvature K_h.
        # V stays Euclidean — we aggregate values by weighted sum in flat space.
        # (Proper manifold averaging = Fréchet mean = expensive. Not needed here.)
        K_vals = self.K                                 # (H,), all negative
        Q_hyp  = exp_map_batched(Q,  K_vals)           # (B, S, H, d_head+1)
        K_hyp  = exp_map_batched(Kp, K_vals)           # (B, S, H, d_head+1)

        # ── Step 4: Lorentz attention scores ─────────────────────────────────
        # Rearrange to (B, H, S, D) for batched matmul
        Q_hyp = Q_hyp.permute(0, 2, 1, 3)             # (B, H, S_q, d_head+1)
        K_hyp = K_hyp.permute(0, 2, 1, 3)             # (B, H, S_k, d_head+1)
        V     =     V.permute(0, 2, 1, 3)             # (B, H, S,   d_head)

        # Split time (index 0) and spatial (index 1:) components
        Q_t, Q_s = Q_hyp[..., :1], Q_hyp[..., 1:]    # (B,H,S,1), (B,H,S,Dh)
        K_t, K_s = K_hyp[..., :1], K_hyp[..., 1:]

        # <Q_i, K_j>_L = -Q_t_i · K_t_j  +  Q_s_i · K_s_j
        # Both computed as batched matmuls → (B, H, S_q, S_k)
        time_term  = -(Q_t @ K_t.transpose(-2, -1))
        space_term =   Q_s @ K_s.transpose(-2, -1)
        lorentz_ip = time_term + space_term

        # score = |K_h| · (−<Q, K>_L) / √d_head
        # Negation: "very negative inner product" → "very high similarity"
        # |K_h| scaling: more curved → sharper attention distribution
        abs_K  = self.K.abs().view(1, self.n_heads, 1, 1)       # (1,H,1,1)
        scores = abs_K * (-lorentz_ip) / math.sqrt(self.d_head) # (B,H,S,S)

        # ── Step 5: Causal mask ───────────────────────────────────────────────
        if mask is None:
            mask = _causal_mask(S, x.device)
        scores = scores.masked_fill(mask, float('-inf'))

        # ── Step 6: Softmax ───────────────────────────────────────────────────
        attn_weights = F.softmax(scores, dim=-1)                 # (B,H,S_q,S_k)

        # ── Step 7: Aggregate V (Euclidean weighted sum) ──────────────────────
        out = attn_weights @ V                                   # (B,H,S_q,Dh)

        # ── Step 8: Concat heads → project to d_model ────────────────────────
        out = out.permute(0, 2, 1, 3).contiguous()              # (B,S,H,Dh)
        out = out.view(B, S, self.n_heads * self.d_head)        # (B,S,H*Dh)
        out = self.W_o(out)                                      # (B,S,d_model)

        return out   # Euclidean — decoder block handles the residual


# ---------------------------------------------------------------------------
# EuclideanAttention — standard GPT-style baseline (identical interface)
# ---------------------------------------------------------------------------

class EuclideanAttention(nn.Module):
    """
    Standard causal dot-product self-attention.

    Interface is identical to LorentzPerHeadAttention — model.py can swap
    between them with a single flag. The only differences:
      - Input x: (B, S, d_model) — no Lorentz +1 time dimension
      - K property returns None (no curvature — flat space)
      - No manifold operations
    """

    def __init__(
        self,
        d_model:  int,
        n_heads:  int,
        d_head:   int,
        init_K:   float = -1.0,   # ignored — kept for interface compatibility
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head  = d_head

        self.W_q = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_k = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_v = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_o = nn.Linear(n_heads * d_head, d_model, bias=False)

    @property
    def K(self):
        return None   # Flat space — no curvature

    def forward(
        self,
        x:    torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x   : (B, S, d_model)
            mask: (1, 1, S, S) bool, or None for auto causal mask
        Returns:
            out : (B, S, d_model)
        """
        B, S, _ = x.shape

        Q = self.W_q(x).view(B, S, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        K = self.W_k(x).view(B, S, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        V = self.W_v(x).view(B, S, self.n_heads, self.d_head).permute(0, 2, 1, 3)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_head)

        if mask is None:
            mask = _causal_mask(S, x.device)
        scores = scores.masked_fill(mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        out = attn_weights @ V

        out = out.permute(0, 2, 1, 3).contiguous().view(B, S, self.n_heads * self.d_head)
        return self.W_o(out)

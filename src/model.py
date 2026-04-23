"""
model.py -- HyperAttn-Nano and GPTNano language models
======================================================

Two decoder-only language models with identical interfaces:

  HyperAttnNano -- uses HyperDecoderBlock (Lorentz manifold, per-head K)
  GPTNano       -- uses standard Euclidean attention and flat FFN

Both accept the same config dict and expose the same forward() signature,
so the training script can swap them with a single flag.

Config keys: d_model, n_layers, n_heads, d_ff, max_seq_len, vocab_size
d_head is always derived as d_model // n_heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from manifolds import exp_map_origin, log_map_origin
from attention import EuclideanAttention, _causal_mask
from blocks    import HyperDecoderBlock, LorentzRMSNorm  # noqa: F401

# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

NANO_CONFIG = {
    'd_model':     128,
    'n_layers':    4,
    'n_heads':     4,
    'd_ff':        512,
    'max_seq_len': 256,
    'vocab_size':  50257,
}

MAIN_CONFIG = {
    'd_model':     256,
    'n_layers':    6,
    'n_heads':     8,
    'd_ff':        1024,
    'max_seq_len': 256,
    'vocab_size':  50257,
}


# ---------------------------------------------------------------------------
# HyperAttnNano
# ---------------------------------------------------------------------------

class HyperAttnNano(nn.Module):
    """
    Decoder-only language model using Lorentz (hyperbolic) attention.

    Forward pass:
        1. Token embedding  ->  (B, S, d_model) Euclidean
        2. exp_map_origin   ->  (B, S, d_model+1) Lorentz
        3. N x HyperDecoderBlock
        4. LorentzRMSNorm
        5. log_map_origin   ->  (B, S, d_model) Euclidean
        6. lm_head          ->  (B, S, vocab_size) logits
        Weight tying: lm_head.weight == embed.weight
    """

    def __init__(
        self,
        config:         dict,
        fixed_curvature: float = -1.0,
        init_K:          float | None = None,   # backward-compat alias
        curvature_init:  float | None = None,   # per-head learnable init (hyper-perhead)
    ):
        super().__init__()
        d_model     = config['d_model']
        n_layers    = config['n_layers']
        n_heads     = config['n_heads']
        d_ff        = config['d_ff']
        vocab_size  = config['vocab_size']
        d_head      = d_model // n_heads

        # init_K kept for backward compatibility; fixed_curvature takes precedence
        if init_K is not None:
            fixed_curvature = init_K
        self.fixed_curvature = fixed_curvature

        # curvature_init: where per-head curvatures start when learnable.
        # Falls back to fixed_curvature so hyper-fixed still works correctly.
        block_init = curvature_init if curvature_init is not None else fixed_curvature

        self.d_model    = d_model
        self.n_layers   = n_layers
        self.n_heads    = n_heads
        self.vocab_size = vocab_size

        self.embed = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embed.weight, std=0.02)

        self.blocks = nn.ModuleList([
            HyperDecoderBlock(d_model, n_heads, d_ff, d_head, curvature_init=block_init)
            for _ in range(n_layers)
        ])

        self.norm = LorentzRMSNorm(d_model)

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

        # Enable float64 manifold ops if requested by config (Experiment B2)
        if config.get("manifold_float64", False):
            import manifolds as _manifolds
            _manifolds.MANIFOLD_FLOAT64 = True

    def forward(self, input_ids, targets=None):
        B, S = input_ids.shape
        emb  = self.embed(input_ids).float()
        x    = exp_map_origin(emb, K=-1.0)
        mask = _causal_mask(S, input_ids.device)
        for block in self.blocks:
            x = block(x, mask)
        x      = self.norm(x)
        x      = log_map_origin(x, K=-1.0)
        logits = self.lm_head(x)
        loss   = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
        return logits, loss

    def get_curvatures(self):
        result = {}
        for i, block in enumerate(self.blocks):
            K_vals = block.attn.K.detach()
            for j in range(self.n_heads):
                result[f"layer_{i}_head_{j}"] = K_vals[j].item()
        return result


# ---------------------------------------------------------------------------
# GPTNano -- standard Euclidean baseline
# ---------------------------------------------------------------------------

class _RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps   = eps
        self.scale = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.scale * (x / rms)


class _EuclideanFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff,    bias=False)
        self.fc2 = nn.Linear(d_ff,    d_model, bias=False)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class _EuclideanDecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, d_head: int):
        super().__init__()
        self.norm1 = _RMSNorm(d_model)
        self.attn  = EuclideanAttention(d_model, n_heads, d_head)
        self.norm2 = _RMSNorm(d_model)
        self.ffn   = _EuclideanFFN(d_model, d_ff)

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x


class GPTNano(nn.Module):
    """Standard Euclidean GPT-style decoder-only language model."""

    def __init__(self, config: dict, init_K: float = -1.0):
        super().__init__()
        d_model    = config['d_model']
        n_layers   = config['n_layers']
        n_heads    = config['n_heads']
        d_ff       = config['d_ff']
        vocab_size = config['vocab_size']
        d_head     = d_model // n_heads

        self.d_model    = d_model
        self.vocab_size = vocab_size

        self.embed = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embed.weight, std=0.02)

        self.blocks = nn.ModuleList([
            _EuclideanDecoderBlock(d_model, n_heads, d_ff, d_head)
            for _ in range(n_layers)
        ])

        self.norm    = _RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

    def forward(self, input_ids, targets=None):
        B, S   = input_ids.shape
        x      = self.embed(input_ids)
        mask   = _causal_mask(S, input_ids.device)
        for block in self.blocks:
            x = block(x, mask)
        x      = self.norm(x)
        logits = self.lm_head(x)
        loss   = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
        return logits, loss

    def get_curvatures(self):
        return {}

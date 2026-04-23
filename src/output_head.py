"""
output_head.py — Spherical output head for HyperAttn-Nano
==========================================================

SphericalOutputHead replaces the standard nn.Linear lm_head with a
cosine-similarity classifier scaled by a learned temperature.

Architecture:
  h_norm = F.normalize(h, dim=-1)           # [B, T, D] — project onto unit sphere
  e_norm = F.normalize(embedding, dim=-1)   # [V, D]    — project onto unit sphere
  logits = exp(log_τ) * (h_norm @ e_norm.T) # [B, T, V]

Motivation (Fisher-Rao / spherical geometry):
  The probability simplex under the square-root embedding is isometric to
  the positive orthant of the unit sphere.  Standard softmax after a
  Euclidean linear layer ignores this geometry.  Cosine-similarity softmax
  respects it, and is known (from face-recognition literature: ArcFace,
  CosFace) to produce better-separated class representations and more
  calibrated probabilities.

Weight tying:
  NOT used for the spherical head.  The head owns its own nn.Parameter
  embedding and normalises it at forward-time; sharing the token embedding
  (which is used un-normalised in the encoder path) would create a
  conflicting gradient signal.

Temperature:
  Stored as log_temperature (a learned scalar).  Init at log(10) ≈ 2.3
  gives a similar logit scale to a standard linear head at the start of
  training.  The optimiser is free to adjust it.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SphericalOutputHead(nn.Module):
    """
    Cosine-similarity output head with learned temperature.

    Parameters
    ----------
    d_model : int
        Hidden size of the final transformer layer.
    vocab_size : int
        Number of output classes (vocabulary).
    temperature_init : float
        Initial value for the temperature scalar τ.  The parameter is stored
        as log(τ) so it stays strictly positive throughout training.
    """

    def __init__(self, d_model: int, vocab_size: int, temperature_init: float = 10.0):
        super().__init__()
        if temperature_init <= 0:
            raise ValueError(f"temperature_init must be positive, got {temperature_init}")
        self.embedding = nn.Parameter(torch.randn(vocab_size, d_model) * 0.02)
        self.log_temperature = nn.Parameter(torch.tensor(math.log(temperature_init)))

    def forward(self, h: Tensor) -> Tensor:
        """
        Parameters
        ----------
        h : Tensor, shape [B, T, D]
            Final hidden states from the transformer body.

        Returns
        -------
        logits : Tensor, shape [B, T, V]
            Temperature-scaled cosine similarities.  Pass through
            F.cross_entropy (which calls log_softmax internally).
        """
        h_norm = F.normalize(h, dim=-1)               # [B, T, D]
        e_norm = F.normalize(self.embedding, dim=-1)  # [V, D]
        logits = self.log_temperature.exp() * (h_norm @ e_norm.t())
        return logits

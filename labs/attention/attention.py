"""Lab 05 — attention.

The one file you edit. Every ``TODO(you)`` is a ``raise NotImplementedError``
for you to replace. Run ``pytest`` until it is green, then drop this file on
the lab page or run ``PREFIX=<you> python verify.py``.

Lessons: M1.L3 (attention), M1.L2 (embeddings and position — RoPE).

Shape conventions (T0.L2): B batch, T sequence length, C model width,
H heads, D = C / H head width. Everything is numpy; no loops over B, T or H
— broadcasting and ``@`` on the last two axes do the batching.
"""

from __future__ import annotations

import numpy as np


def softmax(x, axis=-1):
    """Numerically stable softmax along ``axis`` (O1.L2): subtract the max first.

    Must handle rows that contain ``-np.inf`` (masked positions) — exp(−inf)
    is 0, which is exactly what masking wants.
    """
    # TODO(you)
    raise NotImplementedError("softmax")


def causal_mask(T):
    """A (T, T) boolean array, True where query i may attend to key j: j <= i."""
    # TODO(you)
    raise NotImplementedError("causal_mask")


def attention(q, k, v, mask=None):
    """Scaled dot-product attention.

    q, k, v: (..., T, D). Scores are ``q @ kᵀ / √D`` with shape (..., T, T);
    where ``mask`` is False, set the score to −inf before the softmax; the
    output is ``softmax(scores) @ v`` with shape (..., T, D).

    Transpose only the last two axes of k (``np.swapaxes(k, -1, -2)``), so
    the same code works for (T, D), (B, T, D) and (B, H, T, D).
    """
    # TODO(you)
    raise NotImplementedError("attention")


def split_heads(x, n_heads):
    """(B, T, C) → (B, H, T, D): reshape the last axis into H × D, then move
    H in front of T so each head is an independent (T, D) attention problem."""
    # TODO(you)
    raise NotImplementedError("split_heads")


def merge_heads(x):
    """(B, H, T, D) → (B, T, C): the exact inverse of split_heads."""
    # TODO(you)
    raise NotImplementedError("merge_heads")


def multi_head_attention(x, Wq, Wk, Wv, Wo, n_heads, causal=True):
    """x: (B, T, C); each W: (C, C). Project to q, k, v, split into heads,
    attend (with the causal mask when ``causal``), merge, project with Wo."""
    # TODO(you)
    raise NotImplementedError("multi_head_attention")


def rope(x, base=10000.0):
    """Rotary position embedding (M1.L2) for x of shape (..., T, D), D even.

    Treat dimensions (0, 1), (2, 3), … as pairs. Pair i at position t is
    rotated by the angle ``t · θ_i`` with ``θ_i = base ** (−2i / D)``:

        out[..., 2i]   = x[..., 2i] · cos − x[..., 2i+1] · sin
        out[..., 2i+1] = x[..., 2i] · sin + x[..., 2i+1] · cos

    Positions are 0 … T−1 along the second-to-last axis.
    """
    # TODO(you)
    raise NotImplementedError("rope")

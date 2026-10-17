"""Lab 03 — optimisers, schedules and clipping.

The one file you edit. Every ``TODO(you)`` is a ``raise NotImplementedError``
for you to replace. Run ``pytest`` until it is green, then drop this file on
the lab page or run ``PREFIX=<you> python verify.py``.

Lessons: O1.L3 (optimisers), O1.L4 (learning-rate schedules), P1.L6
(clipping).

Everything here works on plain lists of numpy arrays, so it is independent of
any autograd engine: ``params`` is a list of arrays that you update IN PLACE
(``p -= ...``, ``p *= ...``), and ``step(grads)`` receives a list of arrays of
the same shapes. Updating in place matters: the caller holds references to
the same arrays, exactly as PyTorch's optimizer holds the model's tensors.

The formulas follow PyTorch's ``torch.optim.SGD``, ``Adam`` and ``AdamW``
(default arguments), so your numbers can be compared with theirs.
"""

from __future__ import annotations

import math

import numpy as np


class SGD:
    """p ← p − lr · g, optionally with (heavy-ball) momentum.

    With momentum μ, keep one buffer per parameter, starting at zero:

        buf ← μ · buf + g
        p   ← p − lr · buf

    (This is PyTorch's convention, with dampening 0 and no Nesterov.)
    """

    def __init__(self, params, lr, momentum=0.0):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.buf = [np.zeros_like(p) for p in params]

    def step(self, grads):
        # TODO(you)
        raise NotImplementedError("SGD.step")


class Adam:
    """Adam, and AdamW when ``decoupled=True`` and ``weight_decay > 0``.

    Per parameter, with step counter t starting at 1:

        m ← β1 · m + (1 − β1) · g             first moment: a running mean of g
        v ← β2 · v + (1 − β2) · g²            second moment: a running mean of g²
        m̂ = m / (1 − β1ᵗ)                     bias correction — m and v start at 0
        v̂ = v / (1 − β2ᵗ)
        p ← p − lr · m̂ / (√v̂ + ε)

    Weight decay, two ways (O1.L3):

      * ``decoupled=False`` (Adam + L2): add ``weight_decay · p`` to g BEFORE
        the moments, so the decay is rescaled by √v̂ like any gradient.
      * ``decoupled=True`` (AdamW): shrink p directly, BEFORE the Adam update:
        ``p ← p · (1 − lr · weight_decay)``; g is untouched.
    """

    def __init__(self, params, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, decoupled=True):
        self.params = params
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.decoupled = decoupled
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads):
        # TODO(you)
        raise NotImplementedError("Adam.step")


def lr_at(step, max_lr, warmup, total, min_lr=0.0):
    """Linear warmup, then cosine decay to ``min_lr`` (O1.L4).

    Steps are counted from 0.

      * ``step < warmup``:   ``max_lr · (step + 1) / warmup``
                             (so the last warmup step reaches max_lr exactly)
      * ``step >= total``:   ``min_lr``
      * otherwise:           with ``progress = (step − warmup) / (total − warmup)``,
                             ``min_lr + ½ · (1 + cos(π · progress)) · (max_lr − min_lr)``
    """
    # TODO(you)
    raise NotImplementedError("lr_at")


def clip_grad_norm(grads, max_norm):
    """Clip by GLOBAL norm, in place, and return the norm BEFORE clipping.

    The global norm is √(Σ over every element of every gradient of g²) — one
    number for the whole model, not one per tensor. If it exceeds
    ``max_norm``, multiply every gradient by ``max_norm / (norm + 1e-6)``;
    otherwise leave them alone. (PyTorch's ``clip_grad_norm_`` does the same.)
    """
    # TODO(you)
    raise NotImplementedError("clip_grad_norm")

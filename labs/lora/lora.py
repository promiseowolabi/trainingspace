"""Lab 12 — LoRA and supervised fine-tuning.

The one file you edit. Every ``TODO(you)`` is a ``raise NotImplementedError``
for you to replace. This is a LOCAL lab: it needs PyTorch (CPU build). Run
``pytest`` until it is green, then ``PREFIX=<you> python verify.py`` and
paste the completion code on the lab page.

Lessons: F1.L3 (LoRA), F1.L2 (SFT and chat templates).

LoRA in one line (F1.L3): keep a pretrained weight W frozen and learn a
low-rank correction, so the layer computes

    y = x·Wᵀ + b + (α / r) · x·Aᵀ·Bᵀ        A: (r, in),  B: (out, r)

with B initialised to ZERO — so a freshly wrapped model computes exactly
what the base model did — and A initialised randomly. Only A and B train.
``model.py`` (the GPT of lab 07) is given for the parameter-count check.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Wrap an existing ``nn.Linear``.

    In ``__init__``: keep ``base``; freeze every base parameter
    (``requires_grad_(False)``); store ``r``, ``alpha`` and ``scale = alpha / r``;
    create ``self.A`` as an ``nn.Parameter`` of shape (r, in_features),
    initialised with ``nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))``,
    and ``self.B`` as an ``nn.Parameter`` of zeros, shape (out_features, r).
    """

    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        # TODO(you)
        raise NotImplementedError("LoRALinear.__init__")

    def forward(self, x):
        """base(x) plus the scaled low-rank path. Never form B·A here: two
        thin matmuls (x·Aᵀ, then ·Bᵀ) are what make LoRA cheap."""
        # TODO(you)
        raise NotImplementedError("LoRALinear.forward")

    @torch.no_grad()
    def merged(self) -> nn.Linear:
        """A plain ``nn.Linear`` computing the same function: weight
        W + scale · B·A, the same bias. After merging, inference costs exactly
        what the base model cost (F1.L3)."""
        # TODO(you)
        raise NotImplementedError("LoRALinear.merged")


def apply_lora(model: nn.Module, targets, r: int, alpha: float):
    """Freeze EVERY parameter of ``model``, then replace each ``nn.Linear``
    whose full module name ends with one of ``targets`` (e.g. ``("qkv",
    "proj")``) by a ``LoRALinear`` wrapping it. Return the model.

    Walk ``model.named_modules()``; for each module, look at
    ``module.named_children()`` and use ``setattr(module, child_name, …)``
    to swap a child in place. Full names look like ``"blocks.0.qkv"``.
    """
    # TODO(you)
    raise NotImplementedError("apply_lora")


def count_params(model: nn.Module):
    """Return ``(trainable, total)`` parameter counts."""
    # TODO(you)
    raise NotImplementedError("count_params")


def sft_loss(logits, targets, loss_mask):
    """Supervised fine-tuning loss (F1.L2): cross-entropy averaged over the
    positions where ``loss_mask`` is True only — the response tokens — so the
    model is never trained to predict the prompt.

    logits (B, T, V), targets (B, T) ints, loss_mask (B, T) bool. Use
    ``F.cross_entropy(..., reduction="none")``, zero the masked-out
    positions, divide by the number of positions that count.
    """
    # TODO(you)
    raise NotImplementedError("sft_loss")

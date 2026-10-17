"""Lab 08 — budgeting a training run.

The one file you edit. Every ``TODO(you)`` is a ``raise NotImplementedError``
for you to replace. Pure Python arithmetic. Run ``pytest`` until it is green,
then drop this file on the lab page or run ``PREFIX=<you> python verify.py``.

Lessons: P1.L4 (FLOPs and MFU), P1.L5 (scaling laws), T0.L4 (memory),
M1.L5 (counting a GPT).

Every function is one or two lines. The point is not the code: it is that
after this lab you can price any training run from its config, its token
count and its hardware — and check the numbers in a paper (E1.L4).
Units: FLOPs and FLOP/s as plain floats, memory in GB = 1e9 bytes.
"""

from __future__ import annotations

import math


def param_count(V, T, L, C, tied=True):
    """Exact parameter count of a GPT-2-style model (M1.L5): embeddings
    V·C and T·C, L blocks of 12·C² + 13·C, a final LayerNorm of 2·C. An
    untied output layer (``tied=False``) adds another V·C."""
    # TODO(you)
    raise NotImplementedError("param_count")


def train_flops(n_params, n_tokens):
    """Total training compute: 6 FLOPs per parameter per token (P1.L4) —
    2 forward + 4 backward (A1.L2)."""
    # TODO(you)
    raise NotImplementedError("train_flops")


def forward_flops_per_token(n_params, L, T, C):
    """Forward FLOPs per token including attention (Kaplan et al. 2020):
    ``2·N + 2·L·T·C``."""
    # TODO(you)
    raise NotImplementedError("forward_flops_per_token")


def mfu(n_params, tokens_per_second, peak_flops):
    """Model FLOPs utilisation (P1.L4): the training FLOP/s your token
    throughput implies (6·N per token) divided by the hardware's peak.
    A fraction, e.g. 0.42."""
    # TODO(you)
    raise NotImplementedError("mfu")


def training_days(n_params, n_tokens, n_gpus, peak_flops, mfu_):
    """Wall-clock days to train: total FLOPs ÷ (GPUs × peak × MFU) ÷ 86,400."""
    # TODO(you)
    raise NotImplementedError("training_days")


def state_memory_gb(n_params, bytes_per_param=16):
    """Weights + gradients + optimiser state, in GB (T0.L4)."""
    # TODO(you)
    raise NotImplementedError("state_memory_gb")


def activation_memory_gb(L, s, b, h, a, flash=False):
    """Activation memory in GB (Korthikanti et al. 2022, 16-bit, T0.L4):
    per layer ``s·b·h·34 + 5·a·s²·b`` bytes, for sequence length s,
    micro-batch b, hidden size h and a heads. FlashAttention (``flash=True``)
    never stores the attention matrix, which removes the second term."""
    # TODO(you)
    raise NotImplementedError("activation_memory_gb")


def chinchilla_optimal(compute, tokens_per_param=20):
    """Compute-optimal (N, D) for a FLOP budget (P1.L5): Chinchilla's rule
    of thumb D = 20·N, together with C = 6·N·D. Solve for N, return (N, D)
    as floats."""
    # TODO(you)
    raise NotImplementedError("chinchilla_optimal")

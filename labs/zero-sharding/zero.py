"""Lab 11 — ZeRO sharding over simulated ranks.

The one file you edit. Every ``TODO(you)`` is a ``raise NotImplementedError``
for you to replace. Run ``pytest`` until it is green, then drop this file on
the lab page or run ``PREFIX=<you> python verify.py``.

Lesson: D1.L3 (ZeRO and FSDP).

N simulated ranks train one model whose parameters are a flat 1-D array.
Every rank holds the full parameters (``params[r]``) and computes a full
gradient on its own data (``grads[r]``), exactly like data parallelism. What
ZeRO changes is the OPTIMISER: instead of every rank keeping Adam's m and v
for all parameters, rank r keeps them only for its 1/N shard, updates only
that shard, and the ranks then exchange the updated shards.

The collectives are given (you built real ones in lab 09), and so is the
Adam update rule (lab 03). Your job is the sharding.
"""

from __future__ import annotations

import numpy as np


def reduce_scatter_mean(arrays, n):
    """Given: average N full arrays and return the N shards of the result —
    shard r is what rank r receives from a reduce-scatter."""
    total = sum(arrays) / len(arrays)
    return np.array_split(total, n)


def all_gather(shards):
    """Given: concatenate every rank's shard — what each rank receives from
    an all-gather."""
    return np.concatenate(shards)


def adam_update(p, g, m, v, t, lr, b1=0.9, b2=0.999, eps=1e-8):
    """Given: one in-place Adam step (lab 03) on arrays p, m, v at step t."""
    m *= b1
    m += (1 - b1) * g
    v *= b2
    v += (1 - b2) * g * g
    p -= lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)


def shard(flat, n):
    """Split a flat array into n shards with ``np.array_split`` and return
    COPIES (a shard must be updatable without touching the original)."""
    # TODO(you)
    raise NotImplementedError("shard")


def unshard(shards):
    """The inverse of shard."""
    # TODO(you)
    raise NotImplementedError("unshard")


class ShardState:
    """Rank ``rank``'s Adam state: m and v sized for ITS shard only
    (the length of shard ``rank`` of an ``n_params`` array split ``n_ranks``
    ways), both zeros, and a step counter t = 0."""

    def __init__(self, n_params, n_ranks, rank):
        # TODO(you)
        raise NotImplementedError("ShardState")


def zero_step(params, grads, states, lr):
    """One ZeRO-1 step across N = len(params) simulated ranks.

    1. Reduce-scatter the gradients: rank r gets the MEAN gradient for its
       shard only (``reduce_scatter_mean``).
    2. Each rank r takes its own shard of its parameters, increments its
       state's t, and runs ``adam_update`` on that shard with its m and v.
    3. All-gather the updated shards and write the full result into EVERY
       rank's ``params[r]`` in place (``params[r][...] = full``).

    After the step all ranks hold identical parameters — the same ones a
    single process running Adam on the mean gradient would have.
    """
    # TODO(you)
    raise NotImplementedError("zero_step")


def memory_per_rank_gb(n_params, n_ranks, stage, k=12):
    """Training-state memory per rank, in GB, for ZeRO stage 0–3 (D1.L3).

    Mixed-precision Adam holds 2 bytes of weights, 2 of gradients and k = 12
    of optimiser state (fp32 master, m, v) per parameter. Stage 1 shards the
    optimiser state, stage 2 also the gradients, stage 3 also the weights:

        stage 0: (2 + 2 + k)·Ψ          stage 2: (2 + (2 + k)/N)·Ψ
        stage 1: (2 + 2 + k/N)·Ψ        stage 3: (2 + 2 + k)·Ψ / N
    """
    # TODO(you)
    raise NotImplementedError("memory_per_rank_gb")


def comm_volume(n_params, stage):
    """Elements each rank communicates per step (D1.L3): 2·Ψ for stages
    0–2 — a reduce-scatter plus an all-gather, the same as one all-reduce —
    and 3·Ψ for stage 3, which all-gathers the weights twice (forward and
    backward) and reduce-scatters the gradients once."""
    # TODO(you)
    raise NotImplementedError("comm_volume")

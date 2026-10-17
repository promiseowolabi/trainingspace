"""Lab 09 — ring collectives over simulated ranks.

The one file you edit. Every ``TODO(you)`` is a ``raise NotImplementedError``
for you to replace. Run ``pytest`` until it is green, then drop this file on
the lab page or run ``PREFIX=<you> python verify.py``.

Lesson: D1.L1 (collectives).

N ranks sit in a ring. Rank r can send only to rank (r + 1) % N, and all
ranks send at the same time, once per step — the ``Ring`` class below (given)
enforces that and counts every byte. ``xs[r]`` is rank r's array; every rank
splits its array into N chunks the same way (``chunks_of``, given).

The two phases of a ring all-reduce (D1.L1):

  reduce-scatter  N − 1 steps. Each step, every rank sends ONE chunk's
                  running sum to its neighbour, which adds it to its own copy
                  of that chunk. Afterwards rank r holds the fully reduced
                  (summed over all ranks) chunk r.
  all-gather      N − 1 more steps. Each step, every rank forwards one
                  complete chunk; afterwards every rank has all N.

Each rank sends 2·(N − 1) chunks of size 1/N of the array: 2(N − 1)/N of
the array's bytes in total, whatever N is. That is why the ring is the
bandwidth-optimal all-reduce, and the checks measure it.
"""

from __future__ import annotations

import numpy as np


class Ring:
    """A ring network of n ranks (given — do not edit).

    ``step({src: array, ...})`` delivers each array from rank ``src`` to rank
    ``(src + 1) % n`` and returns ``{dst: array}``. Each call is one
    synchronous step; ``steps`` counts them and ``sent[r]`` counts the bytes
    rank r has sent.
    """

    def __init__(self, n):
        self.n = n
        self.sent = [0] * n
        self.steps = 0

    def step(self, messages):
        if not messages:
            raise ValueError("a step must send at least one message")
        recv = {}
        for src, arr in messages.items():
            if not 0 <= src < self.n:
                raise ValueError(f"no rank {src}")
            arr = np.asarray(arr)
            self.sent[src] += arr.nbytes
            recv[(src + 1) % self.n] = arr.copy()
        self.steps += 1
        return recv


def chunks_of(x, n):
    """Split a 1-D array into n chunks, the same way on every rank (given)."""
    return np.array_split(x, n)


def ring_reduce_scatter(ring, xs):
    """Return a list whose element r is the chunk rank r ends up owning: the
    SUM over all ranks of chunk r.

    Keep a per-rank list of chunk buffers (copies!). At step s (0 … N−2),
    rank r sends its buffer for chunk ``(r − s − 1) % N``; the receiver adds
    the incoming array into ITS buffer for the same chunk index. Work out why
    that index makes rank r finish with chunk r — then do it with
    ``ring.step`` only: no reading another rank's buffers directly.
    """
    # TODO(you)
    raise NotImplementedError("ring_reduce_scatter")


def ring_all_gather(ring, chunks):
    """Rank r starts with only ``chunks[r]``. Return a list of N full arrays
    (``np.concatenate`` of chunks 0 … N−1), one per rank.

    At step s (0 … N−2), rank r forwards chunk ``(r − s) % N`` — the one it
    received at the previous step (or its own, at step 0).
    """
    # TODO(you)
    raise NotImplementedError("ring_all_gather")


def ring_all_reduce(ring, xs):
    """Every rank gets the sum of all ``xs``: reduce-scatter, then all-gather."""
    # TODO(you)
    raise NotImplementedError("ring_all_reduce")


def ring_bytes_per_rank(n_bytes, n):
    """Bytes each rank sends in a ring all-reduce of an ``n_bytes`` array."""
    # TODO(you)
    raise NotImplementedError("ring_bytes_per_rank")


def ring_time(n_bytes, n, bandwidth, latency):
    """Seconds for a ring all-reduce: 2·(N − 1) steps, each paying the
    per-message ``latency`` plus one chunk (n_bytes / N) at ``bandwidth``
    bytes per second."""
    # TODO(you)
    raise NotImplementedError("ring_time")

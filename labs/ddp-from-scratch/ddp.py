"""Lab 10 — data parallelism from scratch.

The one file you edit. Every ``TODO(you)`` is a ``raise NotImplementedError``
for you to replace. This is a LOCAL lab: it needs PyTorch (CPU build), and
the checks start two real processes that talk over ``torch.distributed``'s
``gloo`` backend on your machine — the same API a GPU cluster uses with
``nccl``. Run ``pytest`` until it is green, then ``PREFIX=<you> python
verify.py`` and paste the code on the lab page.

Lesson: D1.L2 (data parallelism).

Inside every function below, you are ONE rank. ``dist.get_rank()`` says
which; ``dist.get_world_size()`` says how many there are. Every rank runs
the same code on a different slice of the batch, and the collectives
(``dist.broadcast``, ``dist.all_reduce``) are where they meet. A collective
must be called by EVERY rank, in the same order, or the program hangs.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def setup(rank, world_size, port):
    """Join the process group: set the environment variables
    ``MASTER_ADDR`` ("127.0.0.1") and ``MASTER_PORT`` (``str(port)``), then
    ``dist.init_process_group("gloo", rank=rank, world_size=world_size)``."""
    # TODO(you)
    raise NotImplementedError("setup")


def cleanup():
    """Leave the process group (given)."""
    dist.destroy_process_group()


def shard_batch(x, y, rank, world_size):
    """This rank's contiguous slice of a global batch whose first dimension
    is divisible by world_size: rank 0 gets the first B/W rows, and so on.
    Pure tensor slicing — no communication."""
    # TODO(you)
    raise NotImplementedError("shard_batch")


def broadcast_params(model):
    """Make every rank's parameters equal to rank 0's: ``dist.broadcast``
    each parameter's ``.data`` in place from ``src=0``. Call it once, before
    training — every replica must start identical."""
    # TODO(you)
    raise NotImplementedError("broadcast_params")


def allreduce_grads(model):
    """Average gradients across ranks: for each parameter with a gradient,
    ``dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)`` then divide by the
    world size. Afterwards every rank holds the SAME gradient — the one a
    single process would have computed on the whole batch."""
    # TODO(you)
    raise NotImplementedError("allreduce_grads")


def plan_buckets(sizes_bytes, bucket_bytes):
    """Group gradient indices, in order, into buckets of at most
    ``bucket_bytes``: start a new bucket when adding the next gradient would
    exceed the limit (a single gradient larger than the limit gets a bucket
    of its own). Return a list of lists of indices.

        plan_buckets([100, 100, 300, 50], 250) -> [[0, 1], [2], [3]]
    """
    # TODO(you)
    raise NotImplementedError("plan_buckets")


def allreduce_grads_bucketed(model, bucket_bytes):
    """Like allreduce_grads, but one all_reduce per BUCKET instead of per
    parameter (D1.L2): for each bucket from ``plan_buckets``, flatten its
    gradients into one tensor (``torch.cat`` of ``g.reshape(-1)``),
    all_reduce it, divide by the world size, and copy the slices back into
    each ``p.grad``. Bucket sizes come from ``g.numel() * g.element_size()``.
    Return the number of all_reduce calls made."""
    # TODO(you)
    raise NotImplementedError("allreduce_grads_bucketed")


def ddp_step(model, optimizer, x, y, loss_fn, bucket_bytes=None):
    """One data-parallel step on the GLOBAL batch (x, y): take this rank's
    shard, forward, ``loss_fn(output, target)``, backward, average gradients
    (bucketed if ``bucket_bytes`` is given), ``optimizer.step()``,
    ``optimizer.zero_grad(set_to_none=True)``. Return the local loss as a float.
    """
    # TODO(you)
    raise NotImplementedError("ddp_step")

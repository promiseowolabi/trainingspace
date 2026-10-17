"""Lab 07 — the training loop.

The one file you edit. Every ``TODO(you)`` is a ``raise NotImplementedError``
for you to replace. This is a LOCAL lab: it needs PyTorch (the CPU build is
enough). Run ``pytest`` until it is green, then ``PREFIX=<you> python
verify.py`` and paste the completion code on the lab page. ``python run.py``
trains a model with your functions and prints samples.

Lessons: P1.L2 (the training loop), P1.L3 (mixed precision), E1.L3
(checkpointing and resumption).

``model.py`` (the GPT, lab 06 in PyTorch) and ``data.py`` (a character-level
tokenizer over this course's own lessons) are given — do not edit them.
"""

from __future__ import annotations

import math

import torch


def get_batch(data, block_size, batch_size, generator):
    """Sample ``batch_size`` random windows from the 1-D LongTensor ``data``.

    Draw start indices with ``torch.randint(len(data) - block_size,
    (batch_size,), generator=generator)``. x is ``data[i : i + block_size]``
    for each start i, and y is the same window shifted one to the right —
    the next token at every position. Return (x, y), each (batch_size,
    block_size). Use ONLY ``generator`` for randomness, so a run can be
    resumed exactly (E1.L3).
    """
    # TODO(you)
    raise NotImplementedError("get_batch")


def configure_optimizer(model, lr, weight_decay, betas=(0.9, 0.95)):
    """AdamW with two parameter groups (O1.L3, P1.L2):

    * every parameter with ``p.dim() >= 2`` (weight matrices, embeddings)
      gets ``weight_decay``;
    * every other parameter (biases, LayerNorm gains) gets 0 — decaying a
      norm's gain towards zero only fights the normalisation.

    Return ``torch.optim.AdamW(groups, lr=lr, betas=betas)``.
    """
    # TODO(you)
    raise NotImplementedError("configure_optimizer")


def lr_at(step, max_lr, warmup, total, min_lr=0.0):
    """Warmup + cosine — lab 03's lr_at (given)."""
    if step < warmup:
        return max_lr * (step + 1) / warmup
    if step >= total:
        return min_lr
    progress = (step - warmup) / (total - warmup)
    return min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (max_lr - min_lr)


def train_step(model, optimizer, micro_batches, grad_clip=1.0, amp=False):
    """One optimiser step over a list of (x, y) micro-batches.

    1. ``model.train()``.
    2. For each micro-batch: forward (``_, loss = model(x, y)``) inside
       ``torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp)``, then
       ``(loss / len(micro_batches)).backward()`` — dividing makes the
       accumulated gradient the MEAN over micro-batches, identical to one
       big batch (gradient accumulation, P1.L2).
    3. Clip with ``torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)``
       and keep the norm it returns (the norm BEFORE clipping).
    4. ``optimizer.step()``, then ``optimizer.zero_grad(set_to_none=True)``.

    Return ``(mean_loss, grad_norm)`` as two Python floats.
    """
    # TODO(you)
    raise NotImplementedError("train_step")


@torch.no_grad()
def evaluate(model, data, block_size, batch_size, n_batches, generator):
    """Mean loss over ``n_batches`` random batches of ``data``, in eval mode.

    Put the model in eval mode, and restore whatever mode it was in before
    you return. ``@torch.no_grad()`` already stops autograd from recording.
    """
    # TODO(you)
    raise NotImplementedError("evaluate")


def save_checkpoint(path, model, optimizer, step, generator):
    """Save everything a bit-exact resume needs (E1.L3) with ``torch.save``:
    the model's state_dict, the optimizer's state_dict (Adam's m and v!),
    the step, and ``generator.get_state()`` (which batches come next)."""
    # TODO(you)
    raise NotImplementedError("save_checkpoint")


def load_checkpoint(path, model, optimizer, generator):
    """Restore what save_checkpoint saved, into existing objects, and return
    the step. ``torch.load(path)`` returns the dict you saved."""
    # TODO(you)
    raise NotImplementedError("load_checkpoint")

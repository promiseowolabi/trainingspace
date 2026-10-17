"""Lab 13 — preference and RL losses.

The one file you edit. Every ``TODO(you)`` is a ``raise NotImplementedError``
for you to replace. Run ``pytest`` until it is green, then drop this file on
the lab page or run ``PREFIX=<you> python verify.py``.

Lessons: F1.L4 (reward models), F1.L5 (RLHF and DPO), F1.L6 (GRPO).

Post-training is a short list of loss functions. Each one here works on
numbers a training loop would already have: summed log-probabilities of a
response under the policy and under a frozen reference model, rewards, and
advantages. Arrays are numpy; "mean" means over the batch.

Numerical rule for the whole file: never write ``np.log(1 / (1 + np.exp(-x)))``.
Use ``log_sigmoid`` (given), which is stable for x = ±1000.
"""

from __future__ import annotations

import numpy as np


def log_sigmoid(x):
    """log σ(x), stable (given): −log(1 + e^(−x)) via np.logaddexp."""
    return -np.logaddexp(0.0, -x)


def sequence_logprob(logits, tokens, mask):
    """Sum of log-probabilities of ``tokens`` under ``logits``, over the
    positions where ``mask`` is 1 (the response, not the prompt).

    logits (B, T, V), tokens (B, T) ints, mask (B, T) 0/1 → (B,).
    Use a stable log-softmax (O1.L2) and ``np.take_along_axis`` to pick each
    position's target log-probability.
    """
    # TODO(you)
    raise NotImplementedError("sequence_logprob")


def bradley_terry_loss(r_chosen, r_rejected):
    """Reward-model loss (F1.L4): mean of −log σ(r_chosen − r_rejected).
    Return a Python float."""
    # TODO(you)
    raise NotImplementedError("bradley_terry_loss")


def dpo_loss(pi_chosen, pi_rejected, ref_chosen, ref_rejected, beta=0.1):
    """Direct Preference Optimisation (F1.L5). Inputs are summed response
    log-probabilities (B,) under the policy (pi_*) and the frozen reference
    (ref_*). With margin = β · [(π_c − ref_c) − (π_r − ref_r)], return the
    mean of −log σ(margin) as a float."""
    # TODO(you)
    raise NotImplementedError("dpo_loss")


def dpo_rewards(pi_logp, ref_logp, beta=0.1):
    """DPO's implicit reward of a response: β · (log π − log π_ref)."""
    # TODO(you)
    raise NotImplementedError("dpo_rewards")


def dpo_grad_weight(pi_chosen, pi_rejected, ref_chosen, ref_rejected, beta=0.1):
    """How strongly DPO pushes on each pair: σ(−margin) — the probability the
    current implicit rewards rank the pair WRONG. Near 1 for a mis-ordered
    pair, near 0 for one already learned. Return an array (B,)."""
    # TODO(you)
    raise NotImplementedError("dpo_grad_weight")


def ppo_clip_objective(logp_new, logp_old, advantages, eps=0.2):
    """PPO's clipped surrogate (F1.L5), to be MAXIMISED. With
    ratio = exp(logp_new − logp_old), return the mean of
    min(ratio · A, clip(ratio, 1 − ε, 1 + ε) · A) as a float."""
    # TODO(you)
    raise NotImplementedError("ppo_clip_objective")


def grpo_advantages(rewards, eps=1e-6):
    """GRPO's group-relative advantages (F1.L6). ``rewards`` has shape
    (n_prompts, group_size): several sampled answers per prompt. Normalise
    within each row: (r − row mean) / (row std + eps), population std
    (numpy's default)."""
    # TODO(you)
    raise NotImplementedError("grpo_advantages")


def kl_k3(logp_policy, logp_ref):
    """The per-token KL estimator used in GRPO-style training (F1.L6):
    with log_ratio = logp_ref − logp_policy,
    ``exp(log_ratio) − 1 − log_ratio``. Always ≥ 0, and 0 exactly when the
    two log-probabilities are equal. Elementwise."""
    # TODO(you)
    raise NotImplementedError("kl_k3")

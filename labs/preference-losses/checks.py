"""Lab 13 checks. The same file runs under pytest, verify.py and the browser.

Each check receives your module as ``m``. Check ids (the function name
without ``check_``) must match src/data/labs.ts.
"""

import math

import numpy as np


def check_sequence_logprob(m):
    r = np.random.default_rng(0)
    logits = r.normal(size=(2, 5, 7)) * 3
    tokens = r.integers(0, 7, size=(2, 5))
    mask = np.array([[0, 0, 1, 1, 1], [0, 1, 1, 1, 1]], dtype=float)
    got = np.asarray(m.sequence_logprob(logits, tokens, mask))
    p = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)
    want = np.array([sum(math.log(p[b, t, tokens[b, t]]) for t in range(5) if mask[b, t]) for b in range(2)])
    assert got.shape == (2,), f"one summed log-probability per sequence; got shape {got.shape}"
    assert np.allclose(got, want), "sum of log p(token) over masked-in positions only"
    huge = np.asarray(m.sequence_logprob(logits * 1000, tokens, mask))
    assert np.all(np.isfinite(huge)), "use a stable log-softmax: logits × 1000 must not give inf/nan"


def check_bradley_terry(m):
    assert math.isclose(m.bradley_terry_loss(np.array([0.0]), np.array([0.0])), math.log(2)), "equal rewards: −log σ(0) = ln 2"
    got = m.bradley_terry_loss(np.array([2.0, -1.0]), np.array([0.0, 1.0]))
    want = (math.log(1 + math.exp(-2)) + math.log(1 + math.exp(2))) / 2
    assert math.isclose(got, want, rel_tol=1e-9), f"expected {want:.6f}, got {got}"
    assert math.isfinite(m.bradley_terry_loss(np.array([-1000.0]), np.array([1000.0]))), "stable for huge margins: use log_sigmoid"


def check_dpo_loss(m):
    args = (np.array([-10.0, -20.0]), np.array([-12.0, -18.0]), np.array([-11.0, -19.0]), np.array([-11.0, -19.0]))
    got = m.dpo_loss(*args, beta=0.5)
    margins = 0.5 * np.array([(-10 + 11) - (-12 + 11), (-20 + 19) - (-18 + 19)])  # [1, -1]
    want = float(np.mean(np.log1p(np.exp(-margins))))
    assert math.isclose(got, want, rel_tol=1e-9), f"expected {want:.6f}, got {got}"
    same = m.dpo_loss(np.array([-5.0]), np.array([-7.0]), np.array([-5.0]), np.array([-7.0]))
    assert math.isclose(same, math.log(2)), "a policy equal to its reference has margin 0 and loss ln 2 — DPO starts there"
    rewards = np.asarray(m.dpo_rewards(np.array([-10.0]), np.array([-12.0]), beta=0.1))
    assert np.allclose(rewards, [0.2]), "implicit reward = β · (log π − log π_ref)"


def check_dpo_gradient(m):
    beta, h = 0.3, 1e-6
    pc, pr, rc, rr = np.array([-9.0, -6.0]), np.array([-8.0, -4.0]), np.array([-10.0, -5.0]), np.array([-7.5, -5.5])  # margins +0.45, −0.75
    w = np.asarray(m.dpo_grad_weight(pc, pr, rc, rr, beta=beta))
    for i in range(2):
        e = np.zeros(2)
        e[i] = h
        d_chosen = (m.dpo_loss(pc + e, pr, rc, rr, beta) - m.dpo_loss(pc - e, pr, rc, rr, beta)) / (2 * h)
        d_rejected = (m.dpo_loss(pc, pr + e, rc, rr, beta) - m.dpo_loss(pc, pr - e, rc, rr, beta)) / (2 * h)
        assert d_chosen < 0 < d_rejected, "lowering the loss means raising log π(chosen) and lowering log π(rejected)"
        assert math.isclose(-d_chosen, beta * w[i] / 2, rel_tol=1e-5), (
            f"pair {i}: ∂loss/∂log π(chosen) should be −β · σ(−margin) / B = {-beta * w[i] / 2:.6f}, finite difference says {d_chosen:.6f}"
        )
    assert w[1] > w[0], "the pair the implicit reward ranks WRONG (pair 1) must get the larger weight"


def check_ppo_clip(m):
    lo = math.log
    assert math.isclose(m.ppo_clip_objective(np.array([lo(1.5)]), np.array([0.0]), np.array([1.0])), 1.2), (
        "ratio 1.5 with A = +1: the gain is capped at (1 + ε)·A = 1.2"
    )
    assert math.isclose(m.ppo_clip_objective(np.array([lo(0.5)]), np.array([0.0]), np.array([-1.0])), -0.8), (
        "ratio 0.5 with A = −1: min(−0.5, −0.8) = −0.8 — the pessimistic bound"
    )
    assert math.isclose(m.ppo_clip_objective(np.array([lo(1.1)]), np.array([0.0]), np.array([2.0])), 2.2), "inside the clip range the objective is ratio · A"


def check_grpo_advantages(m):
    a = np.asarray(m.grpo_advantages(np.array([[1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0], [3.0, 1.0, 2.0, 2.0]])))
    assert a.shape == (3, 4)
    assert np.allclose(a[0], [1, -1, -1, 1], atol=1e-5), "two right, two wrong: +1 for right answers, −1 for wrong"
    assert np.allclose(a[1], 0) and np.all(np.isfinite(a[1])), "a group where every answer scored the same has no signal: advantages 0, not nan"
    assert np.allclose(a[2].mean(), 0, atol=1e-9) and np.isclose(a[2].std(), 1, atol=1e-4), "each row normalised to mean 0, std 1"


def check_kl_k3(m):
    r = np.random.default_rng(1)
    x, y = r.normal(size=50), r.normal(size=50)
    k = np.asarray(m.kl_k3(x, y))
    assert np.all(k >= -1e-12), "the k3 estimator is never negative"
    assert np.allclose(np.asarray(m.kl_k3(x, x)), 0), "zero when the log-probabilities agree"
    p = np.array([0.5, 0.3, 0.15, 0.05])
    q = np.array([0.25, 0.25, 0.25, 0.25])
    samples = r.choice(4, size=400_000, p=p)
    est = float(np.mean(np.asarray(m.kl_k3(np.log(p[samples]), np.log(q[samples])))))
    true_kl = float((p * np.log(p / q)).sum())
    assert abs(est - true_kl) < 0.01, f"averaged over samples from the policy, k3 estimates KL(π‖ref) = {true_kl:.4f}; got {est:.4f}"


CHECKS = [
    check_sequence_logprob,
    check_bradley_terry,
    check_dpo_loss,
    check_dpo_gradient,
    check_ppo_clip,
    check_grpo_advantages,
    check_kl_k3,
]

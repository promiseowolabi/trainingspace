"""Lab 03 checks. The same file runs under pytest, verify.py and the browser.

Each check receives your module as ``m``. Check ids (the function name
without ``check_``) must match src/data/labs.ts. The reference formulas below
are PyTorch's documented update rules (torch.optim.SGD / Adam / AdamW).
"""

import math

import numpy as np


def _ref_adam(p, grads, lr, b1=0.9, b2=0.999, eps=1e-8, wd=0.0, decoupled=True):
    p = p.copy()
    m = np.zeros_like(p)
    v = np.zeros_like(p)
    for t, g in enumerate(grads, start=1):
        if wd and not decoupled:
            g = g + wd * p
        if wd and decoupled:
            p = p * (1 - lr * wd)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        p = p - lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
    return p


def check_sgd_step(m):
    p = np.array([1.0, -2.0, 3.0])
    ref = p
    opt = m.SGD([p], lr=0.1)
    opt.step([np.array([0.5, 0.5, -1.0])])
    assert ref is p, "update the parameter IN PLACE (p -= ...) — the caller holds the same array"
    assert np.allclose(p, [0.95, -2.05, 3.1]), f"p − lr·g: expected [0.95, -2.05, 3.1], got {p}"


def check_momentum(m):
    p = np.zeros(2)
    opt = m.SGD([p], lr=0.1, momentum=0.9)
    g = np.array([1.0, -1.0])
    for _ in range(3):
        opt.step([g.copy()])
    # buf: 1, 1.9, 2.71 → total step 0.1·(1 + 1.9 + 2.71) = 0.561
    assert np.allclose(p, [-0.561, 0.561]), f"after 3 steps of a constant gradient expected ±0.561, got {p}"


def check_adam_first_step(m):
    p = np.array([0.0, 0.0, 0.0])
    opt = m.Adam([p], lr=0.01)
    opt.step([np.array([1e-4, -3.0, 250.0])])
    assert np.allclose(p, [-0.01, 0.01, -0.01], atol=1e-6), (
        f"with bias correction the first Adam step is ≈ lr·sign(g) whatever |g| is; got {p}. "
        "Without bias correction it is (1−β1)/√(1−β2) ≈ 3.2× too large (O1.L3)"
    )


def check_adam_matches_reference(m):
    rng = np.random.default_rng(0)
    p0 = rng.normal(size=(3, 4))
    grads = [rng.normal(size=(3, 4)) for _ in range(20)]
    p = p0.copy()
    opt = m.Adam([p], lr=3e-3, betas=(0.9, 0.95))
    for g in grads:
        opt.step([g.copy()])
    want = _ref_adam(p0, grads, 3e-3, b2=0.95)
    assert np.allclose(p, want, rtol=1e-7, atol=1e-10), "20 steps of Adam differ from the reference update rule"


def check_adamw_decoupled(m):
    p = np.array([2.0, -4.0])
    opt = m.Adam([p], lr=0.1, weight_decay=0.5, decoupled=True)
    opt.step([np.zeros(2)])
    assert np.allclose(p, [1.9, -3.8]), (
        f"AdamW with a zero gradient only shrinks p by (1 − lr·wd) = 0.95: expected [1.9, -3.8], got {p}"
    )
    rng = np.random.default_rng(1)
    p0 = rng.normal(size=5)
    grads = [rng.normal(size=5) for _ in range(10)]
    for decoupled in (True, False):
        p = p0.copy()
        opt = m.Adam([p], lr=1e-2, weight_decay=0.1, decoupled=decoupled)
        for g in grads:
            opt.step([g.copy()])
        want = _ref_adam(p0, grads, 1e-2, wd=0.1, decoupled=decoupled)
        name = "AdamW (decoupled)" if decoupled else "Adam + L2 (decoupled=False)"
        assert np.allclose(p, want, rtol=1e-7, atol=1e-10), f"{name} differs from the reference"


def check_warmup_cosine(m):
    f = lambda s: m.lr_at(s, max_lr=1e-3, warmup=100, total=1100, min_lr=1e-4)  # noqa: E731
    cases = [(0, 1e-5), (49, 5e-4), (99, 1e-3), (600, 5.5e-4), (1100, 1e-4), (5000, 1e-4)]
    for step, want in cases:
        got = f(step)
        assert math.isclose(got, want, rel_tol=1e-9), f"lr_at({step}) = {got}, expected {want}"
    vals = [f(s) for s in range(100, 1100)]
    assert all(a >= b for a, b in zip(vals, vals[1:])), "after warmup the schedule must never increase"


def check_clip_norm(m):
    g1, g2 = np.array([3.0, 0.0]), np.array([[0.0], [4.0]])
    total = m.clip_grad_norm([g1, g2], max_norm=1.0)
    assert math.isclose(total, 5.0, rel_tol=1e-9), f"the global norm of [3, 0] and [[0], [4]] is 5, got {total}"
    assert math.isclose(math.sqrt((g1 ** 2).sum() + (g2 ** 2).sum()), 1.0, rel_tol=1e-5), (
        "after clipping the global norm must be max_norm — scale IN PLACE, every tensor by the same factor"
    )
    assert math.isclose(g1[0] / g2[1, 0], 0.75, rel_tol=1e-9), "the direction must not change: one factor for all tensors"
    small = [np.array([0.1, 0.2])]
    total = m.clip_grad_norm(small, max_norm=1.0)
    assert np.allclose(small[0], [0.1, 0.2]), "gradients under the limit must be left alone"
    assert math.isclose(total, math.sqrt(0.05), rel_tol=1e-9)


def check_conditioning(m):
    H = np.array([1000.0, 1.0])  # curvatures 1000 and 1: condition number 1000

    def run(make, schedule=None):
        x = np.array([1.0, 1.0])
        opt = make(x)
        for s in range(300):
            if schedule:
                opt.lr = schedule(s)
            opt.step([H * x])
        return 0.5 * float((H * x * x).sum())

    sgd = run(lambda x: m.SGD([x], lr=0.0019))
    adam = run(lambda x: m.Adam([x], lr=0.05), lambda s: m.lr_at(s, 0.05, 10, 300))
    assert sgd > 0.1, f"SGD at its largest stable lr should still be stuck (> 0.1) after 300 steps, got {sgd:.3g}"
    assert adam < 1e-6, (
        f"Adam with warmup-cosine should reach < 1e-6 on the same bowl, got {adam:.3g} — "
        "per-coordinate scaling is the point of the second moment"
    )


CHECKS = [
    check_sgd_step,
    check_momentum,
    check_adam_first_step,
    check_adam_matches_reference,
    check_adamw_decoupled,
    check_warmup_cosine,
    check_clip_norm,
    check_conditioning,
]

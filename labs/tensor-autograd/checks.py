"""Lab 02 checks. The same file runs under pytest, verify.py and the browser.

Each check receives your module as ``m``. Check ids (the function name
without ``check_``) must match src/data/labs.ts.
"""

import numpy as np


def rand(*shape, seed=0):
    return np.random.default_rng(seed).normal(size=shape)


def check_unbroadcast(m):
    g = np.arange(12, dtype=float).reshape(3, 4)
    cases = [
        ((4,), g.sum(axis=0)),
        ((1, 4), g.sum(axis=0, keepdims=True)),
        ((3, 1), g.sum(axis=1, keepdims=True)),
        ((), np.array(g.sum())),
        ((3, 4), g),
    ]
    for shape, want in cases:
        got = np.asarray(m.unbroadcast(g.copy(), shape))
        assert got.shape == shape, f"unbroadcast to {shape} returned shape {got.shape}"
        assert np.allclose(got, want), f"unbroadcast to {shape}: wrong values"
    g3 = np.ones((2, 3, 4))
    got = np.asarray(m.unbroadcast(g3, (3, 1)))
    assert got.shape == (3, 1) and np.allclose(got, 8.0), "(2,3,4) → (3,1): drop the leading axis, then sum the stretched one"


def check_add_broadcast(m):
    T = m.Tensor
    x, b = T(rand(5, 3, seed=1)), T(rand(3, seed=2))
    (x + b).sum().backward()
    assert x.grad.shape == (5, 3) and np.allclose(x.grad, 1.0), "x.grad should be all ones"
    assert b.grad.shape == (3,), f"b.grad must keep b's shape (3,), got {b.grad.shape}"
    assert np.allclose(b.grad, 5.0), "b was broadcast over 5 rows, so each element gets gradient 5"


def check_mul(m):
    T = m.Tensor
    xd, yd = rand(4, 3, seed=3), rand(4, 1, seed=4)
    x, y = T(xd), T(yd)
    (x * y).sum().backward()
    assert np.allclose(x.grad, np.broadcast_to(yd, (4, 3))), "d(x·y)/dx = y"
    assert y.grad.shape == (4, 1) and np.allclose(y.grad, xd.sum(axis=1, keepdims=True)), (
        "d(x·y)/dy = x, summed over the axis y was stretched along"
    )


def check_matmul(m):
    T = m.Tensor
    ad, bd = rand(4, 3, seed=5), rand(3, 2, seed=6)
    a, b = T(ad), T(bd)
    w = rand(4, 2, seed=7)
    (a @ b * T(w)).sum().backward()  # upstream gradient G = w
    assert a.grad.shape == (4, 3) and np.allclose(a.grad, w @ bd.T), "dA = G @ Bᵀ"
    assert b.grad.shape == (3, 2) and np.allclose(b.grad, ad.T @ w), "dB = Aᵀ @ G"


def check_sum_axis(m):
    T = m.Tensor
    w = rand(3, seed=8)
    x = T(rand(4, 3, seed=9))
    (x.sum(axis=0) * T(w)).sum().backward()
    assert np.allclose(x.grad, np.broadcast_to(w, (4, 3))), "sum(axis=0): every row receives the (3,) upstream gradient"
    x2 = T(rand(4, 3, seed=10))
    w2 = rand(4, 1, seed=11)
    (x2.sum(axis=1, keepdims=True) * T(w2)).sum().backward()
    assert np.allclose(x2.grad, np.broadcast_to(w2, (4, 3))), "sum(axis=1, keepdims=True)"
    x3 = T(rand(4, 3, seed=12))
    w3 = rand(4, seed=13)
    (x3.sum(axis=1) * T(w3)).sum().backward()
    assert np.allclose(x3.grad, np.broadcast_to(w3[:, None], (4, 3))), "sum(axis=1) without keepdims: expand the axis back first"


def check_relu(m):
    T = m.Tensor
    d = np.array([[-2.0, 0.0, 3.0], [1.0, -0.5, 2.0]])
    x = T(d)
    y = x.relu()
    assert np.allclose(y.data, np.maximum(d, 0)), "relu forward"
    (y * T(np.full((2, 3), 2.0))).sum().backward()
    assert np.allclose(x.grad, 2.0 * (d > 0)), "relu passes the gradient where data > 0, blocks it elsewhere"


def _naive_ce(z, t):
    p = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    return -np.log(p[np.arange(len(t)), t]).mean()


def check_cross_entropy_value(m):
    z = rand(6, 5, seed=14)
    t = np.array([0, 4, 2, 2, 1, 3])
    loss = m.cross_entropy(m.Tensor(z), t)
    assert np.ndim(loss.data) == 0, f"the loss is a scalar, got shape {np.shape(loss.data)}"
    assert np.isclose(float(loss.data), _naive_ce(z, t)), "value differs from -mean(log softmax[target])"
    uniform = m.cross_entropy(m.Tensor(np.zeros((3, 10))), np.array([1, 2, 3]))
    assert np.isclose(float(uniform.data), np.log(10)), "all-zero logits over 10 classes must give ln 10 ≈ 2.303 (E1.L2)"


def check_cross_entropy_stable(m):
    z = rand(4, 5, seed=15) * 1000.0
    t = np.array([1, 0, 3, 4])
    with np.errstate(all="ignore"):
        loss = m.cross_entropy(m.Tensor(z), t)
    assert np.isfinite(loss.data), "logits ×1000 gave inf/nan — subtract the row max before exp (log-sum-exp)"
    shifted = z - z.max(axis=1, keepdims=True)
    want = -(shifted[np.arange(4), t] - np.log(np.exp(shifted).sum(axis=1))).mean()
    assert np.isclose(float(loss.data), want, rtol=1e-6)


def check_cross_entropy_grad(m):
    z = rand(6, 5, seed=16)
    t = np.array([0, 4, 2, 2, 1, 3])
    logits = m.Tensor(z)
    m.cross_entropy(logits, t).backward()
    p = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    p[np.arange(6), t] -= 1
    assert np.allclose(logits.grad, p / 6), "gradient should be (softmax - onehot) / N"


def check_gradcheck(m):
    T = m.Tensor

    def good(x, w):
        return ((x @ w).relu() * (x @ w)).sum()

    x, w = T(rand(3, 4, seed=17)), T(rand(4, 2, seed=18))
    assert m.gradcheck(good, [x, w]) is True, "gradcheck rejected a correct function"
    assert np.allclose(x.data, rand(3, 4, seed=17)), "gradcheck must restore every element it nudges"

    def bad(x):
        out = T(x.data ** 2, (x,), "square")

        def _bw():
            x.grad += out.grad * x.data  # wrong: d(x²)/dx = 2x

        out._backward = _bw
        return out.sum()

    assert m.gradcheck(bad, [T(rand(2, 3, seed=19))]) is False, "gradcheck accepted a backward pass that is off by 2×"


def check_trains_mlp(m):
    T = m.Tensor
    rng = np.random.default_rng(20)
    centers = np.array([[0.0, 2.0], [-2.0, -1.0], [2.0, -1.0]])
    y = np.repeat(np.arange(3), 50)
    X = centers[y] + rng.normal(scale=0.5, size=(150, 2))

    W1 = T(rng.normal(size=(2, 16)) * np.sqrt(2 / 2))
    b1 = T(np.zeros(16))
    W2 = T(rng.normal(size=(16, 3)) * np.sqrt(2 / 16))
    b2 = T(np.zeros(3))
    params = [W1, b1, W2, b2]
    first = None
    for _ in range(150):
        for p in params:
            p.zero_grad()
        h = (T(X) @ W1 + b1).relu()
        loss = m.cross_entropy(h @ W2 + b2, y)
        loss.backward()
        first = float(loss.data) if first is None else first
        for p in params:
            p.data -= 0.5 * p.grad
    final = float(loss.data)
    assert final < 0.15, f"loss went {first:.3f} → {final:.3f} in 150 steps; expected < 0.15 on three separable blobs"


CHECKS = [
    check_unbroadcast,
    check_add_broadcast,
    check_mul,
    check_matmul,
    check_sum_axis,
    check_relu,
    check_cross_entropy_value,
    check_cross_entropy_stable,
    check_cross_entropy_grad,
    check_gradcheck,
    check_trains_mlp,
]

"""Lab 00 checks. The same file runs under pytest, verify.py and the browser.

Each check receives your module as ``m``. Check ids (the function name
without ``check_``) must match src/data/labs.ts.
"""

import numpy as np


def check_broadcast_rules(m):
    cases = [
        ((8, 1, 6), (7, 1), (8, 7, 6)),
        ((5, 4), (1,), (5, 4)),
        ((5, 4), (4,), (5, 4)),
        ((15, 3, 5), (15, 1, 5), (15, 3, 5)),
        ((1, 3), (4, 1), (4, 3)),
        ((), (3, 2), (3, 2)),
        ((1,), (0,), (0,)),
    ]
    for a, b, want in cases:
        got = m.broadcast_shape(a, b)
        assert tuple(got) == want, f"broadcast_shape({a}, {b}) = {got}, expected {want}"
        got_rev = m.broadcast_shape(b, a)
        assert tuple(got_rev) == want, f"broadcast_shape({b}, {a}) = {got_rev}, expected {want} (order must not matter)"


def check_broadcast_errors(m):
    for a, b in [((3,), (4,)), ((2, 1), (8, 4, 3)), ((4, 3), (4,))]:
        try:
            got = m.broadcast_shape(a, b)
        except ValueError:
            continue
        raise AssertionError(f"broadcast_shape({a}, {b}) returned {got}; numpy raises — so must you (ValueError)")


def check_pairwise_dists(m):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5, 3))
    Y = rng.normal(size=(4, 3))
    got = np.asarray(m.pairwise_sq_dists(X, Y))
    assert got.shape == (5, 4), f"shape {got.shape}, expected (5, 4): one row per X row, one column per Y row"
    want = np.array([[((x - y) ** 2).sum() for y in Y] for x in X])
    assert np.allclose(got, want), "values differ from the loop version — check which axis you reduce"


def check_linear_batched(m):
    rng = np.random.default_rng(1)
    W = rng.normal(size=(4, 3))
    b = rng.normal(size=(3,))
    for lead in [(), (2,), (2, 5)]:
        x = rng.normal(size=lead + (4,))
        got = np.asarray(m.linear(x, W, b))
        assert got.shape == lead + (3,), f"x{x.shape} -> {got.shape}, expected {lead + (3,)}"
        assert np.allclose(got, x @ W + b), f"wrong values for x of shape {x.shape}"


def check_matmul_flops(m):
    assert m.matmul_flops(2, 3, 4) == 48, "2·m·k·n: a multiply and an add per term, k terms per output, m·n outputs"
    assert m.matmul_flops(4096, 4096, 4096) == 2 * 4096 ** 3
    assert m.matmul_flops(1, 1, 1) == 2


def check_arithmetic_intensity(m):
    got = m.arithmetic_intensity(4096, 4096, 4096)
    assert abs(got - 4096 / 3) < 1e-6, f"square 4096 matmul in bf16: expected ≈1365.3 FLOP/byte, got {got}"
    got = m.arithmetic_intensity(1, 4096, 4096)
    assert abs(got - 2 * 4096 * 4096 / (2 * (4096 + 4096 * 4096 + 4096))) < 1e-9, (
        f"matrix-vector (m=1): expected ≈1.0 FLOP/byte, got {got} — a batch of one is memory-bound"
    )
    got = m.arithmetic_intensity(64, 64, 64, bytes_per_el=4)
    assert abs(got - 2 * 64 ** 3 / (4 * 3 * 64 * 64)) < 1e-9, "bytes_per_el must scale the bytes moved"


def check_memory_16(m):
    assert m.training_bytes_per_param() == 16, (
        f"got {m.training_bytes_per_param()}: bf16 weights (2) + bf16 grads (2) + fp32 master (4) + Adam m (4) + Adam v (4)"
    )
    got = m.training_memory_gb(7_000_000_000)
    assert abs(got - 112.0) < 1e-6, f"7B params → expected 112 GB before activations, got {got}"


CHECKS = [
    check_broadcast_rules,
    check_broadcast_errors,
    check_pairwise_dists,
    check_linear_batched,
    check_matmul_flops,
    check_arithmetic_intensity,
    check_memory_16,
]

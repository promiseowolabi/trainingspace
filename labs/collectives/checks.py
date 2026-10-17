"""Lab 09 checks. The same file runs under pytest, verify.py and the browser.

Each check receives your module as ``m``. Check ids (the function name
without ``check_``) must match src/data/labs.ts.
"""

import numpy as np


def _xs(n, size=96, seed=0):
    r = np.random.default_rng(seed)
    return [r.normal(size=size) for _ in range(n)]


def check_reduce_scatter(m):
    for n in (2, 3, 4, 6):
        xs = _xs(n)
        before = [x.copy() for x in xs]
        ring = m.Ring(n)
        out = m.ring_reduce_scatter(ring, xs)
        total = np.array_split(sum(xs), n)
        assert len(out) == n, f"N={n}: return one chunk per rank"
        for r in range(n):
            assert np.allclose(out[r], total[r]), f"N={n}: rank {r} must own the fully summed chunk {r}"
        assert ring.steps == n - 1, f"N={n}: reduce-scatter takes N − 1 = {n - 1} steps, you used {ring.steps}"
        assert all(np.array_equal(a, b) for a, b in zip(xs, before)), "do not modify the input arrays — copy the chunks"


def check_all_gather(m):
    for n in (2, 3, 5):
        full = np.arange(n * 8, dtype=float)
        chunks = np.array_split(full, n)
        ring = m.Ring(n)
        out = m.ring_all_gather(ring, chunks)
        assert len(out) == n and all(np.array_equal(o, full) for o in out), f"N={n}: every rank must end with all chunks, in order"
        assert ring.steps == n - 1, f"N={n}: all-gather takes N − 1 steps, you used {ring.steps}"


def check_all_reduce(m):
    for n in (2, 4, 7):
        xs = _xs(n, size=7 * 12, seed=n)
        ring = m.Ring(n)
        out = m.ring_all_reduce(ring, xs)
        s = sum(xs)
        assert len(out) == n and all(np.allclose(o, s) for o in out), f"N={n}: every rank must hold the sum"
        assert ring.steps == 2 * (n - 1), f"N={n}: all-reduce takes 2(N − 1) = {2 * (n - 1)} steps, you used {ring.steps}"


def check_ring_traffic(m):
    for n in (2, 4, 8):
        xs = _xs(n, size=1024, seed=10 + n)
        ring = m.Ring(n)
        m.ring_all_reduce(ring, xs)
        want = 2 * (n - 1) / n * xs[0].nbytes
        assert all(abs(b - want) < 1e-9 for b in ring.sent), (
            f"N={n}: every rank should send 2(N−1)/N × {xs[0].nbytes} = {want:.0f} bytes; sent {ring.sent}"
        )


def check_bytes_formula(m):
    assert m.ring_bytes_per_rank(1000, 2) == 1000
    assert abs(m.ring_bytes_per_rank(1000, 8) - 1750) < 1e-9
    big = m.ring_bytes_per_rank(14e9, 1024)
    assert abs(big - 2 * 1023 / 1024 * 14e9) < 1, "approaches 2 × the array size as N grows: ≈ 28 GB for 14 GB of bf16 gradients"


def check_time_model(m):
    t = m.ring_time(14e9, 8, 400e9, 5e-6)
    want = 2 * 7 * (5e-6 + 14e9 / 8 / 400e9)
    assert abs(t - want) < 1e-12, f"14 GB over 8 ranks at 400 GB/s with 5 µs latency: expected {want:.4f} s, got {t}"
    small = m.ring_time(4e3, 1024, 400e9, 5e-6)
    assert small > 0.0099, "a tiny message on 1,024 ranks is dominated by 2(N − 1) latencies (≈ 10 ms) — that is why gradients are bucketed"


CHECKS = [
    check_reduce_scatter,
    check_all_gather,
    check_all_reduce,
    check_ring_traffic,
    check_bytes_formula,
    check_time_model,
]

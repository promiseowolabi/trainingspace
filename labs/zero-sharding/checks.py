"""Lab 11 checks. The same file runs under pytest, verify.py and the browser.

Each check receives your module as ``m``. Check ids (the function name
without ``check_``) must match src/data/labs.ts.
"""

import numpy as np


def check_shard_roundtrip(m):
    flat = np.arange(10.0)
    s = m.shard(flat, 4)
    assert [len(x) for x in s] == [3, 3, 2, 2], f"np.array_split sizes for 10 over 4 are [3, 3, 2, 2], got {[len(x) for x in s]}"
    s[0][0] = 99
    assert flat[0] == 0, "shard must return copies — updating a shard changed the original"
    assert np.array_equal(m.unshard(m.shard(flat, 3)), flat)


def check_state_is_sharded(m):
    sizes = [m.ShardState(10, 4, r).m.size for r in range(4)]
    assert sizes == [3, 3, 2, 2], f"rank r keeps m only for its own shard: expected [3, 3, 2, 2], got {sizes}"
    st = m.ShardState(1_000_000, 8, 5)
    assert st.m.size == 125_000 and st.v.size == 125_000 and st.t == 0, "1/N of the optimiser state per rank, starting at zero"


def _problem(seed=0, n_ranks=4, d=13):
    r = np.random.default_rng(seed)
    w_true = r.normal(size=d)
    data = []
    for _ in range(n_ranks):
        X = r.normal(size=(32, d))
        data.append((X, X @ w_true + 0.1 * r.normal(size=32)))
    return data, d


def _grad(w, X, y):
    return 2 * X.T @ (X @ w - y) / len(y)


def check_matches_unsharded(m):
    data, d = _problem()
    n = len(data)
    w0 = np.zeros(d)
    ref = w0.copy()
    rm, rv = np.zeros(d), np.zeros(d)
    params = [w0.copy() for _ in range(n)]
    states = [m.ShardState(d, n, r) for r in range(n)]
    for t in range(1, 26):
        g_mean = sum(_grad(ref, X, y) for X, y in data) / n
        rm = 0.9 * rm + 0.1 * g_mean
        rv = 0.999 * rv + 0.001 * g_mean ** 2
        ref = ref - 0.05 * (rm / (1 - 0.9 ** t)) / (np.sqrt(rv / (1 - 0.999 ** t)) + 1e-8)
        grads = [_grad(params[r], *data[r]) for r in range(n)]
        m.zero_step(params, grads, states, lr=0.05)
    for r in range(n):
        assert np.allclose(params[r], ref, atol=1e-10), (
            f"rank {r}: 25 ZeRO steps on 4 ranks must equal 25 Adam steps on the mean gradient — "
            "average (not sum) the gradients, and update only your shard"
        )


def check_ranks_agree(m):
    data, d = _problem(seed=1, n_ranks=3, d=7)
    params = [np.zeros(d) for _ in range(3)]
    ids = [id(p) for p in params]
    states = [m.ShardState(d, 3, r) for r in range(3)]
    m.zero_step(params, [_grad(params[r], *data[r]) for r in range(3)], states, lr=0.1)
    assert [id(p) for p in params] == ids, "write the gathered parameters INTO params[r] (in place) — each rank keeps its own buffer"
    assert all(np.array_equal(params[0], p) for p in params), "after the all-gather every rank must hold the same parameters"
    assert not np.allclose(params[0], 0), "the parameters should have moved"
    assert all(s.t == 1 for s in states), "each rank's step counter advances once per step"


def check_memory_stages(m):
    # ZeRO paper, Figure 1: Ψ = 7.5 B parameters, N = 64 GPUs, K = 12
    got = [m.memory_per_rank_gb(7.5e9, 64, s) for s in range(4)]
    want = [120.0, 31.40625, 16.640625, 1.875]
    for s, (g, w) in enumerate(zip(got, want)):
        assert abs(g - w) < 1e-6, f"stage {s}: expected {w} GB per GPU (ZeRO paper, Figure 1), got {g}"


def check_comm_volume(m):
    assert m.comm_volume(1e9, 0) == 2e9, "plain data parallelism: one all-reduce = 2Ψ"
    assert m.comm_volume(1e9, 1) == 2e9 and m.comm_volume(1e9, 2) == 2e9, "stages 1–2 cost the same as DDP"
    assert m.comm_volume(1e9, 3) == 3e9, "stage 3 adds a second weight all-gather: 1.5× DDP"


CHECKS = [
    check_shard_roundtrip,
    check_state_is_sharded,
    check_matches_unsharded,
    check_ranks_agree,
    check_memory_stages,
    check_comm_volume,
]

"""Lab 05 checks. The same file runs under pytest, verify.py and the browser.

Each check receives your module as ``m``. Check ids (the function name
without ``check_``) must match src/data/labs.ts.
"""

import numpy as np


def rng(seed):
    return np.random.default_rng(seed)


def _ref_attention(q, k, v, mask):
    s = q @ np.swapaxes(k, -1, -2) / np.sqrt(q.shape[-1])
    if mask is not None:
        s = np.where(mask, s, -np.inf)
    s = s - s.max(-1, keepdims=True)
    p = np.exp(s)
    return (p / p.sum(-1, keepdims=True)) @ v


def check_softmax_stable(m):
    x = np.array([[1.0, 2.0, 3.0], [1000.0, 1001.0, 1002.0], [-np.inf, 0.0, 0.0]])
    with np.errstate(all="ignore"):
        p = np.asarray(m.softmax(x, axis=-1))
    assert np.all(np.isfinite(p)), "logits of 1000 overflowed — subtract the row max first"
    assert np.allclose(p.sum(-1), 1.0), "rows must sum to 1"
    assert np.allclose(p[0], p[1]), "softmax is shift-invariant: [1,2,3] and [1000,1001,1002] give the same probabilities"
    assert np.allclose(p[2], [0.0, 0.5, 0.5]), "a −inf entry must get probability exactly 0"
    col = np.asarray(m.softmax(np.array([[1.0, 5.0], [1.0, 5.0]]), axis=0))
    assert np.allclose(col, 0.5), "respect the axis argument"


def check_causal_mask(m):
    got = np.asarray(m.causal_mask(4))
    want = np.tril(np.ones((4, 4), dtype=bool))
    assert got.shape == (4, 4) and got.dtype == bool, f"expected a (4, 4) bool array, got {got.shape} {got.dtype}"
    assert np.array_equal(got, want), "row i must be True for columns 0..i (a token sees itself and the past)"


def check_attention_matches(m):
    r = rng(0)
    for shape in [(5, 8), (2, 5, 8), (2, 3, 5, 8)]:
        q, k, v = r.normal(size=shape), r.normal(size=shape), r.normal(size=shape)
        for mask in (None, np.tril(np.ones((5, 5), dtype=bool))):
            got = np.asarray(m.attention(q, k, v, mask))
            assert got.shape == shape, f"output shape {got.shape}, expected {shape}"
            assert np.allclose(got, _ref_attention(q, k, v, mask)), (
                f"values differ for shape {shape}, mask={'causal' if mask is not None else None} — "
                "did you scale by √D (the head width, q.shape[-1])?"
            )


def check_no_peeking(m):
    r = rng(1)
    B, T, C, H = 2, 6, 16, 4
    x = r.normal(size=(B, T, C))
    W = [r.normal(size=(C, C)) / np.sqrt(C) for _ in range(4)]
    out1 = np.asarray(m.multi_head_attention(x, *W, n_heads=H, causal=True))
    x2 = x.copy()
    x2[:, 4:] = r.normal(size=(B, 2, C))  # change the future
    out2 = np.asarray(m.multi_head_attention(x2, *W, n_heads=H, causal=True))
    assert np.allclose(out1[:, :4], out2[:, :4]), "changing tokens 4–5 changed outputs at positions 0–3: the mask leaks the future"
    assert not np.allclose(out1[:, 4:], out2[:, 4:]), "positions 4–5 should change when their own inputs change"


def check_heads_roundtrip(m):
    x = rng(2).normal(size=(2, 5, 12))
    h = np.asarray(m.split_heads(x, 3))
    assert h.shape == (2, 3, 5, 4), f"split_heads((2, 5, 12), 3) → {h.shape}, expected (2, 3, 5, 4)"
    assert np.allclose(h[:, 1], x[:, :, 4:8]), "head 1 must be channels 4..7 of every token"
    assert np.allclose(np.asarray(m.merge_heads(h)), x), "merge_heads(split_heads(x)) must give back x"


def check_multi_head_matches(m):
    r = rng(3)
    B, T, C, H = 2, 7, 24, 3
    x = r.normal(size=(B, T, C))
    Wq, Wk, Wv, Wo = (r.normal(size=(C, C)) / np.sqrt(C) for _ in range(4))
    D = C // H

    def heads(t):
        return t.reshape(B, T, H, D).transpose(0, 2, 1, 3)

    mask = np.tril(np.ones((T, T), dtype=bool))
    o = _ref_attention(heads(x @ Wq), heads(x @ Wk), heads(x @ Wv), mask)
    want = o.transpose(0, 2, 1, 3).reshape(B, T, C) @ Wo
    got = np.asarray(m.multi_head_attention(x, Wq, Wk, Wv, Wo, n_heads=H, causal=True))
    assert got.shape == (B, T, C), f"shape {got.shape}, expected {(B, T, C)}"
    assert np.allclose(got, want), "multi-head output differs from the reference — check the head split and the √D scale"


def check_rope_rotates(m):
    x = rng(4).normal(size=(2, 9, 8))
    out = np.asarray(m.rope(x))
    assert out.shape == x.shape
    assert np.allclose(out[:, 0], x[:, 0]), "position 0 is rotated by angle 0: unchanged"
    pairs_in = (x.reshape(2, 9, 4, 2) ** 2).sum(-1)
    pairs_out = (out.reshape(2, 9, 4, 2) ** 2).sum(-1)
    assert np.allclose(pairs_in, pairs_out), "a rotation preserves the length of every (2i, 2i+1) pair"
    t, i, D = 3, 1, 8
    th = 10000.0 ** (-2 * i / D) * t
    a, b = x[0, t, 2 * i], x[0, t, 2 * i + 1]
    want = (a * np.cos(th) - b * np.sin(th), a * np.sin(th) + b * np.cos(th))
    assert np.allclose(out[0, t, 2 * i: 2 * i + 2], want), "pair i=1 at position 3: wrong angle or direction"


def check_rope_relative(m):
    r = rng(5)
    D, T = 16, 40
    q, k = r.normal(size=D), r.normal(size=D)
    Q = np.asarray(m.rope(np.tile(q, (T, 1))))
    K = np.asarray(m.rope(np.tile(k, (T, 1))))
    for (i, j) in [(10, 3), (30, 23), (7, 0)]:
        assert np.isclose(Q[i] @ K[j], Q[10] @ K[3]), (
            "after RoPE, q·k must depend only on the distance i − j (here 7), not on i and j themselves"
        )
    assert not np.isclose(Q[10] @ K[3], Q[10] @ K[9]), "different distances should give different scores"


CHECKS = [
    check_softmax_stable,
    check_causal_mask,
    check_attention_matches,
    check_no_peeking,
    check_heads_roundtrip,
    check_multi_head_matches,
    check_rope_rotates,
    check_rope_relative,
]

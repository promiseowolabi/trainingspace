"""Lab 06 checks. The same file runs under pytest, verify.py and the browser.

Each check receives your module as ``m``. Check ids (the function name
without ``check_``) must match src/data/labs.ts. The reference forward pass
below is written independently of your file.
"""

import numpy as np

SMALL = {"vocab_size": 50, "block_size": 16, "n_layer": 2, "n_head": 4, "n_embd": 32}
GPT2_SMALL = {"vocab_size": 50257, "block_size": 1024, "n_layer": 12, "n_head": 12, "n_embd": 768}


def _ln(x, g, b):
    return g * (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-5) + b


def _ref_forward(idx, p, c):
    B, T = idx.shape
    C, H = c["n_embd"], c["n_head"]
    D = C // H
    x = p["wte"][idx] + p["wpe"][:T]
    for i in range(c["n_layer"]):
        k_ = f"h{i}."
        h = _ln(x, p[k_ + "ln1.g"], p[k_ + "ln1.b"])
        qkv = h @ p[k_ + "attn.w_qkv"] + p[k_ + "attn.b_qkv"]
        q, k, v = (qkv[..., j * C:(j + 1) * C].reshape(B, T, H, D).transpose(0, 2, 1, 3) for j in range(3))
        s = q @ k.transpose(0, 1, 3, 2) / np.sqrt(D) + np.triu(np.full((T, T), -1e30), 1)
        s = np.exp(s - s.max(-1, keepdims=True))
        a = ((s / s.sum(-1, keepdims=True)) @ v).transpose(0, 2, 1, 3).reshape(B, T, C)
        x = x + a @ p[k_ + "attn.w_o"] + p[k_ + "attn.b_o"]
        h = _ln(x, p[k_ + "ln2.g"], p[k_ + "ln2.b"])
        u = h @ p[k_ + "mlp.w_fc"] + p[k_ + "mlp.b_fc"]
        u = 0.5 * u * (1 + np.tanh(np.sqrt(2 / np.pi) * (u + 0.044715 * u ** 3)))
        x = x + u @ p[k_ + "mlp.w_proj"] + p[k_ + "mlp.b_proj"]
    return _ln(x, p["ln_f.g"], p["ln_f.b"]) @ p["wte"].T


def _rich_params(m, seed=0):
    p = m.init_params(SMALL, seed=seed)
    r = np.random.default_rng(seed + 100)
    for k in p:
        if k.endswith((".g", ".b", "b_qkv", "b_o", "b_fc", "b_proj")):
            p[k] = p[k] + r.normal(0, 0.1, p[k].shape)
    return p


def check_layer_norm(m):
    r = np.random.default_rng(0)
    x = r.normal(3.0, 2.0, size=(2, 5, 8))
    g, b = r.normal(size=8), r.normal(size=8)
    got = np.asarray(m.layer_norm(x, g, b))
    assert got.shape == x.shape
    assert np.allclose(got, _ln(x, g, b)), "normalise over the LAST axis (each token), then g·x̂ + b"


def check_gelu(m):
    x = np.array([-3.0, -1.0, 0.0, 0.5, 2.0])
    want = 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))
    assert np.allclose(np.asarray(m.gelu(x)), want), "use GPT-2's tanh approximation"


def check_forward_matches(m):
    p = _rich_params(m)
    idx = np.random.default_rng(1).integers(0, SMALL["vocab_size"], size=(2, 11))
    got = np.asarray(m.forward(idx, p, SMALL))
    assert got.shape == (2, 11, SMALL["vocab_size"]), f"logits shape {got.shape}, expected (2, 11, 50)"
    want = _ref_forward(idx, p, SMALL)
    err = np.abs(got - want).max()
    assert err < 1e-8, f"logits differ from the reference by up to {err:.3g}"


def check_causal(m):
    p = _rich_params(m, seed=2)
    r = np.random.default_rng(3)
    idx = r.integers(0, 50, size=(1, 12))
    a = np.asarray(m.forward(idx, p, SMALL))
    idx2 = idx.copy()
    idx2[0, 8:] = (idx2[0, 8:] + 7) % 50
    b = np.asarray(m.forward(idx2, p, SMALL))
    assert np.allclose(a[0, :8], b[0, :8]), "changing tokens 8–11 changed the logits at positions 0–7: the model sees the future"


def check_param_count(m):
    got = m.count_params(GPT2_SMALL)
    assert got == 124_439_808, f"GPT-2 small has 124,439,808 parameters with tied embeddings; count_params says {got:,}"
    actual = sum(v.size for v in m.init_params(SMALL).values())
    assert m.count_params(SMALL) == actual, f"count_params(SMALL) = {m.count_params(SMALL)}, but init_params builds {actual}"


def check_greedy(m):
    p = _rich_params(m, seed=4)
    out = m.generate_greedy([1, 2, 3], p, SMALL, n_new=20)
    assert isinstance(out, list) and len(out) == 23 and out[:3] == [1, 2, 3], "return the prompt plus n_new ids, as a list"
    seq = [1, 2, 3]
    for _ in range(20):
        ctx = np.array([seq[-SMALL["block_size"]:]])
        seq.append(int(np.argmax(_ref_forward(ctx, p, SMALL)[0, -1])))
    assert [int(t) for t in out] == seq, "greedy tokens differ — take the argmax at the LAST position, and crop the context to block_size"


def check_sampling(m):
    logits = np.array([2.0, 1.0, 0.5, -1.0, 3.0])
    rng = np.random.default_rng(0)
    assert m.sample_next(logits, temperature=0, rng=rng) == 4, "temperature 0 is greedy"
    assert all(m.sample_next(logits, top_k=1, rng=rng) == 4 for _ in range(20)), "top_k=1 must always pick the argmax"
    allowed = {4, 0}
    draws = [m.sample_next(logits, top_k=2, rng=rng) for _ in range(300)]
    assert set(draws) <= allowed, f"top_k=2 drew ids outside the two largest logits: {set(draws) - allowed}"
    p = np.exp(logits - logits.max())
    p /= p.sum()  # [0.231, 0.085, 0.052, 0.007, 0.628] — the top two hold 0.858 of the mass
    draws = [m.sample_next(logits, top_p=0.8, rng=rng) for _ in range(300)]
    assert set(draws) == {4, 0}, f"top_p=0.8 keeps {{4, 0}} (0.628 alone is < 0.8, 0.858 reaches it); drew {set(draws)}"
    draws = np.array([m.sample_next(logits, rng=rng) for _ in range(6000)])
    freq = np.bincount(draws, minlength=5) / len(draws)
    assert np.abs(freq - p).max() < 0.03, f"at temperature 1 the draw frequencies {np.round(freq, 3)} should match softmax {np.round(p, 3)}"
    hot = np.array([m.sample_next(logits, temperature=100.0, rng=rng) for _ in range(3000)])
    assert np.abs(np.bincount(hot, minlength=5) / 3000 - 0.2).max() < 0.05, "temperature 100 should be nearly uniform"


CHECKS = [
    check_layer_norm,
    check_gelu,
    check_forward_matches,
    check_causal,
    check_param_count,
    check_greedy,
    check_sampling,
]

"""Lab 06 — a GPT forward pass.

The one file you edit. Every ``TODO(you)`` is a ``raise NotImplementedError``
for you to replace. Run ``pytest`` until it is green, then drop this file on
the lab page or run ``PREFIX=<you> python verify.py``.

Lessons: M1.L4 (the block), M1.L5 (counting a GPT), M1.L6 (sampling).

The architecture is GPT-2's: learned position embeddings, pre-norm
LayerNorm (O1.L5), causal multi-head attention (lab 05), a GELU MLP four
times as wide as the model, a final LayerNorm, and an output layer that
REUSES the token-embedding matrix (``logits = x @ wte.T`` — "tied"
embeddings).

``config`` is a dict: vocab_size V, block_size (max context T), n_layer L,
n_head H, n_embd C. ``params`` is a dict of numpy arrays, laid out by
``init_params`` below (given). Read it: the key names are the spec.
"""

from __future__ import annotations

import numpy as np


def init_params(config, seed=0):
    """GPT-2-style initialisation (given). Shapes are the spec for ``forward``."""
    r = np.random.default_rng(seed)
    V, T, C, L = config["vocab_size"], config["block_size"], config["n_embd"], config["n_layer"]
    std, proj_std = 0.02, 0.02 / np.sqrt(2 * L)
    p = {"wte": r.normal(0, std, (V, C)), "wpe": r.normal(0, std, (T, C))}
    for i in range(L):
        h = f"h{i}."
        p[h + "ln1.g"], p[h + "ln1.b"] = np.ones(C), np.zeros(C)
        p[h + "attn.w_qkv"], p[h + "attn.b_qkv"] = r.normal(0, std, (C, 3 * C)), np.zeros(3 * C)
        p[h + "attn.w_o"], p[h + "attn.b_o"] = r.normal(0, proj_std, (C, C)), np.zeros(C)
        p[h + "ln2.g"], p[h + "ln2.b"] = np.ones(C), np.zeros(C)
        p[h + "mlp.w_fc"], p[h + "mlp.b_fc"] = r.normal(0, std, (C, 4 * C)), np.zeros(4 * C)
        p[h + "mlp.w_proj"], p[h + "mlp.b_proj"] = r.normal(0, proj_std, (4 * C, C)), np.zeros(C)
    p["ln_f.g"], p["ln_f.b"] = np.ones(C), np.zeros(C)
    return p


def _softmax(x):
    """Stable softmax over the last axis (given — you wrote it in lab 05)."""
    z = x - x.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def layer_norm(x, g, b, eps=1e-5):
    """Normalise the last axis to mean 0, variance 1 (biased variance, like
    numpy's default), then scale by g and shift by b (O1.L5)."""
    # TODO(you)
    raise NotImplementedError("layer_norm")


def gelu(x):
    """GPT-2's GELU, tanh approximation:
    ``0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))``"""
    # TODO(you)
    raise NotImplementedError("gelu")


def block(x, p, prefix, n_head):
    """One pre-norm transformer block on x of shape (B, T, C). Keys are
    ``prefix + "ln1.g"`` and so on.

        h = layer_norm(x, ln1)
        q, k, v = split (h @ w_qkv + b_qkv) into three (B, T, C) along the last axis
        a = causal multi-head attention of q, k, v (lab 05), heads merged back to (B, T, C)
        x = x + a @ w_o + b_o
        h = layer_norm(x, ln2)
        x = x + gelu(h @ w_fc + b_fc) @ w_proj + b_proj

    Return the new x.
    """
    # TODO(you)
    raise NotImplementedError("block")


def forward(idx, params, config):
    """Token ids (B, T) → logits (B, T, V).

    x = wte[idx] + wpe[:T]; then every block h0 … h{L−1}; then ln_f; then
    ``x @ wte.T`` (tied output layer).
    """
    # TODO(you)
    raise NotImplementedError("forward")


def count_params(config):
    """The number of parameters ``init_params(config)`` creates — computed
    from the config with arithmetic (M1.L5), not by building the arrays.
    The tied output layer adds no parameters."""
    # TODO(you)
    raise NotImplementedError("count_params")


def generate_greedy(idx, params, config, n_new):
    """Append ``n_new`` tokens to the list ``idx``, each the argmax of the
    logits at the LAST position. Feed the model at most the last
    ``block_size`` tokens. Return the full list of ints."""
    # TODO(you)
    raise NotImplementedError("generate_greedy")


def sample_next(logits, temperature=1.0, top_k=None, top_p=None, rng=None):
    """Sample one token id from a 1-D array of logits (M1.L6).

    1. ``temperature == 0`` → return the argmax (greedy).
    2. Divide the logits by ``temperature``.
    3. ``top_k``: keep the k largest logits, set the rest to −inf.
    4. Softmax.
    5. ``top_p``: sort probabilities descending and keep the smallest prefix
       whose cumulative probability reaches top_p — i.e. keep a token if the
       probability mass BEFORE it is < top_p — zero the rest, renormalise.
    6. Draw with ``rng.choice(len(p), p=p)`` (rng: a numpy Generator; make a
       new one if None). Return a Python int.
    """
    # TODO(you)
    raise NotImplementedError("sample_next")

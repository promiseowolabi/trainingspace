# Lab 05 — Attention, Heads and RoPE

Lessons: M1.L3 (attention), M1.L2 (embeddings and position).

You edit one file: `attention.py` — `softmax`, `causal_mask`, `attention`,
`split_heads`, `merge_heads`, `multi_head_attention` and `rope`. No loops over
batch, time or heads.

```sh
pytest
PREFIX=<you> python verify.py
```

Checks: softmax-stable · causal-mask · attention-matches · no-peeking ·
heads-roundtrip · multi-head-matches · rope-rotates · rope-relative

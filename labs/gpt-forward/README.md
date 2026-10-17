# Lab 06 — A GPT Forward Pass

Lessons: M1.L4 (the block), M1.L5 (counting a GPT), M1.L6 (sampling).

You edit one file: `gpt.py` — `layer_norm`, `gelu`, `block`, `forward`,
`count_params`, `generate_greedy` and `sample_next`. `init_params` is given;
its key names and shapes are the spec.

```sh
pytest
PREFIX=<you> python verify.py
```

Checks: layer-norm · gelu · forward-matches · causal · param-count · greedy ·
sampling

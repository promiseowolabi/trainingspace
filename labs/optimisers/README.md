# Lab 03 — Optimisers, Schedules and Clipping

Lessons: O1.L3 (optimisers), O1.L4 (learning-rate schedules), P1.L6 (stability).

You edit one file: `optim.py` — `SGD.step`, `Adam.step` (Adam and AdamW),
`lr_at` (warmup + cosine) and `clip_grad_norm`. Everything works on plain
lists of numpy arrays, updated in place, following PyTorch's update rules.

```sh
pytest
PREFIX=<you> python verify.py
```

Checks: sgd-step · momentum · adam-first-step · adam-matches-reference ·
adamw-decoupled · warmup-cosine · clip-norm · conditioning

# Lab 08 — Budgeting a Training Run

Lessons: P1.L4 (FLOPs and MFU), P1.L5 (scaling laws), T0.L4, M1.L5.

You edit one file: `budget.py` — eight one-line formulas, checked against
GPT-2, GPT-3 and Chinchilla's published numbers. Pure Python.

```sh
pytest
PREFIX=<you> python verify.py
```

Checks: param-count · train-flops · forward-per-token · mfu · training-days ·
state-memory · activation-memory · chinchilla

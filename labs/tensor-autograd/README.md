# Lab 02 — Autograd for Arrays

Lessons: A1.L4 (tensor autograd), A1.L5 (gradient checking).

You edit one file: `tensor.py`. The graph machinery (`backward`, the
topological sort) is given — you wrote it in lab 01. You write `unbroadcast`,
the backward passes of `+`, `*`, `@`, `sum` and `relu`, a fused stable
`cross_entropy`, and `gradcheck`.

```sh
pytest
PREFIX=<you> python verify.py
```

The last check trains a 2 → 16 → 3 MLP on three blobs using your engine and
expects the loss below 0.15 after 150 steps.

Checks: unbroadcast · add-broadcast · mul · matmul · sum-axis · relu ·
cross-entropy-value · cross-entropy-stable · cross-entropy-grad · gradcheck ·
trains-mlp

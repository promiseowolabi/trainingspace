# Lab 00 — Shapes, FLOPs and Bytes

Lessons: T0.L2 (tensors and shapes), T0.L3 (the cost of a matmul), T0.L4 (where the memory goes).

You edit one file: `shapes.py`. Every `TODO(you)` is a `raise NotImplementedError`.

```sh
pip install -r ../requirements.txt   # numpy, pytest — once
pytest                               # red → green, one test per check
PREFIX=<you> python verify.py        # prints your completion code
```

Or skip the install: paste `shapes.py` into the editor on the lab page and run the
same checks in your browser.

Checks (in `checks.py`, shared by pytest, `verify.py` and the browser):
broadcast-rules · broadcast-errors · pairwise-dists · linear-batched ·
matmul-flops · arithmetic-intensity · memory-16

# Lab 01 — An Autograd Engine in 100 Lines

Lesson: A1.L3 (a scalar engine), built on A1.L2 (the chain rule as a graph).

You edit one file: `engine.py`. The forward passes of `+`, `*` and `**` are given;
you write their backward closures, the four nonlinearities, and `backward()`.

```sh
pytest
PREFIX=<you> python verify.py
```

Two rules the checks enforce: gradients accumulate (`+=`), and `backward()` does
not recurse — a 5,000-node chain must work.

Checks: forward-values · add-mul · accumulates · pow-div · nonlinear ·
topo-order · deep-graph · matches-numeric

"""Lab 01 — a scalar autograd engine.

The one file you edit. Every ``TODO(you)`` is a ``raise NotImplementedError``
for you to replace. Run ``pytest`` until it is green, then drop this file on
the lab page or run ``PREFIX=<you> python verify.py``.

Lesson: A1.L3 (a scalar engine). The shape of the code follows A1.L2: every
operation creates an output node that remembers its inputs (``_prev``) and a
closure (``_backward``) that pushes ``out.grad`` into its inputs' ``.grad``
using the operation's LOCAL derivative. ``backward()`` then calls those
closures in reverse topological order.

Two rules that the checks enforce:

  1. Gradients ACCUMULATE: use ``+=``, never ``=``. A value used twice
     receives gradient from both uses (A1.L3, "the reuse bug").
  2. ``backward()`` must not recurse. A 5000-node chain is a normal graph;
     Python's recursion limit is 1000.
"""

from __future__ import annotations

import math


class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = tuple(_children)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data:.6g}, grad={self.grad:.6g})"

    @staticmethod
    def _lift(other) -> "Value":
        return other if isinstance(other, Value) else Value(other)

    def __add__(self, other):
        other = Value._lift(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # TODO(you): d(out)/d(self) = 1 and d(out)/d(other) = 1.
            raise NotImplementedError("__add__ backward")

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = Value._lift(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            # TODO(you): each input's local derivative is the OTHER input.
            raise NotImplementedError("__mul__ backward")

        out._backward = _backward
        return out

    def __pow__(self, k):
        assert isinstance(k, (int, float)), "only constant exponents"
        out = Value(self.data ** k, (self,), f"**{k}")

        def _backward():
            # TODO(you): the power rule.
            raise NotImplementedError("__pow__ backward")

        out._backward = _backward
        return out

    def exp(self):
        # TODO(you): forward and backward. d/dx e^x = e^x.
        raise NotImplementedError("exp")

    def log(self):
        # TODO(you): natural log, forward and backward. d/dx ln x = 1/x.
        raise NotImplementedError("log")

    def tanh(self):
        # TODO(you): forward and backward. d/dx tanh x = 1 - tanh(x)^2.
        raise NotImplementedError("tanh")

    def relu(self):
        # TODO(you): max(0, x), forward and backward. Use gradient 0 at x == 0.
        raise NotImplementedError("relu")

    def backward(self):
        """Fill ``.grad`` on every node that ``self`` depends on.

        1. Build a topological order of the graph below ``self`` — every node
           after all of its inputs — WITHOUT recursion (use an explicit stack).
        2. Seed ``self.grad = 1.0`` (d self / d self).
        3. Call each node's ``_backward`` in REVERSE topological order, so a
           node pushes gradient to its inputs only after every consumer of it
           has finished pushing gradient to it.
        """
        # TODO(you)
        raise NotImplementedError("backward")

    # Given: everything below is built from the operations above.
    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-Value._lift(other))

    def __truediv__(self, other):
        return self * Value._lift(other) ** -1

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return Value._lift(other) - self

    def __rtruediv__(self, other):
        return Value._lift(other) / self

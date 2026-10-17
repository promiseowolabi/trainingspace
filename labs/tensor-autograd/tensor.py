"""Lab 02 — tensor autograd.

The one file you edit. Every ``TODO(you)`` is a ``raise NotImplementedError``
for you to replace. Run ``pytest`` until it is green, then drop this file on
the lab page or run ``PREFIX=<you> python verify.py``.

Lessons: A1.L4 (tensor autograd) and A1.L5 (gradient checking).

Lab 01's engine, but every node holds a numpy array instead of a float. The
graph machinery (``backward``, the topological sort) is given — you wrote it
in lab 01. What is new is the calculus of shapes:

  * ``.grad`` always has the same shape as ``.data``;
  * when the forward pass BROADCAST an input, the backward pass must SUM the
    gradient back down to that input's shape (``unbroadcast``);
  * matmul's local derivatives are matmuls with a transpose.
"""

from __future__ import annotations

import numpy as np


def unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Sum ``grad`` down to ``shape``, undoing numpy broadcasting.

    Broadcasting can (a) prepend dimensions and (b) stretch size-1 dimensions.
    Undo both: sum away the extra leading axes, then sum (keepdims) every axis
    where ``shape`` has a 1 but ``grad`` does not.

        unbroadcast(np.ones((3, 4)), (4,))    -> [3, 3, 3, 3]
        unbroadcast(np.ones((3, 4)), (3, 1))  -> [[4], [4], [4]]
    """
    # TODO(you)
    raise NotImplementedError("unbroadcast")


class Tensor:
    def __init__(self, data, _children=(), _op=""):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = tuple(_children)
        self._op = _op

    @property
    def shape(self):
        return self.data.shape

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, op={self._op!r})"

    @staticmethod
    def _lift(other) -> "Tensor":
        return other if isinstance(other, Tensor) else Tensor(other)

    def __add__(self, other):
        other = Tensor._lift(other)
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward():
            # TODO(you): like lab 01, but either input may have been broadcast.
            raise NotImplementedError("__add__ backward")

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = Tensor._lift(other)
        out = Tensor(self.data * other.data, (self, other), "*")

        def _backward():
            # TODO(you): elementwise product rule, then unbroadcast.
            raise NotImplementedError("__mul__ backward")

        out._backward = _backward
        return out

    def __matmul__(self, other):
        assert self.data.ndim == 2 and other.data.ndim == 2, "2-D matmul only"
        out = Tensor(self.data @ other.data, (self, other), "@")

        def _backward():
            # TODO(you): out = A @ B, G = out.grad.  dA = G @ Bᵀ,  dB = Aᵀ @ G.
            # Check the shapes: dA must be A's shape, dB must be B's.
            raise NotImplementedError("__matmul__ backward")

        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")

        def _backward():
            # TODO(you): every input element contributed with weight 1, so it
            # receives the gradient of the output element it was summed into.
            # When keepdims is False the summed axis is gone from out.grad:
            # put it back (np.expand_dims) before broadcasting to self.shape.
            raise NotImplementedError("sum backward")

        out._backward = _backward
        return out

    def relu(self):
        # TODO(you): forward and backward. Gradient 0 where data <= 0.
        raise NotImplementedError("relu")

    def backward(self):
        order, seen, stack = [], set(), [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
                continue
            if id(node) in seen:
                continue
            seen.add(id(node))
            stack.append((node, True))
            stack.extend((c, False) for c in node._prev if id(c) not in seen)
        self.grad = np.ones_like(self.data)
        for node in reversed(order):
            node._backward()

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)

    # Given: built from the operations above.
    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-Tensor._lift(other))

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def mean(self, axis=None):
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis) * (1.0 / n)


def cross_entropy(logits: Tensor, targets: np.ndarray) -> Tensor:
    """Mean cross-entropy of ``logits`` (N, C) against integer ``targets`` (N,).

    One fused node, the way real frameworks do it (O1.L2):

    forward   loss = mean over rows of  -log softmax(z)[target]
              Compute log-softmax with the log-sum-exp trick: subtract each
              row's max before ``exp`` so logits of 1000 do not overflow.
    backward  d loss / d z = (softmax(z) - onehot(target)) / N
              — multiplied by ``out.grad``, accumulated into ``logits.grad``.
    """
    # TODO(you)
    raise NotImplementedError("cross_entropy")


def gradcheck(f, inputs, eps=1e-6, atol=1e-5, rtol=1e-3) -> bool:
    """True when backprop agrees with central differences (A1.L5).

    ``f(*inputs)`` returns a scalar Tensor. Zero every input's grad, run
    ``f(*inputs).backward()`` once and keep a copy of each input's ``.grad``.
    Then, for every element of every input: nudge it by +eps, evaluate f,
    nudge it by -eps, evaluate f, RESTORE it, and compare

        numeric = (f(x + eps) - f(x - eps)) / (2 · eps)

    with the analytic gradient. Fail when
    ``|analytic - numeric| > atol + rtol · |numeric|``.

    Hint: ``for idx in np.ndindex(t.data.shape)`` visits every element and
    ``t.data[idx] = …`` edits it in place.
    """
    # TODO(you)
    raise NotImplementedError("gradcheck")

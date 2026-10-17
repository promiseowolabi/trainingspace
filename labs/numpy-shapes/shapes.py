"""Lab 00 — shapes, FLOPs and bytes.

The one file you edit. Replace every ``raise NotImplementedError`` marked
``TODO(you)``. Run ``pytest`` until it is green, then either drop this file on
the lab page or run ``PREFIX=<you> python verify.py`` for a completion code.

Lesson: T0.L2 (tensors and shapes), T0.L3 (the cost of a matmul),
T0.L4 (where the memory goes).
"""

from __future__ import annotations

import numpy as np


def broadcast_shape(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...]:
    """The shape numpy gives ``x + y`` when ``x.shape == a`` and ``y.shape == b``.

    Do it WITHOUT numpy. The rule (T0.L2): align the shapes on the right; each
    pair of dimensions must be equal, or one of them must be 1; a missing
    dimension counts as 1. Raise ``ValueError`` when the shapes are
    incompatible.

        broadcast_shape((8, 1, 6), (7, 1))  ->  (8, 7, 6)
        broadcast_shape((3,), (4,))         ->  ValueError
    """
    # TODO(you)
    raise NotImplementedError("broadcast_shape")


def pairwise_sq_dists(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Squared Euclidean distance between every row of X (n, d) and every row
    of Y (m, d), returned as an (n, m) array.

    No Python loops: insert axes so that broadcasting builds the (n, m, d)
    difference for you, then reduce the last axis.
    """
    # TODO(you)
    raise NotImplementedError("pairwise_sq_dists")


def linear(x: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """A linear layer: ``x @ W + b``.

    ``x`` may have any number of leading batch dimensions — (d_in,), (B, d_in)
    or (B, T, d_in) — with ``W`` of shape (d_in, d_out) and ``b`` of shape
    (d_out,). The result has the same leading dimensions and d_out last.
    """
    # TODO(you)
    raise NotImplementedError("linear")


def matmul_flops(m: int, k: int, n: int) -> int:
    """FLOPs to multiply an (m, k) matrix by a (k, n) matrix.

    Count a multiply and an add as two FLOPs (T0.L3).
    """
    # TODO(you)
    raise NotImplementedError("matmul_flops")


def arithmetic_intensity(m: int, k: int, n: int, bytes_per_el: int = 2) -> float:
    """FLOPs per byte moved for that matmul, assuming each of A (m, k),
    B (k, n) and C (m, n) crosses memory exactly once at ``bytes_per_el``
    bytes per element.
    """
    # TODO(you)
    raise NotImplementedError("arithmetic_intensity")


def training_bytes_per_param() -> int:
    """Bytes of GPU memory per parameter for mixed-precision Adam training
    (T0.L4): bf16 weights, bf16 gradients, an fp32 master copy of the
    weights, and Adam's two fp32 moments. Activations are not included.
    """
    # TODO(you)
    raise NotImplementedError("training_bytes_per_param")


def training_memory_gb(n_params: int) -> float:
    """Training memory, in GB (1e9 bytes), for a model with ``n_params``
    parameters, excluding activations."""
    # TODO(you)
    raise NotImplementedError("training_memory_gb")

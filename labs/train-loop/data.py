"""Lab 07 — the data (given; do not edit).

A character-level tokenizer over ``data/corpus.txt`` — the text of this
course's own T0–M1 lessons, about 159 KB. Character-level keeps the lab
self-contained; swap in your lab 04 tokenizer in the capstone.
"""

from __future__ import annotations

import os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def load(path: str = os.path.join(HERE, "data", "corpus.txt"), val_fraction: float = 0.1):
    text = open(path, encoding="utf-8").read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n_val = int(len(data) * val_fraction)
    return {
        "train": data[:-n_val],
        "val": data[-n_val:],
        "vocab_size": len(chars),
        "encode": lambda s: [stoi[c] for c in s],
        "decode": lambda ids: "".join(chars[i] for i in ids),
    }

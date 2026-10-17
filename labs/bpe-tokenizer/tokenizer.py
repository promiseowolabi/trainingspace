"""Lab 04 — a byte-level BPE tokenizer.

The one file you edit. Every ``TODO(you)`` is a ``raise NotImplementedError``
for you to replace. Pure Python — no numpy. Run ``pytest`` until it is green,
then drop this file on the lab page or run ``PREFIX=<you> python verify.py``.

Lesson: M1.L1 (tokenization).

Byte-level BPE in one paragraph: start from the UTF-8 bytes of the text, so
the 256 byte values are the first 256 token ids and ANY string can be
encoded. Then repeatedly find the most frequent adjacent pair of ids and
replace every occurrence with a new id (256, 257, …). The ordered list of
those replacements — ``merges`` — IS the tokenizer. Encoding new text
replays the merges in the order they were learned; decoding maps each id
back to its bytes.

``merges`` is a dict ``{(a, b): new_id}``. Python dicts keep insertion
order, and the training loop inserts merges in id order.
"""

from __future__ import annotations


def get_stats(ids):
    """Count adjacent pairs: ``[1, 2, 1, 2, 3] → {(1, 2): 2, (2, 1): 1, (2, 3): 1}``.

    Return a dict whose keys appear in the order each pair FIRST occurs in
    ``ids`` — ``train`` relies on that to break ties deterministically.
    """
    # TODO(you)
    raise NotImplementedError("get_stats")


def merge(ids, pair, new_id):
    """Replace every occurrence of ``pair`` in ``ids`` with ``new_id``.

    Scan left to right and do not reuse an id that was just merged:
    ``merge([5, 5, 5], (5, 5), 9) → [9, 5]``. Return a new list.
    """
    # TODO(you)
    raise NotImplementedError("merge")


def train(text, vocab_size):
    """Learn ``vocab_size − 256`` merges from ``text``.

    Start from ``list(text.encode("utf-8"))``. For each new id from 256 up to
    ``vocab_size − 1``: count pairs, pick the most frequent one (on a tie, the
    pair that occurs FIRST — ``max(stats, key=stats.get)`` does exactly this
    on an insertion-ordered dict), merge it everywhere, and record it. Stop
    early if fewer than two ids remain. Return ``merges``.
    """
    # TODO(you)
    raise NotImplementedError("train")


def encode(text, merges):
    """Encode ``text`` to ids by replaying ``merges`` in the order they were learned.

    Start from the UTF-8 bytes. While at least two ids remain: among the
    adjacent pairs present, find the one with the LOWEST merge id (the
    earliest-learned merge); if none of the pairs is in ``merges``, stop;
    otherwise merge it everywhere and repeat.
    """
    # TODO(you)
    raise NotImplementedError("encode")


def vocab(merges):
    """``{id: bytes}`` for every id: 0–255 are single bytes, and each merged
    id is the concatenation of its pair's bytes. Given."""
    v = {i: bytes([i]) for i in range(256)}
    for (a, b), i in sorted(merges.items(), key=lambda kv: kv[1]):
        v[i] = v[a] + v[b]
    return v


def decode(ids, merges):
    """Ids back to text: join each id's bytes, then decode UTF-8.

    A token sequence can end in the middle of a multi-byte character (a model
    can generate anything), so decode with ``errors="replace"`` — never raise.
    """
    # TODO(you)
    raise NotImplementedError("decode")

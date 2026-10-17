"""Lab 04 checks. The same file runs under pytest, verify.py and the browser.

Each check receives your module as ``m``. Check ids (the function name
without ``check_``) must match src/data/labs.ts.
"""

CORPUS = (
    "A model is a function with knobs. Training is nudging every knob downhill. "
    "The loss says how wrong the model is; the gradient says which way is downhill; "
    "the optimiser decides how far to step. A tokenizer turns text into the integers "
    "the model reads, and every one of those integers costs a row of the embedding "
    "matrix and a column of the output layer. Byte pair encoding starts from bytes, "
    "so it can encode anything, and merges the most frequent pairs until the "
    "vocabulary is the size you asked for. The training text decides which pairs win: "
    "train on English and English words become single tokens; train on code and "
    "indentation does. Tokenizers are trained once and frozen, because the model's "
    "embedding rows are tied to their ids. Change the tokenizer and you must retrain "
    "the model. The tokenizer is the one part of a language model you cannot fix later. "
) * 4


def check_stats_counts(m):
    got = m.get_stats([1, 2, 1, 2, 3])
    assert dict(got) == {(1, 2): 2, (2, 1): 1, (2, 3): 1}, f"got {got}"
    assert list(got)[:1] == [(1, 2)], "keys must appear in order of first occurrence"
    assert dict(m.get_stats([7])) == {} and dict(m.get_stats([])) == {}, "fewer than two ids: no pairs"


def check_merge_replaces(m):
    assert m.merge([5, 5, 5], (5, 5), 9) == [9, 5], "merge left to right, without reusing a merged id"
    assert m.merge([1, 2, 3, 1, 2], (1, 2), 7) == [7, 3, 7]
    original = [1, 2, 1]
    m.merge(original, (1, 2), 7)
    assert original == [1, 2, 1], "return a new list; do not modify the input"


def check_train_classic(m):
    merges = m.train("aaabdaaabac", 260)
    want = {(97, 97): 256, (256, 97): 257, (257, 98): 258, (258, 100): 259}
    assert dict(merges) == want, (
        f"on 'aaabdaaabac' expected {want}, got {dict(merges)} — "
        "check the tie-break: equal counts go to the pair that occurs first"
    )
    assert list(merges.values()) == [256, 257, 258, 259], "merges must be inserted in id order"


def check_encode_replays(m):
    merges = m.train("aaabdaaabac", 260)
    assert m.encode("aaabdaaabac", merges) == [259, 258, 97, 99], (
        "encoding the training text must reproduce training's final sequence"
    )
    assert m.encode("", merges) == [] and m.encode("x", merges) == [120]
    assert m.encode("aaab", merges) == [258], "replay merges by rank: aa → 256, then 256·a → 257, then 257·b → 258"


def check_roundtrip(m):
    merges = m.train(CORPUS, 300)
    for text in [CORPUS, "unseen words: zygote quixotic", "naïve café — 東京 🚀", "  \n\t indentation\n    def f():"]:
        ids = m.encode(text, merges)
        assert all(isinstance(i, int) and 0 <= i < 300 for i in ids), "every id must be in [0, vocab_size)"
        back = m.decode(ids, merges)
        assert back == text, f"decode(encode(x)) != x for {text[:30]!r}: got {back[:30]!r}"


def check_compression(m):
    merges = m.train(CORPUS, 400)
    n_bytes = len(CORPUS.encode("utf-8"))
    n_tokens = len(m.encode(CORPUS, merges))
    ratio = n_bytes / n_tokens
    assert ratio > 2.5, f"{n_bytes} bytes → {n_tokens} tokens = {ratio:.2f} bytes/token; expected > 2.5 (the reference gets 2.87) with 144 merges on its own training text"


def check_decode_partial(m):
    merges = m.train("héllo", 258)
    e_acute = list("é".encode("utf-8"))  # two bytes
    out = m.decode([e_acute[0]], merges)
    assert out == "�", f"a lone first byte of a 2-byte character must decode to U+FFFD, got {out!r} — use errors='replace'"


CHECKS = [
    check_stats_counts,
    check_merge_replaces,
    check_train_classic,
    check_encode_replays,
    check_roundtrip,
    check_compression,
    check_decode_partial,
]

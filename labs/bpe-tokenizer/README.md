# Lab 04 — A Byte-Level BPE Tokenizer

Lesson: M1.L1 (tokenization). Pure Python — no numpy needed.

You edit one file: `tokenizer.py` — `get_stats`, `merge`, `train`, `encode`
and `decode`. `vocab` is given.

```sh
pytest
PREFIX=<you> python verify.py
```

Checks: stats-counts · merge-replaces · train-classic · encode-replays ·
roundtrip · compression · decode-partial

Keep your tokenizer: the capstone trains one on your own corpus.

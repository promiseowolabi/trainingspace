# Lab 11 — ZeRO, Sharded

Lesson: D1.L3 (ZeRO and FSDP).

You edit one file: `zero.py` — `shard`, `unshard`, `ShardState`, `zero_step`,
`memory_per_rank_gb` and `comm_volume`. The collectives and the Adam update
are given.

```sh
pytest
PREFIX=<you> python verify.py
```

Checks: shard-roundtrip · state-is-sharded · matches-unsharded · ranks-agree ·
memory-stages · comm-volume

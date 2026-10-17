# Lab 09 — Ring All-Reduce

Lesson: D1.L1 (collectives).

You edit one file: `collectives.py` — `ring_reduce_scatter`, `ring_all_gather`,
`ring_all_reduce`, `ring_bytes_per_rank` and `ring_time`. The `Ring` network
simulator is given: each rank may send only to its right-hand neighbour, once
per step, and every byte is counted.

```sh
pytest
PREFIX=<you> python verify.py
```

Checks: reduce-scatter · all-gather · all-reduce · ring-traffic ·
bytes-formula · time-model

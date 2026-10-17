# Lab 10 — Data Parallelism from Scratch

Lesson: D1.L2 (data parallelism). A LOCAL lab: needs PyTorch (CPU build).

You edit one file: `ddp.py`. The checks start two real processes that talk
over `torch.distributed` with the `gloo` backend on 127.0.0.1 — the same API a
GPU cluster uses with `nccl` — and compare their training with one process.

```sh
pip install torch --index-url https://download.pytorch.org/whl/cpu   # once
pytest                          # ~5 s
PREFIX=<you> python verify.py
```

If a check reports "hung", one rank is waiting in a collective the other
never called. If your firewall asks about Python accepting connections,
the traffic stays on 127.0.0.1.

Checks: shard-batch · plan-buckets · broadcast-sync ·
matches-single-process · bucketed-matches · bucket-calls

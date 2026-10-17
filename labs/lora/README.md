# Lab 12 — LoRA and the SFT Loss

Lessons: F1.L3 (LoRA), F1.L2 (SFT and chat templates). A LOCAL lab: needs
PyTorch (CPU build).

You edit one file: `lora.py` — `LoRALinear` (init, forward, merged),
`apply_lora`, `count_params` and `sft_loss`. `model.py` is lab 07's GPT, given.

```sh
pytest
PREFIX=<you> python verify.py
```

Checks: zero-init · frozen-base · scale · merge-equal · apply-counts ·
sft-masking · learns-low-rank

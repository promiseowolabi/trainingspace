# Lab 07 — A Training Loop You Can Resume

Lessons: P1.L2 (the training loop), P1.L3 (mixed precision), E1.L3 (checkpointing).

A LOCAL lab: it needs PyTorch. The CPU build is enough:

```sh
pip install -r ../requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
pytest                          # ~10 s
PREFIX=<you> python verify.py   # completion code
python run.py                   # a full ~2-minute training run with your loop
```

You edit one file: `train.py`. `model.py` (lab 06's GPT in PyTorch),
`data.py` (a character-level tokenizer) and `data/corpus.txt` (the text of
this course's T0–M1 lessons) are given.

Checks: get-batch · optimizer-groups · initial-loss · accumulation ·
clipping · bf16-master · learns · resume-exact

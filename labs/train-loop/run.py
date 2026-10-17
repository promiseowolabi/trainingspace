#!/usr/bin/env python3
"""Lab 07 — a full training run with YOUR train.py (given; edit freely).

    python run.py                       # ~2 minutes on a laptop CPU
    python run.py --steps 3000 --resume # continue from runs/ckpt.pt

Prints train/val loss, gradient norm, tokens per second and achieved FLOP/s
(6·N per token, P1.L4), checkpoints to runs/, and samples text at the end.
"""

from __future__ import annotations

import argparse
import os
import time

import torch

import data as dataset
import train as T
from model import GPT, GPTConfig

p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, default=2000)
p.add_argument("--batch", type=int, default=32)
p.add_argument("--accum", type=int, default=1)
p.add_argument("--lr", type=float, default=3e-3)
p.add_argument("--warmup", type=int, default=100)
p.add_argument("--layers", type=int, default=4)
p.add_argument("--width", type=int, default=128)
p.add_argument("--heads", type=int, default=4)
p.add_argument("--context", type=int, default=128)
p.add_argument("--amp", action="store_true", help="bf16 autocast (slow on CPUs without bf16 support, P1.L3)")
p.add_argument("--eval-every", type=int, default=250)
p.add_argument("--resume", action="store_true")
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

torch.manual_seed(args.seed)
D = dataset.load()
cfg = GPTConfig(D["vocab_size"], args.context, args.layers, args.heads, args.width)
model = GPT(cfg)
opt = T.configure_optimizer(model, args.lr, weight_decay=0.1)
gen = torch.Generator().manual_seed(args.seed)
os.makedirs("runs", exist_ok=True)
ckpt = os.path.join("runs", "ckpt.pt")
start = T.load_checkpoint(ckpt, model, opt, gen) if args.resume and os.path.exists(ckpt) else 0

N = model.num_params()
tokens_per_step = args.batch * args.accum * args.context
print(f"{N:,} parameters · vocab {D['vocab_size']} · {tokens_per_step:,} tokens/step · first loss should be ≈ ln V")

t0, toks = time.perf_counter(), 0
for step in range(start, args.steps):
    for group in opt.param_groups:
        group["lr"] = T.lr_at(step, args.lr, args.warmup, args.steps, args.lr / 10)
    batches = [T.get_batch(D["train"], args.context, args.batch, gen) for _ in range(args.accum)]
    loss, norm = T.train_step(model, opt, batches, grad_clip=1.0, amp=args.amp)
    toks += tokens_per_step
    if step % args.eval_every == 0 or step == args.steps - 1:
        val = T.evaluate(model, D["val"], args.context, args.batch, 20, torch.Generator().manual_seed(1))
        dt = time.perf_counter() - t0
        print(f"step {step:5d}  train {loss:.3f}  val {val:.3f}  |g| {norm:6.2f}  "
              f"{toks / dt:8.0f} tok/s  {6 * N * toks / dt / 1e9:6.1f} GFLOP/s")
        T.save_checkpoint(ckpt, model, opt, step + 1, gen)

prompt = torch.tensor([D["encode"]("The gradient")])
out = model.generate(prompt, 300, temperature=0.8, generator=torch.Generator().manual_seed(0))
print("\n" + D["decode"](out[0].tolist()))

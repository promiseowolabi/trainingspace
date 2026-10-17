# TRAININGSPACE — build plan, design, and resume state

> Written 2026-10-16. A course in the kernelspace series: kernelspace taught
> how a model is *served*; trainingspace teaches how one is *trained*.
> This file is the single source of truth; keep the RESUME POINT at the bottom
> current.

## Mission

Take a backend engineer who writes Python every day — and has never trained
a model — and make them someone who can explain, implement and debug every
step between "a pile of text" and "a model that follows instructions":

1. **Build it from scratch.** Gradients, an autograd engine, an optimiser, a
   tokenizer, attention and a GPT forward pass — in numpy, by hand, graded in
   the browser.
2. **Scale it.** The PyTorch training loop, mixed precision, the FLOPs and
   memory arithmetic, scaling laws, and data / tensor / pipeline / ZeRO
   parallelism — on a laptop CPU, with simulated ranks where a cluster would
   be needed.
3. **Shape it.** SFT, LoRA, reward models, RLHF, DPO and GRPO; evals,
   contamination, and debugging a run that has gone wrong.

**Python is the course language.** Every code block defaults to Python;
every lab is a Python package; other languages appear only as a comparison.

**Audience:** Java/Python backend engineers. Comfortable with Python; no ML,
no calculus beyond "slope", no linear algebra beyond "a table of numbers"
assumed. Ends at the level of reading — and reproducing the core of — a
frontier training paper.

**Compute assumption:** a laptop CPU. Every required lab finishes on CPU in
minutes. Multi-GPU topics are taught with arithmetic, simulated ranks
(numpy) and real multi-process `torch.distributed` on the `gloo` backend.
A GPU is an optional stretch goal, never a requirement.

## Platform

Forked from **functionspace** (the lean engine: lesson block system,
manifest + agent surface, progress store, completion codes, test suite).
kernelspace contributes *ideas*, not code: the takeaway-number discipline,
isomorphism panels, and "the terminal and the site run the identical checks".

What is new here:

- **Two lab lanes, one set of checks.**
  - **browser** labs (numpy / pure Python): `pytest` locally, *and* the lab
    page runs the very same `checks.py` against your file in **Pyodide**
    (CPython compiled to WebAssembly, loaded in a Web Worker). Drop your
    `.py` file on the page; all green → lab complete. This is kernelspace's
    "drop the `.wasm`", in Python.
  - **local** labs (PyTorch, `torch.distributed`): `pytest` + `verify.py`
    locally; `verify.py` prints a completion code the site accepts.
- The Knative/DataEngine labs, `verify.sh` and the kind/VAST platforms are
  gone.

## Naming, domain, repo

- Course name: **trainingspace**.
- Repo: `github.com/promiseowolabi/trainingspace` (to be created; private
  like functionspace).
- Deploy: GitHub Pages project site →
  `https://promiseowolabi.github.io/trainingspace/`, `VITE_BASE=/trainingspace/`.
- **No `naigap` reference anywhere**, and no sibling-course identity leaking
  into copy. Enforced by `tests/branding.test.ts`.
- Lab files ship as zips under `public/labs/` (the repo is private).

## The rot rule (inherited)

Framework- or model-specific facts — a PyTorch default, a published model's
token count, a paper's hyper-parameter — live in a `vendor` block with the
month verified and the source, or they do not appear. Numbers in prose are
(a) arithmetic the reader can redo, (b) a cited figure, or (c) measured on
the reference run (CPU, PyTorch 2.x, Python 3.12) — and say which.

## Curriculum — 43 lessons, 8 tracks + capstone

| Track | Name | Lessons | Level | Part |
|---|---|---|---|---|
| T0 | The Training Contract | 4 | 200 | build |
| A1 | Autograd from Scratch | 5 | 200 | build |
| O1 | Networks & Optimisation | 6 | 300 | build |
| M1 | The Transformer | 6 | 300 | build |
| P1 | Pretraining | 6 | 400 | scale |
| D1 | Distributed Training | 6 | 400 | scale |
| F1 | Post-Training | 6 | 400 | shape |
| E1 | Evals & Debugging | 4 | 400 | shape |
| T* | Capstone: Train It, Defend It | — | 400 | — |

### T0 — The Training Contract
1. `what-training-is` — a model is a function with knobs; training is "nudge every knob downhill". Loss, gradient, step.
2. `tensors-and-shapes` — arrays, shapes, broadcasting, batch dimensions; the shape is the spec.
3. `the-cost-of-a-matmul` — 2·m·n·k FLOPs, bytes moved, arithmetic intensity.
4. `where-the-memory-goes` — params, grads, optimiser state, activations; the 16-bytes-per-parameter rule.

### A1 — Autograd from Scratch
1. `derivatives-as-sensitivity` — finite differences, the slope as "how much does the loss care".
2. `the-chain-rule-as-a-graph` — computational graphs, local gradients, reverse mode vs forward mode.
3. `a-scalar-engine` — micrograd: `Value`, `_backward`, topological sort, gradient accumulation.
4. `tensor-autograd` — matmul backward, broadcasting backward (unbroadcast), sum/mean.
5. `gradient-checking` — central differences, relative error, the bugs gradcheck catches.

### O1 — Networks & Optimisation
1. `the-mlp` — layers, nonlinearity, why depth; initialisation (Xavier/Kaiming) by variance arithmetic.
2. `cross-entropy` — softmax, log-sum-exp stability, why the gradient is `p - y`.
3. `optimisers` — SGD, momentum, Adam, AdamW (decoupled weight decay).
4. `learning-rate-schedules` — warmup, cosine, WSD; why warmup exists.
5. `normalisation` — LayerNorm vs RMSNorm, pre-norm vs post-norm, residual streams.
6. `generalisation` — train/val split, overfitting, dropout, weight decay as a prior.

### M1 — The Transformer
1. `tokenization` — bytes, BPE merges, vocab size trade-offs.
2. `embeddings-and-position` — token embeddings, learned positions, RoPE.
3. `attention` — Q/K/V, scaled dot product, the causal mask, multi-head.
4. `the-block` — attention + MLP, residuals, pre-norm; the shape of a GPT.
5. `counting-a-gpt` — parameter count from a config; 12·L·d² and the embedding.
6. `sampling` — logits → tokens: temperature, top-k, top-p; why next-token loss produces a generator.

### P1 — Pretraining
1. `data-is-the-model` — crawl → filter → dedup → mixture; tokens as the currency.
2. `the-training-loop` — PyTorch: model, loader, AdamW, schedule, clip, eval, checkpoint.
3. `mixed-precision` — fp32/bf16/fp16/fp8 bit layouts, loss scaling, master weights.
4. `flops-and-mfu` — 6·N·D, tokens/s → MFU, the honest throughput number.
5. `scaling-laws` — Kaplan → Chinchilla, compute-optimal N and D, why everyone over-trains now.
6. `stability` — loss spikes, gradient clipping, z-loss, QK-norm, what a divergence looks like.

### D1 — Distributed Training
1. `collectives` — all-reduce, reduce-scatter, all-gather; the ring and its 2(N−1)/N bytes.
2. `data-parallel` — DDP: replicate, all-reduce gradients, buckets and overlap.
3. `zero-and-fsdp` — shard optimiser state, gradients, then parameters; memory per stage.
4. `tensor-parallel` — column/row splits of the MLP and attention; two all-reduces per block.
5. `pipeline-parallel` — stages, micro-batches, the bubble (p−1)/(m+p−1), 1F1B.
6. `choosing-a-layout` — 3D/4D parallelism, activation checkpointing, a memory+comm budget.

### F1 — Post-Training
1. `from-base-to-assistant` — what pretraining leaves undone; the post-training stack.
2. `sft-and-chat-templates` — instruction data, templates, loss masking.
3. `lora` — low-rank adapters, rank and alpha, memory saved, merge at inference.
4. `reward-models` — Bradley–Terry, pairwise loss, reward hacking.
5. `rlhf-and-dpo` — PPO with a KL penalty; DPO as the closed form; when each wins.
6. `grpo-and-verifiable-rewards` — group-relative advantages, RL on checkable tasks.

### E1 — Evals & Debugging
1. `what-evals-measure` — perplexity vs benchmarks, contamination, variance of a score.
2. `debugging-a-run` — the checklist: overfit one batch, check init loss = ln V, watch grad norms.
3. `checkpointing-and-resumption` — what state a resume needs; determinism.
4. `reading-a-training-paper` — the table you build from any paper: N, D, C, layout, recipe.

## Labs — 14, two lanes

| # | id | lane | lesson | what the reader builds |
|---|---|---|---|---|
| 00 | `numpy-shapes` | browser | T0.L2 | broadcasting, matmul FLOPs, memory accounting |
| 01 | `micrograd` | browser | A1.L3 | scalar autograd engine |
| 02 | `tensor-autograd` | browser | A1.L4 | numpy tensor autograd + gradcheck |
| 03 | `optimisers` | browser | O1.L3 | SGD/momentum/Adam/AdamW + warmup-cosine + clipping |
| 04 | `bpe-tokenizer` | browser | M1.L1 | train merges, encode, decode |
| 05 | `attention` | browser | M1.L3 | stable softmax, causal multi-head attention, RoPE |
| 06 | `gpt-forward` | browser | M1.L5 | GPT forward pass from a weights dict, param count |
| 07 | `train-loop` | local | P1.L2 | PyTorch training loop, bf16 autocast, clip, resume |
| 08 | `flops-and-memory` | browser | P1.L4 | 6ND, MFU, memory per parameter, Chinchilla |
| 09 | `collectives` | browser | D1.L1 | ring all-reduce over simulated ranks |
| 10 | `ddp-from-scratch` | local | D1.L2 | gradient all-reduce over `gloo`, matches 1-process |
| 11 | `zero-sharding` | browser | D1.L3 | ZeRO-1/2/3 over simulated ranks, equality to unsharded |
| 12 | `lora` | local | F1.L3 | LoRA layer, merge/unmerge, fine-tune |
| 13 | `preference-losses` | browser | F1.L5 | DPO loss, PPO clip, GRPO advantages, KL estimators |

### Lab layout (every lab)

```
labs/common/tscheck.py      check runner + completion code (shared)
labs/<id>/README.md         the brief
labs/<id>/<module>.py       the one file you edit — TODO(you) markers
labs/<id>/checks.py         CHECKS = [check_x, ...]; each raises AssertionError
labs/<id>/test_lab.py        pytest wrapper: one test per check
labs/<id>/verify.py         runs CHECKS, prints TS-<LAB>-xxxxxxxx when all pass
labs/_solutions/<id>/       reference solutions — local only (gitignored), never packed
```

`checks.py` is the single source of truth: pytest, `verify.py` and the
browser all run it. Check ids are the function names minus `check_`;
`tests/labs.test.ts` keeps them equal to `src/data/labs.ts`, and
`scripts/check-labs.py` proves every template fails and every solution
passes.

Completion code: `sha256("trainingspace:<lab>:<prefix>")[:8]` →
`TS-<LAB>-<8 hex>`. A progress marker, not an exam — the course says so.

## RESUME POINT

- [x] Plan written.
- [x] Fork stripped and rebranded; Pyodide grading lane (worker + editor + file drop).
- [x] All 43 lessons: T0 (4), A1 (5), O1 (6), M1 (6), P1 (6), D1 (6), F1 (6), E1 (4).
- [x] All 14 labs: template, checks, pytest wrapper, verify.py, README, guide,
      reference solution; `scripts/check-labs.py` passes for every lab.
- [x] Browser lane graded end to end in headless Chrome for all 10 browser labs.
- [x] Capstone steps written.
- [ ] Repo created (github.com/promiseowolabi/trainingspace, private), first commit, Pages deployed.

## Measurements behind the lessons (reference machine)

Apple M5 laptop (10 cores), Python 3.12, numpy 2.5.4, PyTorch 2.12, CPU only.
Python loop dot 85 MFLOP/s; numpy matmul 2.0 TFLOP/s (n = 2048, fp32);
matrix-vector 36 GFLOP/s at 72.6 GB/s. Lab 07's 828k-parameter GPT: 19
steps/s, ~79k tokens/s, val loss 4.98 → 1.75 at step 1,000, then overfits.
bf16 autocast on this CPU ~27× slower than fp32. gloo all-reduce, 2
processes: 4 KB 0.14 ms, 100 MB 17 ms (5.9 GB/s). Activation checkpointing
+38% step time. Each lesson states its own setup next to its numbers.

# trainingspace

**From one gradient to a post-trained GPT.**

A hands-on course on how language models are trained, in Python, for backend
engineers who write Python every day and have never trained a model:

- **Build it** — gradients, an autograd engine, optimisers, a tokenizer,
  attention and a GPT forward pass, from scratch in numpy.
- **Scale it** — the PyTorch training loop, mixed precision, FLOPs and MFU,
  scaling laws, and data / ZeRO / tensor / pipeline parallelism.
- **Shape it** — SFT, LoRA, reward models, RLHF, DPO and GRPO; evals,
  debugging and reading training papers.

Site: https://promiseowolabi.github.io/trainingspace/ (once deployed)

## Curriculum — 43 lessons, 8 tracks, 14 labs

| Track | Name | Lessons |
|---|---|---|
| T0 | The Training Contract | 4 |
| A1 | Autograd from Scratch | 5 |
| O1 | Networks & Optimisation | 6 |
| M1 | The Transformer | 6 |
| P1 | Pretraining | 6 |
| D1 | Distributed Training | 6 |
| F1 | Post-Training | 6 |
| E1 | Evals & Debugging | 4 |
| T* | Capstone: Train It, Defend It | — |

## Labs — Python, two lanes, one set of checks

Every lab is a folder with one file to edit (`TODO(you)` markers), a
`checks.py`, a pytest wrapper and a `verify.py`. `checks.py` is the single
source of truth:

- **browser lane** (10 numpy / pure-Python labs): the lab page runs the
  lab's own `checks.py` against your file in **Pyodide** (CPython compiled to
  WebAssembly) in a Web Worker. Write the code on the page or drop your
  local file on it; all green marks the lab done. `pytest` runs the same checks.
- **local lane** (4 PyTorch labs: training loop, DDP over `gloo`, LoRA):
  `pytest` + `verify.py` on your machine; a laptop CPU is enough. `verify.py`
  prints a `TS-<LAB>-xxxxxxxx` completion code you paste on the lab page.

Every number in a lesson is arithmetic you can redo, a dated and sourced
figure (`vendor` blocks), or a measurement from the reference machine
(Apple M5 laptop, numpy 2.5, PyTorch 2.12, CPU only) — and says which.

## Rules this repo enforces

- **No third-party domain, sibling-course identity or internal environment
  names** — `tests/branding.test.ts`.
- **The rot rule** — framework and model facts live in dated, sourced
  `vendor` blocks.
- **The lab gate** — `scripts/check-labs.py`: every template fails its
  checks, every reference solution (`labs/_solutions/`, never shipped) passes
  them and `verify.py` prints its code. `tests/labs.test.ts` keeps check ids,
  completion codes and wrappers in step with `src/data/labs.ts`.

## Development

Node 22+ and Python 3.10+ (numpy, pytest; torch CPU for the local labs).

```sh
npm ci
npm run dev                          # vite dev server on :3000
npm run export                       # lesson manifest, lessons-md/, llms.txt
python3 scripts/pack-labs.py         # public/labs/<id>.zip
python scripts/check-labs.py         # the lab gate (needs numpy + torch)
npm run lint && npm run typecheck
VITE_BASE=/trainingspace/ npm run build
npm run test                         # after build: includes bundle checks
npm run report                       # what is built and what is outstanding
```

## Layout

```
src/data/lessons/<track>/   lesson content (typed block files)
src/data/lab-guides/        step-by-step lab guides (same block system)
src/data/labs.ts            lab registry — check ids must match checks.py
src/workers/                the Pyodide grading worker
labs/<id>/                  lab folders: module, checks.py, test_lab.py, verify.py
labs/common/tscheck.py      shared check runner + completion codes
labs/_solutions/            reference solutions — local only (gitignored), never packed or bundled
scripts/                    manifest/markdown export, lab packing, lab gate, report
tests/                      registry, branding, labs, site, bundle
```

Pushes to `main` build and deploy to GitHub Pages.

import type { LabGuide } from './types'

const guide: LabGuide = {
  id: 'train-loop',
  requires: ['Python 3.10+', 'numpy, pytest', 'PyTorch (CPU build is enough)', 'about 1 GB of free RAM'],
  blocks: [
    {
      type: 'prose',
      md: `## What you will build

The training loop of P1.L2, piece by piece, around a model and data you are given: lab 06's GPT rewritten in PyTorch (\`model.py\`) and a character-level tokenizer over this course's own lessons (\`data.py\`, 159 KB of text, 146 distinct characters). Seven functions in \`train.py\`; then \`run.py\` uses them to train an 828k-parameter model for 2,000 steps in about two minutes on a laptop CPU.

Install PyTorch's CPU build once (\`pip install torch --index-url https://download.pytorch.org/whl/cpu\`); a GPU is never used. The checks take about ten seconds.`,
    },
    {
      type: 'prose',
      md: `## 1. get_batch

Random windows of the token stream. One \`torch.randint\` call with the generator for the start indices, then \`torch.stack\` of slices for x and of the slices shifted by one for y. Never call \`torch.randint\` without \`generator=\`: the resume check restores the generator and expects the same batches to come next.

## 2. configure_optimizer

Two parameter groups by dimensionality: \`p.dim() >= 2\` decays, everything else does not. The check walks \`model.named_parameters()\` and tells you which tensor landed in the wrong group.

## 3. train_step

The four steps in the docstring. The \`accumulation\` check is the one to understand: it runs one batch of 8 and two micro-batches of 4 through models with identical weights and an SGD optimiser, and requires identical updates. That only works if each micro-batch's loss is divided by the number of micro-batches before \`backward()\` — gradients add up across calls (A1.L3), so the division turns the sum into a mean.

Autocast is a context manager around the **forward** only: \`with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):\`. The backward runs outside it. The \`bf16-master\` check confirms the parameters are still float32 afterwards.`,
    },
    {
      type: 'callout',
      variant: 'warning',
      title: 'bf16 is slow on most CPUs',
      md: `On the reference laptop a bf16-autocast step was about **27× slower** than float32 (0.7 against 19 steps per second), because the CPU emulates bf16 arithmetic. Mixed precision is a GPU optimisation (P1.L3); here it is a correctness exercise. \`run.py --amp\` exists so you can see the cost yourself — do not use it for real runs on a CPU.`,
    },
    {
      type: 'prose',
      md: `## 4. evaluate

Remember \`model.training\`, switch to \`model.eval()\`, average the losses of \`n_batches\` batches, switch back. The \`initial-loss\` check expects a fresh model to score ln 146 ≈ 4.98, the uniform-guess loss of O1.L2.

## 5. Checkpoints

Save a dict of four things; load them back into the objects you are given and return the step. The \`resume-exact\` check loads into a model with a **different** random init and a generator with a **different** seed, so anything you forget to save shows up as a mismatch. The usual omission is the optimizer state: without Adam's m and v the resumed run takes different steps.

## 6. Run it

\`python run.py\`. On the reference laptop, 2,000 steps took about 105 s at 19 steps per second (about 79,000 tokens/s, 390 GFLOP/s by the 6·N rule). Watch the validation column: it reaches its minimum near step 1,000 and then **rises** while training loss keeps falling. 2,000 steps × 4,096 tokens is about 58 passes over a 140,000-character training set — O1.L6's memorisation, in a language model. P1.L1 explains why real pretraining never sees its data that many times.`,
    },
  ],
  cleanup: [
    {
      type: 'prose',
      md: `- Stop at the validation minimum: add early stopping to \`run.py\` (keep the best checkpoint separately).
- Halve the batch and set \`--accum 2\`: the loss curve should be nearly identical, the memory lower (T0.L4).
- Kill \`run.py\` halfway with Ctrl-C and rerun it with \`--resume\`; confirm the step counter and the loss continue where they left off.
- Delete \`runs/\` when you are done; checkpoints are about 10 MB each.`,
    },
  ],
}

export default guide

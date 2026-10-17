import type { LabGuide } from './types'

const guide: LabGuide = {
  id: 'optimisers',
  requires: ['a browser — or Python 3.10+ with numpy and pytest'],
  blocks: [
    {
      type: 'prose',
      md: `## What you will build

The four pieces between "I have gradients" and "the parameters moved": an optimiser's update rule, the schedule that sets its learning rate each step, and the clip that protects it from a bad batch. Each one is a few lines; the checks pin the exact formulas PyTorch uses, so after this lab \`torch.optim.AdamW\` holds no surprises.

Everything operates on plain lists of numpy arrays and updates them **in place** — the first check fails if you rebind \`p\` instead of mutating it.`,
    },
    {
      type: 'prose',
      md: `## 1. SGD and momentum

Without momentum: \`p -= lr * g\`. With momentum, keep the buffer: \`buf *= μ; buf += g; p -= lr * buf\`. The momentum check applies a constant gradient three times; the buffer grows 1 → 1.9 → 2.71, which is momentum's whole point — a consistent direction picks up speed.

## 2. Adam

Increment \`self.t\` once per \`step\`, not once per parameter. Then for each parameter: update \`m\` and \`v\` in place, bias-correct them into \`m_hat\` and \`v_hat\`, and step. The \`adam-first-step\` check feeds gradients of 0.0001, −3 and 250 and expects each parameter to move by exactly \`lr\`: on step 1, \`m̂ = g\` and \`√v̂ = |g|\`, so the update is \`lr · g/|g|\`. Forget the bias correction and the step is (1 − β1)/√(1 − β2) ≈ 3.2× too large — the early steps that most need care get the biggest kick (O1.L3).

## 3. Weight decay, two ways

\`decoupled=False\` adds \`wd · p\` to the gradient **before** the moments. \`decoupled=True\` multiplies \`p\` by \`(1 − lr · wd)\` **before** the Adam update and leaves the gradient alone. The check starts with a zero gradient so you can see the difference: AdamW shrinks the weights by 5%; Adam + L2 would turn the decay into a full-size, sign-only Adam step.`,
    },
    {
      type: 'prose',
      md: `## 4. lr_at

Three branches, straight from the docstring. The check pins six points, including the end of warmup (exactly \`max_lr\`), the middle of the cosine (halfway between max and min) and everything past \`total\` (\`min_lr\`).

## 5. clip_grad_norm

One norm across **all** tensors: square-sum each, add, square-root. If it exceeds \`max_norm\`, multiply every tensor by the same factor. Return the norm from **before** clipping — that is the number you log in a real run to spot instability (P1.L6).

## 6. The race

\`conditioning\` runs your SGD at its largest stable learning rate and your Adam with your schedule on a bowl that is 1,000× steeper in one direction than the other. Nothing new to write: it is the picture from O1.L3, computed with your code.`,
    },
  ],
  cleanup: [
    {
      type: 'prose',
      md: `- Plug \`Adam\` into lab 02's training loop (\`params = [W1.data, b1.data, …]\`, \`grads = [W1.grad, …]\`) and compare its loss curve with plain gradient descent.
- Plot \`lr_at\` for warmup 100, total 1100, then try a warmup of 0 with Adam on the same MLP and a learning rate 10× larger. Watch the first steps.
- If you have PyTorch: run \`torch.optim.AdamW\` and your \`Adam(decoupled=True)\` side by side on the same tensors and diff them after 100 steps.`,
    },
  ],
}

export default guide

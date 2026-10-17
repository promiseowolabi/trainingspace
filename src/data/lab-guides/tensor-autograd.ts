import type { LabGuide } from './types'

const guide: LabGuide = {
  id: 'tensor-autograd',
  requires: ['a browser — or Python 3.10+ with numpy and pytest', 'lab 01 done (recommended)'],
  blocks: [
    {
      type: 'prose',
      md: `## What you will build

Lab 01's engine at array granularity. \`Tensor\` holds a float64 numpy array; \`backward()\` and the topological sort are given. You write seven pieces, and the checks are ordered so each one builds on the last:

| piece | lesson | checks |
|---|---|---|
| \`unbroadcast\` | A1.L4 | unbroadcast |
| \`+\`, \`*\` backward | A1.L4 | add-broadcast, mul |
| \`@\` backward | A1.L4 | matmul |
| \`sum\` backward | A1.L4 | sum-axis |
| \`relu\` | A1.L4 | relu |
| \`cross_entropy\` | O1.L2 preview | cross-entropy-value, -stable, -grad |
| \`gradcheck\` | A1.L5 | gradcheck |

The final check, \`trains-mlp\`, uses all of them to train a small network.`,
    },
    {
      type: 'prose',
      md: `## 1. unbroadcast

Two loops. First, while \`grad\` has more dimensions than \`shape\`, sum away axis 0. Then, for every axis where \`shape\` has a 1 and \`grad\` does not, sum that axis with \`keepdims=True\`. Finish with \`reshape(shape)\`, which also handles the scalar case \`shape == ()\`.

## 2. + and *

Exactly lab 01's rules, each wrapped in \`unbroadcast(…, input.shape)\`. For \`*\`, the local derivative is the other input's **array**, multiplied elementwise with \`out.grad\` before unbroadcasting.

## 3. @

\`dA = G @ B.T\` and \`dB = A.T @ G\`. If you are unsure which is which, print the shapes: there is only one arrangement that type-checks.

## 4. sum

The backward of a sum is a broadcast. When \`axis\` was given and \`keepdims\` was False, \`np.expand_dims(out.grad, axis)\` restores the summed axis as size 1; then \`np.broadcast_to(g, self.shape)\` stretches it.`,
    },
    {
      type: 'prose',
      md: `## 5. cross_entropy — one fused node

Forward, for logits z of shape (N, C) and integer targets of shape (N,):

\`\`\`python
shifted = z - z.max(axis=1, keepdims=True)                 # the log-sum-exp trick
logp = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
loss = -logp[np.arange(N), targets].mean()
\`\`\`

Backward: \`p = exp(logp)\`, subtract 1 at each row's target, divide by N, multiply by \`out.grad\`, and **add** into \`logits.grad\`. The \`stable\` check multiplies the logits by 1000; without the max subtraction, \`exp\` overflows to inf.`,
    },
    {
      type: 'callout',
      variant: 'info',
      title: 'Why the loss of all-zero logits is ln C',
      md: `With every logit equal, softmax is uniform: each class gets probability 1/C, and −log(1/C) = ln C. The check uses C = 10 and expects 2.303. Remember this number — E1.L2 uses "first loss = ln(vocab size)" as the first test of every training run.`,
    },
    {
      type: 'prose',
      md: `## 6. gradcheck

Follow the docstring: zero grads, one backward, copy the grads, then nudge every element by ±ε with \`np.ndindex\`, **restore it**, and compare with \`atol + rtol · |numeric|\`. The check gives you a correct function, which must return \`True\`, and a node whose backward is off by 2×, which must return \`False\` — and then verifies you put every element back.

## 7. trains-mlp

Nothing to write: if the six pieces are right, a 2 → 16 → 3 MLP goes from about 1.6 to below 0.15 in 150 steps of plain gradient descent at learning rate 0.5. If everything else is green and this is red, look at \`+\` with broadcasting — the bias gradients are the usual culprit.`,
    },
  ],
  cleanup: [
    {
      type: 'prose',
      md: `- Add \`tanh\` and \`exp\` and gradient-check them with your own \`gradcheck\`.
- Replace the learning rate 0.5 with 5.0 in a copy of the training loop and watch the loss diverge (T0.L1).
- Keep \`tensor.py\`: lab 03 builds optimisers that update its parameters.`,
    },
  ],
}

export default guide

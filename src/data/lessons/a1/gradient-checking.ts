import type { Lesson } from '../types'

const lesson: Lesson = {
  id: 'a1.l5',
  slug: 'gradient-checking',
  trackId: 'a1',
  index: 5,
  title: 'Gradient Checking',
  minutes: 15,
  hook: 'A wrong backward pass does not crash. It trains — slower, worse, or not at all — and looks like a hyper-parameter problem for a week. Comparing it with two forward passes per element takes a minute and ends the argument.',
  exercise: 'read+quiz',
  takeaway: {
    number: '10⁻¹⁰ vs 0.39',
    claim: 'In float64 a correct cross-entropy gradient matches central differences to a relative error of about 10⁻¹⁰; the same check in float32 at ε = 10⁻⁶ reports 0.39 on a correct gradient — so check in float64 or trust nothing.',
  },
  blocks: [
    {
      type: 'prose',
      md: `You now have two independent ways to get a gradient: the engine's backward pass, and nudging (A1.L1). The backward pass is fast and might be wrong. Nudging is slow and is right up to a known error. A **gradient check** compares them:

\`\`\`python
for every element i of every input x:
    x[i] += eps;  hi = f()
    x[i] -= 2*eps; lo = f()
    x[i] += eps                       # restore it!
    numeric = (hi - lo) / (2 * eps)
    compare numeric with x.grad[i]
\`\`\`

Lab 02's \`gradcheck\` is this loop. The details that make it trustworthy are all in *how* you compare and *what precision* you compare in.`,
    },
    {
      type: 'prose',
      md: `## Compare relative, not absolute

A gradient element of 10⁻⁸ that is off by 10⁻⁸ is 100% wrong; one of 300 off by 10⁻⁶ is fine. So compare with a tolerance that scales with the size of the numbers:

\`\`\`
|analytic − numeric|  ≤  atol + rtol · |numeric|        (lab 02: atol 1e-5, rtol 1e-3)
\`\`\`

or report the **relative error** \`max|a − n| / max(|a|, |n|)\`. As a rule of thumb in float64: below 10⁻⁷ is correct, 10⁻⁷ to 10⁻⁴ deserves a look (a kink, or ε too big), above 10⁻² is a bug.

## The measurements

Checking the fused cross-entropy from lab 02 on a \`(4, 5)\` batch of logits, on the reference laptop:`,
    },
    {
      type: 'prose',
      md: `| precision | ε | relative error | verdict |
|---|---|---|---|
| float64 | 10⁻² | 1.7 × 10⁻⁶ | truncation error from a large step |
| float64 | 10⁻⁴ | **1.7 × 10⁻¹⁰** | correct |
| float64 | 10⁻⁶ | 7.2 × 10⁻¹⁰ | correct |
| float64 | 10⁻⁸ | 8.0 × 10⁻⁸ | rounding error starting to show |
| float32 | 10⁻² | 3.2 × 10⁻⁵ | looks suspicious — is not |
| float32 | 10⁻⁴ | 3.1 × 10⁻³ | looks like a bug — is not |
| float32 | 10⁻⁶ | **0.39** | pure noise on a correct gradient |
| float64, backward missing the 1/N | 10⁻⁶ | **0.75** | the real bug |

The correct gradient checked in float32 at ε = 10⁻⁶ scores *worse* than nothing, while the genuinely broken one (a forgotten \`/ N\`, so every gradient is 4× too big) scores 0.75. Run gradient checks in **float64**, with **ε around 10⁻⁵ to 10⁻⁶**, on **small** inputs. That is why lab 02's \`Tensor\` stores float64 even though training will run in bf16 (P1.L3).`,
    },
    {
      type: 'statline',
      stats: [
        { value: '1.7·10⁻¹⁰', label: 'correct, float64', hint: 'cross-entropy gradient vs central differences, ε = 10⁻⁴' },
        { value: '0.75', label: 'missing 1/N', hint: 'every gradient 4× too large on a batch of 4 — a typical scaling bug' },
        { value: '0.39', label: 'correct, float32', hint: 'ε = 10⁻⁶: f(x+ε) − f(x−ε) is mostly rounding' },
        { value: '2 × size', label: 'forward passes', hint: 'two per element of every input — keep checked tensors tiny' },
      ],
    },
    {
      type: 'prose',
      md: `## What a check catches — and what it does not

**It catches:** a missing or extra factor (\`/N\`, the 2 in d(x²)/dx), a transpose in the wrong place, a forgotten \`unbroadcast\`, \`=\` instead of \`+=\`, a sign error, a gradient sent to the wrong input. In practice, almost every backward-pass bug.

**It does not catch:**

- **A forward pass that is wrong.** The check verifies that backward is the derivative of *your* forward. If forward computes the wrong function, both agree on the wrong answer. Test forward values separately against a reference — lab 02's \`cross-entropy-value\` does.
- **Kinks.** At exactly x = 0, relu has no derivative; if a nudge crosses zero the numeric value is between 0 and 1. Random inputs almost never land there; when a check fails on one element of a relu layer, look at its value before you look at your code.
- **Randomness.** Dropout draws a new mask each call, so \`f(x + ε)\` and \`f(x − ε)\` are different functions. Fix the seed, or check with dropout off.
- **Mutation.** A check that does not restore \`x[i]\` corrupts every later element's result. Lab 02 tests for that too.`,
    },
    {
      type: 'callout',
      variant: 'isomorphism',
      title: 'A differential test',
      md: `You have written this test before. When you replace a slow, obviously-correct implementation with a fast, clever one — a cache, a hand-rolled parser, a vectorised query — you run both on the same inputs and diff the outputs. Gradient checking is a differential test where the oracle is "nudge it and see", and it belongs in the same place: the unit tests of every new operation, never the training loop.`,
    },
    {
      type: 'code',
      filename: 'using it',
      code: `from tensor import Tensor, gradcheck
import numpy as np

x = Tensor(np.random.randn(3, 4))
w = Tensor(np.random.randn(4, 2))
assert gradcheck(lambda x, w: (x @ w).relu().sum(), [x, w])

# PyTorch ships the same tool, and insists on float64 for the same reason:
# torch.autograd.gradcheck(fn, (x.double().requires_grad_(),))`,
    },
    {
      type: 'quiz',
      questions: [
        {
          q: 'A new custom layer gradient-checks with relative error 3 × 10⁻³ in float32 at ε = 1e-4. What should you do?',
          options: [
            'Ship it — 3 × 10⁻³ is small',
            'Rewrite the backward pass — it is wrong',
            'Re-run the check in float64 with ε ≈ 1e-6 before concluding anything: in float32 that error is what a correct gradient looks like',
            'Use a larger ε until the error goes down',
          ],
          correct: [2],
          explanation: 'A correct cross-entropy gradient scored 3.1 × 10⁻³ under exactly those settings. Float32 cannot tell a correct gradient from a slightly wrong one at that step size. In float64 a correct gradient scores around 10⁻¹⁰ and a real bug stays large, so the verdict becomes unambiguous.',
        },
        {
          q: 'Every gradient in a model is exactly 2× what gradcheck says. Which bug is most consistent with that?',
          options: [
            'A transpose in the matmul backward',
            'A constant factor error — for example, the mean\'s 1/N applied once in forward but the loss summed, or a node\'s gradient added twice',
            'Overflow in the softmax',
            'The learning rate is 2× too high',
          ],
          correct: [1],
          explanation: 'A uniform scale points to a constant factor: a missing or extra 1/N or 2, or a contribution accumulated twice (for example, calling _backward on one node two times). A transpose bug gives a different pattern per element, not a clean ratio; the learning rate never enters the gradient.',
        },
        {
          q: 'A layer with dropout fails gradcheck on random elements, a different set each run. Most likely explanation?',
          options: [
            'Dropout\'s backward pass is wrong',
            'Each call to f draws a new dropout mask, so f(x+ε) and f(x−ε) are not the same function — fix the seed or disable dropout for the check',
            'ε is too small',
            'Dropout cannot be differentiated',
          ],
          correct: [1],
          explanation: 'Central differences assume the function is deterministic. A fresh random mask per call makes the numeric gradient noise. With the mask fixed, dropout\'s gradient (multiply by the same mask and scale) checks cleanly.',
        },
      ],
    },
    {
      type: 'deepdive',
      title: 'Going deeper',
      md: `- CS231n, "Gradient checks" in the *Neural Networks Part 3* notes — the relative-error rule of thumb, kinks, and checking only a few dimensions of a large tensor.
- PyTorch's \`torch.autograd.gradcheck\` and \`gradgradcheck\` (second derivatives): the same algorithm, with \`eps=1e-6\`, \`atol=1e-5\` and \`rtol=1e-3\` by default — lab 02's defaults.
- This closes A1: you have an engine you trust. O1 now uses it to train something worth training.`,
    },
  ],
}

export default lesson

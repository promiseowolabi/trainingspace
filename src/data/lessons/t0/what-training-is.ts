import type { Lesson } from '../types'

const lesson: Lesson = {
  id: 't0.l1',
  slug: 'what-training-is',
  trackId: 't0',
  index: 1,
  title: 'What Training Is',
  minutes: 15,
  hook: 'A model is a function with knobs. Training is four lines in a loop — run it, score it, ask each knob which way is downhill, nudge it — repeated until the score stops improving.',
  exercise: 'read+quiz',
  takeaway: {
    number: '27.7 → 0.001 in 10 steps',
    claim: 'Gradient descent on y = w·x + b took the loss from 27.7 to 0.001 in ten steps at learning rate 0.05, and to 50,640 — divergence — at 0.2: the same four lines, one knob of difference.',
  },
  blocks: [
    {
      type: 'prose',
      md: `Strip the vocabulary off any training run — a 7-billion-parameter language model, a fraud classifier, the line of best fit in a spreadsheet — and the same four steps are left:

1. **Forward.** Run the model on some inputs and get its predictions.
2. **Loss.** Score the predictions against the right answers with one number: how wrong, on average.
3. **Backward.** For every knob (parameter) in the model, work out which direction makes the loss smaller, and how strongly the loss responds. That vector of sensitivities is the **gradient**.
4. **Step.** Move every parameter a small distance in the downhill direction.

Repeat until the loss stops falling. Everything else in this course — autograd, Adam, attention, bf16, ZeRO, DPO — is either a better way to do one of those four steps, or a way to do them on more data, faster, with less memory.`,
    },
    {
      type: 'code',
      filename: 'the_loop.py',
      code: `for batch in data:
    predictions = model(batch.inputs)            # 1. forward
    loss = loss_fn(predictions, batch.targets)   # 2. loss
    grads = gradient(loss, model.parameters)     # 3. backward
    for p, g in zip(model.parameters, grads):    # 4. step
        p -= learning_rate * g`,
      chips: ['the whole course', '4 steps'],
    },
    {
      type: 'isomorphism',
      title: 'You already run this loop',
      pairs: [
        {
          os: 'a load test tuning a connection pool',
          osLine: 'change the pool size, rerun, read p99, keep the change if p99 fell',
          llm: 'one training step',
          llmLine: 'change the weights, rerun the batch, read the loss, keep the change because the gradient promised it would fall',
        },
        {
          os: 'p99 latency',
          osLine: 'one number that summarises how bad things are',
          llm: 'the loss',
          llmLine: 'one number that summarises how wrong the predictions are',
        },
        {
          os: 'a flame graph',
          osLine: 'which function is responsible for how much of the time',
          llm: 'the gradient',
          llmLine: 'which parameter is responsible for how much of the loss — and in which direction',
        },
      ],
    },
    {
      type: 'prose',
      md: `The difference from your load test is step 3. You find out which way to turn a pool-size knob by trying both directions. A model has millions of knobs; trying each one both ways would cost two forward passes per parameter per step. The gradient gives you **all of them at once**, for roughly the price of one extra forward pass. A1 is about how.

## The smallest model there is

Three data points that happen to lie on a line: \`(1, 3)\`, \`(2, 5)\`, \`(3, 7)\`. The model is \`y = w·x + b\` — two knobs. The loss is **mean squared error**: the average of (prediction − target)².

Start with both knobs at zero. Every prediction is 0, the errors are −3, −5 and −7, and the loss is (9 + 25 + 49) / 3 = **27.67**.

For this loss the gradient can be written down by hand (A1.L1 shows where it comes from):

\`\`\`
dL/dw = (2/3) · Σ (pred − y) · x  = (2/3) · (−3·1 − 5·2 − 7·3) = −22.67
dL/db = (2/3) · Σ (pred − y)      = (2/3) · (−3 − 5 − 7)       = −10.00
\`\`\`

Both are negative: increasing either knob decreases the loss. With a **learning rate** of 0.05, the step moves \`w\` by 0.05 × 22.67 = 1.13 and \`b\` by 0.05 × 10 = 0.5.`,
    },
    {
      type: 'statline',
      stats: [
        { value: '27.67', label: 'loss at step 0', hint: 'w = 0, b = 0: every prediction is 0' },
        { value: '5.49', label: 'after 1 step', hint: 'w = 1.13, b = 0.50' },
        { value: '1.09', label: 'after 2 steps', hint: 'w = 1.64, b = 0.72' },
        { value: '0.0011', label: 'after 10 steps', hint: 'w = 2.04, b = 0.91 — close to the true 2 and 1' },
      ],
    },
    {
      type: 'prose',
      md: `Arithmetic you can redo with a calculator, or with the twelve lines of Python below. Notice two things.

**The loss falls fast, then slowly.** Most of the improvement is in the first few steps; after step 10, \`w\` and \`b\` crawl towards 2 and 1 because the gradient — the slope — is nearly flat near the bottom. Every real loss curve has this shape.

**The learning rate is not a detail.** Run the same code at 0.2 and after 20 steps the loss is **50,640**; at 0.3 it is about **2 × 10¹⁵**. Each step overshoots the bottom by more than it started, so the next gradient is bigger, so the next overshoot is bigger. This is **divergence**, and it is the first failure mode you will meet in a real run (P1.L6).`,
    },
    {
      type: 'code',
      filename: 'fit_a_line.py',
      code: `xs, ys = [1, 2, 3], [3, 5, 7]
w, b, lr = 0.0, 0.0, 0.05

for step in range(11):
    preds = [w * x + b for x in xs]                                  # forward
    loss = sum((p - y) ** 2 for p, y in zip(preds, ys)) / len(xs)    # loss
    dw = 2 / len(xs) * sum((p - y) * x for p, y, x in zip(preds, ys, xs))
    db = 2 / len(xs) * sum(p - y for p, y in zip(preds, ys))         # backward
    print(f"step {step:2}  w={w:.3f}  b={b:.3f}  loss={loss:.4f}")
    w -= lr * dw                                                     # step
    b -= lr * db`,
    },
    {
      type: 'diagram',
      caption: 'fig 1 — one training step',
      height: 64,
      nodes: [
        { id: 'params', x: 2, y: 6, w: 18, h: 12, label: 'parameters', sub: 'w, b' },
        { id: 'fwd', x: 28, y: 6, w: 18, h: 12, label: 'forward', sub: 'preds = w·x + b' },
        { id: 'loss', x: 54, y: 6, w: 18, h: 12, label: 'loss', sub: 'mean (pred − y)²' },
        { id: 'data', x: 80, y: 6, w: 18, h: 12, label: 'data', sub: 'x, y' },
        { id: 'bwd', x: 54, y: 40, w: 18, h: 12, label: 'backward', sub: 'dL/dw, dL/db' },
        { id: 'step', x: 28, y: 40, w: 18, h: 12, label: 'step', sub: 'p −= lr · g' },
      ],
      edges: [
        { from: 'params', to: 'fwd' },
        { from: 'data', to: 'loss' },
        { from: 'fwd', to: 'loss' },
        { from: 'loss', to: 'bwd' },
        { from: 'bwd', to: 'step' },
        { from: 'step', to: 'params' },
      ],
      steps: [
        { caption: 'Forward: the current parameters turn inputs into predictions. Nothing is learned here; this is just running the function.', active: ['params', 'fwd'], edges: ['params->fwd'] },
        { caption: 'Loss: predictions are compared with the targets from the data and collapsed into one number — mean squared error for this model.', active: ['fwd', 'loss', 'data'], edges: ['fwd->loss', 'data->loss'] },
        { caption: 'Backward: the gradient says, for each parameter, how much the loss changes per unit nudge, and in which direction. One number per parameter.', active: ['loss', 'bwd'], edges: ['loss->bwd'] },
        { caption: 'Step: every parameter moves against its gradient by learning-rate × gradient. Then the loop starts again with the updated parameters.', active: ['bwd', 'step', 'params'], edges: ['bwd->step', 'step->params'] },
      ],
    },
    {
      type: 'prose',
      md: `## From two knobs to billions

A language model runs the identical loop. What changes is the size of every box:

| | the line | a small GPT (capstone) | a 7B model |
|---|---|---|---|
| parameters | 2 | ~5 million | 7 × 10⁹ |
| one example | a number | 256 tokens of text | 4,096+ tokens |
| loss | squared error | cross-entropy over the vocabulary (O1.L2) | same |
| backward | by hand | autograd (A1) | autograd, sharded over GPUs (D1) |
| step | \`p -= lr * g\` | AdamW with a schedule (O1.L3–L4) | same, in mixed precision (P1.L3) |

Each row of that table is a track of this course. The loop is the contract they all honour.`,
    },
    {
      type: 'callout',
      variant: 'warning',
      title: 'A falling loss is not a working model',
      md: `The loss measures how wrong the model is **on the data you score it on**. A model with enough knobs can drive training loss towards zero by memorising the training set and still be useless on anything new. That is why every real run holds some data back and reports a **validation loss** as well (O1.L6). The line above has two knobs and three points, so it cannot memorise — your GPT can.`,
    },
    {
      type: 'quiz',
      questions: [
        {
          q: 'A run\'s loss goes 2.1, 1.8, 3.9, 47, 9.1e5, nan over its first five logged steps. What is the most likely cause, and the first thing to change?',
          options: [
            'The model is too small — add layers',
            'The learning rate is too large — each step overshoots and the overshoot grows; lower it',
            'The data is shuffled badly — reshuffle',
            'The loss function is wrong — switch to mean absolute error',
          ],
          correct: [1],
          explanation: 'Loss that falls briefly and then grows geometrically is divergence: each step overshoots further, so the next gradient is bigger. The line in this lesson does exactly that at lr 0.2. Adding layers or changing the loss does not stop an oversized step; lowering the learning rate (and, later, warmup and clipping) does.',
        },
        {
          q: 'Why do frameworks compute gradients instead of nudging each parameter up and down and measuring the loss?',
          options: [
            'Nudging is less accurate',
            'Nudging needs two forward passes per parameter per step — billions of passes for a large model — while the gradient for all parameters costs about one extra pass',
            'Gradients work only for neural networks',
            'Nudging cannot handle negative numbers',
          ],
          correct: [1],
          explanation: 'The nudge method (finite differences) is perfectly accurate enough to check gradients with (A1.L5), but its cost scales with the number of parameters. Reverse-mode autograd returns every partial derivative for a small constant multiple of one forward pass, which is the only reason training large models is feasible.',
        },
        {
          q: 'Your training loss is 0.02 and still falling, but validation loss has risen for the last three evaluations. What is happening?',
          options: [
            'The model is still learning — keep going',
            'The learning rate is too low',
            'The model has started memorising the training set: training loss no longer predicts performance on new data',
            'Validation loss is computed incorrectly; ignore it',
          ],
          correct: [2],
          explanation: 'Training loss measures fit to examples the model has seen. When it keeps falling while validation loss rises, the extra fit is memorisation, not learning. Stopping, regularising or adding data are the responses (O1.L6); "keep going" makes it worse.',
        },
      ],
    },
    {
      type: 'deepdive',
      title: 'Going deeper',
      md: `- Run \`fit_a_line.py\` with learning rates 0.01, 0.05, 0.1, 0.15 and 0.2 and print the loss every step. Find the largest rate that still converges — about 0.18 for this data, set by the curvature of the loss: 2 ÷ the largest eigenvalue of its Hessian, 11.09 (O1.L4).
- T0.L2 introduces the arrays that let you write the forward pass without the Python loops — the same loop, about 24,000× faster on the reference laptop (T0.L3).
- Michael Nielsen, *Neural Networks and Deep Learning*, chapter 1 (neuralnetworksanddeeplearning.com) — the same idea, built up from a single neuron.`,
    },
  ],
}

export default lesson

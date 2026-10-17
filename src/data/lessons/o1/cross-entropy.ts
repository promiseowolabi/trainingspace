import type { Lesson } from '../types'

const lesson: Lesson = {
  id: 'o1.l2',
  slug: 'cross-entropy',
  trackId: 'o1',
  index: 2,
  title: 'Cross-Entropy',
  minutes: 17,
  hook: 'Every language model is trained on one loss: minus the log-probability it gave the right next token. Its gradient is the simplest expression in deep learning — predicted minus actual — and computing it naively overflows at a logit of 89.',
  exercise: 'read+quiz',
  takeaway: {
    number: 'ln 50,257 ≈ 10.82',
    claim: 'A language model that guesses uniformly over a 50,257-token vocabulary has cross-entropy ln 50,257 ≈ 10.82 nats; every real run starts there, and the gradient of the loss with respect to the logits is just softmax − onehot.',
  },
  blocks: [
    {
      type: 'prose',
      md: `T0.L1 used squared error for a model that predicts a number. A classifier — and a language model is a classifier over its vocabulary, asked once per position — predicts a **probability distribution** instead. It needs two pieces.

## Softmax: scores to probabilities

The last layer outputs one unbounded score per class, the **logits** \`z\`. Softmax turns them into probabilities:

\`\`\`
p_i = exp(z_i) / Σ_j exp(z_j)
\`\`\`

Every \`p_i\` is positive and they sum to 1. Adding the same constant to every logit changes nothing — the constant cancels between numerator and denominator. Keep that fact; it is what makes the computation safe.

## Cross-entropy: score the right answer

If the correct class is \`y\`, the loss is

\`\`\`
L = −log p_y
\`\`\`

Give the right answer probability 1 and the loss is 0. Give it 0.5 and the loss is 0.69. Give it 0.001 and the loss is 6.9 — confident and wrong is punished hard, without limit. Averaged over a batch (and, for a language model, over every position in every sequence), that is the loss of every model in the rest of this course.`,
    },
    {
      type: 'statline',
      stats: [
        { value: 'ln C', label: 'uniform guess', hint: 'p_y = 1/C for every class, so L = −log(1/C) = ln C' },
        { value: '10.82', label: 'ln 50,257', hint: 'GPT-2\'s vocabulary: the loss of a model that knows nothing' },
        { value: '11.76', label: 'ln 128,256', hint: 'the vocabulary size of the Llama 3 tokenizer' },
        { value: '0.69', label: 'p = 0.5', hint: '−ln 0.5: a coin flip on the right answer' },
      ],
    },
    {
      type: 'prose',
      md: `**ln C is your first sanity check.** A freshly initialised model has small, nearly equal logits, so its first loss should be within a few percent of ln(vocabulary size). If a GPT-2-vocabulary model logs 25 at step 0, its initial logits are far too large, and the first hundred steps will be spent undoing that. E1.L2 makes this the first line of its debugging checklist; lab 02 checks it for C = 10.

Read the loss as **nats per token** (natural log). Divide by ln 2 for bits per token; \`exp(L)\` is the **perplexity** — "as uncertain as a uniform choice among exp(L) options". A loss of 3.0 is a perplexity of 20.

## The gradient is predicted minus actual

Combine the two formulas and differentiate with respect to the logits. Everything cancels to:

\`\`\`
∂L/∂z_i = p_i − 1[i = y]          (divide by N for a batch mean)
\`\`\`

The gradient pushes the correct class's logit up by (1 − p_y) and every other logit down by its own probability. A confident correct prediction gives a near-zero gradient; a confident wrong one gives the largest possible. It is the vector lab 02's \`cross_entropy\` returns, and it is why frameworks fuse softmax and the loss into one node: the fused backward needs only \`p\`, not the chain of \`exp\`, \`sum\`, \`divide\` and \`log\` nodes.`,
    },
    {
      type: 'prose',
      md: `## Computing it without overflowing

\`exp\` grows fast. Measured with numpy: \`exp(709)\` is 8.2 × 10³⁰⁷ in float64 and \`exp(710)\` is **inf**; in float32 the wall is between 88 and **89**. Logits of 100 are routine in a large model's last layer, so the textbook formula produces inf / inf = **nan**.

The fix uses the shift-invariance above. Subtract each row's maximum logit before exponentiating — the largest exponent is then exactly 0 — and compute the log directly:

\`\`\`python
shifted = z - z.max(axis=-1, keepdims=True)                         # largest entry is 0
log_probs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
loss = -log_probs[np.arange(N), y].mean()
\`\`\`

This is the **log-sum-exp trick**. Never compute \`log(softmax(z))\` as two steps: when a probability underflows to 0, its log is −inf even though the log-probability itself was a perfectly ordinary −120.`,
    },
    {
      type: 'code',
      filename: 'in PyTorch',
      code: `import torch.nn.functional as F

# logits: (B, T, V) from the model; targets: (B, T) token ids
loss = F.cross_entropy(logits.view(-1, V), targets.view(-1))
# one fused, stable kernel: log-softmax + negative log-likelihood + mean.
# Pass logits — never softmax(logits) — or you apply softmax twice.`,
      chips: ['fused', 'expects raw logits'],
    },
    {
      type: 'callout',
      variant: 'segfault',
      title: 'Softmax applied twice',
      md: `\`F.cross_entropy\` applies log-softmax itself. Passing it \`softmax(logits)\` squashes every input into [0, 1], so the second softmax sees nearly equal scores and the loss barely moves below ln C. Training "works" — the loss falls a little and then plateaus. If a classifier's loss stalls just under ln C, look for a softmax at the end of the model.`,
    },
    {
      type: 'quiz',
      questions: [
        {
          q: 'A new language model with a 32,000-token vocabulary logs a loss of 41.7 at step 0. What does that tell you?',
          options: [
            'Normal — losses start high',
            'Something is wrong with the initial logits: a model that knows nothing should start near ln 32,000 ≈ 10.4, so these logits are far too large (bad init, missing scaling, or a bug in the output layer)',
            'The vocabulary is too large',
            'The learning rate is too high',
          ],
          correct: [1],
          explanation: 'With small, nearly equal logits the softmax is close to uniform and the loss is close to ln V. A loss four times larger means the model starts confidently wrong, which costs steps to unlearn and often hints at an initialisation bug. The learning rate cannot have acted yet at step 0.',
        },
        {
          q: 'For one example the model assigns p = [0.7, 0.2, 0.1] and the correct class is 1 (the 0.2). What is the gradient of the loss with respect to the logits?',
          options: ['[0.7, 0.2, 0.1]', '[0.7, −0.8, 0.1]', '[−0.3, 0.8, −0.9]', '[0, −1, 0]'],
          correct: [1],
          explanation: '∂L/∂z = p − onehot(y) = [0.7, 0.2 − 1, 0.1] = [0.7, −0.8, 0.1]. Gradient descent subtracts it, so the correct logit rises and the others fall in proportion to the probability they stole.',
        },
        {
          q: 'A loss computed as -np.log(softmax(z)[y]) returns inf for some examples even though the model is training fine. Why, and what is the fix?',
          options: [
            'The model is diverging — lower the learning rate',
            'p_y underflowed to exactly 0 for very negative logits, and log(0) = −inf; compute log-softmax directly with the max-shift (log-sum-exp) instead of log of softmax',
            'softmax should be replaced with sigmoid',
            'Use float16 to get more range',
          ],
          correct: [1],
          explanation: 'A log-probability of −800 is a valid, finite number, but exp(−800) underflows to 0 in float64, and log(0) is −inf. Computing shifted − log Σ exp(shifted) never forms the tiny probability. Float16 has less range, not more.',
        },
      ],
    },
    {
      type: 'deepdive',
      title: 'Going deeper',
      md: `- Derive the gradient yourself: write L = −z_y + log Σ exp(z_j) and differentiate each term — five lines.
- Why cross-entropy and not squared error for classifiers: squared error on a sigmoid output has gradients that vanish exactly when the model is confidently wrong. Bishop, *Pattern Recognition and Machine Learning*, §4.3.
- Label smoothing replaces the one-hot target with 0.9 on the right class and the rest spread evenly; it changes the gradient to p − smoothed_target. Check that your fused backward handles it.`,
    },
  ],
}

export default lesson

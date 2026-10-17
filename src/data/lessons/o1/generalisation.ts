import type { Lesson } from '../types'

const lesson: Lesson = {
  id: 'o1.l6',
  slug: 'generalisation',
  trackId: 'o1',
  index: 6,
  title: 'Generalisation',
  minutes: 16,
  hook: 'Training loss measures how well the model fits what it has seen. The number you actually care about is how it does on what it has not — and the two can move in opposite directions for thousands of steps while accuracy does not change at all.',
  exercise: 'read+quiz',
  takeaway: {
    number: 'val 0.62 → 0.95, accuracy flat',
    claim: 'An MLP that memorised 200 noisy examples (training loss 0.000) saw validation loss climb from 0.62 to 0.95 while validation accuracy stayed at 87% — it grew more confident, not more right, and only a held-out set could show it.',
  },
  blocks: [
    {
      type: 'prose',
      md: `Hold some data back. Never train on it; evaluate on it every few hundred steps. That **validation set** is the only honest measure of whether the model learned something general or memorised its training set — the difference T0.L1 warned about.

## An experiment in memorisation

A 20-dimensional classification task with noisy labels — 200 training examples, 2,000 validation examples — and an MLP with two hidden layers of 256 units: about 72,000 parameters for 200 examples. AdamW at 10⁻³, full-batch, measured on the reference laptop:`,
    },
    {
      type: 'prose',
      md: `| step | train loss | val loss | val accuracy |
|---|---|---|---|
| 1 | 0.702 | 0.672 | 61.4% |
| 250 | — | **0.623** (best) | — |
| 500 | 0.000 | 0.703 | 87.3% |
| 1,250 | 0.000 | 0.819 | 86.9% |
| 3,000 | 0.000 | **0.950** | 87.1% |

By step 500 training loss is zero to three decimal places: the network has memorised all 200 labels, noise included. From then on, validation loss rises steadily while validation accuracy does not move. The model is not getting more examples wrong — it is getting the ones it gets wrong **more confidently** wrong, and cross-entropy (O1.L2) punishes confidence without limit.`,
    },
    {
      type: 'statline',
      stats: [
        { value: '~72,000', label: 'parameters', hint: '20·256 + 256·256 + 256·2, plus biases' },
        { value: '200', label: 'training examples', hint: 'with label noise — some labels are wrong' },
        { value: '0.000', label: 'train loss by step 500', hint: 'every training label, noise included, memorised' },
        { value: '0.62 → 0.95', label: 'val loss, step 250 → 3,000', hint: 'validation accuracy 87.3% → 87.1% over the same steps' },
      ],
    },
    {
      type: 'prose',
      md: `## The tools, and what they did here

**Early stopping.** Keep the checkpoint with the best validation loss — step 250 above. It is free, and the reason every training loop evaluates periodically and checkpoints (P1.L2).

**Weight decay.** Pulling weights towards zero (AdamW, O1.L3) penalises the large weights that sharp, memorising fits need. Same experiment, final validation loss after 3,000 steps:

| regulariser | best val loss | val loss at 3,000 |
|---|---|---|
| none | 0.623 | 0.950 |
| weight decay 0.1 | 0.605 | 0.820 |
| weight decay 1.0 | **0.510** | **0.659** |
| dropout 0.5 | 0.582 | 1.560 |

**Dropout** — zero a random half of the hidden units on each training step — improved the best checkpoint but made the long-run loss *worse* here: with full-batch training and nothing to stop it, the network still memorised, and the evaluation-time network (dropout off) was more overconfident than the one it trained. Regularisers are not interchangeable, and the only way to know which helps is to measure on held-out data.

**More data** beats all of them. The ratio here — 360 parameters per example — is extreme. Pretraining sits at the other end.`,
    },
    {
      type: 'callout',
      variant: 'isomorphism',
      title: 'Why LLM pretraining rarely overfits',
      md: `A 7B model trained on 2 trillion tokens sees about **285 tokens per parameter**, and usually sees each token **once**. A model cannot memorise examples it never repeats, so pretraining train and validation losses track each other closely, and dropout is typically switched off. Overfitting returns in the places where data is small and repeated: fine-tuning (F1), multiple epochs over scarce data, and benchmark contamination (E1.L1) — where the "validation" set leaked into training and the score measures memory.`,
    },
    {
      type: 'diagram',
      caption: 'fig 1 — reading two loss curves',
      height: 60,
      nodes: [
        { id: 'both', x: 2, y: 6, w: 22, h: 12, label: 'both falling', sub: 'learning' },
        { id: 'gap', x: 30, y: 6, w: 22, h: 12, label: 'gap opens', sub: 'val flattens' },
        { id: 'diverge', x: 58, y: 6, w: 22, h: 12, label: 'val rises', sub: 'memorising' },
        { id: 'stop', x: 58, y: 38, w: 22, h: 12, label: 'best checkpoint', sub: 'early stopping' },
        { id: 'fix', x: 30, y: 38, w: 22, h: 12, label: 'regularise / more data', sub: 'next run' },
      ],
      edges: [
        { from: 'both', to: 'gap' },
        { from: 'gap', to: 'diverge' },
        { from: 'diverge', to: 'stop' },
        { from: 'stop', to: 'fix' },
      ],
      steps: [
        { caption: 'Early in training both losses fall together: what the model learns from the training set transfers to data it has not seen.', active: ['both'] },
        { caption: 'A gap opens: training loss keeps falling, validation flattens. The model is starting to fit details specific to the training examples.', active: ['both', 'gap'], edges: ['both->gap'] },
        { caption: 'Validation loss rises while training loss heads to zero — memorisation. Accuracy may not move at all; the loss is reporting overconfidence first.', active: ['gap', 'diverge'], edges: ['gap->diverge'] },
        { caption: 'Keep the checkpoint from the validation minimum, then fix the cause for the next run: more data, weight decay, a smaller model or fewer epochs.', active: ['diverge', 'stop', 'fix'], edges: ['diverge->stop', 'stop->fix'] },
      ],
    },
    {
      type: 'quiz',
      questions: [
        {
          q: 'A fine-tuning run\'s validation loss rises for 2,000 steps while validation accuracy stays flat. A teammate says "accuracy is fine, so nothing is wrong". What is the better reading?',
          options: [
            'They are right — loss and accuracy measure the same thing',
            'The model is becoming more confident on the examples it gets wrong; it is overfitting, and calibrated probabilities (which downstream uses like sampling and ranking depend on) are getting worse — stop at the validation-loss minimum',
            'The validation set is broken',
            'Raise the learning rate to escape the plateau',
          ],
          correct: [1],
          explanation: 'Accuracy only counts the argmax; cross-entropy also measures confidence. Rising validation loss with flat accuracy is the signature of growing overconfidence from memorisation, as in this lesson\'s experiment. For a language model the probabilities ARE the product — sampling (M1.L6) draws from them.',
        },
        {
          q: 'Why do large pretraining runs usually train without dropout?',
          options: [
            'Dropout is incompatible with attention',
            'Each token is typically seen once, so there is little opportunity to memorise; train and validation losses track each other and dropout would only slow learning',
            'Dropout uses too much memory',
            'Weight decay replaces it exactly',
          ],
          correct: [1],
          explanation: 'Overfitting needs repetition. With trillions of tokens and roughly one epoch, the model never sees an example twice, so regularising against memorisation costs learning speed for no benefit. It returns when data is small and repeated, as in fine-tuning.',
        },
        {
          q: 'Which checkpoint should you keep from this lesson\'s unregularised run?',
          options: ['Step 1', 'Step 250 — lowest validation loss', 'Step 500 — training loss reached zero', 'Step 3,000 — the final one'],
          correct: [1],
          explanation: 'The validation minimum (0.623 at step 250) is the best estimate of performance on new data. Zero training loss marks memorisation, not success; the final checkpoint is the most overconfident.',
        },
      ],
    },
    {
      type: 'deepdive',
      title: 'Going deeper',
      md: `- Zhang et al., "Understanding deep learning requires rethinking generalization" (2016) — networks that memorise random labels perfectly, and what that implies.
- Nakkiran et al., "Deep Double Descent" (2019) — why bigger models can generalise better even past the point of memorising.
- Rerun the experiment with 2,000 training examples instead of 200 and watch the gap close.
- This closes O1. M1 builds the model all of this will train.`,
    },
  ],
}

export default lesson

import type { Lesson } from '../types'

const lesson: Lesson = {
  id: 'p1.l6',
  slug: 'stability',
  trackId: 'p1',
  index: 6,
  title: 'Stability',
  minutes: 18,
  hook: 'Small models forgive almost anything. Large runs spike: the loss jumps by a whole nat in one step, sometimes recovers, sometimes diverges to nan — weeks into a run that cannot simply be restarted. Stability is a set of cheap insurance policies, each aimed at one way the numbers blow up.',
  exercise: 'read+quiz',
  takeaway: {
    number: '~20 spikes in one run',
    claim: 'PaLM 540B saw roughly 20 loss spikes during pretraining and handled them by restarting about 100 steps earlier and skipping 200–500 batches — while on lab 07\'s 0.8M-parameter model, clipping made almost no difference at all.',
  },
  blocks: [
    {
      type: 'prose',
      md: `## What instability looks like

Four signatures, from mild to fatal:

1. **A spike.** Loss jumps sharply for a few steps, then recovers. The gradient norm usually spikes first.
2. **A plateau after a spike.** The run recovers to a worse loss than it had, and stays there.
3. **Divergence.** Loss climbs geometrically to inf or nan (T0.L1's learning rate of 0.2, at scale).
4. **Silent collapse.** Loss flatlines at the uniform-guess value — O1.L4's post-norm runs at 3·10⁻³ never left ln 32.

Divergence in a small experiment costs minutes. In a run on thousands of GPUs it costs the compute since the last good checkpoint, plus the time to work out what happened. So large runs stack defences, each of which removes one route to blowing up.`,
    },
    {
      type: 'prose',
      md: `## The defences

| defence | what it bounds | cost |
|---|---|---|
| **warmup** (O1.L4) | early steps, while Adam's statistics settle | a few thousand steps at reduced rate |
| **pre-norm** (O1.L5) | the gradient path through depth | none — it is the default |
| **gradient clipping** at 1.0 | the size of any single update | none when healthy; hides problems if it is always active |
| **β2 = 0.95** instead of 0.999 (O1.L3) | stale second-moment estimates when gradients grow suddenly | slightly noisier steps |
| **weight decay** | slow growth of weight norms | a hyper-parameter |
| **z-loss**: + 10⁻⁴ · (log Σ exp z)² | output logits drifting to huge values, which make the softmax overconfident | one extra term in the loss |
| **QK-norm**: normalise q and k before the dot product | attention logits growing until the softmax saturates (M1.L3) | two norms per layer |

Two of these are worth a sentence more. **z-loss** penalises the log of the softmax normaliser, keeping the output logits near zero mean scale without changing which token is predicted; PaLM used it. **QK-norm** applies a LayerNorm or RMSNorm to queries and keys, so their dot product cannot grow without bound — the attention version of the √D argument, applied continuously instead of only at initialisation.`,
    },
    {
      type: 'vendor',
      snapshot: '2026-10',
      title: 'How a real run handled spikes',
      md: `The PaLM paper reports that, despite gradient clipping, training the 540B model saw loss spikes **roughly 20 times**, at irregular intervals and sometimes late in training. They were not caused by bad data alone: replaying the same batches from a different starting state did not reproduce them. The mitigation was operational — **restart from a checkpoint about 100 steps before the spike and skip roughly 200–500 data batches**. PaLM also used z-loss (coefficient 10⁻⁴) to stabilise the softmax normaliser.`,
      systems: ['palm'],
      sources: ['https://arxiv.org/abs/2204.02311'],
    },
    {
      type: 'prose',
      md: `## An honest null result

On lab 07's 828k-parameter pre-norm model, 300 steps per setting, the reference laptop:

| peak lr | clip at 1.0 | no clipping |
|---|---|---|
| 1 × 10⁻² | val 2.764 | val 2.798 |
| 3 × 10⁻² | val 2.929 | val 2.963 |
| 1 × 10⁻¹ | val 3.044 | val 3.028 |

Clipping changed the result by about 0.03 either way, and nothing diverged even at 10⁻¹. Small, shallow, pre-norm models are forgiving; their gradient norms rarely do anything dramatic. That is exactly why stability problems surprise teams: they appear only at the scale where they are expensive. Every defence in the table above is cheap to adopt early and painful to retrofit.`,
    },
    {
      type: 'callout',
      variant: 'segfault',
      title: 'Clipping that is always on',
      md: `Clipping at 1.0 is insurance for rare bad steps. If the logged pre-clip gradient norm is **above** the threshold on most steps, clipping has become a second learning-rate schedule that you did not choose, and it is hiding the real problem — usually a learning rate that is too high, or a layer whose activations are growing. Log the pre-clip norm (lab 07's \`train_step\` returns it) and treat "clipping every step" as an alert.`,
    },
    {
      type: 'diagram',
      caption: 'fig 1 — anatomy of a spike',
      height: 60,
      nodes: [
        { id: 'logit', x: 2, y: 6, w: 22, h: 12, label: 'logits grow', sub: 'attention or output' },
        { id: 'sat', x: 30, y: 6, w: 22, h: 12, label: 'softmax saturates', sub: 'one-hot, tiny grads' },
        { id: 'jump', x: 58, y: 6, w: 20, h: 12, label: 'a batch hits it', sub: 'huge gradient' },
        { id: 'spike', x: 82, y: 6, w: 16, h: 12, label: 'loss spike', sub: 'or nan' },
        { id: 'fix', x: 30, y: 40, w: 48, h: 12, label: 'QK-norm · z-loss · clipping · β2 = 0.95', sub: 'each cuts one link' },
      ],
      edges: [
        { from: 'logit', to: 'sat' },
        { from: 'sat', to: 'jump' },
        { from: 'jump', to: 'spike' },
        { from: 'fix', to: 'sat' },
      ],
      steps: [
        { caption: 'Slowly, over thousands of steps, some logits grow — attention scores in a few heads, or the output layer\'s scores.', active: ['logit'] },
        { caption: 'Large logits saturate the softmax: it becomes nearly one-hot, and gradients through it become tiny — until they suddenly are not.', active: ['logit', 'sat'], edges: ['logit->sat'] },
        { caption: 'A batch arrives where the confident prediction is badly wrong; the gradient is enormous, and Adam\'s slow second moment has not seen anything like it.', active: ['sat', 'jump', 'spike'], edges: ['sat->jump', 'jump->spike'] },
        { caption: 'The defences each break one link: QK-norm and z-loss stop logits growing, β2 = 0.95 lets v react faster, and clipping bounds the step that follows.', active: ['fix', 'sat'], edges: ['fix->sat'] },
      ],
    },
    {
      type: 'quiz',
      questions: [
        {
          q: 'A run\'s logged pre-clip gradient norm is between 3 and 8 on almost every step, with clipping at 1.0. The loss is falling slowly. What does that suggest?',
          options: [
            'Everything is healthy — clipping is doing its job',
            'Clipping is active on nearly every step, so it is silently acting as the real learning-rate control; investigate a learning rate that is too high or a layer whose activations are growing',
            'The clipping threshold should be raised to 10 so it never triggers',
            'Gradient norms above 1 mean the model has diverged',
          ],
          correct: [1],
          explanation: 'Clipping is meant to catch rare outliers. When it fires constantly, every step is rescaled by an amount you did not choose, and the underlying cause stays hidden. Raising the threshold blindly removes the insurance without diagnosing anything.',
        },
        {
          q: 'What does z-loss penalise, and why does that help stability?',
          options: [
            'The L2 norm of the weights, like weight decay',
            'The squared log of the softmax normaliser, which keeps output logits from drifting to large magnitudes where the softmax saturates and gradients become erratic',
            'The difference between train and validation loss',
            'The gradient norm',
          ],
          correct: [1],
          explanation: 'z-loss adds 10⁻⁴·(log Σ exp z)² to the loss. It does not change which token wins; it discourages the overall scale of the logits from growing, which removes one of the main routes to a saturated softmax and the spikes that follow.',
        },
        {
          q: 'Lab 07\'s small model trained fine without clipping at a learning rate of 0.1. What should you conclude for a 70B run?',
          options: [
            'Clipping is unnecessary at any scale',
            'Very little: instabilities grow with depth, width and training length, so small-model robustness says almost nothing about large runs — adopt the standard defences anyway',
            'Use a learning rate of 0.1 for the 70B run',
            'Remove warmup as well',
          ],
          correct: [1],
          explanation: 'This lesson\'s null result is a warning about small experiments, not evidence for large ones. PaLM 540B spiked about 20 times despite clipping. The defences are cheap; discovering you needed them three weeks in is not.',
        },
      ],
    },
    {
      type: 'deepdive',
      title: 'Going deeper',
      md: `- Chowdhery et al., "PaLM" (2022), section 5.1 — the loss-spike account and the restart-and-skip procedure.
- Wortsman et al., "Small-scale proxies for large-scale Transformer training instabilities" (2023, arxiv.org/abs/2309.14322) — reproducing large-model instabilities in small models by raising the learning rate, and what QK-norm and z-loss fix.
- Dehghani et al., "Scaling Vision Transformers to 22 Billion Parameters" (2023) — QK-norm at scale.
- This closes P1. D1 takes the same loop onto many devices.`,
    },
  ],
}

export default lesson

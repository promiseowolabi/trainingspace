import type { Lesson } from '../types'

const lesson: Lesson = {
  id: 'o1.l4',
  slug: 'learning-rate-schedules',
  trackId: 'o1',
  index: 4,
  title: 'Learning-Rate Schedules',
  minutes: 17,
  hook: 'The learning rate is not a constant. It ramps up for the first few thousand steps, holds, and decays towards zero — and skipping the ramp can leave a transformer stuck at the loss of a uniform guess for its whole run.',
  exercise: 'read+quiz',
  takeaway: {
    number: '3 of 3 seeds stuck at ln 32',
    claim: 'A 6-layer post-norm transformer at learning rate 3·10⁻³ stayed at the uniform-guess loss ln 32 ≈ 3.47 on all three seeds without warmup, and learned the task on all three with 150 warmup steps.',
  },
  blocks: [
    {
      type: 'prose',
      md: `Lab 03's \`lr_at\` encodes the schedule nearly every LLM is trained with:

1. **Warmup** — rise linearly from about 0 to the peak over the first steps (hundreds to a few thousand).
2. **Decay** — fall from the peak to a floor (often 10% of the peak) along a cosine curve, ending at the last step.

Each half solves a different problem.

## Why decay

A stochastic gradient is the true gradient plus noise: each batch is a sample. Near the bottom of the loss, the true gradient is small and the noise is not, so a constant step keeps jostling the parameters around the minimum. The size of that jostle scales with the learning rate. Lowering the learning rate late in training shrinks it — which is why loss curves typically drop noticeably during the final decay, after looking flat for a long time. Cosine is one smooth way to do this; linear decay works about as well. What matters most is that the rate ends low.`,
    },
    {
      type: 'prose',
      md: `## Why warmup: an experiment

A small transformer — 6 layers, width 64, 4 heads, the original **post-norm** layout (O1.L5) — learns a toy task: predict \`(x_t + x_{t−1}) mod 32\` from a random token sequence. AdamW, batch 64, 600 steps, three seeds per setting, PyTorch on the reference laptop. Final losses (ln 32 = 3.47 is "learned nothing"):

| peak lr | no warmup | 150 warmup steps |
|---|---|---|
| 1 × 10⁻³ | 0.55 · 0.76 · 2.10 | 0.77 · 0.03 · 0.02 |
| 3 × 10⁻³ | **3.47 · 3.47 · 3.47** | **0.42 · 0.78 · 2.10** |
| 1 × 10⁻² | 3.47 · 3.47 · 3.47 | 3.47 · 3.47 · 3.47 |

At 3 × 10⁻³ every run without warmup stays at the loss of a uniform guess for 600 steps; with warmup every run learns. At 10⁻² warmup is not enough on its own.

Why the first steps are dangerous:

- **Adam's statistics are not ready.** v is an average over about 1/(1 − β2) steps; after 3 steps it has seen 3 gradients. Bias correction makes it unbiased, not reliable, so the per-parameter step sizes are noisy exactly when the weights are random.
- **Early gradients are large and correlated.** A freshly initialised network has big, consistent gradients in a few directions. A full-size step along them can push LayerNorm inputs, attention logits or the output layer into a regime — saturated softmax, collapsed attention — that it does not climb out of. The stuck runs above are that failure: the loss does not explode, it flatlines.

Warmup buys the statistics time to settle and lets the model take its first steps small.`,
    },
    {
      type: 'diagram',
      caption: 'fig 1 — warmup, cosine decay, and WSD',
      height: 60,
      nodes: [
        { id: 'w', x: 2, y: 30, w: 18, h: 12, label: 'warmup', sub: '0 → peak, linear' },
        { id: 'peak', x: 26, y: 6, w: 18, h: 12, label: 'peak lr', sub: 'most learning' },
        { id: 'cos', x: 50, y: 18, w: 20, h: 12, label: 'cosine decay', sub: 'to ~10% of peak' },
        { id: 'end', x: 78, y: 40, w: 20, h: 12, label: 'last step', sub: 'loss drops here' },
        { id: 'wsd', x: 50, y: 42, w: 20, h: 12, label: 'WSD', sub: 'hold, then short decay' },
      ],
      edges: [
        { from: 'w', to: 'peak' },
        { from: 'peak', to: 'cos' },
        { from: 'cos', to: 'end' },
        { from: 'peak', to: 'wsd' },
        { from: 'wsd', to: 'end' },
      ],
      steps: [
        { caption: 'Warmup ramps the learning rate linearly from near zero to the peak over the first hundreds or thousands of steps, while Adam\'s moment estimates fill in.', active: ['w', 'peak'], edges: ['w->peak'] },
        { caption: 'Most of the learning happens near the peak. The peak value is the single most important hyper-parameter of the run; it is usually found by a sweep on a small model.', active: ['peak'] },
        { caption: 'Cosine decay lowers the rate smoothly to a floor at the final step. The total length must be fixed in advance — the curve depends on it.', active: ['peak', 'cos', 'end'], edges: ['peak->cos', 'cos->end'] },
        { caption: 'Warmup-stable-decay holds the peak and decays only in a short final phase, so a run can be extended without restarting, and checkpoints from the stable phase can each be decayed into a finished model.', active: ['peak', 'wsd', 'end'], edges: ['peak->wsd', 'wsd->end'] },
      ],
    },
    {
      type: 'prose',
      md: `## Warmup-stable-decay

Cosine has an operational flaw: its shape depends on the total number of steps, so you must decide the length of the run before you start, and a cosine run cannot be extended without re-warming. **WSD** (warmup → stable → decay) holds the peak for most of the run and decays over a short final stretch, often around 10% of the steps. Reported results match cosine at the end, and any stable-phase checkpoint can be decayed into a finished model — useful for scaling-law sweeps (P1.L5), where you want models trained for many different token counts from one run.

## Choosing the peak

The peak learning rate is found empirically: sweep it on a small model, a factor of about 3 between trials, and take the largest value that trains stably. It generally **falls as models grow** — published large-model peaks are typically around 10⁻⁴ to a few × 10⁻⁴, small models train at 10⁻³ and above. The warmup length is much less sensitive; a few hundred to a few thousand steps is common.`,
    },
    {
      type: 'callout',
      variant: 'warning',
      title: 'A schedule is part of the experiment',
      md: `Two runs with different total lengths and a cosine schedule have different learning rates at every step after warmup. Comparing their losses at step 10,000 compares two different schedules, not two models. Compare final losses, or use WSD and compare decayed checkpoints.`,
    },
    {
      type: 'quiz',
      questions: [
        {
          q: 'A team doubles the planned length of a cosine-scheduled run halfway through by editing total_steps and resuming. What happens to the learning rate at the resume point?',
          options: [
            'Nothing — the schedule continues smoothly',
            'It jumps upward: halfway through the old schedule becomes a quarter of the way through the new one, which sits higher on the cosine — an abrupt increase that can destabilise the run',
            'It drops to the minimum',
            'Warmup restarts from zero',
          ],
          correct: [1],
          explanation: 'Cosine is a function of step / total. Changing the total moves the current step to an earlier point on the curve, where the rate is higher. WSD avoids this: in the stable phase the rate does not depend on the total length.',
        },
        {
          q: 'In this lesson\'s experiment, why did warmup rescue the post-norm runs at 3·10⁻³ but not at 10⁻²?',
          options: [
            'Warmup only works below 5·10⁻³ by definition',
            'Warmup protects the first steps; at 10⁻² the steps are still too large once warmup ends, so the run fails after the ramp — warmup lowers the risk of a high peak, it does not remove it',
            'The 10⁻² runs needed more seeds',
            'AdamW disables warmup above 10⁻²',
          ],
          correct: [1],
          explanation: 'Warmup changes only the first 150 steps. If the peak itself is too large for the architecture, the instability arrives at the end of the ramp instead. Architecture changes (pre-norm, O1.L5) widen the stable range far more — the same experiment with pre-norm learned at every rate tested.',
        },
        {
          q: 'Why does the loss of a long run often drop sharply during the final decay phase?',
          options: [
            'The model suddenly learns new features',
            'Gradient noise keeps the parameters bouncing around the minimum at a size set by the learning rate; lowering the rate shrinks the bounce, so the parameters settle closer to the bottom',
            'Evaluation changes at the end of training',
            'The data gets easier',
          ],
          correct: [1],
          explanation: 'At a constant learning rate the stochastic gradient noise sets a floor on how close the iterate gets to the minimum. Decaying the rate lowers that floor, so the loss falls even though the "direction" learned has not changed much.',
        },
      ],
    },
    {
      type: 'deepdive',
      title: 'Going deeper',
      md: `- Goyal et al., "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour" (2017) — where linear warmup became standard.
- Xiong et al., "On Layer Normalization in the Transformer Architecture" (2020, arxiv.org/abs/2002.04745) — why post-norm transformers need warmup and pre-norm ones much less.
- Hu et al., "MiniCPM" (2024, arxiv.org/abs/2404.06395) — the warmup-stable-decay schedule.
- Rerun the experiment: \`nn.TransformerEncoderLayer(..., norm_first=False)\` versus \`norm_first=True\`, three seeds each.`,
    },
  ],
}

export default lesson

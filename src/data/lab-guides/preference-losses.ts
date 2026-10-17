import type { LabGuide } from './types'

const guide: LabGuide = {
  id: 'preference-losses',
  requires: ['a browser — or Python 3.10+ with numpy and pytest'],
  blocks: [
    {
      type: 'prose',
      md: `## What you will build

Every post-training method in F1 is a loss function over quantities a training loop already has. You implement each one on plain numpy arrays, so the formulas are the whole lab. Use \`log_sigmoid\` (given) everywhere a log of a sigmoid appears: several checks feed margins of ±1000.`,
    },
    {
      type: 'prose',
      md: `## 1. sequence_logprob

The quantity everything else consumes: log π(response | prompt) as a sum over response tokens. Stable log-softmax over the vocabulary axis, \`np.take_along_axis(logp, tokens[..., None], axis=-1)[..., 0]\` to pick each target, multiply by the mask, sum over time.

## 2. Bradley–Terry and DPO

Both are \`-log_sigmoid(margin).mean()\`; only the margin differs. For a reward model it is r(chosen) − r(rejected). For DPO it is β times the difference of two log-ratios against the reference model. A policy identical to its reference has margin 0 and loss ln 2 — every DPO run starts there.

\`dpo_grad_weight\` is σ(−margin) — written \`1 / (1 + exp(margin))\` — and the \`dpo-gradient\` check verifies by finite differences that the loss's derivative with respect to log π(chosen) is −β times that weight over the batch size. Pairs the model already ranks correctly get a small weight; mis-ranked pairs a large one.`,
    },
    {
      type: 'prose',
      md: `## 3. PPO, GRPO and KL

\`ppo_clip_objective\`: ratio = exp(new − old); take the element-wise minimum of the unclipped and clipped terms, then the mean. The two cases in the check show why it is a minimum: for good actions the gain is capped at 1 + ε, and for bad actions the pessimistic (more negative) term wins.

\`grpo_advantages\`: normalise each row — one prompt's group of sampled answers — to mean 0 and standard deviation 1. A row where every answer got the same reward must come out as zeros, which the \`eps\` in the denominator guarantees.

\`kl_k3\`: one line. The check samples 400,000 tokens from a policy and confirms the estimator's average equals the exact KL to within 0.01.`,
    },
  ],
  cleanup: [
    {
      type: 'prose',
      md: `- Plot \`dpo_grad_weight\` against the margin from −5 to 5 for β = 0.1 and β = 1: β sets how quickly DPO stops pushing on pairs it has learned.
- Compare k3 with the naive estimator \`logp_policy − logp_ref\` on the same samples: both average to the KL, but one can be negative per sample and has higher variance.
- Use \`sequence_logprob\` and \`dpo_loss\` with lab 12's model to run DPO on a handful of preference pairs.`,
    },
  ],
}

export default guide

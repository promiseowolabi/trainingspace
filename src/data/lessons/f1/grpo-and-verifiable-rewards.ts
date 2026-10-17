import type { Lesson } from '../types'

const lesson: Lesson = {
  id: 'f1.l6',
  slug: 'grpo-and-verifiable-rewards',
  trackId: 'f1',
  index: 6,
  title: 'GRPO and Verifiable Rewards',
  minutes: 19,
  hook: 'When the answer can be checked — a maths result, a passing test suite — you do not need a learned reward model at all. Sample several answers to each problem, reward the right ones, and push the policy towards whatever beat its own average. No value model, no preference data.',
  exercise: 'read+quiz',
  takeaway: {
    number: '0.06 → 0.94 in 40 steps',
    claim: 'In a toy GRPO run with groups of 8 samples, the probability of the one correct answer out of 16 rose from 1/16 to 0.94 in 40 steps; with groups of 2, most groups had no reward variance and it barely moved for 20 steps.',
  },
  blocks: [
    {
      type: 'prose',
      md: `## Rewards you can compute

A reward model (F1.L4) is a learned guess at human preference, and RL against it invites reward hacking. For a large and valuable class of tasks there is no need to guess: **the output can be checked**. Did the final answer to the maths problem match? Do the unit tests pass? Does the proof verify? Is the JSON valid against the schema? A **verifiable reward** is a program — typically 1 for correct, 0 otherwise, sometimes with a small bonus for following a required format. Training against it is **RLVR** (reinforcement learning with verifiable rewards).

The reward is sparse — one number per complete answer — and hard to hack in the usual way (the checker does not get tired or impressed by length), though models still find holes in weak checkers, such as tests that do not test enough.

## GRPO: the group is the baseline

PPO (F1.L5) needs a **value model** to say how good a response was *relative to expectation* — the advantage. **Group Relative Policy Optimization** (GRPO, from DeepSeekMath) replaces it with a simpler estimate: for each prompt, sample a **group** of G answers from the current policy, score them all, and use the group itself as the baseline:

\`\`\`
A_i = ( r_i − mean(r_1 … r_G) ) / ( std(r_1 … r_G) + ε )
\`\`\`

Answers better than their siblings get positive advantage, worse ones negative, and every token of an answer shares its answer's advantage. Then a PPO-style clipped update with those advantages, plus a KL penalty to the reference estimated per token with the **k3** estimator. Lab 13's \`grpo_advantages\` and \`kl_k3\` are exactly these two pieces. No value model: one fewer 16-byte-per-parameter model in memory than PPO.`,
    },
    {
      type: 'prose',
      md: `## A toy run

A "policy" that is just a softmax over 16 candidate answers to one question, one of which is correct (reward 1, all others 0), starting uniform: p(correct) = 1/16 = 0.0625. Each step samples a group, computes GRPO advantages and takes a policy-gradient step on the logits. Measured with numpy on the reference laptop, 60 steps:

| group size | p(correct) at step 0 | 10 | 20 | 40 | 59 | groups with no signal |
|---|---|---|---|---|---|---|
| 2 | 0.062 | 0.062 | 0.062 | 0.42 | 0.91 | 41 of 60 |
| 8 | 0.062 | 0.11 | 0.31 | **0.94** | 0.97 | 30 of 60 |

The last column is the important one. When every answer in a group gets the same reward — all wrong, early on — every advantage is zero and the step teaches nothing. With groups of 2 and a 6% success rate, 41 of 60 steps were wasted that way, and the policy did not move at all for the first 20. Larger groups find a correct answer sooner and give a usable signal more often. Real RLVR runs sample groups of 8 to 64 per prompt and filter out prompts that are always or never solved.`,
    },
    {
      type: 'diagram',
      caption: 'fig 1 — one GRPO step for one prompt',
      height: 62,
      nodes: [
        { id: 'p', x: 2, y: 24, w: 14, h: 12, label: 'prompt', sub: 'a maths problem' },
        { id: 's', x: 22, y: 24, w: 18, h: 12, label: 'sample G answers', sub: 'current policy' },
        { id: 'v', x: 46, y: 6, w: 16, h: 12, label: 'verifier', sub: 'right = 1, wrong = 0' },
        { id: 'a', x: 46, y: 42, w: 16, h: 12, label: 'group advantages', sub: '(r − mean) / std' },
        { id: 'u', x: 70, y: 24, w: 28, h: 12, label: 'clipped update − β·KL', sub: 'no value model' },
      ],
      edges: [
        { from: 'p', to: 's' },
        { from: 's', to: 'v' },
        { from: 'v', to: 'a' },
        { from: 'a', to: 'u' },
      ],
      steps: [
        { caption: 'For each prompt, the current policy samples a group of G complete answers, with full reasoning, at a non-zero temperature so they differ.', active: ['p', 's'], edges: ['p->s'] },
        { caption: 'A program checks each answer: the final number matches, or the tests pass. No reward model and no human in the loop.', active: ['s', 'v'], edges: ['s->v'] },
        { caption: 'Rewards are normalised within the group: right answers in a mostly-wrong group get a large positive advantage; a group with identical rewards contributes nothing.', active: ['v', 'a'], edges: ['v->a'] },
        { caption: 'A PPO-style clipped update raises the probability of every token in above-average answers and lowers it for below-average ones, with a KL penalty to the reference.', active: ['a', 'u'], edges: ['a->u'] },
      ],
    },
    {
      type: 'vendor',
      snapshot: '2026-10',
      title: 'RL on verifiable rewards at scale',
      md: `GRPO was introduced in DeepSeekMath (Shao et al., 2024) as a lighter alternative to PPO for mathematical reasoning. DeepSeek-R1 (2025) reported that **DeepSeek-R1-Zero**, trained with GRPO from a base model using only rule-based rewards for answer accuracy and output format — no SFT first — improved its pass@1 on the AIME 2024 maths benchmark from **15.6% to 71.0%** over the course of RL training, while spontaneously producing longer reasoning. The released DeepSeek-R1 added a small amount of SFT data before RL to fix readability problems R1-Zero showed.`,
      systems: ['deepseek-math', 'deepseek-r1'],
      sources: ['https://arxiv.org/abs/2402.03300', 'https://arxiv.org/abs/2501.12948'],
    },
    {
      type: 'callout',
      variant: 'warning',
      title: 'The checker is the specification',
      md: `RLVR optimises exactly what the verifier rewards. A unit-test reward with weak tests teaches code that passes weak tests; a maths reward that only checks the final number teaches the model to reach the number, however it gets there. Verifiers deserve the same review as production code — they are the objective function.`,
    },
    {
      type: 'quiz',
      questions: [
        {
          q: 'In a GRPO batch, one prompt\'s 8 sampled answers are all wrong. What does that prompt contribute to the update?',
          options: [
            'A strong negative signal on all 8 answers',
            'Nothing: all rewards are equal, so every group-relative advantage is 0 — the samples cost compute and teach nothing',
            'A random update',
            'A positive signal, because the model tried',
          ],
          correct: [1],
          explanation: 'Advantages are rewards minus the group mean; with no variance they are all zero. That is why RLVR pipelines filter out prompts the model always or never solves, and why larger groups (more chances for one success) learn faster on hard prompts — as the toy run showed.',
        },
        {
          q: 'What does GRPO remove from the PPO recipe, and what replaces it?',
          options: [
            'The KL penalty, replaced by clipping',
            'The value model (critic): the mean and standard deviation of rewards within each prompt\'s group of samples act as the baseline for advantages',
            'The reference model, replaced by the reward model',
            'Sampling, replaced by a fixed dataset',
          ],
          correct: [1],
          explanation: 'PPO estimates advantages with a learned value model — another full-size model to train and hold in memory. GRPO uses the group of answers to the same prompt as a Monte-Carlo baseline instead, which fits verifiable rewards well since each answer gets one score.',
        },
        {
          q: 'Why do verifiable rewards suit RL better than a learned reward model, for maths and code?',
          options: [
            'They are always denser',
            'A program\'s judgement does not drift or reward superficial features the way a learned approximation of preference does, so strong optimisation against it is far less prone to reward hacking — as long as the checker itself is correct and complete',
            'They make the KL penalty unnecessary in all cases',
            'They allow training without sampling',
          ],
          correct: [1],
          explanation: 'Reward hacking exploits the gap between a proxy and the real goal. When the reward is the real goal — the answer is right, the tests pass — the gap shrinks to the checker\'s own blind spots. The KL penalty is still commonly used, to keep outputs readable and the policy close to a sensible reference.',
        },
      ],
    },
    {
      type: 'deepdive',
      title: 'Going deeper',
      md: `- Shao et al., "DeepSeekMath" (2024, arxiv.org/abs/2402.03300), section 4 — GRPO's objective, including the k3 KL term.
- DeepSeek-AI, "DeepSeek-R1" (2025, arxiv.org/abs/2501.12948) — RL with rule-based rewards at scale.
- Schulman, "Approximating KL Divergence" (joschu.net/blog/kl-approx.html) — k1, k2 and k3, and why k3 is unbiased and non-negative.
- Rerun the toy: a 16-way softmax, a group size, \`grpo_advantages\`, and the policy-gradient step \`z += lr · Σ A_i · (onehot(a_i) − p) / G\`.
- This closes F1. E1 is about knowing whether any of it worked.`,
    },
  ],
}

export default lesson

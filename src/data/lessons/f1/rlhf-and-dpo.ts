import type { Lesson } from '../types'

const lesson: Lesson = {
  id: 'f1.l5',
  slug: 'rlhf-and-dpo',
  trackId: 'f1',
  index: 5,
  title: 'RLHF and DPO',
  minutes: 22,
  hook: 'RLHF samples responses, scores them with a reward model and pushes the policy towards high scores without drifting far from where it started — four models in memory and a notoriously finicky loop. DPO shows the same objective has a closed form you can train with a classification loss and two models.',
  exercise: 'lab+quiz',
  takeaway: {
    number: '4 models vs 2',
    claim: 'PPO-based RLHF keeps a policy, a value model, a reward model and a frozen reference in memory — about 250 GB of state for a 7B policy — while DPO optimises the same KL-regularised objective from preference pairs with just the policy and the reference.',
  },
  blocks: [
    {
      type: 'prose',
      md: `## The objective

Both methods optimise the same thing. Find a policy π (the model being trained) that earns high reward while staying close to a frozen **reference** π_ref (the SFT model):

\`\`\`
maximise   E_{y ~ π(·|x)} [ r(x, y) ]  −  β · KL( π(·|x) ‖ π_ref(·|x) )
\`\`\`

The KL term is the defence against reward hacking (F1.L4): the policy may move only as far from the reference as the reward gain justifies, at an exchange rate set by β. Without it, the policy drifts into regions where the reward model is wrong.

## RLHF with PPO

The InstructGPT recipe treats generation as reinforcement learning:

1. **Sample** responses from the current policy for a batch of prompts.
2. **Score** each with the reward model, and subtract β times the per-token log-ratio log π/π_ref as a KL penalty.
3. **Estimate advantages** — how much better each token's choice was than expected — using a learned **value model** (a critic) that predicts future reward from each position.
4. **Update** the policy with PPO's clipped objective for a few epochs over these samples, then go back to step 1.

PPO's clipped surrogate (lab 13's \`ppo_clip_objective\`) limits how far one update can move the policy: with ratio ρ = π_new/π_old, it maximises min(ρ·A, clip(ρ, 1 − ε, 1 + ε)·A). Good actions gain at most a factor 1 + ε; bad actions are penalised with the pessimistic bound.`,
    },
    {
      type: 'prose',
      md: `## The cost of PPO

Four models live in memory at once, for a 7B policy:

| model | trains? | memory (T0.L4) |
|---|---|---|
| policy π | yes | ~112 GB (16 B/param) |
| value model | yes | ~112 GB |
| reward model | no | ~14 GB (bf16) |
| reference π_ref | no | ~14 GB (bf16) |
| **total** | | **~250 GB**, before activations and generation |

Plus a generation loop inside the training loop: sampling is slow, memory-bound decoding (T0.L3), and it dominates wall-clock time. And the hyper-parameters — β, ε, the value-loss weight, the number of PPO epochs, the generation temperature — interact. PPO-based RLHF works, and it is hard.`,
    },
    {
      type: 'prose',
      md: `## DPO: the closed form

Rafailov et al. (2023) noticed that the KL-regularised objective above has a known optimal policy:

\`\`\`
π*(y|x) = π_ref(y|x) · exp( r(x, y) / β ) / Z(x)
\`\`\`

Solve that for the reward:

\`\`\`
r(x, y) = β · log( π*(y|x) / π_ref(y|x) )  +  β · log Z(x)
\`\`\`

Now substitute it into the Bradley–Terry likelihood of the preference data (F1.L4). Bradley–Terry only looks at the **difference** of two rewards for the same prompt, so the intractable β·log Z(x) cancels, and what remains is a loss on the policy directly:

\`\`\`
L_DPO = −log σ( β · [ log π(y_w|x)/π_ref(y_w|x)  −  log π(y_l|x)/π_ref(y_l|x) ] )
\`\`\`

y_w is the chosen response, y_l the rejected one. Every term is a summed sequence log-probability (lab 13's \`sequence_logprob\`), computed by one forward pass of the policy and one of the frozen reference. No reward model, no value model, no sampling: **a classification loss on preference pairs.** The quantity β·log π/π_ref acts as an implicit reward (\`dpo_rewards\`), and the gradient on each pair is weighted by σ(−margin) — large for pairs the model currently ranks wrong, near zero for pairs it has learned (\`dpo_grad_weight\`).`,
    },
    {
      type: 'diagram',
      caption: 'fig 1 — the two pipelines',
      height: 64,
      nodes: [
        { id: 'prefs', x: 2, y: 26, w: 18, h: 12, label: 'preference pairs', sub: 'chosen / rejected' },
        { id: 'rm', x: 28, y: 6, w: 18, h: 12, label: 'reward model', sub: 'F1.L4' },
        { id: 'ppo', x: 54, y: 6, w: 20, h: 12, label: 'PPO loop', sub: 'sample · score · update' },
        { id: 'dpo', x: 40, y: 44, w: 26, h: 12, label: 'DPO loss', sub: 'policy + reference only' },
        { id: 'pol', x: 82, y: 26, w: 16, h: 12, label: 'tuned policy', sub: 'max r − β·KL' },
      ],
      edges: [
        { from: 'prefs', to: 'rm' },
        { from: 'rm', to: 'ppo' },
        { from: 'ppo', to: 'pol' },
        { from: 'prefs', to: 'dpo' },
        { from: 'dpo', to: 'pol' },
      ],
      steps: [
        { caption: 'Both methods start from the same data: prompts with a chosen and a rejected response, and the same goal — high reward, small KL from the reference.', active: ['prefs'] },
        { caption: 'RLHF first distils the preferences into a reward model, then runs an online PPO loop: generate, score, estimate advantages with a value model, update.', active: ['prefs', 'rm', 'ppo', 'pol'], edges: ['prefs->rm', 'rm->ppo', 'ppo->pol'] },
        { caption: 'DPO skips the reward model and the sampling: the optimal-policy identity turns the same objective into a loss computed directly on the preference pairs.', active: ['prefs', 'dpo', 'pol'], edges: ['prefs->dpo', 'dpo->pol'] },
        { caption: 'Same target, different trade-offs: PPO learns from fresh samples of the current policy; DPO is simpler and cheaper but learns only from the fixed pairs it was given.', active: ['ppo', 'dpo', 'pol'] },
      ],
    },
    {
      type: 'prose',
      md: `## When each wins

**DPO** is simple, stable and cheap — half the models, no generation in the loop, a loss as easy to monitor as SFT's — and has become a default for preference tuning. Its limitation is that it is **offline**: it learns only from the pairs in its dataset, which were usually sampled from some other model. Variants generate fresh pairs from the current policy and label them (iterative or online DPO) to recover some of RL's advantage.

**PPO** (and RL generally) is **online**: it learns from the current policy's own samples, so it can discover and reinforce good behaviour that no dataset contained. That matters most when the reward can be computed for any output — the setting of F1.L6, where RL is now the main tool.`,
    },
    {
      type: 'lab',
      lab: 'preference-losses',
      brief: 'Implement summed sequence log-probabilities, the Bradley–Terry loss, DPO with its implicit rewards and gradient weights (checked by finite differences), PPO\'s clipped objective, GRPO advantages and the k3 KL estimator.',
    },
    {
      type: 'quiz',
      questions: [
        {
          q: 'At the start of DPO training the policy equals the reference. What is the loss, and why?',
          options: [
            '0, because nothing needs to change',
            'ln 2 ≈ 0.693: both log-ratios are 0, the margin is 0, and −log σ(0) = ln 2 — the loss of a coin flip',
            'It is undefined',
            'It equals the SFT loss',
          ],
          correct: [1],
          explanation: 'log π/π_ref = 0 for every response when π = π_ref, so every pair has margin 0. Lab 13\'s dpo-loss check requires exactly ln 2 here; a DPO run that starts elsewhere has a bug in its log-probabilities.',
        },
        {
          q: 'Why does the partition function Z(x) disappear from the DPO loss?',
          options: [
            'It is approximated as 1',
            'It depends only on the prompt, and the Bradley–Terry likelihood depends only on the difference of two rewards for the same prompt, so β·log Z(x) cancels',
            'It is computed by the reference model',
            'DPO ignores the KL term',
          ],
          correct: [1],
          explanation: 'Z(x) is a sum over all possible responses — intractable — but it is the same for the chosen and the rejected response to a given prompt. Preference models only see reward differences, so it subtracts out. That cancellation is the whole trick.',
        },
        {
          q: 'A team has a fixed dataset of 50,000 human preference pairs and a modest GPU budget. Which method is the natural first choice?',
          options: [
            'PPO, because RL is always better',
            'DPO: it trains directly on the pairs with two models and no generation loop — simpler, cheaper and stable; consider online variants or RL later if the offline data limits quality',
            'Neither — preferences cannot be used without a reward model',
            'More SFT on the rejected responses',
          ],
          correct: [1],
          explanation: 'With a fixed offline dataset, DPO optimises the same KL-regularised objective at a fraction of PPO\'s memory and engineering cost. PPO\'s advantage — learning from fresh samples — matters most when a reward can be computed for new outputs.',
        },
      ],
    },
    {
      type: 'deepdive',
      title: 'Going deeper',
      md: `- Schulman et al., "Proximal Policy Optimization Algorithms" (2017, arxiv.org/abs/1707.06347).
- Ouyang et al., "InstructGPT" (2022), section 3.5 — PPO with a per-token KL penalty for language models.
- Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (2023, arxiv.org/abs/2305.18290) — the derivation above, in full.`,
    },
  ],
}

export default lesson

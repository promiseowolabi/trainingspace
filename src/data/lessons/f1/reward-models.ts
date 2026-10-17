import type { Lesson } from '../types'

const lesson: Lesson = {
  id: 'f1.l4',
  slug: 'reward-models',
  trackId: 'f1',
  index: 4,
  title: 'Reward Models',
  minutes: 17,
  hook: 'People cannot write down a score for "a good answer", but they can say which of two answers is better. A reward model turns thousands of those comparisons into a function that scores any response — and a model optimised hard enough against it will find every flaw that function has.',
  exercise: 'read+quiz',
  takeaway: {
    number: 'σ(1) = 73%',
    claim: 'Under the Bradley–Terry model a reward gap of 1 means a 73% chance the higher-scored response is preferred and a gap of 2 means 88% — rewards are only meaningful as differences, and only near the responses they were trained on.',
  },
  blocks: [
    {
      type: 'prose',
      md: `SFT (F1.L2) needs demonstrations: someone must write the ideal response. That is expensive, and it caps quality at what the writer can produce. Judging is easier than writing — most people can tell which of two answers is better without being able to write the better one. **Preference data** exploits that: show a labeller a prompt and two responses, record which one they preferred.

## The Bradley–Terry model

To learn from comparisons, assume each response y to prompt x has a hidden scalar quality r(x, y), and that the probability a person prefers y₁ over y₂ is

\`\`\`
P(y₁ ≻ y₂) = σ(r(x, y₁) − r(x, y₂))          σ = the logistic sigmoid
\`\`\`

Only **differences** matter: adding a constant to every reward changes nothing. A gap of 0 is a coin flip; a gap of 1 is σ(1) = **73%**; a gap of 2 is **88%**; a gap of 4 is 98%.

Training a reward model r_θ is then maximum likelihood on the comparisons — which is binary cross-entropy on the margin:

\`\`\`
loss = −log σ(r_θ(x, y_chosen) − r_θ(x, y_rejected))       averaged over pairs
\`\`\`

Lab 13's \`bradley_terry_loss\` is this line. With equal rewards it is ln 2 = 0.693, the loss of knowing nothing.`,
    },
    {
      type: 'statline',
      stats: [
        { value: '50%', label: 'gap 0', hint: 'σ(0): the reward model has no opinion' },
        { value: '73%', label: 'gap 1', hint: 'σ(1) = 0.731' },
        { value: '88%', label: 'gap 2', hint: 'σ(2) = 0.881' },
        { value: 'ln 2', label: 'initial loss', hint: '−log σ(0) = 0.693 nats per pair' },
      ],
    },
    {
      type: 'prose',
      md: `## Building one

A reward model is usually the SFT model itself with its vocabulary output layer replaced by a **single scalar head**: run the prompt and response through the transformer, read the final token's hidden state, project it to one number. Starting from a strong language model matters: judging a response requires understanding it.

Preference data is noisy. Labellers disagree with each other a substantial fraction of the time on close pairs, which caps the accuracy any reward model can reach on held-out comparisons — commonly around 65–75% for general chat data, higher on pairs with a clear winner. A reward model that scores 95% on its validation pairs is more likely to have learned a shortcut (such as "longer is better") than human taste.`,
    },
    {
      type: 'diagram',
      caption: 'fig 1 — from comparisons to a scorer',
      height: 60,
      nodes: [
        { id: 'pair', x: 2, y: 22, w: 20, h: 12, label: 'prompt + 2 responses', sub: 'human picks one' },
        { id: 'rm', x: 30, y: 22, w: 18, h: 12, label: 'reward model', sub: 'LM + scalar head' },
        { id: 'rc', x: 56, y: 6, w: 16, h: 12, label: 'r(chosen)', sub: 'scalar' },
        { id: 'rr', x: 56, y: 38, w: 16, h: 12, label: 'r(rejected)', sub: 'scalar' },
        { id: 'bt', x: 80, y: 22, w: 18, h: 12, label: '−log σ(Δr)', sub: 'Bradley–Terry' },
      ],
      edges: [
        { from: 'pair', to: 'rm' },
        { from: 'rm', to: 'rc' },
        { from: 'rm', to: 'rr' },
        { from: 'rc', to: 'bt' },
        { from: 'rr', to: 'bt' },
      ],
      steps: [
        { caption: 'Each training example is a prompt, two responses, and a label saying which one a person preferred.', active: ['pair'] },
        { caption: 'The same reward model scores both responses independently: a language model whose last layer outputs one number instead of a vocabulary distribution.', active: ['pair', 'rm', 'rc', 'rr'], edges: ['pair->rm', 'rm->rc', 'rm->rr'] },
        { caption: 'The loss pushes the chosen score above the rejected one: −log σ of the difference, binary cross-entropy on the margin.', active: ['rc', 'rr', 'bt'], edges: ['rc->bt', 'rr->bt'] },
        { caption: 'After training, the model scores any response to any prompt — which is what lets RL optimise against it without a human in the loop for every sample (F1.L5).', active: ['rm'] },
      ],
    },
    {
      type: 'prose',
      md: `## Reward hacking

A reward model is a learned approximation of human preference, accurate near the kind of responses it was trained on. Optimise a policy hard against it and the policy drifts into responses the reward model has never seen, where its scores are wrong — and the optimiser seeks out precisely the places where they are wrong in its favour. Classic symptoms: responses get longer and longer, more confident, more flattering, full of formatting that labellers once liked — while their real quality stops improving or falls.

Gao et al. measured this directly with a "gold" reward model standing in for humans: as a policy was optimised against a smaller proxy reward model, the proxy score rose steadily while the gold score rose, peaked and then **declined**. The defences are the ones F1.L5 builds in: a KL penalty that keeps the policy close to its starting point, early stopping on held-out human evaluation, and refreshing the reward model with comparisons of the new policy's outputs.`,
    },
    {
      type: 'quiz',
      questions: [
        {
          q: 'A reward model scores response A at 3.2 and response B at 1.2. Under Bradley–Terry, how likely is a labeller to prefer A?',
          options: ['about 62%', 'about 73%', 'about 88%', 'about 96%'],
          correct: [2],
          explanation: 'Only the difference matters: Δ = 2.0, and σ(2) ≈ 0.881. The absolute values 3.2 and 1.2 carry no meaning on their own — adding 100 to both would predict the same preference.',
        },
        {
          q: 'During RLHF, the reward model\'s scores keep rising while human ratings of samples have flattened and the responses have doubled in length. What is happening?',
          options: [
            'The policy is improving in ways humans cannot see',
            'Reward hacking: the policy has found responses the reward model over-scores (here, length), outside the distribution it was trained on — stop or strengthen the KL penalty, and retrain the reward model on the new outputs',
            'The learning rate is too low',
            'The human raters are wrong',
          ],
          correct: [1],
          explanation: 'A proxy reward diverging from the real objective is the signature of over-optimisation. Length is the classic exploited feature because longer answers were often preferred in the training comparisons. The gold-versus-proxy experiments show exactly this shape.',
        },
        {
          q: 'Your reward model reaches 97% accuracy on held-out preference pairs where labellers agree with each other about 70% of the time. What should you suspect first?',
          options: [
            'An excellent reward model',
            'A shortcut or leakage: no model should agree with noisy labels far more often than the labellers agree with each other — check for duplicate prompts across the split, or a feature such as length that predicts the label',
            'Too small a learning rate',
            'Too few epochs',
          ],
          correct: [1],
          explanation: 'Inter-annotator agreement is a ceiling on meaningful accuracy. Exceeding it by a wide margin almost always means the evaluation is contaminated or the model has found a spurious cue — which will then be exploited during RL.',
        },
      ],
    },
    {
      type: 'deepdive',
      title: 'Going deeper',
      md: `- Bradley and Terry, "Rank Analysis of Incomplete Block Designs" (1952) — the original pairwise-comparison model.
- Stiennon et al., "Learning to summarize from human feedback" (2020, arxiv.org/abs/2009.01325) — reward models for RLHF, before InstructGPT.
- Gao, Schulman and Hilton, "Scaling Laws for Reward Model Overoptimization" (2022, arxiv.org/abs/2210.10760).`,
    },
  ],
}

export default lesson

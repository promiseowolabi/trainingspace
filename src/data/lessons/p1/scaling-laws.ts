import type { Lesson } from '../types'

const lesson: Lesson = {
  id: 'p1.l5',
  slug: 'scaling-laws',
  trackId: 'p1',
  index: 5,
  title: 'Scaling Laws',
  minutes: 20,
  hook: 'Loss falls as a smooth power law in parameters and in tokens, so the best split of a compute budget between model size and data can be computed before the run. The answer — about 20 tokens per parameter — is also the answer almost nobody uses any more, for a good reason.',
  exercise: 'read+quiz',
  takeaway: {
    number: '20 tokens / parameter',
    claim: 'Chinchilla\'s compute-optimal recipe trains about 20 tokens per parameter; current models train hundreds or thousands per parameter on purpose, because a smaller model at nearly the same loss is far cheaper to serve.',
  },
  blocks: [
    {
      type: 'prose',
      md: `Train a family of models of different sizes on different amounts of data, plot final loss against size or tokens on log-log axes, and the points fall on straight lines. That empirical fact — loss is a **power law** in N (parameters) and D (tokens), over many orders of magnitude — is what makes it possible to plan the largest runs with experiments that cost a thousandth of them.

## The question scaling laws answer

You have a compute budget C. Since C ≈ 6·N·D (P1.L4), you can spend it on a big model trained briefly or a small model trained long. Which gives the lowest loss?

**Kaplan et al. (2020)** fitted power laws and concluded that model size should grow faster than data: for a 10× compute increase, make the model about 5× bigger and train on about 2× more data. GPT-3 (175B parameters, 300B tokens: under 2 tokens per parameter) followed that advice.

**Hoffmann et al. (2022), "Chinchilla"**, redid the experiments with learning-rate schedules matched to each run's length (O1.L4's warning, made consequential) and found that N and D should grow **equally**: for a 10× compute increase, about 3.2× more of each. At the optimum, a model wants about **20 tokens per parameter**. They proved it by training Chinchilla — 70B parameters on 1.4 trillion tokens — with the same compute as their 280B Gopher model, and Chinchilla beat Gopher across the board.`,
    },
    {
      type: 'prose',
      md: `## The fitted loss

Chinchilla's third fitting approach gives a formula you can evaluate:

\`\`\`
L(N, D) = E + A / N^α + B / D^β
        = 1.69 + 406.4 / N^0.34 + 410.7 / D^0.28
\`\`\`

E is the irreducible loss of the text itself; the other two terms are what a finite model and finite data leave on the table. Evaluated (the constants are specific to their data and tokenizer, so read the *differences*):

| run | N | D | tokens / param | compute | predicted loss |
|---|---|---|---|---|---|
| Gopher | 280 B | 300 B | 1.1 | 5.0 × 10²³ | 1.993 |
| Chinchilla | 70 B | 1.4 T | 20 | 5.9 × 10²³ | **1.937** |
| 8B, compute-optimal | 8 B | 160 B | 20 | 7.7 × 10²¹ | 2.164 |
| 8B, Llama-3-style | 8 B | 15 T | 1,875 | 7.2 × 10²³ | **1.949** |
| optimal for that compute | 77 B | 1.55 T | 20 | 7.2 × 10²³ | 1.929 |

The last two rows are the modern story. Spending 7.2 × 10²³ FLOPs on an 8B model trained on 15 trillion tokens predicts a loss only **0.02** worse than the compute-optimal 77B model — and the 8B model is nearly **10× cheaper to run** for every token it ever generates.`,
    },
    {
      type: 'statline',
      stats: [
        { value: '~20', label: 'tokens / param', hint: 'Chinchilla\'s compute-optimal ratio' },
        { value: '1.7', label: 'GPT-3', hint: '300 B tokens / 175 B parameters — Kaplan-era sizing' },
        { value: '1,875', label: 'Llama-3-style 8B', hint: '15 T tokens / 8 B parameters — deliberately over-trained' },
        { value: '0.02', label: 'loss gap', hint: 'predicted: 8B @ 15T vs a 77B model with the same compute' },
      ],
    },
    {
      type: 'prose',
      md: `## Why everyone over-trains now

"Compute-optimal" minimises **training** compute for a given loss. But a successful model is trained once and then served billions of times, and serving cost scales with N, not D. Once inference dominates the lifetime bill, the right trade is a smaller model trained on far more data than Chinchilla recommends: you pay more training compute per point of loss, and recover it many times over at inference. The table shows why that is cheap — past the optimum, the D term shrinks slowly but steadily.

The limit is data. At thousands of tokens per parameter, a few hundred billion parameters would need more high-quality text than exists, which is why data quality, repetition and synthetic data (P1.L1) are where much of the frontier effort has moved.

## How scaling laws are used in practice

1. Train a ladder of small models (say 10M to 1B parameters) at several token counts, with schedules matched to each length (WSD makes this cheap, O1.L4).
2. Fit the power law.
3. Extrapolate to choose N, D and the learning rate of the big run, and predict its final loss before it starts.

If the big run lands off the predicted curve, something is wrong — the data, the code, or the hardware. A scaling law is a forecast, a budget and a regression test at once.`,
    },
    {
      type: 'diagram',
      caption: 'fig 1 — spending a fixed compute budget',
      height: 60,
      nodes: [
        { id: 'c', x: 2, y: 22, w: 18, h: 12, label: 'budget C', sub: '= 6·N·D' },
        { id: 'big', x: 30, y: 4, w: 22, h: 12, label: 'big N, small D', sub: 'GPT-3 style' },
        { id: 'opt', x: 30, y: 22, w: 22, h: 12, label: 'D ≈ 20·N', sub: 'compute-optimal' },
        { id: 'small', x: 30, y: 40, w: 22, h: 12, label: 'small N, huge D', sub: 'over-trained' },
        { id: 'train', x: 62, y: 10, w: 36, h: 12, label: 'lowest loss per training FLOP', sub: 'Chinchilla' },
        { id: 'serve', x: 62, y: 36, w: 36, h: 12, label: 'cheapest to serve, ~same loss', sub: 'most current models' },
      ],
      edges: [
        { from: 'c', to: 'big' },
        { from: 'c', to: 'opt' },
        { from: 'c', to: 'small' },
        { from: 'opt', to: 'train' },
        { from: 'small', to: 'serve' },
      ],
      steps: [
        { caption: 'A compute budget is a product, 6·N·D: you choose how to split it between model size and tokens.', active: ['c'] },
        { caption: 'Kaplan-era models spent most of it on parameters and stopped early: under 2 tokens per parameter for GPT-3.', active: ['c', 'big'], edges: ['c->big'] },
        { caption: 'Chinchilla showed the training-loss-optimal split grows N and D together, at about 20 tokens per parameter.', active: ['c', 'opt', 'train'], edges: ['c->opt', 'opt->train'] },
        { caption: 'Counting inference cost, a smaller model trained far past 20 tokens per parameter reaches nearly the same loss and is many times cheaper to serve.', active: ['c', 'small', 'serve'], edges: ['c->small', 'small->serve'] },
      ],
    },
    {
      type: 'quiz',
      questions: [
        {
          q: 'You have 1 × 10²² FLOPs. By Chinchilla\'s rule of thumb, what model size and token count minimise training loss?',
          options: ['about 9 B parameters, 180 B tokens', 'about 90 B parameters, 18 B tokens', 'about 900 M parameters, 1.8 T tokens', 'about 30 B parameters, 55 B tokens'],
          correct: [0],
          explanation: 'C = 6·N·D with D = 20·N gives C = 120·N², so N = √(10²² / 120) ≈ 9.1 × 10⁹ and D ≈ 1.8 × 10¹¹. Lab 08\'s chinchilla_optimal computes it.',
        },
        {
          q: 'A product team will serve a model to millions of users. Why might they choose an 8B model trained on 15 T tokens over the compute-optimal ~77B model for the same training compute?',
          options: [
            'The 8B model has much lower loss',
            'The predicted loss is almost the same (about 0.02 worse), and every generated token costs roughly 10× less to serve — over the model\'s lifetime the inference saving dwarfs the training premium',
            'Small models do not need evaluation',
            'Chinchilla scaling does not apply to 8B models',
          ],
          correct: [1],
          explanation: 'Compute-optimality minimises training FLOPs per point of loss. Lifetime cost includes inference, which scales with N. Over-training buys a smaller model at nearly the same quality — the reasoning behind most current open models.',
        },
        {
          q: 'Why did Kaplan et al. and Hoffmann et al. reach different conclusions about how to split compute?',
          options: [
            'They used different definitions of a token',
            'A main factor: Kaplan\'s runs used a learning-rate schedule not matched to each run\'s length, so shorter runs were undertrained relative to their own optimum; Chinchilla tuned the cosine length per run',
            'Chinchilla used a different architecture',
            'Kaplan measured validation loss and Chinchilla training loss',
          ],
          correct: [1],
          explanation: 'Comparing runs whose schedules do not end where the run ends measures the schedule as much as the model (O1.L4). Matching the cosine decay to each run\'s token count changed the fitted exponents, and with them the recommended split.',
        },
      ],
    },
    {
      type: 'deepdive',
      title: 'Going deeper',
      md: `- Kaplan et al., "Scaling Laws for Neural Language Models" (2020, arxiv.org/abs/2001.08361).
- Hoffmann et al., "Training Compute-Optimal Large Language Models" (2022, arxiv.org/abs/2203.15556) — Chinchilla; the loss fit above is their equation 10 with the approach-3 constants.
- Sardana and Frankle, "Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws" (2023, arxiv.org/abs/2401.00448).
- Run a mini scaling study with lab 07: three widths (64, 128, 256) at three step counts, plot final validation loss against 6·N·D.`,
    },
  ],
}

export default lesson

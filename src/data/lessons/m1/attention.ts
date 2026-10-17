import type { Lesson } from '../types'

const lesson: Lesson = {
  id: 'm1.l3',
  slug: 'attention',
  trackId: 'm1',
  index: 3,
  title: 'Attention',
  minutes: 22,
  hook: 'Attention is a soft database lookup: every token issues a query, every earlier token offers a key and a value, and the answer is a weighted average. Three matmuls and a softmax — plus one division by √D that decides whether it trains at all.',
  exercise: 'lab+quiz',
  takeaway: {
    number: 'variance 64 → 1',
    claim: 'Dot products of 64-dimensional random vectors have variance 64, which saturates the softmax onto one token (78% of the weight); dividing by √64 brings the variance to 1 and spreads it (7%) — that is the whole reason for the scale.',
  },
  blocks: [
    {
      type: 'prose',
      md: `Every token in a sequence needs information from other tokens: a pronoun needs its noun, a closing bracket its opening one. Attention lets each position pull in what it needs from every earlier position, with the choice of *what* learned from data.

## Query, key, value

Each token's vector \`x\` is projected three ways by three learned matrices:

- **query** \`q = x @ Wq\` — what this token is looking for;
- **key** \`k = x @ Wk\` — what this token offers to be found by;
- **value** \`v = x @ Wv\` — what this token hands over if it is chosen.

The match between token i's query and token j's key is a dot product. A softmax over j turns token i's matches into weights summing to 1, and token i's output is the weighted average of the values:

\`\`\`
scores  = q @ kᵀ / √D           (T, T)   every query against every key
weights = softmax(scores)       (T, T)   each row sums to 1
out     = weights @ v           (T, D)   a blend of values per token
\`\`\`

It is a key–value lookup where every key matches a little and the result is a blend — differentiable, so training can shape what queries and keys mean.`,
    },
    {
      type: 'isomorphism',
      title: 'A lookup you already know',
      pairs: [
        {
          os: 'dict lookup: cache[key]',
          osLine: 'exactly one key matches; you get exactly its value',
          llm: 'attention',
          llmLine: 'every key matches to a degree; you get a weighted blend of all values',
        },
        {
          os: 'a search engine',
          osLine: 'query terms scored against document index terms; results ranked by score',
          llm: 'q · k',
          llmLine: 'query vector scored against each key vector; softmax turns scores into weights',
        },
      ],
    },
    {
      type: 'prose',
      md: `## Why divide by √D

If q and k have D independent components of variance 1, their dot product is a sum of D such products and has variance **D**. With D = 64, scores have a standard deviation of 8, and a softmax over numbers spread that widely puts almost all its weight on the largest one. Measured over 200 random rows of 128 keys:

| scores | score variance | largest weight (mean) |
|---|---|---|
| q · k | 64.3 | **0.78** |
| q · k / √64 | 1.0 | **0.071** |

Uniform attention over 128 keys would be 0.008 per key. Unscaled, one random key takes 78% of the weight before anything has been learned. A saturated softmax has near-zero gradients for every other key (O1.L1's tanh problem again), so the model cannot learn to look elsewhere. Dividing by √D restores unit variance and a softmax that can learn.

## The causal mask

A language model predicts token t+1 from tokens 0…t. During training all positions are computed at once, so position t must not see positions after it, or it could copy the answer. Before the softmax, set every score where j > i to −∞; \`exp(−∞) = 0\` gives those keys exactly zero weight. The mask is a lower-triangular (T, T) pattern — lab 05's \`causal_mask\` — and its correctness is testable: changing a future token must not change any earlier output (\`no-peeking\`).`,
    },
    {
      type: 'diagram',
      caption: 'fig 1 — one attention head, causal',
      height: 64,
      nodes: [
        { id: 'x', x: 2, y: 26, w: 12, h: 12, label: 'x', sub: '(T, C)' },
        { id: 'q', x: 20, y: 4, w: 14, h: 12, label: 'q = x·Wq', sub: '(T, D)' },
        { id: 'k', x: 20, y: 26, w: 14, h: 12, label: 'k = x·Wk', sub: '(T, D)' },
        { id: 'v', x: 20, y: 48, w: 14, h: 12, label: 'v = x·Wv', sub: '(T, D)' },
        { id: 's', x: 42, y: 14, w: 18, h: 12, label: 'q·kᵀ / √D', sub: '(T, T)' },
        { id: 'm', x: 66, y: 14, w: 16, h: 12, label: 'mask + softmax', sub: 'rows sum to 1' },
        { id: 'o', x: 66, y: 44, w: 16, h: 12, label: 'weights · v', sub: '(T, D)' },
      ],
      edges: [
        { from: 'x', to: 'q' },
        { from: 'x', to: 'k' },
        { from: 'x', to: 'v' },
        { from: 'q', to: 's' },
        { from: 'k', to: 's' },
        { from: 's', to: 'm' },
        { from: 'm', to: 'o' },
        { from: 'v', to: 'o' },
      ],
      steps: [
        { caption: 'Three learned projections turn each token vector into a query, a key and a value. All three are ordinary matmuls over every position at once.', active: ['x', 'q', 'k', 'v'], edges: ['x->q', 'x->k', 'x->v'] },
        { caption: 'Every query is scored against every key: a (T, T) matrix, scaled by √D so its entries have variance about 1.', active: ['q', 'k', 's'], edges: ['q->s', 'k->s'] },
        { caption: 'Scores for future positions are set to −∞, then each row is softmaxed: token i\'s weights over tokens 0…i, summing to 1.', active: ['s', 'm'], edges: ['s->m'] },
        { caption: 'Each token\'s output is its weights times the value vectors: a blend of what the tokens it attended to had to offer.', active: ['m', 'v', 'o'], edges: ['m->o', 'v->o'] },
      ],
    },
    {
      type: 'prose',
      md: `## Many heads

One attention pattern per layer is limiting: a token may need its subject, the previous token, and the matching bracket all at once. **Multi-head attention** splits the C-dimensional projections into H heads of D = C/H dimensions, runs attention independently in each, concatenates the results and mixes them with an output projection \`Wo\`. Same FLOPs as one wide head — the split is a reshape (lab 05's \`split_heads\`) — but H independent patterns. GPT-2 small uses 12 heads of 64.

## What it costs

The projections cost what any linear layer costs: about \`8·T·C²\` FLOPs for q, k, v and the output projection. The new term is the score matrix: \`2·T²·D\` FLOPs per head for \`q·kᵀ\` and the same again for \`weights·v\`. More important is its **memory**: a (T, T) matrix per head, per layer, per sequence. For GPT-2 small at T = 1,024 in bf16, that is 12 heads × 1,024² × 2 bytes ≈ **25 MB per layer per sequence** — the 5·a·s² term in T0.L4's activation formula.`,
    },
    {
      type: 'vendor',
      snapshot: '2026-10',
      title: 'FlashAttention',
      md: `FlashAttention (Dao et al., 2022) computes exactly the same output without ever writing the (T, T) score matrix to GPU memory: it streams blocks of keys and values through on-chip SRAM and keeps a running softmax (the "online softmax" trick — the same max-shift as O1.L2, updated block by block). Attention memory becomes linear in T instead of quadratic, and because the operation was memory-bound (T0.L3), it also runs several times faster. In PyTorch it is used automatically by \`torch.nn.functional.scaled_dot_product_attention\` on supported GPUs.`,
      systems: ['flashattention', 'pytorch'],
      sources: [
        'https://arxiv.org/abs/2205.14135',
        'https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html',
      ],
    },
    {
      type: 'lab',
      lab: 'attention',
      brief: 'Write a stable softmax, the causal mask, batched scaled dot-product attention, the head split, multi-head attention and RoPE — then prove the mask never leaks the future and that RoPE scores depend only on distance.',
    },
    {
      type: 'quiz',
      questions: [
        {
          q: 'A colleague drops the √D scaling "because the model will learn the right scale". With D = 128, what happens at initialisation?',
          options: [
            'Nothing — the softmax is scale-invariant',
            'Scores have variance about 128, so each softmax row concentrates on one essentially random key and the gradients to the other keys nearly vanish — learning is slow or stalls',
            'The scores overflow to inf',
            'Attention becomes uniform',
          ],
          correct: [1],
          explanation: 'Softmax is invariant to adding a constant, not to multiplying. Scores with standard deviation ≈ 11 make the softmax nearly one-hot, and a saturated softmax passes almost no gradient to the keys it ignores. The measured case at D = 64 already put 78% of the weight on one key.',
        },
        {
          q: 'Training loss drops to nearly zero within a few hundred steps on a language-modelling task, but generation produces garbage. Which bug fits best?',
          options: [
            'The learning rate is too low',
            'The causal mask is missing or wrong, so each position can see the token it is supposed to predict and simply copies it',
            'Too many heads',
            'The tokenizer vocabulary is too small',
          ],
          correct: [1],
          explanation: 'A leaky mask turns next-token prediction into copying, which is trivially easy, so training loss collapses. At generation time the future does not exist, and the model has learned nothing useful. The no-peeking test catches this in seconds.',
        },
        {
          q: 'Doubling the context length from 2,048 to 4,096 multiplies the attention score memory per layer (without FlashAttention) by how much?',
          options: ['1×', '2×', '4×', '8×'],
          correct: [2],
          explanation: 'The score matrix is T × T per head. Doubling T quadruples it. The projections and MLP scale linearly in T, so at long context the quadratic term dominates — the motivation for FlashAttention\'s linear-memory formulation.',
        },
      ],
    },
    {
      type: 'deepdive',
      title: 'Going deeper',
      md: `- Vaswani et al., "Attention Is All You Need" (2017, arxiv.org/abs/1706.03762) — section 3.2 introduces scaled dot-product and multi-head attention.
- Jay Alammar, "The Illustrated Transformer" — the same mechanism, drawn.
- Milakov and Gimelshein, "Online normalizer calculation for softmax" (2018) — the trick FlashAttention is built on.`,
    },
  ],
}

export default lesson

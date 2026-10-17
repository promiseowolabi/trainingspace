import type { Lesson } from '../types'

const lesson: Lesson = {
  id: 'm1.l2',
  slug: 'embeddings-and-position',
  trackId: 'm1',
  index: 2,
  title: 'Embeddings and Position',
  minutes: 17,
  hook: 'An embedding is a lookup table you train. Attention, on its own, cannot tell "dog bites man" from "man bites dog" — so position has to be added, and how it is added decides how far a model can read.',
  exercise: 'read+quiz',
  takeaway: {
    number: 'θᵢ = 10000^(−2i/D)',
    claim: 'Rotary embeddings rotate each pair of query and key dimensions by position × θᵢ, so the attention score between two tokens depends only on their distance — with no position parameters at all.',
  },
  blocks: [
    {
      type: 'prose',
      md: `## Embeddings: a trained lookup table

The first layer of a language model is a matrix \`wte\` of shape \`(V, C)\`: one row of C numbers per token in the vocabulary. "Embedding token 3797" means "take row 3797":

\`\`\`python
x = wte[idx]          # idx: (B, T) ints  ->  x: (B, T, C)
\`\`\`

Mathematically this is a matmul with a one-hot vector — \`onehot(3797) @ wte\` — which is how gradients flow into it: only the rows of tokens that appeared in the batch receive gradient. Nothing about a row is designed. It starts random (std 0.02 in GPT-2) and training moves tokens that behave alike to nearby rows, because nearby rows produce similar downstream computations.

## Attention cannot see order

Attention (M1.L3) computes, for each token, a weighted average of other tokens' values, with weights from query–key dot products. Shuffle the input tokens and every output is shuffled the same way — nothing else changes. Without help, "dog bites man" and "man bites dog" are the same bag of three vectors. The causal mask leaks a little order (a token knows how many tokens precede it), but not enough. Position has to be injected.`,
    },
    {
      type: 'prose',
      md: `## Three ways to add position

**Learned absolute (GPT-2).** A second table \`wpe\` of shape \`(block_size, C)\`, added to the token embeddings: \`x = wte[idx] + wpe[:T]\`. Simple, and lab 06 uses it. GPT-2's is 1,024 × 768 = 786,432 parameters, and it has a hard limit: there is no row 1,025, so the model cannot read beyond the length it was trained with.

**Sinusoidal (the 2017 transformer).** The same addition, but the position vectors are fixed sine and cosine waves of different frequencies instead of learned rows. No parameters, and defined for any position — though models still degrade beyond trained lengths.

**Rotary, RoPE (Llama and most current models).** Nothing is added to x. Instead, inside every attention layer, each query and key vector is **rotated** by an angle proportional to its position, pair of dimensions by pair of dimensions:

\`\`\`
for pair i = (dims 2i, 2i+1), at position t:   rotate by  t · θᵢ,   θᵢ = 10000^(−2i/D)
\`\`\`

Low pairs rotate fast (θ₀ = 1 radian per position), high pairs slowly (the last pair once in tens of thousands of positions), like the hands of a clock.`,
    },
    {
      type: 'diagram',
      caption: 'fig 1 — why RoPE gives relative position',
      height: 62,
      nodes: [
        { id: 'q', x: 2, y: 6, w: 20, h: 12, label: 'q at position m', sub: 'rotated by m·θ' },
        { id: 'k', x: 2, y: 40, w: 20, h: 12, label: 'k at position n', sub: 'rotated by n·θ' },
        { id: 'dot', x: 38, y: 22, w: 24, h: 12, label: 'q · k', sub: 'angle between them' },
        { id: 'rel', x: 76, y: 22, w: 22, h: 12, label: 'depends on m − n', sub: 'only the distance' },
      ],
      edges: [
        { from: 'q', to: 'dot' },
        { from: 'k', to: 'dot' },
        { from: 'dot', to: 'rel' },
      ],
      steps: [
        { caption: 'The query of the token at position m has each of its dimension pairs rotated by m·θᵢ. Nothing else about the vector changes; rotations preserve length.', active: ['q'] },
        { caption: 'The key of the token at position n is rotated the same way by n·θᵢ.', active: ['k'] },
        { caption: 'A dot product between two rotated 2-D vectors depends only on the angle between them: rotating both by the same extra amount changes nothing.', active: ['q', 'k', 'dot'], edges: ['q->dot', 'k->dot'] },
        { caption: 'So the score depends on (m − n)·θᵢ: relative position, computed inside attention, with zero parameters. Lab 05\'s rope-relative check tests exactly this.', active: ['dot', 'rel'], edges: ['dot->rel'] },
      ],
    },
    {
      type: 'prose',
      md: `RoPE's properties are why it won:

- **Relative by construction.** The score between tokens 10 and 3 equals the score between 30 and 23 (for the same content). What matters for language is mostly "how far back", not "at absolute index 512".
- **No parameters, no length table.** Position is a computation, so there is no row that does not exist.
- **Extendable.** Because θᵢ is a formula, a model trained at 8k tokens can be adapted to longer contexts by rescaling the angles (raising the base, or interpolating positions) and fine-tuning briefly — the trick behind most long-context releases.

## Tied embeddings

The output layer maps a C-vector back to V scores — a \`(C, V)\` matrix, the same shape as \`wteᵀ\`. GPT-2 **ties** them: \`logits = x @ wte.T\`, so one matrix both reads and writes tokens and the model saves V·C parameters (38.6 M for GPT-2 small, 31% of its total). Larger models usually untie them; at 8B parameters the saving is small and separate matrices train slightly better.`,
    },
    {
      type: 'quiz',
      questions: [
        {
          q: 'A model trained with GPT-2-style learned position embeddings and block_size 2,048 is given a 4,000-token input. What happens?',
          options: [
            'It works, slightly worse',
            'It cannot run as-is: there are no position embeddings for positions 2,048 and beyond — wpe has exactly 2,048 rows',
            'Positions wrap around automatically',
            'Attention ignores position, so it works perfectly',
          ],
          correct: [1],
          explanation: 'wpe[:T] needs T rows. Learned absolute embeddings fix the maximum length at training time. RoPE has no such table, which is one reason long-context models use it — though they still need adaptation to work well past their trained length.',
        },
        {
          q: 'Why does a transformer with no position information treat "dog bites man" and "man bites dog" almost identically?',
          options: [
            'The tokenizer merges them',
            'Attention is a weighted average whose weights depend on content only, so permuting the inputs just permutes the outputs — only the causal mask carries any order',
            'Embeddings are sorted alphabetically',
            'The MLP removes order',
          ],
          correct: [1],
          explanation: 'Query–key scores are functions of the vectors, not of where they sit. Without position signals the model sees a set of tokens. The causal mask reveals how many tokens precede each one, which is too weak to recover full word order reliably.',
        },
        {
          q: 'What does tying the input embedding and the output layer save in GPT-2 small (V = 50,257, C = 768)?',
          options: ['786,432 parameters', 'about 38.6 million parameters', 'about 85 million parameters', 'nothing — they are different shapes'],
          correct: [1],
          explanation: 'The output layer would be another (768, 50,257) matrix: 38,597,376 parameters, about 31% of GPT-2 small\'s 124.4 M. The shapes are transposes of each other, which is what makes tying possible. 786,432 is the size of wpe.',
        },
      ],
    },
    {
      type: 'deepdive',
      title: 'Going deeper',
      md: `- Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021, arxiv.org/abs/2104.09864).
- Press and Wolf, "Using the Output Embedding to Improve Language Models" (2016) — weight tying.
- Chen et al., "Extending Context Window of Large Language Models via Positional Interpolation" (2023) — rescaling RoPE angles for longer context.`,
    },
  ],
}

export default lesson

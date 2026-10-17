import type { Lesson } from '../types'

const lesson: Lesson = {
  id: 'm1.l5',
  slug: 'counting-a-gpt',
  trackId: 'm1',
  index: 5,
  title: 'Counting a GPT',
  minutes: 17,
  hook: 'From five numbers in a config — vocabulary, context, layers, heads, width — you can derive a model\'s parameter count to the last digit, its memory, and the FLOPs of every token it will ever process. Do it once by hand and every model card becomes readable.',
  exercise: 'lab+quiz',
  takeaway: {
    number: '124,439,808',
    claim: 'GPT-2 small\'s parameter count follows exactly from its config — 38.6 M of embeddings, 12 blocks of 7.09 M, one final norm — and its forward pass costs about 2 FLOPs per parameter per token.',
  },
  blocks: [
    {
      type: 'prose',
      md: `GPT-2 small's config: V = 50,257, T = 1,024, L = 12, H = 12, C = 768. Count every array lab 06's \`init_params\` creates:

| part | shape | parameters |
|---|---|---|
| token embedding \`wte\` | 50,257 × 768 | 38,597,376 |
| position embedding \`wpe\` | 1,024 × 768 | 786,432 |
| per block: ln1 (g, b) | 2 × 768 | 1,536 |
| per block: attention qkv + bias | 768 × 2,304 + 2,304 | 1,771,776 |
| per block: attention output + bias | 768 × 768 + 768 | 590,592 |
| per block: ln2 (g, b) | 2 × 768 | 1,536 |
| per block: MLP up + bias | 768 × 3,072 + 3,072 | 2,362,368 |
| per block: MLP down + bias | 3,072 × 768 + 768 | 2,360,064 |
| **one block** | | **7,087,872** |
| 12 blocks | | 85,054,464 |
| final LayerNorm | 2 × 768 | 1,536 |
| output layer | tied to \`wte\` | 0 |
| **total** | | **124,439,808** |

That number is the "124M" in "GPT-2 124M", and lab 06's \`count_params\` must hit it exactly.`,
    },
    {
      type: 'statline',
      stats: [
        { value: '12·L·C²', label: 'blocks, approx.', hint: '12 × 12 × 768² = 84.9 M against an exact 85.1 M — biases and norms are the difference' },
        { value: '31%', label: 'embeddings, GPT-2 small', hint: '38.6 M of 124.4 M — at small widths the vocabulary is a big share' },
        { value: '2·N', label: 'forward FLOPs / token', hint: 'one multiply-add per weight per token, plus attention\'s T-dependent term' },
        { value: '6·N', label: 'training FLOPs / token', hint: 'forward + backward ≈ 3× forward (A1.L2) — the basis of P1.L4' },
      ],
    },
    {
      type: 'prose',
      md: `## The shortcut: 12·L·C²

Biases and norms are O(C) per block; the matrices are O(C²). Dropping the small terms, each block is 12·C² (M1.L4), so:

\`\`\`
N ≈ 12 · L · C²  +  V · C          (blocks + embedding)
\`\`\`

This is accurate to a fraction of a percent for the blocks, and it explains how models scale: **doubling the width quadruples** the parameters per layer, doubling the depth doubles them. As models grow, the V·C embedding term shrinks in relative terms — 31% of GPT-2 small, a few percent of a 70B model.

## FLOPs per token

Every weight in a matmul participates in one multiply and one add per token that passes through it (T0.L3). So a forward pass costs about **2·N FLOPs per token**. Attention adds a term that grows with context — computing scores against T earlier keys and blending T values — of about 2·L·T·C per token. For GPT-2 small at T = 1,024 that is 18.9 M against 2N = 249 M: about 8%. For long contexts or small models it matters; for most training budgets it is folded into the approximation.

Measured with lab 06's numpy forward pass on GPT-2 small's shapes (random float32 weights), the reference laptop processed a 1,024-token sequence in **1.56 s** — 656 tokens/s, or about **160 GFLOP/s** by the 2·N rule. The same laptop's matmul peak was 2 TFLOP/s (T0.L3); the difference is LayerNorm, softmax and GELU at memory speed, float32 temporaries, and numpy's single-operation-at-a-time execution — exactly the overheads PyTorch's fused kernels exist to remove.`,
    },
    {
      type: 'code',
      filename: 'count_params, the short version',
      code: `def count_params(V, T, L, C):
    per_block = 12 * C * C + 13 * C      # 4 attention + 8 MLP weight matrices, biases + 2 norms
    return V * C + T * C + L * per_block + 2 * C

count_params(50257, 1024, 12, 768)       # 124,439,808`,
      chips: ['exact for GPT-2', 'tied output'],
    },
    {
      type: 'prose',
      md: `Check the 13·C: biases are 3C (qkv) + C (attention out) + 4C (MLP up) + C (MLP down) = 9C, and the two LayerNorms add 4C. Lab 06 asks you to derive it yourself; the code above is the answer, so do the derivation first.

## Memory, from the count

T0.L4's rule turns the count into bytes. GPT-2 small: 124.4 M × 2 bytes = 249 MB of bf16 weights to run, and 124.4 M × 16 = **2.0 GB** of training state before activations. A laptop can train it — the capstone's model is smaller still.`,
    },
    {
      type: 'lab',
      lab: 'gpt-forward',
      brief: 'Assemble LayerNorm, GELU, the pre-norm block and tied embeddings into a GPT-2-architecture forward pass that matches a reference to 10⁻⁸, derive count_params until it says 124,439,808, then add greedy decoding and temperature, top-k and top-p sampling (M1.L6).',
    },
    {
      type: 'quiz',
      questions: [
        {
          q: 'A model has L = 32, C = 4,096 and V = 32,000 with untied embeddings. Approximately how many parameters?',
          options: ['about 1.3 B', 'about 6.7 B', 'about 13 B', 'about 4.3 B'],
          correct: [1],
          explanation: '12 × 32 × 4,096² ≈ 6.44 B for the blocks, plus two V·C embedding matrices (input and untied output) of 131 M each: ≈ 6.7 B. These are LLaMA-7B\'s shapes; its published count is 6.7 B.',
        },
        {
          q: 'You double a model\'s width C and keep everything else fixed. Roughly what happens to its block parameters and its forward FLOPs per token?',
          options: ['Both double', 'Both quadruple', 'Parameters quadruple, FLOPs double', 'Parameters double, FLOPs quadruple'],
          correct: [1],
          explanation: 'Block parameters are 12·L·C², so doubling C multiplies them by 4. Forward FLOPs are about 2N per token, so they also quadruple. Width is the expensive knob; depth scales both linearly.',
        },
        {
          q: 'What does "a forward pass costs 2N FLOPs per token" leave out?',
          options: [
            'Nothing — it is exact',
            'The attention score and blend computations, which grow with context length (about 2·L·T·C per token), plus cheap elementwise operations',
            'The embedding lookup, which costs V·C FLOPs',
            'The backward pass, which is free',
          ],
          correct: [1],
          explanation: '2N counts the matmuls with weights. Attention\'s q·kᵀ and weights·v involve no parameters but cost FLOPs proportional to T. An embedding lookup is indexing, not a matmul, so it costs almost nothing despite holding V·C parameters.',
        },
      ],
    },
    {
      type: 'deepdive',
      title: 'Going deeper',
      md: `- Kaplan et al., "Scaling Laws for Neural Language Models" (2020, arxiv.org/abs/2001.08361), Table 1 — the parameter and FLOP accounting, term by term, including the attention term.
- Load a real model card's config (any Hugging Face \`config.json\`) and predict its parameter count before reading it.
- P1.L4 turns 6·N per token into the cost of a whole training run.`,
    },
  ],
}

export default lesson

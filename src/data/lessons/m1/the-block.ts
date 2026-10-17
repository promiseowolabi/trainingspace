import type { Lesson } from '../types'

const lesson: Lesson = {
  id: 'm1.l4',
  slug: 'the-block',
  trackId: 'm1',
  index: 4,
  title: 'The Block',
  minutes: 18,
  hook: 'A GPT is one block repeated: normalise, attend, add; normalise, MLP, add. Attention moves information between tokens; the MLP — two thirds of the parameters — processes each token on its own. Stack it L times and you have the model.',
  exercise: 'read+quiz',
  takeaway: {
    number: '8C² of 12C²',
    claim: 'In a GPT block the MLP holds 8·C² of the 12·C² weights and attention the other 4·C²: attention routes information between positions, the MLP does most of the per-token computation.',
  },
  blocks: [
    {
      type: 'prose',
      md: `Everything is in place. A GPT block is the pre-norm residual layer of O1.L5 with attention (M1.L3) in one branch and an MLP (O1.L1) in the other:

\`\`\`python
def block(x):                                # x: (B, T, C), the residual stream
    x = x + attention(layer_norm_1(x))       # mix information ACROSS positions
    x = x + mlp(layer_norm_2(x))             # transform each position on its own
    return x

def mlp(h):                                  # GPT-2: expand 4x, GELU, project back
    return gelu(h @ W_fc + b_fc) @ W_proj + b_proj      # (C -> 4C -> C)
\`\`\`

And the model is the block in a loop:

\`\`\`python
x = wte[idx] + wpe[:T]                       # (B, T, C)    M1.L2
for blk in blocks:                           # L times
    x = blk(x)
logits = layer_norm_f(x) @ wte.T             # (B, T, V)    tied output, M1.L2
\`\`\`

Lab 06 is these ten lines, made exact.`,
    },
    {
      type: 'prose',
      md: `## Two kinds of work

The two sub-blocks do different jobs, and it helps to keep them apart:

- **Attention is the only place tokens interact.** It reads other positions' vectors and writes a blend into this position's stream. Without it, each token's prediction would depend on that token alone.
- **The MLP sees one token at a time.** The same weights apply independently at every position: no information crosses between tokens. It is where most per-token transformation happens — and where interpretability research finds much of a model's stored factual association.

Both write into the **residual stream** by addition. Think of the stream as a C-dimensional bus every layer reads from and writes to; each block adds its contribution and passes the sum on.

## Where the parameters are

For width C (ignoring biases and norms, which are O(C)):

| part | matrices | weights |
|---|---|---|
| attention: q, k, v, output | four C × C | **4·C²** |
| MLP: up and down | C × 4C and 4C × C | **8·C²** |
| **block** | | **12·C²** |

For GPT-2 small (C = 768): 12 × 768² = 7.08 M per block; the exact count with biases and LayerNorms is 7,087,872. Twelve blocks hold 85 M of the model's 124.4 M parameters; the embeddings hold most of the rest (M1.L5).`,
    },
    {
      type: 'diagram',
      caption: 'fig 1 — the model as a stack',
      height: 66,
      nodes: [
        { id: 'emb', x: 2, y: 50, w: 22, h: 12, label: 'embed', sub: 'wte[idx] + wpe' },
        { id: 'b1', x: 30, y: 50, w: 18, h: 12, label: 'block 1', sub: 'attn + MLP' },
        { id: 'dots', x: 30, y: 28, w: 18, h: 12, label: '…', sub: 'L blocks' },
        { id: 'bl', x: 30, y: 6, w: 18, h: 12, label: 'block L', sub: 'attn + MLP' },
        { id: 'lnf', x: 56, y: 6, w: 18, h: 12, label: 'ln_f', sub: 'final norm' },
        { id: 'out', x: 80, y: 6, w: 18, h: 12, label: 'logits', sub: 'x · wteᵀ' },
      ],
      edges: [
        { from: 'emb', to: 'b1' },
        { from: 'b1', to: 'dots' },
        { from: 'dots', to: 'bl' },
        { from: 'bl', to: 'lnf' },
        { from: 'lnf', to: 'out' },
      ],
      steps: [
        { caption: 'Token ids become vectors: a row of the token table plus a row of the position table. This (B, T, C) array is the residual stream.', active: ['emb'] },
        { caption: 'Each block reads the stream, lets tokens exchange information through attention, transforms each token with the MLP, and adds both results back.', active: ['emb', 'b1'], edges: ['emb->b1'] },
        { caption: 'The same structure repeats L times with different weights — 12 in GPT-2 small, 32 in an 8B-class model. Depth is how a model composes simple steps into complex ones.', active: ['b1', 'dots', 'bl'], edges: ['b1->dots', 'dots->bl'] },
        { caption: 'A final LayerNorm, then the embedding matrix used in reverse scores every vocabulary entry at every position: (B, T, V) logits for the cross-entropy loss.', active: ['bl', 'lnf', 'out'], edges: ['bl->lnf', 'lnf->out'] },
      ],
    },
    {
      type: 'vendor',
      snapshot: '2026-10',
      title: 'The Llama-style block',
      md: `Most current open models keep this skeleton and swap three parts: **RMSNorm** instead of LayerNorm (O1.L5), **RoPE** inside attention instead of a position table (M1.L2), and a **SwiGLU** MLP: \`(silu(h·W1) ⊙ (h·W3)) · W2\`, three matrices instead of two. To keep roughly 8·C² MLP weights, the hidden width shrinks from 4·C to about ⅔·4·C — LLaMA 7B uses C = 4,096 and a hidden width of 11,008. Biases are dropped throughout.`,
      systems: ['llama'],
      sources: ['https://arxiv.org/abs/2302.13971', 'https://arxiv.org/abs/2002.05202'],
    },
    {
      type: 'callout',
      variant: 'info',
      title: 'Why every block is the same shape',
      md: `Identical blocks are an engineering choice as much as a modelling one: one kernel configuration, one memory layout and one communication pattern repeated L times. Pipeline parallelism (D1.L5) cuts the stack between blocks; tensor parallelism (D1.L4) splits inside each block the same way L times; activation checkpointing (D1.L6) saves one tensor per block boundary. A uniform stack makes all three simple.`,
    },
    {
      type: 'quiz',
      questions: [
        {
          q: 'Which sub-block lets information move from token 3 to token 9?',
          options: [
            'The MLP',
            'Attention — it is the only operation that reads other positions; the MLP, LayerNorm and residual adds act on each position separately',
            'The final LayerNorm',
            'The embedding',
          ],
          correct: [1],
          explanation: 'MLPs, norms and residual additions are position-wise: the same function applied independently at each token. Only attention combines vectors from different positions. Remove it and each prediction depends on one token.',
        },
        {
          q: 'A GPT block has C = 4,096. Roughly how many weights are in its MLP (GPT-2 style, 4× expansion)?',
          options: ['16.8 M', '67 M', '134 M', '201 M'],
          correct: [2],
          explanation: '8·C² = 8 × 4,096² ≈ 134 M. Attention adds 4·C² ≈ 67 M, for about 201 M per block. Thirty-two such blocks is about 6.4 B — most of a "7B" model.',
        },
        {
          q: 'Why does the block ADD each sub-block\'s output to x instead of replacing x?',
          options: [
            'Addition is cheaper than assignment',
            'The residual stream keeps an identity path for the gradient through every block (O1.L5) and lets each block make an incremental contribution rather than rebuild the representation',
            'It halves the parameter count',
            'It makes attention causal',
          ],
          correct: [1],
          explanation: 'x + f(x) has derivative 1 + f\'(x), so gradients reach early blocks undiminished, and each block only has to learn a correction. Replacing x would make the stack a plain deep network, which does not train at this depth.',
        },
      ],
    },
    {
      type: 'deepdive',
      title: 'Going deeper',
      md: `- Radford et al., "Language Models are Unsupervised Multitask Learners" (GPT-2, 2019) — the pre-norm change and the final LayerNorm.
- Shazeer, "GLU Variants Improve Transformer" (2020, arxiv.org/abs/2002.05202) — SwiGLU.
- Andrej Karpathy, *nanoGPT* (github.com/karpathy/nanoGPT) \`model.py\` — the same block in PyTorch in about 100 lines; compare it with lab 06.`,
    },
  ],
}

export default lesson

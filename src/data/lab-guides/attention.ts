import type { LabGuide } from './types'

const guide: LabGuide = {
  id: 'attention',
  requires: ['a browser — or Python 3.10+ with numpy and pytest'],
  blocks: [
    {
      type: 'prose',
      md: `## What you will build

The one operation that makes a transformer a transformer, in about thirty lines of numpy — and the position encoding most current models use with it. The rule for this lab: **no Python loops over batch, time or heads**. Every function works on whole arrays; \`@\` already batches over every axis except the last two.`,
    },
    {
      type: 'prose',
      md: `## 1. softmax and the mask

\`softmax\` is O1.L2's stable version with an \`axis\` argument. Masked positions arrive as \`-inf\`; after the max-shift they are still \`-inf\` and \`exp\` turns them into exactly 0 — no special case needed. \`causal_mask\` is one call to \`np.tril\`.

## 2. attention

Three lines: scores, mask, weighted sum. Scale by \`√D\` where D is \`q.shape[-1]\` — the **head** width, not the model width. Use \`np.swapaxes(k, -1, -2)\`, not \`k.T\`: \`.T\` reverses every axis, which is right for a 2-D array and wrong for every batched one.

## 3. Heads

\`split_heads\` is a reshape and a transpose: \`(B, T, C) → (B, T, H, D) → (B, H, T, D)\`. Once heads sit in front of T, your \`attention\` treats each one as an independent problem automatically. \`merge_heads\` reverses both steps. Then \`multi_head_attention\` is: three projections, split, attend with the mask, merge, project.

The \`no-peeking\` check changes the last two tokens of a sequence and requires that outputs 0–3 do not move. If it fails while \`multi-head-matches\` passes for the causal case, your mask is being broadcast against the wrong axes.`,
    },
    {
      type: 'prose',
      md: `## 4. RoPE

Build the angle table once: \`theta = base ** (-np.arange(0, D, 2) / D)\` has shape \`(D/2,)\`, and \`angle = positions[:, None] * theta[None, :]\` has shape \`(T, D/2)\`. Take the even and odd dimensions with \`x[..., 0::2]\` and \`x[..., 1::2]\`, rotate each pair with the 2-D rotation formula from the docstring, and write them back into the same slots of a new array.

The last check is the reason RoPE exists: it builds the same query at every position and the same key at every position, and requires that the score between positions 10 and 3 equals the score between 30 and 23 — attention that sees only *relative* distance.`,
    },
    {
      type: 'callout',
      variant: 'info',
      title: 'Two RoPE conventions',
      md: `This lab rotates **adjacent** pairs (0, 1), (2, 3), …, as in the original RoFormer paper. Many implementations instead pair dimension i with dimension i + D/2 (the "rotate half" layout). The two are the same rotation applied to a permuted vector, and give identical attention scores — but weights trained with one layout produce wrong outputs if loaded into code that uses the other.`,
    },
  ],
  cleanup: [
    {
      type: 'prose',
      md: `- Time \`attention\` for T = 512, 1024 and 2048 with D = 64 and watch the (T, T) score matrix dominate memory: T² × 8 bytes per head in float64.
- Write the same attention with \`np.einsum('bhqd,bhkd->bhqk', q, k)\` and compare.
- Apply \`rope\` to q and k inside \`multi_head_attention\` (after the split, before the scores) — that is exactly where Llama-style models apply it.`,
    },
  ],
}

export default guide

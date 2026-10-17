import type { Lesson } from '../types'

const lesson: Lesson = {
  id: 'd1.l4',
  slug: 'tensor-parallel',
  trackId: 'd1',
  index: 4,
  title: 'Tensor Parallelism',
  minutes: 18,
  hook: 'When a single layer is too big or too slow for one GPU, split its matrices. Cut the first MLP matrix by columns and the second by rows, and the whole block needs exactly one all-reduce — but that all-reduce happens in every layer, twice per step, which is why tensor parallelism never leaves the server.',
  exercise: 'read+quiz',
  takeaway: {
    number: '2 all-reduces per block, forward',
    claim: 'Megatron-style tensor parallelism splits each MLP and attention block so that it needs one all-reduce in forward and one in backward per sub-block — activation-sized messages in every layer, fast enough only over NVLink inside one server.',
  },
  blocks: [
    {
      type: 'prose',
      md: `Data parallelism and ZeRO split the **batch** and the **storage**; every rank still computes whole layers. **Tensor parallelism** (TP) splits the computation of each layer across devices, so every GPU holds and multiplies only a slice of each weight matrix. It reduces both memory and per-GPU compute per layer, and it is how one layer too large for a single device can run at all.

## The MLP, split two ways

Take the block's MLP (M1.L4): \`Y = gelu(X · A) · B\`, with A of shape (C, 4C) and B of shape (4C, C). On t GPUs:

1. **Split A by columns**: GPU i holds \`A_i\`, a (C, 4C/t) slice. Each computes \`H_i = gelu(X · A_i)\` — a slice of the hidden layer. GELU is elementwise, so it needs nothing from the other slices. **No communication.**
2. **Split B by rows**: GPU i holds \`B_i\`, the (4C/t, C) rows matching its hidden slice, and computes \`Y_i = H_i · B_i\` — a full-size (B, T, C) output that is a **partial sum**.
3. **All-reduce** the \`Y_i\`: \`Y = Σ Y_i\`, the exact result.`,
    },
    {
      type: 'code',
      filename: 'check it in numpy',
      code: `import numpy as np
gelu = lambda z: 0.5 * z * (1 + np.tanh(0.7978845608 * (z + 0.044715 * z**3)))
X, A, B = np.random.randn(4, 8), np.random.randn(8, 32), np.random.randn(32, 8)
full = gelu(X @ A) @ B

t = 4                                                       # "GPUs"
A_cols, B_rows = np.split(A, t, axis=1), np.split(B, t, axis=0)
partials = [gelu(X @ Ai) @ Bi for Ai, Bi in zip(A_cols, B_rows)]   # no communication
print(np.allclose(sum(partials), full))                     # True — the sum is the all-reduce`,
      chips: ['column-parallel → row-parallel', '1 all-reduce'],
    },
    {
      type: 'prose',
      md: `The order matters. Splitting A by **rows** instead would make each GPU produce a partial *pre-activation*, and GELU of a partial sum is not a partial GELU — you would need an all-reduce before the nonlinearity as well. Column-then-row puts the only synchronisation after the second matmul.

**Attention splits the same way, by heads.** Heads are independent (M1.L3), so GPU i takes heads 4i…4i + 3: the q, k, v projections are column-parallel (each GPU computes its heads' q, k, v), attention runs locally, and the output projection is row-parallel, followed by one all-reduce. Lab 05's \`split_heads\` is literally the sharding axis.

So a transformer block under TP has **one all-reduce after attention and one after the MLP** in the forward pass, and the mirror-image pair in backward.`,
    },
    {
      type: 'diagram',
      caption: 'fig 1 — the Megatron MLP on 2 GPUs',
      height: 62,
      nodes: [
        { id: 'x', x: 2, y: 24, w: 12, h: 12, label: 'X', sub: 'replicated' },
        { id: 'a1', x: 20, y: 6, w: 20, h: 12, label: 'gelu(X·A₁)', sub: 'GPU 0, cols' },
        { id: 'a2', x: 20, y: 42, w: 20, h: 12, label: 'gelu(X·A₂)', sub: 'GPU 1, cols' },
        { id: 'b1', x: 46, y: 6, w: 18, h: 12, label: 'H₁·B₁', sub: 'partial Y' },
        { id: 'b2', x: 46, y: 42, w: 18, h: 12, label: 'H₂·B₂', sub: 'partial Y' },
        { id: 'ar', x: 72, y: 24, w: 26, h: 12, label: 'all-reduce', sub: 'Y = Y₁ + Y₂' },
      ],
      edges: [
        { from: 'x', to: 'a1' },
        { from: 'x', to: 'a2' },
        { from: 'a1', to: 'b1' },
        { from: 'a2', to: 'b2' },
        { from: 'b1', to: 'ar' },
        { from: 'b2', to: 'ar' },
      ],
      steps: [
        { caption: 'Both GPUs hold the same input activations X. Each holds half of A\'s columns and half of B\'s rows — half the MLP\'s weights each.', active: ['x'] },
        { caption: 'Column-parallel first matmul: each GPU computes its own slice of the hidden layer and applies GELU locally. Nothing is exchanged.', active: ['x', 'a1', 'a2'], edges: ['x->a1', 'x->a2'] },
        { caption: 'Row-parallel second matmul: each GPU multiplies its hidden slice by the matching rows of B, producing a full-size but partial output.', active: ['a1', 'a2', 'b1', 'b2'], edges: ['a1->b1', 'a2->b2'] },
        { caption: 'One all-reduce sums the partial outputs. Every GPU now holds the exact Y — and the next layer starts from replicated activations again.', active: ['b1', 'b2', 'ar'], edges: ['b1->ar', 'b2->ar'] },
      ],
    },
    {
      type: 'prose',
      md: `## Why it stays inside a server

Each of those all-reduces carries a full activation tensor: B·T·C values. For one 4,096-token sequence at C = 8,192 in bf16 that is 67 MB, four times per layer per step (two forward, two backward), times 80 layers: about **21 GB per sequence per step**, on the critical path — the next layer cannot start until the all-reduce finishes, so there is little to overlap it with.

Over NVLink at roughly 900 GB/s per GPU (D1.L1) that is tolerable; over a 50 GB/s network link it would dominate the step. So TP is almost always limited to the GPUs of one server — typically **t ≤ 8** — and combined with data parallelism or ZeRO across servers. Sequence parallelism, a refinement, additionally shards the LayerNorm and dropout activations along the sequence axis and turns each all-reduce into a reduce-scatter plus an all-gather, saving activation memory for the same traffic.`,
    },
    {
      type: 'quiz',
      questions: [
        {
          q: 'Why is the first MLP matrix split by columns and the second by rows, rather than both by columns?',
          options: [
            'Columns are faster to transfer',
            'Column-splitting the first matmul gives each GPU a complete slice of the hidden layer, so the elementwise GELU needs no communication; row-splitting the second then turns the output into a sum of partials needing exactly one all-reduce',
            'It halves the parameter count',
            'Row-splitting is required for backward',
          ],
          correct: [1],
          explanation: 'GELU(Σ partials) ≠ Σ GELU(partials), so any split that leaves the hidden layer as a partial sum forces a synchronisation before the nonlinearity. Column-then-row keeps the hidden slice exact and defers all communication to one all-reduce at the end.',
        },
        {
          q: 'A team proposes tensor parallelism of degree 16 across two 8-GPU servers connected by 50 GB/s links. What is the main concern?',
          options: [
            'TP cannot exceed 8 by definition',
            'Every layer\'s activation all-reduces would cross the slow inter-server link on the critical path, so the step time would be dominated by communication; keep TP within the NVLink domain and use data/pipeline parallelism across servers',
            'The model would become less accurate',
            'Memory usage would double',
          ],
          correct: [1],
          explanation: 'TP communicates activation-sized tensors several times per layer and the next layer waits for them. That works at NVLink speed and fails at network speed. The standard layout puts TP inside a server and the more communication-tolerant schemes across servers (D1.L6).',
        },
        {
          q: 'How does tensor parallelism split multi-head attention?',
          options: [
            'By sequence position',
            'By heads: each GPU computes q, k, v and attention for a subset of heads (column-parallel projections), then the row-parallel output projection and one all-reduce combine them',
            'By layers',
            'It cannot be applied to attention',
          ],
          correct: [1],
          explanation: 'Heads are independent computations, so they are the natural sharding axis: no head needs another head\'s keys or values. The pattern is the same as the MLP — column-parallel in, row-parallel out, one all-reduce.',
        },
      ],
    },
    {
      type: 'deepdive',
      title: 'Going deeper',
      md: `- Shoeybi et al., "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism" (2019, arxiv.org/abs/1909.08053) — the column/row split, figure 3.
- Korthikanti et al. (2022) — sequence parallelism and its activation savings.
- PyTorch's \`torch.distributed.tensor.parallel\` (\`ColwiseParallel\`, \`RowwiseParallel\`) names the two halves exactly as this lesson does.`,
    },
  ],
}

export default lesson

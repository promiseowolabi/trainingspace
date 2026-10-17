import type { Lesson } from '../types'

const lesson: Lesson = {
  id: 'd1.l3',
  slug: 'zero-and-fsdp',
  trackId: 'd1',
  index: 3,
  title: 'ZeRO and FSDP',
  minutes: 21,
  hook: 'Data parallelism keeps N identical copies of 16 bytes per parameter. ZeRO keeps one copy, cut into N pieces: first the optimiser state, then the gradients, then the weights themselves — for the same communication as plain data parallelism, or 1.5× of it.',
  exercise: 'lab+quiz',
  takeaway: {
    number: '120 → 1.9 GB per GPU',
    claim: 'Sharding everything (ZeRO-3) over 64 GPUs cuts a 7.5B model\'s training state from 120 GB to 1.9 GB per GPU, for 1.5× the communication of data parallelism; sharding only the optimiser state (ZeRO-1) gets to 31.4 GB for no extra communication.',
  },
  blocks: [
    {
      type: 'prose',
      md: `D1.L2 ended with a problem: a 7B model needs 112 GB of training state per replica and an H100 has 80 GB. Data parallelism makes it worse, not better — 64 GPUs hold 64 identical copies. **ZeRO** (Zero Redundancy Optimizer, Rajbhandari et al. 2019) removes the redundancy in three stages. With Ψ parameters, N ranks and K = 12 bytes of optimiser state per parameter (fp32 master weight, m and v):

| stage | sharded | bytes per parameter per GPU | 7.5 B params, 64 GPUs |
|---|---|---|---|
| 0 — data parallel | nothing | 2 + 2 + 12 = 16 | **120 GB** |
| 1 — ZeRO-1 | optimiser state | 2 + 2 + 12/N | **31.4 GB** |
| 2 — ZeRO-2 | + gradients | 2 + 14/N | **16.6 GB** |
| 3 — ZeRO-3 / FSDP | + weights | 16/N | **1.9 GB** |

These are Figure 1 of the ZeRO paper, reproduced exactly by lab 11's \`memory_per_rank_gb\`. Activations are not included: they are per rank and depend on the micro-batch (T0.L4).`,
    },
    {
      type: 'prose',
      md: `## Stage 1: shard the optimiser

The optimiser state is the biggest term — 12 of 16 bytes — and it is only touched in the update. So split the parameters into N slices and let rank r own the optimiser state for slice r only. One step:

1. Forward and backward as in data parallelism, full weights on every rank.
2. **Reduce-scatter** the gradients: rank r receives the averaged gradient for its slice only.
3. Rank r updates its slice with its slice of m, v and the master weights.
4. **All-gather** the updated slices so every rank has the full new weights.

Steps 2 and 4 are exactly the two halves of the all-reduce data parallelism already performed (D1.L1). ZeRO-1 moves the optimiser step to the middle of it: **same bytes, 4× less memory** in the 64-GPU example. Stage 2 also discards the gradient outside each rank's slice after the reduce-scatter — again free.

Lab 11 implements stage 1 on simulated ranks and proves it computes exactly the same parameters as unsharded Adam.`,
    },
    {
      type: 'prose',
      md: `## Stage 3: shard the weights too

Now no rank holds the full model even between steps. Each layer's weights live as N slices; just before a layer runs, its slices are **all-gathered** into a temporary full copy, used, and freed:

- forward: all-gather layer k's weights, compute, free — layer by layer;
- backward: all-gather again (they were freed), compute gradients, **reduce-scatter** them to their owners, free.

Weights cross the network twice (forward and backward) and gradients once: **3Ψ** per step against data parallelism's 2Ψ — 1.5× the communication for 16/N of the memory. Prefetching the next layer's all-gather while the current one computes hides much of it. PyTorch's implementation is **FSDP** (Fully Sharded Data Parallel); DeepSpeed's is ZeRO stage 3. Same idea, different APIs.`,
    },
    {
      type: 'diagram',
      caption: 'fig 1 — what each rank holds',
      height: 62,
      nodes: [
        { id: 'dp', x: 2, y: 6, w: 22, h: 12, label: 'data parallel', sub: '16 B/param × N copies' },
        { id: 'z1', x: 28, y: 6, w: 22, h: 12, label: 'ZeRO-1', sub: 'optimiser ÷ N' },
        { id: 'z2', x: 54, y: 6, w: 20, h: 12, label: 'ZeRO-2', sub: '+ grads ÷ N' },
        { id: 'z3', x: 78, y: 6, w: 20, h: 12, label: 'ZeRO-3', sub: '+ weights ÷ N' },
        { id: 'c2', x: 28, y: 42, w: 46, h: 12, label: 'communication 2Ψ', sub: 'same as data parallel' },
        { id: 'c3', x: 78, y: 42, w: 20, h: 12, label: '3Ψ', sub: '1.5×' },
      ],
      edges: [
        { from: 'dp', to: 'z1' },
        { from: 'z1', to: 'z2' },
        { from: 'z2', to: 'z3' },
        { from: 'z1', to: 'c2' },
        { from: 'z3', to: 'c3' },
      ],
      steps: [
        { caption: 'Plain data parallelism stores weights, gradients and optimiser state in full on every rank: 16 bytes per parameter, N times over.', active: ['dp'] },
        { caption: 'ZeRO-1 shards the 12 bytes of optimiser state. The gradient all-reduce is split into a reduce-scatter and an all-gather around the sharded update — no extra bytes.', active: ['dp', 'z1', 'c2'], edges: ['dp->z1', 'z1->c2'] },
        { caption: 'ZeRO-2 keeps only each rank\'s slice of the reduced gradient. Still the same two collectives, still 2Ψ.', active: ['z1', 'z2', 'c2'], edges: ['z1->z2'] },
        { caption: 'ZeRO-3 shards the weights as well and gathers each layer just in time, in forward and again in backward: 16/N bytes per parameter for 3Ψ of traffic.', active: ['z2', 'z3', 'c3'], edges: ['z2->z3', 'z3->c3'] },
      ],
    },
    {
      type: 'callout',
      variant: 'isomorphism',
      title: 'Sharding a table you already know',
      md: `A database that replicates a whole table to every node reads fast and wastes storage; one that shards it stores one copy and pays a network hop to fetch a row it does not own. ZeRO-1/2 are replication for reads (weights) with sharding for the cold data (optimiser state, touched once per step). ZeRO-3 shards everything and fetches each layer on demand — and prefetching the next layer is the read-ahead of a sequential scan.`,
    },
    {
      type: 'lab',
      lab: 'zero-sharding',
      brief: 'Shard Adam\'s state across simulated ranks, reduce-scatter the gradients, update only your slice, all-gather the result — and prove it matches unsharded Adam while reproducing the ZeRO paper\'s 120 / 31.4 / 16.6 / 1.9 GB.',
    },
    {
      type: 'quiz',
      questions: [
        {
          q: 'A 70B model must be fully fine-tuned on 64 H100s (80 GB). Which ZeRO stage is the minimum that fits the training state, ignoring activations?',
          options: [
            'Stage 0 — 64 GPUs is plenty',
            'Stage 1: (4 + 12/64) × 70 B ≈ 293 GB — does not fit',
            'Stage 2: (2 + 14/64) × 70 B ≈ 155 GB — does not fit; stage 3 is needed: 16/64 × 70 B = 17.5 GB',
            'No stage can fit it',
          ],
          correct: [2],
          explanation: 'Stages 1 and 2 still keep a full copy of the bf16 weights (and, for stage 1, the gradients) on every GPU: 140 GB of weights alone exceeds 80 GB. Only stage 3 shards the weights: 70 × 10⁹ × 16 / 64 = 17.5 GB, leaving room for activations.',
        },
        {
          q: 'Why does ZeRO-1 cost no more communication than data parallelism?',
          options: [
            'It compresses the gradients',
            'Data parallelism\'s all-reduce is already a reduce-scatter followed by an all-gather; ZeRO-1 performs the optimiser update between the two halves and gathers updated weights instead of reduced gradients — the same bytes',
            'It skips the gradient synchronisation',
            'It only communicates every other step',
          ],
          correct: [1],
          explanation: 'all-reduce = reduce-scatter + all-gather (D1.L1). ZeRO-1 keeps both halves and moves the work between them, so the 2Ψ traffic is unchanged while the 12 bytes per parameter of optimiser state are split N ways.',
        },
        {
          q: 'In ZeRO-3/FSDP, why are a layer\'s weights all-gathered twice per step?',
          options: [
            'Once for each micro-batch',
            'They are freed after the forward pass to save memory, and the backward pass needs them again to compute gradients',
            'Once in fp32 and once in bf16',
            'For redundancy against failures',
          ],
          correct: [1],
          explanation: 'Holding gathered weights from forward until backward would bring back most of the memory ZeRO-3 removed. So each layer is gathered, used and freed in forward, and gathered again in backward — the extra Ψ of traffic that makes stage 3 cost 3Ψ instead of 2Ψ.',
        },
      ],
    },
    {
      type: 'deepdive',
      title: 'Going deeper',
      md: `- Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models" (2019, arxiv.org/abs/1910.02054).
- Zhao et al., "PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel" (2023, arxiv.org/abs/2304.11277).
- ZeRO-Offload and ZeRO-Infinity push optimiser state to CPU memory or NVMe — the same sharding, with a slower tier.`,
    },
  ],
}

export default lesson

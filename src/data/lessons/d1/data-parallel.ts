import type { Lesson } from '../types'

const lesson: Lesson = {
  id: 'd1.l2',
  slug: 'data-parallel',
  trackId: 'd1',
  index: 2,
  title: 'Data Parallelism',
  minutes: 19,
  hook: 'Give every GPU a full copy of the model and a different slice of the batch, average the gradients, take the same step everywhere. It is the simplest way to use N devices, the maths is exactly single-device training — and it does nothing at all for memory.',
  exercise: 'lab+quiz',
  takeaway: {
    number: '28 GB per rank per step',
    claim: 'Data-parallel training of a 7B model all-reduces 14 GB of bf16 gradients every step, so each rank sends about 28 GB — 0.56 s on a 50 GB/s link, which must hide behind the backward pass or it eats the step.',
  },
  blocks: [
    {
      type: 'prose',
      md: `## The algorithm

N ranks, each with a full replica of the model:

1. **Broadcast** the initial weights from rank 0, so every replica starts identical.
2. Each step, split the global batch into N shards; rank r runs forward and backward on shard r.
3. **All-reduce** the gradients and divide by N: every rank now holds the average gradient over the whole batch.
4. Every rank applies the same optimiser step to the same weights with the same gradient — so the replicas stay identical without ever exchanging weights again.

Because the mean of per-shard mean gradients equals the full-batch mean gradient, this is **exactly** single-device training with batch N × B. Lab 10 checks it: two processes on 8 examples each must reproduce one process on 16 to within 10⁻⁶.`,
    },
    {
      type: 'diagram',
      caption: 'fig 1 — one data-parallel step on 2 ranks',
      height: 62,
      nodes: [
        { id: 'batch', x: 36, y: 2, w: 28, h: 10, label: 'global batch', sub: '16 examples' },
        { id: 'a', x: 6, y: 20, w: 24, h: 12, label: 'rank 0', sub: 'fwd + bwd, rows 0–7' },
        { id: 'b', x: 70, y: 20, w: 24, h: 12, label: 'rank 1', sub: 'fwd + bwd, rows 8–15' },
        { id: 'ar', x: 36, y: 34, w: 28, h: 10, label: 'all-reduce / N', sub: 'mean gradient' },
        { id: 'sa', x: 6, y: 48, w: 24, h: 12, label: 'same step', sub: 'rank 0' },
        { id: 'sb', x: 70, y: 48, w: 24, h: 12, label: 'same step', sub: 'rank 1' },
      ],
      edges: [
        { from: 'batch', to: 'a' },
        { from: 'batch', to: 'b' },
        { from: 'a', to: 'ar' },
        { from: 'b', to: 'ar' },
        { from: 'ar', to: 'sa' },
        { from: 'ar', to: 'sb' },
      ],
      steps: [
        { caption: 'The global batch is split by rows. Each rank holds the complete model, so no activations ever cross between ranks.', active: ['batch', 'a', 'b'], edges: ['batch->a', 'batch->b'] },
        { caption: 'Each rank computes the gradient of its own shard\'s loss. The two gradients differ — they saw different data.', active: ['a', 'b'] },
        { caption: 'An all-reduce sums the gradients and each rank divides by N. Both now hold the gradient one device would have computed on all 16 examples.', active: ['a', 'b', 'ar'], edges: ['a->ar', 'b->ar'] },
        { caption: 'Identical weights plus identical gradients give identical updates: the replicas stay in lock-step without ever sending weights.', active: ['ar', 'sa', 'sb'], edges: ['ar->sa', 'ar->sb'] },
      ],
    },
    {
      type: 'prose',
      md: `## What it costs

**Communication.** One all-reduce of every gradient, every step: for N parameters in bf16, 2·N bytes of gradient and about 2 × that sent per rank (D1.L1). A 7B model: 14 GB of gradients, **about 28 GB sent per rank per step**. On a 50 GB/s inter-server link that is 0.56 s. A step processing 32,768 tokens per GPU costs 6 × 7 × 10⁹ × 32,768 ≈ 1.4 × 10¹⁵ FLOPs — about 3.5 s on an H100 at 40% MFU. Done naively after the backward pass, communication adds 16% to every step.

**Overlap** hides it. Gradients for the last layers are ready first, so DDP all-reduces each ~25 MB bucket the moment its gradients are complete, while the backward pass continues into earlier layers. Lab 10 builds the buckets; real DDP also fires them from backward hooks. Well-overlapped data parallelism scales close to linearly — until the batch gets too large.

**Memory: nothing saved.** Every rank holds the full 16 bytes per parameter plus activations for its shard. A 7B model needs 112 GB of training state per GPU (T0.L4), more than an 80 GB H100 has — so plain data parallelism cannot train it at all. D1.L3 fixes that by sharding.`,
    },
    {
      type: 'statline',
      stats: [
        { value: '14 GB', label: '7B gradients, bf16', hint: '7 × 10⁹ × 2 bytes, all-reduced every step' },
        { value: '~28 GB', label: 'sent per rank', hint: '2(N − 1)/N × 14 GB for large N' },
        { value: '0.56 s', label: 'at 50 GB/s', hint: '400 Gb/s per GPU — a typical inter-server link' },
        { value: '10⁻⁶', label: 'lab 10 agreement', hint: '2 processes × 8 examples vs 1 process × 16, after 5 SGD steps' },
      ],
    },
    {
      type: 'prose',
      md: `## The batch-size ceiling

Data parallelism scales by making the global batch bigger: N ranks × B per rank × accumulation steps (P1.L2). That works only up to a point. Beyond a model- and data-dependent **critical batch size**, doubling the batch no longer halves the number of steps needed — each step's gradient is already a good estimate of the true gradient, and averaging more examples adds little. Past that point extra ranks burn compute without speeding up training. This is one reason large runs combine data parallelism with the model-splitting schemes of D1.L4 and D1.L5 rather than adding replicas forever.`,
    },
    {
      type: 'lab',
      lab: 'ddp-from-scratch',
      brief: 'Two real processes over torch.distributed\'s gloo backend: broadcast the initial weights, shard the batch, average gradients per tensor and then in DDP-style buckets — and match single-process training to 10⁻⁶.',
    },
    {
      type: 'quiz',
      questions: [
        {
          q: 'A team moves a run from 1 GPU with batch 32 to 8 GPUs with 32 per GPU and changes nothing else. What changed about the training maths?',
          options: [
            'Nothing — data parallelism is invisible',
            'The global batch is now 256, so each step averages 8× more examples; the learning rate and schedule tuned for batch 32 may no longer be right, and the run takes 8× fewer steps per epoch',
            'Each GPU now trains a different model',
            'The gradients are now 8× larger',
          ],
          correct: [1],
          explanation: 'Averaging over ranks makes the effective batch N × B. Gradients are averaged, not summed, so they are not 8× larger, but the step count per epoch and the noise per step change, and learning-rate recipes are tied to the batch size. Data parallelism is exact for the batch it computes, which is not the batch you had before.',
        },
        {
          q: 'Why do data-parallel replicas never need to exchange weights after the first broadcast?',
          options: [
            'They do — weights are all-reduced every step',
            'Identical starting weights, an identical averaged gradient and an identical deterministic optimiser give identical new weights on every rank',
            'Only rank 0 updates its weights',
            'Weights are stored on a parameter server',
          ],
          correct: [1],
          explanation: 'The update is a deterministic function of weights, gradient and optimiser state, all of which are equal across ranks after the all-reduce. That is why the initial broadcast matters so much: if replicas start different, they stay different.',
        },
        {
          q: 'Your 7B data-parallel run spends 0.56 s per step in gradient all-reduce and 3.5 s computing. Profiling shows the all-reduce starts only after the whole backward pass ends. What is the main improvement available?',
          options: [
            'Use fp32 gradients',
            'Overlap: launch each bucket\'s all-reduce as soon as its gradients are ready during backward, hiding most of the 0.56 s behind the remaining computation',
            'Reduce the number of ranks',
            'Make buckets as small as possible',
          ],
          correct: [1],
          explanation: 'Gradients for later layers are finished early in the backward pass. Bucketed, hook-driven all-reduces overlap communication with the computation still running, which is how DDP keeps scaling efficiency high. Tiny buckets would add latency instead.',
        },
      ],
    },
    {
      type: 'deepdive',
      title: 'Going deeper',
      md: `- Li et al., "PyTorch Distributed: Experiences on Accelerating Data Parallel Training" (VLDB 2020, arxiv.org/abs/2006.15704) — buckets, hooks and overlap in DDP.
- McCandlish et al., "An Empirical Model of Large-Batch Training" (2018, arxiv.org/abs/1812.06162) — the critical batch size.
- PyTorch's DDP tutorial: \`torchrun\` launches the same processes lab 10's checks spawn by hand.`,
    },
  ],
}

export default lesson

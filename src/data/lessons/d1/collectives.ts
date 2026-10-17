import type { Lesson } from '../types'

const lesson: Lesson = {
  id: 'd1.l1',
  slug: 'collectives',
  trackId: 'd1',
  index: 1,
  title: 'Collectives',
  minutes: 19,
  hook: 'Distributed training is ordinary training plus a handful of group operations — all-reduce, reduce-scatter, all-gather, broadcast. Learn what each one moves and what it costs, and every parallelism scheme in this track becomes a choice of which collective to call when.',
  exercise: 'lab+quiz',
  takeaway: {
    number: '2(N − 1)/N',
    claim: 'A ring all-reduce sends 2(N − 1)/N of the data per rank — never more than twice the array, however many ranks there are — so its time is set by the slowest link, plus a latency per step that dominates small messages.',
  },
  blocks: [
    {
      type: 'prose',
      md: `When N devices cooperate on one training step they need to combine data. Point-to-point sends are too low-level to reason about; instead, every framework exposes a few **collectives** — operations every rank calls together, with a defined result:

| collective | before | after | used for |
|---|---|---|---|
| **broadcast** | rank 0 has x | every rank has x | initial weights (D1.L2) |
| **all-reduce** | rank r has xᵣ | every rank has Σ xᵣ | averaging gradients (D1.L2) |
| **reduce-scatter** | rank r has xᵣ | rank r has chunk r of Σ xᵣ | sharded gradients (D1.L3) |
| **all-gather** | rank r has chunk r | every rank has all chunks | sharded weights (D1.L3) |

The last two compose into the second: **all-reduce = reduce-scatter + all-gather**. That identity is the hinge of ZeRO.`,
    },
    {
      type: 'prose',
      md: `## The ring

Arrange N ranks in a circle; each talks only to its neighbour. Split every rank's array into N chunks.

- **Reduce-scatter**, N − 1 steps: each step, every rank passes one chunk's running sum to its neighbour, which adds its own contribution. After N − 1 steps, each chunk has visited every rank and rank r holds the complete sum of chunk r.
- **All-gather**, N − 1 more steps: each rank forwards finished chunks around the ring until everyone has all of them.

Per rank, that is 2(N − 1) messages of size 1/N of the array:

\`\`\`
bytes sent per rank = 2 · (N − 1) / N · array_bytes     → 1× at N = 2, 1.75× at N = 8, → 2× as N grows
\`\`\`

The total never exceeds twice the array, no matter how many ranks join. Lab 09 builds both phases on a simulated ring and measures the bytes.`,
    },
    {
      type: 'diagram',
      caption: 'fig 1 — ring reduce-scatter, 4 ranks',
      height: 62,
      nodes: [
        { id: 'r0', x: 4, y: 4, w: 18, h: 12, label: 'rank 0', sub: 'chunks a b c d' },
        { id: 'r1', x: 78, y: 4, w: 18, h: 12, label: 'rank 1', sub: 'chunks a b c d' },
        { id: 'r2', x: 78, y: 44, w: 18, h: 12, label: 'rank 2', sub: 'chunks a b c d' },
        { id: 'r3', x: 4, y: 44, w: 18, h: 12, label: 'rank 3', sub: 'chunks a b c d' },
        { id: 'sum', x: 34, y: 24, w: 32, h: 12, label: 'after 3 steps', sub: 'rank r owns Σ of chunk r' },
      ],
      edges: [
        { from: 'r0', to: 'r1' },
        { from: 'r1', to: 'r2' },
        { from: 'r2', to: 'r3' },
        { from: 'r3', to: 'r0' },
      ],
      steps: [
        { caption: 'Each rank holds a full gradient, split into four chunks. Every link carries one chunk per step, all four links in parallel.', active: ['r0', 'r1', 'r2', 'r3'] },
        { caption: 'Step 1: each rank sends one chunk to its right-hand neighbour, which adds its own copy of that chunk. Four partial sums of two contributions each now exist.', active: ['r0', 'r1', 'r2', 'r3'], edges: ['r0->r1', 'r1->r2', 'r2->r3', 'r3->r0'] },
        { caption: 'Steps 2 and 3 forward the running sums one hop further. Every chunk travels three hops and collects all four contributions on the way.', active: ['r0', 'r1', 'r2', 'r3'], edges: ['r0->r1', 'r1->r2', 'r2->r3', 'r3->r0'] },
        { caption: 'After N − 1 = 3 steps each rank owns one fully summed chunk. Three more steps of all-gather would give every rank all four — a complete all-reduce.', active: ['sum'] },
      ],
    },
    {
      type: 'prose',
      md: `## Latency and bandwidth

Each step costs a fixed **latency** α (start-up, synchronisation, network hops) plus the time to push one chunk at link bandwidth β:

\`\`\`
T_allreduce ≈ 2 · (N − 1) · (α + array_bytes / (N · β))
\`\`\`

For large arrays the second term dominates and the time is almost independent of N. For small arrays the first term dominates and grows with N. Measured with \`torch.distributed\` on the \`gloo\` backend, two processes on the reference laptop:

| array | time per all-reduce | effective rate |
|---|---|---|
| 4 KB | 0.14 ms | 0.03 GB/s |
| 1 MB | 0.42 ms | 2.4 GB/s |
| 100 MB | 17 ms | 5.9 GB/s |

The 4 KB all-reduce moved data 200 times slower than the 100 MB one: it was all latency. That is why DDP packs gradients into buckets of about 25 MB instead of reducing each tensor separately (lab 10).`,
    },
    {
      type: 'vendor',
      snapshot: '2026-10',
      title: 'Real interconnects',
      md: `Within one server, NVIDIA H100 SXM GPUs are connected by **NVLink**, about **900 GB/s** of bandwidth per GPU. Between servers, clusters typically use InfiniBand or RoCE at **400 Gb/s (50 GB/s) per network port**, usually one port per GPU — roughly an order of magnitude less than NVLink. This gap shapes every layout decision in D1.L6: the parallelism that communicates most (tensor parallelism) stays inside a server; the one that communicates least per step (data parallelism) spans servers. NCCL, NVIDIA's collectives library, chooses ring or tree algorithms per message size and topology.`,
      systems: ['nvidia-h100', 'nccl'],
      sources: ['https://www.nvidia.com/en-us/data-center/h100/', 'https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/'],
    },
    {
      type: 'lab',
      lab: 'collectives',
      brief: 'Build ring reduce-scatter, all-gather and all-reduce on a simulated ring that only lets each rank talk to its neighbour, then measure that every rank sends exactly 2(N − 1)/N of the data — and model when latency dominates.',
    },
    {
      type: 'quiz',
      questions: [
        {
          q: 'You double the number of data-parallel ranks from 64 to 128. Roughly what happens to the bytes each rank sends in a ring all-reduce of the same gradient?',
          options: ['They double', 'They halve', 'They stay almost the same: 2·63/64 ≈ 1.97× versus 2·127/128 ≈ 1.98× the gradient', 'They grow by 64×'],
          correct: [2],
          explanation: 'The ring\'s per-rank traffic approaches twice the array size and stops growing. What does grow is the number of steps, 2(N − 1), so the latency term doubles — negligible for large buckets, painful for tiny ones.',
        },
        {
          q: 'A team all-reduces each of their model\'s 400 parameter tensors separately, many of them a few kilobytes. Communication takes far longer than the bandwidth suggests. Why, and what is the fix?',
          options: [
            'Bandwidth is lower for small tensors — buy faster network cards',
            'Each small all-reduce is latency-bound: 2(N − 1) steps of fixed cost regardless of size. Concatenate gradients into large buckets so the latency is paid a few times instead of 400',
            'All-reduce cannot handle small tensors',
            'Use broadcast instead of all-reduce',
          ],
          correct: [1],
          explanation: 'The time model has a latency term multiplied by the number of steps, independent of message size. Measured locally, a 4 KB all-reduce achieved 0.03 GB/s against 5.9 GB/s for 100 MB. Bucketing — DDP\'s default is about 25 MB — amortises the latency.',
        },
        {
          q: 'Which identity makes ZeRO possible without extra communication compared with data parallelism?',
          options: [
            'broadcast = all-gather',
            'all-reduce = reduce-scatter + all-gather: splitting the all-reduce lets each rank stop after the reduce-scatter, update only its shard, and then all-gather the updated weights instead of the gradients',
            'reduce-scatter is free',
            'all-gather is the inverse of broadcast',
          ],
          correct: [1],
          explanation: 'Data parallelism already pays a reduce-scatter and an all-gather inside its all-reduce. ZeRO-1 and -2 rearrange the same two halves around a sharded optimiser step, so they cost the same bytes as DDP (D1.L3).',
        },
      ],
    },
    {
      type: 'deepdive',
      title: 'Going deeper',
      md: `- Patarasuk and Yuan, "Bandwidth optimal all-reduce algorithms for clusters of workstations" (2009) — the ring's optimality proof.
- NCCL's \`nccl-tests\` README explains "algorithm bandwidth" versus "bus bandwidth" — the 2(N − 1)/N factor, as a benchmark convention.
- Run lab 10, then time \`dist.all_reduce\` yourself on 2 and 4 local processes for sizes from 4 KB to 100 MB.`,
    },
  ],
}

export default lesson

import type { Lesson } from '../types'

const lesson: Lesson = {
  id: 'd1.l6',
  slug: 'choosing-a-layout',
  trackId: 'd1',
  index: 6,
  title: 'Choosing a Layout',
  minutes: 21,
  hook: 'Real runs combine everything: tensor parallelism inside a server, pipeline stages across servers, data parallelism with sharded optimiser state across the rest, and activation checkpointing wherever memory is still short. Choosing the numbers is a memory budget and a communication budget, solved together.',
  exercise: 'read+quiz',
  takeaway: {
    number: 'TP 8 × PP 16 × DP 128 = 16,384',
    claim: 'Llama 3 405B was trained on 16,384 GPUs as 8-way tensor × 16-way pipeline × 128-way data parallelism: the most communication-hungry split inside each 8-GPU server, the least across the whole cluster.',
  },
  blocks: [
    {
      type: 'prose',
      md: `Each scheme in this track splits something different and pays in a different currency:

| scheme | splits | saves memory on | communicates | over |
|---|---|---|---|---|
| data parallel (DP) | the batch | nothing | gradients, once per step (2Ψ) | any network, overlapped |
| ZeRO-1/2 | optimiser state, gradients | 12–14 of 16 bytes | same as DP | any network |
| ZeRO-3 / FSDP | weights too | all 16 bytes | 1.5 × DP | fast network, prefetched |
| tensor parallel (TP) | each layer's matrices | weights *and* activations | activations, every layer, critical path | NVLink only |
| pipeline parallel (PP) | the stack of layers | weights | boundary activations, point-to-point | any network; costs a bubble |

Combining them is called **3D parallelism** (DP × TP × PP); add context parallelism — splitting long sequences across GPUs for attention — and it becomes 4D. The product of the degrees is the number of GPUs.`,
    },
    {
      type: 'prose',
      md: `## The fourth tool: activation checkpointing

Everything above shards weights and optimiser state. Activations — the T0.L4 term that grows with micro-batch and sequence length — need their own tool. **Activation checkpointing** (also called recomputation) keeps only each block's *input* during the forward pass and discards everything inside the block. In the backward pass, each block's forward is **re-run** from its saved input to regenerate what the gradients need.

Cost: roughly one extra forward pass, about a third more compute. Measured on lab 07's model scaled to 8 layers of width 256 (batch 16 × 256 tokens, PyTorch on the reference laptop): **229 ms per step without, 317 ms with checkpointing every block — 38% slower.** Memory: activations shrink from "everything inside every block" to "one tensor per block boundary", often several-fold. Selective checkpointing recomputes only the cheap but memory-hungry parts, such as attention scores, for most of the saving at a fraction of the cost.`,
    },
    {
      type: 'prose',
      md: `## A worked layout: 70B on 64 GPUs

Eight servers of eight H100s (80 GB each), a 70B model, sequences of 4,096 tokens.

1. **Does plain DP fit?** 70 × 10⁹ × 16 bytes = 1.12 TB per replica. No.
2. **TP = 8 inside each server.** Each GPU holds 1/8 of every layer: bf16 weights 17.5 GB, gradients 17.5 GB. TP's all-reduces stay on NVLink.
3. **DP = 8 across servers, with ZeRO-1.** The 12 bytes of optimiser state are split over TP × DP = 64 GPUs: 70 × 10⁹ × 12 / 64 = 13.1 GB.
4. **Total so far**: 17.5 + 17.5 + 13.1 ≈ **48 GB**, leaving about 32 GB for activations and buffers. With sequence parallelism and selective checkpointing that is workable; if not, add PP = 2 (and halve DP) or move to ZeRO-3.
5. **Check the bubble and the batch.** No pipeline, so no bubble. DP = 8 with, say, 4 micro-batches of one sequence each per replica gives a global batch of 32 × 4,096 ≈ 131k tokens — modest for a 70B run, so gradient accumulation can raise it.

Every number is lab 08 arithmetic. The art is in the order: put TP where the bandwidth is, use PP only when memory demands it, give everything else to DP, and shard optimiser state across all of it.`,
    },
    {
      type: 'diagram',
      caption: 'fig 1 — mapping a layout onto a cluster',
      height: 64,
      nodes: [
        { id: 'gpu', x: 2, y: 6, w: 22, h: 12, label: 'GPU', sub: '80 GB' },
        { id: 'node', x: 30, y: 6, w: 24, h: 12, label: 'server: 8 GPUs', sub: 'NVLink ~900 GB/s' },
        { id: 'tp', x: 30, y: 40, w: 24, h: 12, label: 'TP = 8', sub: 'per-layer all-reduces' },
        { id: 'cluster', x: 62, y: 6, w: 36, h: 12, label: 'cluster', sub: 'network ~50 GB/s per GPU' },
        { id: 'pp', x: 62, y: 32, w: 16, h: 12, label: 'PP', sub: 'boundaries' },
        { id: 'dp', x: 82, y: 32, w: 16, h: 12, label: 'DP + ZeRO', sub: 'gradients' },
      ],
      edges: [
        { from: 'gpu', to: 'node' },
        { from: 'node', to: 'tp' },
        { from: 'node', to: 'cluster' },
        { from: 'cluster', to: 'pp' },
        { from: 'cluster', to: 'dp' },
      ],
      steps: [
        { caption: 'Start from the memory of one GPU: weights, gradients, optimiser shards and activations for its micro-batch must fit in 80 GB.', active: ['gpu'] },
        { caption: 'Inside a server, GPUs share NVLink at about 900 GB/s each — the only place fast enough for tensor parallelism\'s per-layer all-reduces.', active: ['gpu', 'node', 'tp'], edges: ['gpu->node', 'node->tp'] },
        { caption: 'Between servers, each GPU gets roughly 50 GB/s. Pipeline stages exchange only boundary activations, so PP can span servers when memory requires it.', active: ['node', 'cluster', 'pp'], edges: ['node->cluster', 'cluster->pp'] },
        { caption: 'Data parallelism takes all remaining GPUs: its one gradient all-reduce per step overlaps with backward, and ZeRO shards optimiser state across the whole cluster.', active: ['cluster', 'dp'], edges: ['cluster->dp'] },
      ],
    },
    {
      type: 'vendor',
      snapshot: '2026-10',
      title: 'A published layout',
      md: `Meta's Llama 3 paper reports training the 405B model on up to **16,384 H100 GPUs** with 4D parallelism: **tensor parallelism 8** within each server, **pipeline parallelism 16**, **data parallelism 128** (with FSDP-style sharding of optimiser state and gradients), and context parallelism added for the long-context stage. They report **38–43% bf16 MFU** for this setup.`,
      systems: ['llama-3'],
      sources: ['https://arxiv.org/abs/2407.21783'],
    },
    {
      type: 'quiz',
      questions: [
        {
          q: 'A team has 32 GPUs (4 servers × 8) and a 13B model. Training state is 208 GB. Which layout is the most natural first try?',
          options: [
            'TP = 32 across all servers',
            'PP = 32, one layer per GPU',
            'DP = 32 with ZeRO-3/FSDP (208 GB / 32 ≈ 6.5 GB per GPU), or TP = 2–4 within servers plus DP with ZeRO-1 if activations are tight',
            'Plain DP = 32',
          ],
          correct: [2],
          explanation: '13B is small enough that sharding alone solves memory: FSDP across 32 GPUs puts 6.5 GB of state on each. TP across servers would put per-layer all-reduces on the slow network; a 32-stage pipeline would have a huge bubble; plain DP needs 208 GB per GPU.',
        },
        {
          q: 'Activation checkpointing every block made lab 07\'s scaled-up model 38% slower per step. Why would a team accept that?',
          options: [
            'It improves accuracy',
            'It cuts activation memory to one tensor per block boundary, which can allow a larger micro-batch, a longer sequence, or fewer model-parallel GPUs — often worth more throughput than the recomputation costs',
            'It reduces communication',
            'It is required by ZeRO',
          ],
          correct: [1],
          explanation: 'Recomputation trades about one extra forward pass for most of the activation memory. If that memory lets you drop a pipeline stage (removing a bubble) or double the micro-batch (raising arithmetic intensity), the run can end up faster overall.',
        },
        {
          q: 'Why is TP usually the innermost dimension (within a server) and DP the outermost (across the cluster)?',
          options: [
            'Convention only',
            'TP communicates activations in every layer on the critical path and needs NVLink bandwidth; DP communicates once per step, overlapped with backward, and tolerates the slower network — so each goes where its traffic fits',
            'DP cannot run inside a server',
            'TP needs more memory',
          ],
          correct: [1],
          explanation: 'Layouts are matched to bandwidth: the most frequent, latency-critical traffic on the fastest links. Swapping them would put per-layer all-reduces on 50 GB/s links and waste NVLink on infrequent gradient syncs.',
        },
      ],
    },
    {
      type: 'deepdive',
      title: 'Going deeper',
      md: `- Narayanan et al. (2021, arxiv.org/abs/2104.04473) — section 3 derives the trade-offs between TP, PP and DP that this lesson summarises.
- Chen et al., "Training Deep Nets with Sublinear Memory Cost" (2016) — activation checkpointing.
- The Hugging Face "Ultra-Scale Playbook" — a long, practical walk through every parallelism dimension with measurements.
- This closes D1. F1 turns the pretrained model into an assistant.`,
    },
  ],
}

export default lesson

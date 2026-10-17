import type { LabGuide } from './types'

const guide: LabGuide = {
  id: 'zero-sharding',
  requires: ['a browser — or Python 3.10+ with numpy and pytest'],
  blocks: [
    {
      type: 'prose',
      md: `## What you will build

ZeRO stage 1 on simulated ranks: the optimiser state — Adam's m and v, 8 of T0.L4's 16 bytes per parameter — is split across N ranks so each holds 1/N of it. The forward and backward passes are unchanged from data parallelism. Only the update changes: each rank updates only its slice of the parameters, and the slices are then exchanged.

The collectives (\`reduce_scatter_mean\`, \`all_gather\`) and the Adam formula (\`adam_update\`) are given. The model is linear regression on a flat weight vector, so the checks can compare your sharded training with plain Adam exactly.`,
    },
    {
      type: 'prose',
      md: `## 1. shard, unshard, ShardState

\`np.array_split\` decides the shard sizes (uneven when N does not divide the length: 10 over 4 is 3, 3, 2, 2). \`ShardState\` sizes its m and v from the same split, so rank r's state matches rank r's shard exactly.

## 2. zero_step

Three moves, matching the docstring:

1. \`reduce_scatter_mean(grads, n)\` — rank r receives the averaged gradient for its shard only. (In DDP it would receive the whole averaged gradient; reduce-scatter is half an all-reduce, lab 09.)
2. Each rank takes **its** shard of **its** parameters with \`shard(params[r], n)[r]\`, bumps its step counter, and calls \`adam_update\` on that shard with its own m and v.
3. \`all_gather\` the updated shards and write the result into every \`params[r]\` with \`params[r][...] = full\`, so each rank's buffer is updated in place.

\`matches-unsharded\` runs 25 steps on 4 ranks and compares with Adam on the mean gradient: the update is the same arithmetic, just distributed.`,
    },
    {
      type: 'prose',
      md: `## 3. The accounting

\`memory_per_rank_gb\` is the four-row formula from D1.L3; the check reproduces Figure 1 of the ZeRO paper (7.5 B parameters on 64 GPUs: 120, 31.4, 16.6 and 1.9 GB). \`comm_volume\` is one line: stages 0–2 move 2Ψ elements per rank per step, stage 3 moves 3Ψ.`,
    },
  ],
  cleanup: [
    {
      type: 'prose',
      md: `- Extend \`zero_step\` to stage 2: keep only the reduce-scattered gradient shard instead of the full gradient. What changes in memory, and what does not change in the result?
- Use \`memory_per_rank_gb\` to find the smallest number of 80 GB GPUs on which a 70 B model's training state fits at each stage (ignoring activations).
- Read PyTorch's FSDP2 documentation (\`fully_shard\`): it is ZeRO-3 with per-module all-gathers.`,
    },
  ],
}

export default guide

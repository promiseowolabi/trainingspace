import type { LabGuide } from './types'

const guide: LabGuide = {
  id: 'ddp-from-scratch',
  requires: ['Python 3.10+', 'numpy, pytest', 'PyTorch (CPU build is enough)'],
  blocks: [
    {
      type: 'prose',
      md: `## What you will build

Data parallelism as PyTorch's \`DistributedDataParallel\` does it, minus the engineering: every rank holds a full copy of the model, processes its own slice of each batch, and averages gradients with an all-reduce before the identical optimiser step. The checks start **two real Python processes** with \`multiprocessing\`, connect them through \`torch.distributed\` on the \`gloo\` backend over 127.0.0.1, run your functions in both, and compare the final weights with a single process that trained on the whole batch.

Everything you write here would run unchanged on a GPU cluster with \`nccl\` instead of \`gloo\`.`,
    },
    {
      type: 'prose',
      md: `## 1. setup and shard_batch

\`setup\` sets two environment variables and calls \`init_process_group\`; every rank calls it with the same port. \`shard_batch\` is slicing: B / W rows per rank, in order.

## 2. broadcast_params

The ranks build their models from different random seeds in the checks — rank 1's weights start different on purpose. One \`dist.broadcast(p.data, src=0)\` per parameter makes them identical. Without it the replicas train different models from the first step, and averaging their gradients is meaningless.

## 3. allreduce_grads

\`dist.all_reduce\` sums in place across ranks; divide by the world size to get the mean. With equal shard sizes and a mean-reduced loss, the mean of the per-rank gradients **is** the full-batch gradient, so \`matches-single-process\` demands agreement to 10⁻⁶ after five SGD steps.`,
    },
    {
      type: 'callout',
      variant: 'segfault',
      title: 'Hangs are collective-order bugs',
      md: `A collective completes only when every rank has called it. If rank 0 skips a parameter whose \`.grad\` is \`None\` and rank 1 does not, rank 1 waits forever. The checks give up after 90 s and report "hung". Make every rank follow exactly the same sequence of collectives — iterate parameters in the same order, with the same conditions, on every rank.`,
    },
    {
      type: 'prose',
      md: `## 4. Buckets

One all-reduce per parameter tensor means hundreds of small messages per step, each paying a latency (lab 09's \`ring_time\`). DDP instead concatenates gradients into buckets of about 25 MB and reduces each bucket once. \`plan_buckets\` is the grouping rule; \`allreduce_grads_bucketed\` flattens each bucket with \`torch.cat\`, reduces it, divides, and copies the slices back with \`copy_\` and \`view_as\`. The last check counts your all-reduce calls for three bucket sizes.

## 5. ddp_step

Shard, forward, backward, average, step, zero. Every rank ends the step with the same weights.`,
    },
  ],
  cleanup: [
    {
      type: 'prose',
      md: `- Wrap the same model in \`torch.nn.parallel.DistributedDataParallel\` inside a spawned worker and check it matches your version.
- Real DDP overlaps communication with the backward pass: it launches a bucket's all-reduce as soon as the hooks report all of that bucket's gradients ready. Add gradient hooks (\`p.register_post_accumulate_grad_hook\`) and reduce buckets as they fill.
- With 4 processes instead of 2, does anything in your code change? (It should not.)`,
    },
  ],
}

export default guide

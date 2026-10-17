import type { LabGuide } from './types'

const guide: LabGuide = {
  id: 'collectives',
  requires: ['a browser — or Python 3.10+ with numpy and pytest'],
  blocks: [
    {
      type: 'prose',
      md: `## What you will build

The algorithm under every data-parallel training run: the ring all-reduce of D1.L1. A given \`Ring\` class simulates N ranks wired in a circle. Its one method, \`step\`, delivers one array from every sending rank to its right-hand neighbour and records the bytes. You may not read another rank's buffers directly — everything must cross the network through \`step\`, so the traffic the checks measure is the traffic your algorithm would really send.`,
    },
    {
      type: 'prose',
      md: `## 1. Reduce-scatter

Keep one list of chunk buffers per rank: \`bufs[r] = [c.copy() for c in chunks_of(xs[r], n)]\`. Then N − 1 steps. In step s, rank r sends \`bufs[r][(r - s - 1) % n]\`. The receiver \`dst\` knows its sender is \`(dst - 1) % n\`, so it can work out which chunk index arrived and add the array into its own buffer for that chunk.

Trace chunk c on paper for N = 4: rank c + 1 sends it at step 0, rank c + 2 adds its contribution and sends it on at step 1, and so on, until the sum of all four contributions arrives at rank c at step N − 2. That is why the offset is −1: it makes rank r finish owning chunk r.

## 2. All-gather

Rank r starts with \`{r: chunks[r]}\`. In step s it forwards chunk \`(r - s) % n\` — its own at step 0, then whatever it received at the previous step. After N − 1 steps every rank has every chunk; return each rank's chunks concatenated in index order.

## 3. All-reduce, and the arithmetic

\`ring_all_reduce\` is the two calls in sequence. Then two formulas from D1.L1: bytes per rank, and the time model where every one of the 2(N − 1) steps pays a latency plus one chunk at the link bandwidth.`,
    },
    {
      type: 'callout',
      variant: 'info',
      title: 'What the traffic check proves',
      md: `\`ring-traffic\` runs all-reduce on 2, 4 and 8 ranks and requires every rank to have sent exactly 2(N − 1)/N times the array size — 1.0× for N = 2, 1.5× for 4, 1.75× for 8. The cost per rank barely grows with N. A naive scheme in which every rank sends its whole array to every other rank would cost (N − 1)× instead.`,
    },
  ],
  cleanup: [
    {
      type: 'prose',
      md: `- Use \`ring_time\` to compare one 14 GB all-reduce with 10,000 all-reduces of 1.4 MB each on 64 ranks: the second pays 10,000 × 2 × 63 latencies. That is the case for bucketing (lab 10).
- Implement a naive all-reduce on the same \`Ring\` (every rank's array goes all the way round) and compare \`ring.sent\`.
- Lab 10 replaces the simulator with real processes and \`torch.distributed\`.`,
    },
  ],
}

export default guide

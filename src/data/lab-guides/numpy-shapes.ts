import type { LabGuide } from './types'

const guide: LabGuide = {
  id: 'numpy-shapes',
  requires: ['a browser — or Python 3.10+ with numpy and pytest'],
  blocks: [
    {
      type: 'prose',
      md: `## What you will build

Seven small functions in \`shapes.py\`, in two groups. The first three make the broadcasting rule of T0.L2 automatic: you implement it by hand, then use it to delete a loop. The last four put numbers on T0.L3 and T0.L4: FLOPs, arithmetic intensity, and the 16 bytes per parameter that decide whether a model can be trained on a given GPU.

Work top to bottom and run the checks after each function; the checks are independent, so a red one never hides a green one.`,
    },
    {
      type: 'prose',
      md: `## 1. broadcast_shape — the rule, without numpy

Walk both shapes **from the right**. At each position take the dimension from each shape, or 1 if that shape has run out. If the two are equal, or one is 1, the result is the other one; otherwise raise \`ValueError\`. Build the result backwards and reverse it at the end.`,
    },
    {
      type: 'code',
      filename: 'hint — the loop header',
      code: `for i in range(1, max(len(a), len(b)) + 1):
    x = a[-i] if i <= len(a) else 1
    y = b[-i] if i <= len(b) else 1
    ...`,
    },
    {
      type: 'callout',
      variant: 'warning',
      title: 'The zero-size edge case',
      md: `\`broadcast_shape((1,), (0,))\` is \`(0,)\`: a size-1 dimension stretches to **any** size, including 0. \`max(x, y)\` gets this wrong; "if x is 1, take y" gets it right.`,
    },
    {
      type: 'prose',
      md: `## 2. pairwise_sq_dists — broadcasting instead of loops

Insert a new axis into each input so their shapes are \`(n, 1, d)\` and \`(1, m, d)\`. Subtract, square, and sum the last axis. One line is enough; T0.L2 shows it. The check compares your answer with a double loop.

## 3. linear — any number of batch dimensions

\`x @ W + b\` already does the right thing for \`(d_in,)\`, \`(B, d_in)\` and \`(B, T, d_in)\`: \`@\` contracts the last axis of \`x\` with the first of \`W\`, and \`b\` broadcasts over every leading axis. The point of this check is that you **do not** reshape or loop — and that you can explain why you did not need to.

## 4–7. The arithmetic

- \`matmul_flops\`: 2·m·k·n.
- \`arithmetic_intensity\`: FLOPs ÷ (bytes_per_el × (m·k + k·n + m·n)). The check tries a square matmul (≈ 1365 FLOP/byte at n = 4096) and a matrix-vector product (≈ 1).
- \`training_bytes_per_param\`: the table in T0.L4.
- \`training_memory_gb\`: parameters × bytes ÷ 10⁹. The check uses 7 × 10⁹ parameters and expects 112.`,
    },
    {
      type: 'callout',
      variant: 'info',
      title: 'Reading a failing check',
      md: `Every check says what it expected. \`broadcast_shape((5, 4), (4,)) = (4, 5), expected (5, 4)\` means you built the result in the right order and forgot to reverse it. Read the message before you read your code.`,
    },
  ],
  cleanup: [
    {
      type: 'prose',
      md: `- Time \`pairwise_sq_dists\` for n = m = 2,000 and d = 64, then rewrite it with the identity |x − y|² = |x|² − 2x·y + |y|² so the work is one matmul. Compare the time and the peak memory.
- Use \`arithmetic_intensity\` to find the batch size at which a (B, 4096) × (4096, 4096) bf16 matmul crosses an H100's ridge point of about 295 FLOP/byte (T0.L3).
- Keep \`shapes.py\`. Lab 08 builds a full FLOP-and-memory calculator on top of these functions.`,
    },
  ],
}

export default guide

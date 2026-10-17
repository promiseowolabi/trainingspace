import type { LabGuide } from './types'

const guide: LabGuide = {
  id: 'flops-and-memory',
  requires: ['a browser — or Python 3.10+ and pytest'],
  blocks: [
    {
      type: 'prose',
      md: `## What you will build

A calculator for training runs: eight functions, each a line or two of arithmetic from T0.L4, M1.L5, P1.L4 and P1.L5. The checks are published numbers — GPT-2's parameter counts, GPT-3's training FLOPs, Chinchilla's compute-optimal model — so a green run means your formulas reproduce real papers.

Keep units straight: FLOPs and FLOP/s are plain numbers (989 TFLOP/s is \`989e12\`), memory is in GB of 10⁹ bytes, MFU is a fraction.`,
    },
    {
      type: 'prose',
      md: `## The formulas, in one table

| function | formula | lesson |
|---|---|---|
| \`param_count\` | V·C + T·C + L·(12C² + 13C) + 2C, plus V·C if untied | M1.L5 |
| \`train_flops\` | 6·N·D | P1.L4 |
| \`forward_flops_per_token\` | 2·N + 2·L·T·C | M1.L5 |
| \`mfu\` | 6·N·tokens_per_second / peak | P1.L4 |
| \`training_days\` | train_flops / (GPUs · peak · MFU) / 86,400 | P1.L4 |
| \`state_memory_gb\` | N · bytes_per_param / 10⁹ | T0.L4 |
| \`activation_memory_gb\` | L · (34·s·b·h + 5·a·s²·b) / 10⁹, without the second term under FlashAttention | T0.L4 |
| \`chinchilla_optimal\` | N = √(C / (6 · 20)), D = 20·N | P1.L5 |

Each docstring repeats its formula. The only one with algebra in it is the last: substitute D = 20·N into C = 6·N·D to get C = 120·N², and solve for N.`,
    },
    {
      type: 'callout',
      variant: 'info',
      title: 'Why these numbers are worth memorising',
      md: `With \`training_days\` you can sanity-check any announcement: 7B parameters on 2 T tokens on 1,024 H100s at 40% MFU is about **2.4 days**. If a claim is off from your estimate by 10×, either the MFU, the token count or the hardware is not what was stated — and E1.L4 is about finding which.`,
    },
  ],
  cleanup: [
    {
      type: 'prose',
      md: `- Use your own measurements from lab 07's \`run.py\` (parameters, tokens/s) and your laptop's matmul peak from T0.L3 to compute your MFU.
- Ask \`chinchilla_optimal\` for the compute of your lab 07 run (6 × 828,416 × the tokens you trained on) and compare its recommended N with the model you trained.
- Keep \`budget.py\` for E1.L4: it is the toolkit for reading training papers.`,
    },
  ],
}

export default guide

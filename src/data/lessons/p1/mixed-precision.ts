import type { Lesson } from '../types'

const lesson: Lesson = {
  id: 'p1.l3',
  slug: 'mixed-precision',
  trackId: 'p1',
  index: 3,
  title: 'Mixed Precision',
  minutes: 19,
  hook: 'Training in 16-bit numbers halves memory traffic and runs on a GPU\'s fastest units — and silently throws away every update smaller than 1/128 of the weight it is added to. Mixed precision is the set of tricks that keeps the speed and loses nothing.',
  exercise: 'read+quiz',
  takeaway: {
    number: '1 + 0.001 = 1 in bf16',
    claim: 'bf16 has 8 bits of precision, so adding 0.001 to 1.0 returns 1.0 — a thousand such updates left the weight unchanged on the reference laptop — which is why the optimiser keeps an fp32 master copy of every weight.',
  },
  blocks: [
    {
      type: 'prose',
      md: `A floating-point number is a sign, an **exponent** (which sets the range) and a **mantissa** (which sets the precision). Fewer bits means fewer bytes to move — the memory-bound operations of T0.L3 run proportionally faster — and GPUs have dedicated matrix units that run 16- and 8-bit formats many times faster than 32-bit. The question is what you lose.

| format | sign · exponent · mantissa | largest value | step size at 1.0 |
|---|---|---|---|
| fp32 | 1 · 8 · 23 | 3.4 × 10³⁸ | 1.2 × 10⁻⁷ |
| bf16 | 1 · 8 · 7 | 3.4 × 10³⁸ | **0.0078** |
| fp16 | 1 · 5 · 10 | **65,504** | 0.00098 |
| fp8 E4M3 | 1 · 4 · 3 | 448 | 0.125 |
| fp8 E5M2 | 1 · 5 · 2 | 57,344 | 0.25 |

**bf16** ("brain float") keeps fp32's exponent — same range, no overflow — and gives up precision. **fp16** keeps more precision and gives up range. Those two choices produce the two classic failure modes.`,
    },
    {
      type: 'prose',
      md: `## Failure 1: updates that vanish (bf16)

Measured in PyTorch on the reference laptop: start a weight at 1.0 and add 0.001 a thousand times.

| dtype | 1.0 + 0.001 | after 1,000 additions | should be |
|---|---|---|---|
| fp32 | 1.0010000 | 2.0000467 | 2.0 |
| fp16 | 1.0009766 | 1.9765625 | 2.0 |
| bf16 | **1.0** | **1.0** | 2.0 |

bf16's spacing near 1.0 is 1/128 ≈ 0.0078, so any increment below half of that rounds away — every single time. A learning rate of 10⁻⁴ times a gradient of order 1 is exactly such an increment. A model whose weights live only in bf16 stops learning the moment its updates get small, with no error and no nan.

**The fix: fp32 master weights.** Keep an fp32 copy of every parameter. Run forward and backward in bf16 for speed, but apply the optimiser update to the fp32 copy, where 0.001 registers, and cast back to bf16 for the next forward. That is the 4-byte "master weight" in T0.L4's 16 bytes per parameter. In PyTorch you get it for free with \`torch.autocast\`: parameters stay fp32 (lab 07's \`bf16-master\` check), and only the matmuls inside the autocast region run in bf16.`,
    },
    {
      type: 'prose',
      md: `## Failure 2: values that overflow or vanish (fp16)

fp16's largest value is 65,504. Measured: \`torch.tensor(70000.0).half()\` is **inf**, and \`torch.tensor(1e-8).half()\` is **0.0** — while bf16 represents 10⁻⁸ without trouble. Gradients of a deep network are routinely smaller than fp16's smallest normal number (about 6 × 10⁻⁵), so in fp16 they underflow to zero.

**The fix: loss scaling.** Multiply the loss by a large factor S (say 2¹⁶) before \`backward()\`; every gradient is S times larger and stays representable. Divide by S before the optimiser step. If any gradient overflowed to inf, skip the step and halve S; if many steps pass cleanly, double it. PyTorch's \`GradScaler\` does exactly this. bf16 has fp32's range, so it needs no scaler — the main reason bf16 replaced fp16 for training on hardware that supports it.

## What stays in fp32 anyway

Some operations are kept in fp32 even inside an autocast region, because they sum many values or exponentiate: softmax, log-softmax and the cross-entropy loss, LayerNorm statistics, and the gradient all-reduce in many setups. Autocast keeps an internal list of which operations run in low precision; lab 07's model also calls \`.float()\` on the logits before the loss.`,
    },
    {
      type: 'vendor',
      snapshot: '2026-10',
      title: 'Why the GPU cares',
      md: `On an NVIDIA H100 SXM, dense bf16 tensor-core throughput is specified at about **989 TFLOP/s**, against about **67 TFLOP/s** for plain fp32 arithmetic — roughly 15× (TF32 tensor cores sit in between). FP8 tensor cores double bf16 again, to about 1,979 TFLOP/s dense. FP8 training needs per-tensor (or finer) scaling factors, like fp16's loss scaling but for every tensor; NVIDIA's Transformer Engine and recent large runs (DeepSeek-V3 reports FP8 training at scale) do this. On a CPU without bf16 hardware, the opposite holds: a bf16 autocast step in lab 07 ran about **27× slower** than fp32 on the reference laptop, because the arithmetic is emulated.`,
      systems: ['nvidia-h100', 'pytorch', 'deepseek-v3'],
      sources: [
        'https://www.nvidia.com/en-us/data-center/h100/',
        'https://arxiv.org/abs/2209.05433',
        'https://arxiv.org/abs/2412.19437',
      ],
    },
    {
      type: 'diagram',
      caption: 'fig 1 — one mixed-precision step',
      height: 62,
      nodes: [
        { id: 'master', x: 2, y: 6, w: 22, h: 12, label: 'fp32 master', sub: '4 B/param' },
        { id: 'cast', x: 30, y: 6, w: 18, h: 12, label: 'cast', sub: 'to bf16' },
        { id: 'fb', x: 54, y: 6, w: 22, h: 12, label: 'forward + backward', sub: 'bf16 matmuls' },
        { id: 'grad', x: 54, y: 40, w: 22, h: 12, label: 'gradients', sub: 'bf16 → fp32' },
        { id: 'adam', x: 2, y: 40, w: 22, h: 12, label: 'AdamW update', sub: 'fp32 m, v, master' },
      ],
      edges: [
        { from: 'master', to: 'cast' },
        { from: 'cast', to: 'fb' },
        { from: 'fb', to: 'grad' },
        { from: 'grad', to: 'adam' },
        { from: 'adam', to: 'master' },
      ],
      steps: [
        { caption: 'The fp32 master weights are the source of truth. They change in increments as small as the optimiser asks for.', active: ['master'] },
        { caption: 'A bf16 copy feeds the forward and backward passes, so the matmuls run on the fast tensor cores and activations take half the memory.', active: ['master', 'cast', 'fb'], edges: ['master->cast', 'cast->fb'] },
        { caption: 'Gradients come out of the backward pass in low precision and are accumulated or reduced in fp32 before the optimiser sees them.', active: ['fb', 'grad'], edges: ['fb->grad'] },
        { caption: 'AdamW updates its fp32 moments and the fp32 master weights. An update of 0.001 on a weight of 1.0 survives here; it would have vanished in bf16.', active: ['grad', 'adam', 'master'], edges: ['grad->adam', 'adam->master'] },
      ],
    },
    {
      type: 'quiz',
      questions: [
        {
          q: 'A team trains with all parameters stored in bf16 (no fp32 master copy). The loss falls normally for a while, then flattens far above what fp32 runs reach, with no nan or error. What is the most likely cause?',
          options: [
            'The model is too small',
            'Updates smaller than about half of bf16\'s spacing at each weight\'s magnitude round to zero, so once the learning rate decays or gradients shrink, weights stop changing',
            'bf16 overflowed',
            'The data loader repeated batches',
          ],
          correct: [1],
          explanation: 'bf16\'s 8 bits of precision mean 1.0 + 0.001 = 1.0. Early, large updates register; later, small ones vanish silently. The fix is fp32 master weights (or stochastic rounding), which is standard practice.',
        },
        {
          q: 'Why does fp16 training need a loss scaler while bf16 training does not?',
          options: [
            'fp16 is less precise than bf16',
            'fp16\'s 5-bit exponent cannot represent very small gradients (they underflow to 0) or values above 65,504; scaling the loss moves gradients into range. bf16 has fp32\'s 8-bit exponent and the same range',
            'bf16 uses a different backward pass',
            'Loss scaling only works with bf16',
          ],
          correct: [1],
          explanation: 'The scaler fixes a range problem, not a precision one — fp16 is actually the more precise of the two. bf16 trades precision for range, which removes the need for scaling and introduces the vanishing-update problem master weights solve.',
        },
        {
          q: 'Your laptop run with --amp (bf16 autocast) is far slower than fp32. Is mixed precision broken?',
          options: [
            'Yes — autocast is incorrect on CPUs',
            'No — without bf16 hardware the CPU emulates the arithmetic (about 27× slower on the reference laptop); the speedup exists on GPUs with tensor cores, where bf16 runs about 15× faster than fp32',
            'Yes — the master weights are being copied every step',
            'No — the first epoch is always slower',
          ],
          correct: [1],
          explanation: 'Mixed precision is a hardware optimisation. Its benefit comes from bf16 tensor-core throughput and halved memory traffic on accelerators. On hardware without native bf16 matrix units the conversion overhead dominates.',
        },
      ],
    },
    {
      type: 'deepdive',
      title: 'Going deeper',
      md: `- Micikevicius et al., "Mixed Precision Training" (2017, arxiv.org/abs/1710.03740) — master weights and loss scaling.
- Micikevicius et al., "FP8 Formats for Deep Learning" (2022, arxiv.org/abs/2209.05433) — E4M3 and E5M2.
- Reproduce the table: \`w = torch.tensor(1.0, dtype=torch.bfloat16)\`, add \`1e-3\` in a loop, print \`w\`. Then try \`torch.finfo(torch.bfloat16)\`.`,
    },
  ],
}

export default lesson

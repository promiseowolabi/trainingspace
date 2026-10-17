import type { Lesson } from '../types'

const lesson: Lesson = {
  id: 'f1.l3',
  slug: 'lora',
  trackId: 'f1',
  index: 3,
  title: 'LoRA',
  minutes: 18,
  hook: 'Fine-tuning every weight of a 7B model needs 112 GB of training state. Freeze the weights and learn a low-rank correction to a few matrices instead, and the trainable part shrinks to a quarter of a percent — with no extra cost at inference once it is merged back in.',
  exercise: 'lab+quiz',
  takeaway: {
    number: '16.8 M of 6.7 B = 0.25%',
    claim: 'LoRA of rank 16 on the four attention projections of a 7B, 32-layer model trains 16.8 million parameters — 0.25% of the model — so gradients and optimiser state nearly vanish and only the frozen bf16 weights remain.',
  },
  blocks: [
    {
      type: 'prose',
      md: `## The idea

Full fine-tuning updates a weight matrix W (d_out × d_in) to W + ΔW. LoRA's hypothesis (Hu et al., 2021) is that for adapting a pretrained model, the useful ΔW has **low rank**: it can be written as a product of two thin matrices,

\`\`\`
ΔW = (α / r) · B · A          A: r × d_in,   B: d_out × r,   r ≪ d
\`\`\`

So freeze W and train only A and B. The layer computes

\`\`\`
y = W·x + (α / r) · B·(A·x)
\`\`\`

through a bottleneck of width r — two thin matmuls that cost a sliver of the base layer.

Two details make it work:

- **B starts at zero.** The adapter is a no-op at step 0, so fine-tuning starts exactly from the pretrained model. A starts random, so that B receives a non-zero gradient on the first step. (Lab 12's \`zero-init\` and \`frozen-base\` check both.)
- **α / r scaling.** Dividing by r keeps the adapter's effect roughly independent of rank, so a learning rate tuned at r = 8 still works at r = 16.`,
    },
    {
      type: 'prose',
      md: `## The arithmetic

A 7B-class model with 32 layers of width 4,096, LoRA rank 16 on the q, k, v and output projections:

\`\`\`
per matrix:  r · (d_in + d_out) = 16 · (4,096 + 4,096) = 131,072
per layer:   4 matrices                                  = 524,288
model:       32 layers                                    = 16.8 M   (0.25% of 6.7 B)
\`\`\`

Training memory (T0.L4), before activations:

| | full fine-tuning | LoRA r = 16 |
|---|---|---|
| bf16 weights | 13.4 GB | 13.4 GB (frozen) |
| gradients | 13.4 GB | 0.03 GB |
| fp32 master + Adam m, v | 80.4 GB | 0.2 GB |
| **total** | **≈ 107 GB** | **≈ 13.6 GB** |

From "a multi-GPU job" to "one 24 GB card". Activations are unchanged — the forward and backward passes still go through every frozen layer — so long sequences still need activation memory.

**QLoRA** goes one step further: store the frozen base in 4-bit precision (about 3.4 GB for 6.7 B parameters), dequantise each layer on the fly for the matmul, and keep the small adapters in bf16.`,
    },
    {
      type: 'diagram',
      caption: 'fig 1 — a LoRA layer',
      height: 60,
      nodes: [
        { id: 'x', x: 2, y: 22, w: 12, h: 12, label: 'x', sub: '(…, d_in)' },
        { id: 'w', x: 24, y: 4, w: 28, h: 12, label: 'W · x', sub: 'frozen, d_out × d_in' },
        { id: 'a', x: 24, y: 40, w: 12, h: 12, label: 'A · x', sub: 'r × d_in' },
        { id: 'b', x: 40, y: 40, w: 14, h: 12, label: 'B · (Ax)', sub: 'd_out × r' },
        { id: 'sum', x: 64, y: 22, w: 14, h: 12, label: '+', sub: '× α / r' },
        { id: 'y', x: 86, y: 22, w: 12, h: 12, label: 'y', sub: '(…, d_out)' },
      ],
      edges: [
        { from: 'x', to: 'w' },
        { from: 'x', to: 'a' },
        { from: 'a', to: 'b' },
        { from: 'w', to: 'sum' },
        { from: 'b', to: 'sum' },
        { from: 'sum', to: 'y' },
      ],
      steps: [
        { caption: 'The pretrained weight W computes its usual output. It is frozen: no gradient, no optimiser state, stored in bf16 (or 4-bit for QLoRA).', active: ['x', 'w'], edges: ['x->w'] },
        { caption: 'In parallel, A projects x down to r dimensions and B projects it back up — a rank-r path whose cost is r·(d_in + d_out) multiply-adds per token.', active: ['x', 'a', 'b'], edges: ['x->a', 'a->b'] },
        { caption: 'The adapter\'s output is scaled by α/r and added. Because B starts at zero, the sum equals W·x on the first step.', active: ['w', 'b', 'sum', 'y'], edges: ['w->sum', 'b->sum', 'sum->y'] },
        { caption: 'After training, W + (α/r)·B·A is a single matrix of the original shape: merged, the layer costs exactly what the base layer cost.', active: ['w', 'sum'] },
      ],
    },
    {
      type: 'prose',
      md: `## Merge or keep separate

After training, \`W' = W + (α/r)·B·A\` can be computed once and stored — a merged model with **zero** inference overhead (lab 12's \`merged\`). Or the adapters can be kept separate: one base model in memory, many small adapters (tens of megabytes each) swapped per request or per customer. Both are common.

## What it is good and bad at

LoRA matches full fine-tuning closely for adapting behaviour, format and domain style — most SFT and preference tuning. It is weaker when the task needs substantial new capability or knowledge, where the low-rank constraint bites; applying it to the MLP matrices as well as attention, or raising r, narrows the gap. As always: measure on held-out data.`,
    },
    {
      type: 'vendor',
      snapshot: '2026-10',
      title: 'What the papers reported',
      md: `Hu et al. (2021) reported that, on GPT-3 175B, LoRA reduced the number of trainable parameters by about **10,000×** and GPU memory during training by about **3×** compared with full fine-tuning with Adam, while matching or exceeding its quality on their benchmarks, with no added inference latency. Dettmers et al. ("QLoRA", 2023) fine-tuned a **65B** model on a single **48 GB** GPU by quantising the frozen base to 4 bits.`,
      systems: ['lora', 'qlora'],
      sources: ['https://arxiv.org/abs/2106.09685', 'https://arxiv.org/abs/2305.14314'],
    },
    {
      type: 'lab',
      lab: 'lora',
      brief: 'Build LoRALinear with a zero-initialised B, freeze the base, apply it by module name to lab 07\'s GPT and count what trains, merge it back into a plain nn.Linear, write the masked SFT loss — then train a rank-2 adapter to learn a rank-1 change exactly.',
    },
    {
      type: 'quiz',
      questions: [
        {
          q: 'A LoRA adapter is initialised with both A and B equal to zero. What happens in training?',
          options: [
            'It trains normally',
            'Nothing ever changes: with B = 0 the gradient to A is zero, and with A = 0 the gradient to B is zero, so both stay at zero forever',
            'It diverges',
            'It learns twice as fast',
          ],
          correct: [1],
          explanation: '∂y/∂A involves B and ∂y/∂B involves A·x. If both are zero, both gradients are zero. That is why one factor (B) starts at zero — to make the adapter a no-op — and the other (A) starts random so learning can begin.',
        },
        {
          q: 'Where does LoRA save memory, and where does it not?',
          options: [
            'It saves activation memory only',
            'It removes nearly all gradient and optimiser-state memory (they exist only for A and B) but keeps the frozen weights in memory and does not reduce activations',
            'It shrinks the model weights by r / d',
            'It saves nothing during training, only at inference',
          ],
          correct: [1],
          explanation: 'The 12 bytes per parameter of optimiser state and the 2 of gradients apply only to trainable parameters — 0.25% of the model in this lesson\'s example. The frozen weights still have to be held for the forward pass, and every layer\'s activations are still stored for the backward pass.',
        },
        {
          q: 'A service needs 50 customer-specific fine-tunes of the same 7B model. Why is LoRA attractive beyond training cost?',
          options: [
            'Each adapter is a few tens of megabytes, so one copy of the base model can serve all 50 by swapping adapters, instead of storing and loading 50 full 13 GB models',
            'LoRA models are more accurate',
            'Adapters run without the base model',
            'It removes the need for a tokenizer',
          ],
          correct: [0],
          explanation: 'r·(d_in + d_out) per adapted matrix is tiny next to the base weights. Keeping adapters separate lets many fine-tunes share one base model in memory; merging is only needed when a single variant must have zero overhead.',
        },
      ],
    },
    {
      type: 'deepdive',
      title: 'Going deeper',
      md: `- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021, arxiv.org/abs/2106.09685).
- Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs" (2023, arxiv.org/abs/2305.14314).
- Biderman et al., "LoRA Learns Less and Forgets Less" (2024, arxiv.org/abs/2405.09673) — where the low-rank constraint costs quality, and where it helps.
- Hugging Face \`peft\` implements exactly lab 12's \`LoRALinear\`, with the same zero-initialised B.`,
    },
  ],
}

export default lesson

import type { Lesson } from '../types'

const lesson: Lesson = {
  id: 't0.l4',
  slug: 'where-the-memory-goes',
  trackId: 't0',
  index: 4,
  title: 'Where the Memory Goes',
  minutes: 16,
  hook: 'A 7-billion-parameter model is 14 GB to serve and 112 GB to train — before a single activation is stored. The other 98 GB are gradients and optimiser state, and knowing which is which is how every memory-saving technique in D1 is designed.',
  exercise: 'read+quiz',
  takeaway: {
    number: '16 bytes / parameter',
    claim: 'Mixed-precision Adam training holds 16 bytes per parameter — bf16 weights and gradients, an fp32 master copy and two fp32 Adam moments — so one 80 GB GPU cannot even hold the training state of a 7B model.',
  },
  blocks: [
    {
      type: 'prose',
      md: `To **run** a model you need its weights. To **train** it you need the weights and everything the four-step loop (T0.L1) creates along the way. Count them for one parameter, in the standard recipe — mixed-precision training with the Adam optimiser:

| what | why it exists | dtype | bytes |
|---|---|---|---|
| weight | the parameter, used in forward and backward | bf16 | 2 |
| gradient | step 3's output, one per parameter | bf16 | 2 |
| master weight | the update \`p −= lr·g\` is too small to register in bf16 (P1.L3) | fp32 | 4 |
| Adam first moment *m* | running average of the gradient (O1.L3) | fp32 | 4 |
| Adam second moment *v* | running average of the squared gradient | fp32 | 4 |
| **total** | | | **16** |

Twelve of the sixteen bytes exist only because we are training. Lab 00's \`training_bytes_per_param\` is this table.`,
    },
    {
      type: 'statline',
      stats: [
        { value: '14 GB', label: '7B weights, bf16', hint: '7 × 10⁹ × 2 bytes — enough to serve it' },
        { value: '112 GB', label: '7B training state', hint: '7 × 10⁹ × 16 bytes, no activations' },
        { value: '80 GB', label: 'one H100', hint: 'HBM3 capacity of an H100 SXM (T0.L3 vendor note)' },
        { value: '5 B', label: 'params per 80 GB', hint: '80 × 10⁹ ÷ 16 — the largest model whose training state fits, with zero room for activations' },
      ],
    },
    {
      type: 'prose',
      md: `Variants exist — some recipes keep fp32 gradients (18 bytes), some keep no master copy and use stochastic rounding (12), some optimisers keep one moment instead of two — but 16 is the number to carry into a design review. It is the one the ZeRO paper uses, and it is why "the model is 14 GB" and "the model fits on a GPU" are different claims.

## The fourth consumer: activations

The backward pass needs the intermediate values the forward pass computed — the input to every matmul, the output of every nonlinearity — because local derivatives are functions of them (A1.L2). So the forward pass **stores** them, and they live until backward consumes them. That is activation memory, and unlike the 16 bytes it scales with the batch:

\`\`\`
activations ∝ layers × batch × sequence length × hidden size   (+ a sequence² term for naive attention)
\`\`\``,
    },
    {
      type: 'vendor',
      snapshot: '2026-10',
      title: 'How big activations get: the Megatron estimate',
      md: `Korthikanti et al. (2022) estimate the activation memory of one transformer layer, in 16-bit, as about

**s·b·h·(34 + 5·a·s / h)** bytes

for sequence length *s*, micro-batch *b*, hidden size *h* and *a* attention heads. For a 7B-shaped model (h = 4096, a = 32, 32 layers) at s = 2048, b = 1:

- the 34·s·b·h part: 34 × 2048 × 4096 = **0.29 GB** per layer;
- the 5·a·s²·b part (the stored attention matrix): 5 × 32 × 2048² = **0.67 GB** per layer;
- × 32 layers ≈ **31 GB for a single sequence**.

FlashAttention (M1.L3) never stores the s² attention matrix, which removes the second term: ≈ 9 GB per sequence. Activation recomputation (D1.L6) trades FLOPs to shrink the rest.`,
      systems: ['megatron-lm'],
      sources: ['https://arxiv.org/abs/2205.05198'],
    },
    {
      type: 'diagram',
      caption: 'fig 1 — memory over one training step',
      height: 62,
      nodes: [
        { id: 'static', x: 2, y: 4, w: 30, h: 12, label: 'static: 16 B/param', sub: 'weights · grads · master · m · v' },
        { id: 'fwd', x: 38, y: 4, w: 26, h: 12, label: 'forward', sub: 'activations pile up' },
        { id: 'peak', x: 70, y: 4, w: 28, h: 12, label: 'peak', sub: 'end of forward' },
        { id: 'bwd', x: 70, y: 40, w: 28, h: 12, label: 'backward', sub: 'activations freed layer by layer' },
        { id: 'step', x: 38, y: 40, w: 26, h: 12, label: 'optimizer step', sub: 'reads m, v, master' },
        { id: 'next', x: 2, y: 40, w: 30, h: 12, label: 'next step', sub: 'back to static only' },
      ],
      edges: [
        { from: 'static', to: 'fwd' },
        { from: 'fwd', to: 'peak' },
        { from: 'peak', to: 'bwd' },
        { from: 'bwd', to: 'step' },
        { from: 'step', to: 'next' },
      ],
      steps: [
        { caption: 'Between steps, memory holds the static training state: 16 bytes per parameter, independent of batch size. For 7B parameters that is 112 GB.', active: ['static'] },
        { caption: 'The forward pass runs layer by layer and keeps each layer\'s inputs for the backward pass. Memory climbs with every layer and every token in the batch.', active: ['static', 'fwd'], edges: ['static->fwd'] },
        { caption: 'Peak memory is at the end of the forward pass: static state plus every stored activation. Out-of-memory errors almost always happen here or early in backward.', active: ['fwd', 'peak'], edges: ['fwd->peak'] },
        { caption: 'Backward walks the layers in reverse, consuming each layer\'s stored activations and freeing them while gradients fill in.', active: ['peak', 'bwd'], edges: ['peak->bwd'] },
        { caption: 'The optimiser step reads the gradients, both Adam moments and the master weights, writes new weights, and memory returns to the static floor for the next step.', active: ['bwd', 'step', 'next'], edges: ['bwd->step', 'step->next'] },
      ],
    },
    {
      type: 'prose',
      md: `## What you can trade

Every memory technique in this course removes one row of the table, or the activation term, and pays for it with something else:

| technique | removes | pays with | lesson |
|---|---|---|---|
| smaller micro-batch + gradient accumulation | activations | more, smaller steps — same maths | P1.L2 |
| activation checkpointing | most activations | ~33% more compute (forward run twice) | D1.L6 |
| ZeRO-1 / -2 / -3, FSDP | optimiser state, grads, weights — sharded over N GPUs | communication | D1.L3 |
| LoRA | grads and optimiser state for frozen weights | adapts a low-rank slice only | F1.L3 |
| 8-bit optimisers | half of *m* and *v* | quantisation error in the moments | P1.L3 |

None of them is free. A design review that says "we'll just use FSDP" without saying what it costs in bandwidth is missing half the sentence.`,
    },
    {
      type: 'callout',
      variant: 'isomorphism',
      title: 'The JVM heap you already size',
      md: `You size a JVM heap as "steady-state live set + headroom for the allocation burst of one request". Training memory has the same shape: the **16 bytes per parameter** are the live set — always resident — and **activations** are the per-request burst, proportional to batch × sequence, allocated in the forward pass and released in the backward. OOM in training, like OOM in a service, is almost always the burst on top of a live set you forgot was already 80% of the heap.`,
    },
    {
      type: 'quiz',
      questions: [
        {
          q: 'Your team has 8 H100s (80 GB each) and wants to fine-tune all weights of a 13B model with plain data parallelism (every GPU holds a full copy of everything). Will it fit?',
          options: [
            'Yes — 13B in bf16 is 26 GB',
            'Yes — 8 GPUs give 640 GB in total',
            'No — each replica needs 13 × 10⁹ × 16 B = 208 GB of training state, and plain data parallelism puts all of it on every GPU',
            'No — but only because of activations',
          ],
          correct: [2],
          explanation: '26 GB is the serving footprint. Training needs 16 bytes per parameter, 208 GB, and data parallelism replicates it on each device, so 640 GB of aggregate memory does not help. Sharding the state across the 8 GPUs (ZeRO-3/FSDP, D1.L3) brings it to 26 GB per GPU plus activations.',
        },
        {
          q: 'A run with micro-batch 16 runs out of memory at the end of the forward pass. Which change keeps the same effective batch and the same training maths while cutting peak memory?',
          options: [
            'Halve the learning rate',
            'Micro-batch 8 with 2 gradient-accumulation steps',
            'Switch from Adam to SGD with momentum',
            'Double the sequence length and halve the batch',
          ],
          correct: [1],
          explanation: 'Activations scale with the micro-batch; static state does not. Two micro-batches of 8, with gradients summed before one optimiser step, compute the same update as one batch of 16 at roughly half the activation memory. Changing the optimiser alters the maths; doubling the sequence length makes the attention term worse.',
        },
        {
          q: 'Which part of training memory does NOT grow when you double the batch size?',
          options: ['Stored activations', 'The attention score matrices', 'Adam\'s two moment buffers', 'The inputs saved for each matmul\'s backward pass'],
          correct: [2],
          explanation: 'The optimiser state is one entry per parameter — 8 bytes per parameter for m and v in fp32 — regardless of how many examples are in the batch. Everything the forward pass stores for the backward pass scales with batch size.',
        },
      ],
    },
    {
      type: 'deepdive',
      title: 'Going deeper',
      md: `- Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models" (2019, arxiv.org/abs/1910.02054) — section 3 is the 16-bytes-per-parameter accounting this lesson uses.
- Korthikanti et al., "Reducing Activation Recomputation in Large Transformer Models" (2022) — the activation formula above, derived term by term.
- In PyTorch, \`torch.cuda.max_memory_allocated()\` after one step on a GPU shows your peak; compare it with 16 × parameters to see your activation share.`,
    },
  ],
}

export default lesson

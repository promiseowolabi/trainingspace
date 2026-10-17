import type { Lesson } from '../types'

const lesson: Lesson = {
  id: 'p1.l4',
  slug: 'flops-and-mfu',
  trackId: 'p1',
  index: 4,
  title: 'FLOPs and MFU',
  minutes: 18,
  hook: 'The cost of a training run is one multiplication: 6 × parameters × tokens. Divide it by what your hardware actually delivers and you have the run\'s length before it starts; divide what you measured by what the hardware could do and you have MFU — the honest throughput number.',
  exercise: 'lab+quiz',
  takeaway: {
    number: 'C = 6·N·D',
    claim: 'Training compute is 6 FLOPs per parameter per token — 3.15 × 10²³ for GPT-3 — and MFU, the fraction of peak those FLOPs achieve, is how you compare runs across hardware: 19% for lab 07 on a laptop, around 40% for well-tuned large GPU runs.',
  },
  blocks: [
    {
      type: 'prose',
      md: `## Where the 6 comes from

Three facts from earlier lessons multiply together:

- a forward pass costs about **2 FLOPs per parameter per token** — one multiply and one add per weight (M1.L5);
- the backward pass costs about **twice the forward** — for every matmul, one for the input gradient and one for the weight gradient (A1.L2);
- so one training step costs about **3 × 2 = 6 FLOPs per parameter per token**.

Over a whole run of D tokens on an N-parameter model:

\`\`\`
C ≈ 6 · N · D
\`\`\`

GPT-3: 6 × 175 × 10⁹ × 300 × 10⁹ = **3.15 × 10²³ FLOPs**. Lab 07's run: 6 × 828,416 × (2,000 × 4,096) = 4.1 × 10¹³ FLOPs — ten orders of magnitude apart, the same formula. It ignores attention's context-dependent term (M1.L5) and the optimiser's elementwise work; both are small for typical training shapes.`,
    },
    {
      type: 'prose',
      md: `## From FLOPs to days

Hardware has a peak FLOP rate, and no real workload reaches it: kernels launch, data moves, GPUs wait on each other. The fraction actually achieved is the **model FLOPs utilisation**:

\`\`\`
MFU = (6 · N · tokens_per_second) / peak_FLOP_per_second
\`\`\`

It uses the FLOPs the *model* needs, not whatever the hardware executed — recomputation for activation checkpointing (D1.L6) does not count — so it cannot be inflated by doing redundant work. Turn it around and you can plan:

\`\`\`
days = 6 · N · D / (GPUs · peak · MFU) / 86,400
\`\`\`

A 7B model on 2 trillion tokens, on 1,024 H100s at 989 TFLOP/s and 40% MFU: 8.4 × 10²² / 4.05 × 10¹⁷ ≈ 207,000 s ≈ **2.4 days**. Lab 08's \`training_days\` computes exactly this.`,
    },
    {
      type: 'statline',
      stats: [
        { value: '3.15·10²³', label: 'GPT-3 training FLOPs', hint: '6 × 175 B parameters × 300 B tokens' },
        { value: '~390 GFLOP/s', label: 'lab 07, achieved', hint: '6 × 828,416 × 79,198 tokens/s on the reference laptop' },
        { value: '~19%', label: 'lab 07 MFU', hint: 'against the laptop\'s measured 2 TFLOP/s float32 matmul peak (T0.L3) — a measured peak, not a datasheet one' },
        { value: '2.4 days', label: '7B × 2T tokens', hint: 'on 1,024 H100s at 40% MFU' },
      ],
    },
    {
      type: 'prose',
      md: `## Why MFU is never 100%

Everything the roofline taught (T0.L3) shows up here:

- **Memory-bound operations.** LayerNorm, softmax, GELU and the optimiser step do few FLOPs per byte and run at bandwidth speed, contributing time but almost no model FLOPs. Fusion (\`torch.compile\`, FlashAttention) attacks this.
- **Small matmuls.** A small model's matmuls are too small to saturate the hardware — lab 07's 128-wide layers are an extreme case.
- **Communication.** In distributed training, time spent sending gradients and activations between GPUs (D1) is time not computing, unless it is overlapped.
- **Overheads.** Python, kernel launches, data loading, evaluation, checkpoint writes, and restarts after failures.

So MFU is a single number that summarises how well a stack uses its hardware, and it is the number to ask for when someone quotes "tokens per second".`,
    },
    {
      type: 'vendor',
      snapshot: '2026-10',
      title: 'Published MFU',
      md: `Google reported **46.2% MFU** for PaLM 540B on TPU v4 (and introduced the term MFU in the same paper). Meta reported **38–43% bf16 MFU** for Llama 3 405B pretraining on up to 16,384 H100 GPUs. These are well-tuned, large-scale runs; 30–50% is a realistic range for large dense models, and much lower is common for small models or unoptimised code.`,
      systems: ['palm', 'llama-3'],
      sources: ['https://arxiv.org/abs/2204.02311', 'https://arxiv.org/abs/2407.21783'],
    },
    {
      type: 'callout',
      variant: 'warning',
      title: 'Tokens per second is not a benchmark by itself',
      md: `"We train at 50,000 tokens per second" means nothing without the model size and the hardware. 50,000 tokens/s on a 1B model is 3 × 10¹⁴ FLOP/s; on a 70B model it is 2.1 × 10¹⁶. Convert every throughput claim to MFU before comparing — lab 08's \`mfu\` is one line.`,
    },
    {
      type: 'lab',
      lab: 'flops-and-memory',
      brief: 'Eight one-line formulas — parameter counts, 6·N·D, forward FLOPs with attention, MFU, wall-clock days, training-state and activation memory, and Chinchilla\'s compute-optimal N and D — each checked against a published number.',
    },
    {
      type: 'quiz',
      questions: [
        {
          q: 'A 13B model trains on 1 trillion tokens. Roughly how many FLOPs is that?',
          options: ['7.8 × 10²¹', '7.8 × 10²²', '1.3 × 10²²', '2.6 × 10²²'],
          correct: [1],
          explanation: '6 × 13 × 10⁹ × 10¹² = 7.8 × 10²². The tempting 2.6 × 10²² is 2·N·D — the forward pass only; training is three times that.',
        },
        {
          q: 'Team A reports 45% MFU; team B reports 60% "hardware FLOPs utilisation" with activation recomputation on. Which is more efficient at training the model?',
          options: [
            'Team B — 60% is higher',
            'Cannot tell from these numbers alone: B\'s figure counts recomputed forward passes, which MFU deliberately excludes; B\'s MFU could be lower than 45%',
            'Team A — MFU is always higher than HFU',
            'They are identical by definition',
          ],
          correct: [1],
          explanation: 'Hardware FLOPs utilisation counts every FLOP executed, including the forward passes redone by activation checkpointing (roughly a third more work). MFU counts only the 6·N·D the model needs, so it measures progress per second. Compare like with like.',
        },
        {
          q: 'Your 1B model trains at 20,000 tokens/s on 8 GPUs with a peak of 989 TFLOP/s each. What is the MFU?',
          options: ['about 1.5%', 'about 15%', 'about 45%', 'about 0.15%'],
          correct: [0],
          explanation: '6 × 10⁹ × 20,000 = 1.2 × 10¹⁴ FLOP/s, against 8 × 989 × 10¹² = 7.9 × 10¹⁵. That is about 1.5% — a sign of a badly configured run (tiny batches, a data-loading bottleneck, or no mixed precision), well worth investigating before spending the budget.',
        },
      ],
    },
    {
      type: 'deepdive',
      title: 'Going deeper',
      md: `- Kaplan et al., "Scaling Laws for Neural Language Models" (2020), section 2.1 — the C ≈ 6·N·D derivation, with the attention term.
- Chowdhery et al., "PaLM" (2022), appendix B — the definition of MFU and why it beats hardware FLOPs utilisation.
- Measure your own MFU: run lab 07's \`run.py\`, take its GFLOP/s column, and divide by your machine's matmul peak from T0.L3.`,
    },
  ],
}

export default lesson

import type { Lesson } from '../types'

const lesson: Lesson = {
  id: 'd1.l5',
  slug: 'pipeline-parallel',
  trackId: 'd1',
  index: 5,
  title: 'Pipeline Parallelism',
  minutes: 18,
  hook: 'Give each GPU a contiguous slice of the layers and pass activations down the line. The communication is tiny and point-to-point — but the pipeline has to fill and drain every step, and while it does, GPUs sit idle. That idle time has a formula.',
  exercise: 'read+quiz',
  takeaway: {
    number: 'bubble = (p − 1)/(m + p − 1)',
    claim: 'With p pipeline stages and m micro-batches per step, the idle fraction is (p − 1)/(m + p − 1): 47% for 8 stages and 8 micro-batches, under 10% with 64 — which is why pipelines need many micro-batches.',
  },
  blocks: [
    {
      type: 'prose',
      md: `**Pipeline parallelism** (PP) splits the model **by depth**: with p stages, GPU 1 holds layers 1–10, GPU 2 layers 11–20, and so on. A stage receives activations from the previous one, runs its layers, and sends its output on; in the backward pass gradients flow back the same way. Because every block of a GPT is the same shape (M1.L4), slicing the stack into equal stages is easy.

What PP is good at:

- **Memory**: each GPU stores only its stages' weights, gradients and optimiser state — 1/p of the model.
- **Communication**: only the activations at stage boundaries cross between GPUs, point-to-point, once per micro-batch in each direction. That is far less than tensor parallelism's per-layer all-reduces, so PP tolerates slower links and can span servers.

What it is bad at is keeping everyone busy.`,
    },
    {
      type: 'prose',
      md: `## The bubble

Run one batch through four stages naively: stage 1 computes while stages 2–4 wait; then stage 2 computes while 1, 3 and 4 wait. Three quarters of the hardware is idle all the time.

The fix is to split each batch into **m micro-batches** and stream them: as soon as stage 1 finishes micro-batch 1 it starts micro-batch 2 while stage 2 works on micro-batch 1. The pipeline still has to **fill** at the start of the step (later stages wait for their first input) and **drain** at the end (earlier stages have nothing left to do), and those idle slots — the **bubble** — are a fixed cost per step. With p stages and m micro-batches of equal cost, the idle fraction is:

\`\`\`
bubble fraction = (p − 1) / (m + p − 1)
\`\`\`

| stages p | micro-batches m | bubble |
|---|---|---|
| 4 | 4 | 3/7 = **43%** |
| 8 | 8 | 7/15 = **47%** |
| 8 | 32 | 7/39 = **18%** |
| 8 | 64 | 7/71 = **9.9%** |

The lesson of the table: you need **m ≫ p**. More micro-batches means a bigger global batch (or smaller micro-batches, which hurt arithmetic intensity, T0.L3) — PP trades against the batch-size ceiling of D1.L2.`,
    },
    {
      type: 'diagram',
      caption: 'fig 1 — fill, steady state, drain (p = 4)',
      height: 62,
      nodes: [
        { id: 's1', x: 2, y: 4, w: 20, h: 10, label: 'stage 1', sub: 'layers 1–8' },
        { id: 's2', x: 2, y: 18, w: 20, h: 10, label: 'stage 2', sub: 'layers 9–16' },
        { id: 's3', x: 2, y: 32, w: 20, h: 10, label: 'stage 3', sub: 'layers 17–24' },
        { id: 's4', x: 2, y: 46, w: 20, h: 10, label: 'stage 4', sub: 'layers 25–32' },
        { id: 'fill', x: 30, y: 18, w: 18, h: 12, label: 'fill', sub: 'later stages idle' },
        { id: 'steady', x: 54, y: 18, w: 20, h: 12, label: 'steady state', sub: 'all stages busy' },
        { id: 'drain', x: 80, y: 18, w: 18, h: 12, label: 'drain', sub: 'early stages idle' },
      ],
      edges: [
        { from: 's1', to: 's2' },
        { from: 's2', to: 's3' },
        { from: 's3', to: 's4' },
        { from: 'fill', to: 'steady' },
        { from: 'steady', to: 'drain' },
      ],
      steps: [
        { caption: 'The model is cut into four stages of equal depth, one per GPU. Activations flow down the stages in forward and gradients flow back up in backward.', active: ['s1', 's2', 's3', 's4'], edges: ['s1->s2', 's2->s3', 's3->s4'] },
        { caption: 'Fill: micro-batch 1 must pass through stages 1, 2 and 3 before stage 4 has anything to do. That costs p − 1 slots of idle time on the later stages.', active: ['fill', 's2', 's3', 's4'] },
        { caption: 'Steady state: with enough micro-batches in flight, every stage is working on a different one. This is the only phase that runs at full utilisation.', active: ['steady', 's1', 's2', 's3', 's4'], edges: ['fill->steady'] },
        { caption: 'Drain: the last micro-batches empty out and the early stages fall idle — another p − 1 slots. Fill plus drain against m micro-batches gives the (p − 1)/(m + p − 1) bubble.', active: ['drain', 's1', 's2', 's3'], edges: ['steady->drain'] },
      ],
    },
    {
      type: 'prose',
      md: `## Schedules: GPipe and 1F1B

**GPipe** runs all m forward passes, then all m backward passes. Simple, but every stage must keep the activations of all m micro-batches until their backward pass — activation memory grows with m, exactly when you want m large.

**1F1B** (one-forward-one-backward, from PipeDream) starts each micro-batch's backward pass as soon as possible and then alternates forward and backward in the steady state. The bubble is the same, but a stage never holds more than about p micro-batches' activations in flight, independent of m. It is the default in Megatron-style training.

**Interleaved 1F1B** gives each GPU several non-adjacent chunks of layers (virtual stages), which cuts the bubble roughly by the number of chunks at the cost of more point-to-point messages. Newer "zero-bubble" schedules split the backward pass into its input-gradient and weight-gradient halves (A1.L4's two matmuls) and use the weight-gradient work to fill the bubble.`,
    },
    {
      type: 'callout',
      variant: 'warning',
      title: 'Balanced stages or nothing',
      md: `The bubble formula assumes every stage takes the same time. If the last stage also holds the output layer — a (C, V) matmul over a 128k vocabulary is as big as several transformer blocks — it runs slower than the others, and every other stage waits for it on every micro-batch. Real pipelines give the first and last stages fewer transformer blocks to compensate for the embedding and the output layer.`,
    },
    {
      type: 'quiz',
      questions: [
        {
          q: 'An 8-stage pipeline runs 16 micro-batches per step. What fraction of the GPU time is bubble?',
          options: ['about 7%', 'about 30%', 'about 47%', 'about 88%'],
          correct: [1],
          explanation: '(p − 1)/(m + p − 1) = 7/23 ≈ 30%. Doubling to 32 micro-batches brings it to 7/39 ≈ 18%. Halving the stages is the other lever: 4 stages and 16 micro-batches give 3/19 ≈ 16%.',
        },
        {
          q: 'Why can pipeline parallelism span servers when tensor parallelism should not?',
          options: [
            'Pipelines use compression',
            'Pipeline stages exchange only boundary activations, point-to-point, once per micro-batch per direction; tensor parallelism all-reduces activation-sized tensors inside every layer, on the critical path',
            'Pipelines do not need gradients',
            'Tensor parallelism uses more memory',
          ],
          correct: [1],
          explanation: 'Per layer, TP communicates far more than PP, and the next computation waits for it. PP\'s messages are few and can overlap with other micro-batches\' work, so they tolerate a 50 GB/s network link.',
        },
        {
          q: 'What does 1F1B improve over GPipe?',
          options: [
            'It removes the bubble entirely',
            'Peak activation memory: by starting backward passes early and alternating, each stage holds about p micro-batches of activations instead of all m',
            'It reduces communication to zero',
            'It allows unequal stages',
          ],
          correct: [1],
          explanation: 'GPipe and 1F1B have the same bubble. GPipe keeps every micro-batch\'s activations until the backward phase starts; 1F1B frees them sooner, so memory no longer grows with the number of micro-batches — which matters because a small bubble needs many of them.',
        },
      ],
    },
    {
      type: 'deepdive',
      title: 'Going deeper',
      md: `- Huang et al., "GPipe" (2018, arxiv.org/abs/1811.06965) and Narayanan et al., "PipeDream" (2019) — the two schedules.
- Narayanan et al., "Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM" (2021, arxiv.org/abs/2104.04473) — interleaved schedules and the bubble analysis.
- Qi et al., "Zero Bubble Pipeline Parallelism" (2023, arxiv.org/abs/2401.10241).`,
    },
  ],
}

export default lesson

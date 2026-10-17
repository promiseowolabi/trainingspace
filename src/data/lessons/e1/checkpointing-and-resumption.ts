import type { Lesson } from '../types'

const lesson: Lesson = {
  id: 'e1.l3',
  slug: 'checkpointing-and-resumption',
  trackId: 'e1',
  index: 3,
  title: 'Checkpointing and Resumption',
  minutes: 16,
  hook: 'At cluster scale, interruptions are not exceptions — they are a rate. A long run is a sequence of resumes, and a resume that is not exact quietly changes the experiment. Know what state a step depends on, save all of it, and choose how often from the failure rate.',
  exercise: 'read+quiz',
  takeaway: {
    number: '466 interruptions in 54 days',
    claim: 'Meta reported 466 job interruptions in a 54-day window of Llama 3 405B pretraining — one every few hours — so checkpoints must capture everything the next step depends on, and their interval should come from the failure rate.',
  },
  blocks: [
    {
      type: 'prose',
      md: `## What a step depends on

Lab 07's \`resume-exact\` check trains 20 steps straight and compares with 10 steps, a save, a load into a *differently initialised* model, and 10 more. Bit-for-bit equality requires every input of the next step to be restored:

| state | why | if you forget it |
|---|---|---|
| model weights | obviously | — |
| optimiser state (Adam's m, v, step count; fp32 master weights) | the update depends on them (O1.L3) | the first steps after resume behave like a fresh Adam: bias correction restarts, steps are mis-sized |
| scheduler position | the learning rate is a function of the step (O1.L4) | warmup reruns, or the cosine jumps |
| data position and sampler RNG | which batches come next | batches repeat or are skipped; the data mixture shifts |
| other RNG (dropout, sampling) | stochastic layers | not bit-exact, though usually harmless |
| gradient-accumulation and loss-scaler state | mid-step state, fp16 scaler (P1.L3) | a wrong first update |

The model weights are the smallest part. For a 7B model, a resumable checkpoint holds bf16 weights (14 GB) plus fp32 master weights, m and v (3 × 28 GB): about **98 GB**, seven times the inference checkpoint.`,
    },
    {
      type: 'vendor',
      snapshot: '2026-10',
      title: 'How often large runs stop',
      md: `The Llama 3 paper reports that during a **54-day** snapshot of 405B pretraining on up to 16,384 H100 GPUs there were **466 job interruptions**: 47 planned (maintenance, upgrades) and **419 unexpected**, roughly 78% of them attributed to confirmed or suspected hardware issues — GPUs and their HBM most often. That is an unexpected interruption about every **3 hours**. Despite this the team reported over 90% effective training time, which depends on frequent, fast checkpointing and automated restarts.`,
      systems: ['llama-3'],
      sources: ['https://arxiv.org/abs/2407.21783'],
    },
    {
      type: 'prose',
      md: `## How often to checkpoint

Checkpoint too rarely and each failure loses hours of compute; too often and the writes themselves eat the time. If writing a checkpoint stalls training for C and failures arrive on average every M, the classic approximation (Young, 1974) for the interval that minimises total waste is

\`\`\`
interval ≈ √(2 · C · M)
\`\`\`

With failures every 186 minutes (419 in 54 days) and a 2-minute checkpoint stall: √(2 × 2 × 186) ≈ **27 minutes**. Halve the stall and the optimal interval drops to about 19 minutes and the waste falls with it — which is why large runs invest in making checkpoints cheap:

- **Sharded checkpoints.** Each rank writes only its own shard of the state (D1.L3), in parallel, instead of gathering 98 GB onto one process.
- **Asynchronous checkpoints.** Copy the state to CPU memory (fast), resume training, and write to storage in the background.
- **In-memory redundancy.** Keep a recent copy in another node's RAM so a single-node failure restores from memory, not from storage.`,
    },
    {
      type: 'diagram',
      caption: 'fig 1 — the cost of a failure',
      height: 60,
      nodes: [
        { id: 'ck', x: 2, y: 22, w: 18, h: 12, label: 'last checkpoint', sub: 'step s' },
        { id: 'work', x: 26, y: 22, w: 20, h: 12, label: 'work since', sub: 'lost on failure' },
        { id: 'fail', x: 52, y: 22, w: 14, h: 12, label: 'failure', sub: 'node dies' },
        { id: 'restart', x: 72, y: 6, w: 26, h: 12, label: 'restart + load', sub: 'detect, reschedule, read' },
        { id: 'redo', x: 72, y: 40, w: 26, h: 12, label: 'recompute', sub: 'steps since s' },
      ],
      edges: [
        { from: 'ck', to: 'work' },
        { from: 'work', to: 'fail' },
        { from: 'fail', to: 'restart' },
        { from: 'fail', to: 'redo' },
      ],
      steps: [
        { caption: 'Every checkpoint costs a stall while state is captured — the C in √(2·C·M). Sharded, asynchronous checkpoints shrink it to seconds.', active: ['ck'] },
        { caption: 'Every step after the last checkpoint is at risk. On average half an interval of work is lost per failure.', active: ['ck', 'work', 'fail'], edges: ['ck->work', 'work->fail'] },
        { caption: 'After a failure the job must be detected, rescheduled on healthy nodes and reloaded — minutes that automation and fast storage reduce.', active: ['fail', 'restart'], edges: ['fail->restart'] },
        { caption: 'Then the lost steps are recomputed. A resume that restores every piece of state makes them identical to the steps that were lost; anything less changes the run.', active: ['fail', 'redo'], edges: ['fail->redo'] },
      ],
    },
    {
      type: 'callout',
      variant: 'segfault',
      title: 'The silent resume bug',
      md: `A resume that restores weights but not the data sampler starts reading the dataset from the beginning again. The loss curve looks perfectly continuous — the model has seen that data before, so it is if anything slightly lower — and the run quietly repeats its first tokens and never reaches the last ones. Lab 07's generator state is in the checkpoint for exactly this reason; at scale the equivalent is the data loader's position in each shard.`,
    },
    {
      type: 'quiz',
      questions: [
        {
          q: 'A team resumes from a checkpoint that contains only the model\'s state_dict, with a freshly constructed AdamW. What goes wrong?',
          options: [
            'Nothing — the weights are what matters',
            'Adam\'s moments restart at zero and its step count at 1: bias correction makes the first updates about lr · sign(g) per parameter, a burst of large, poorly scaled steps that can cause a loss spike',
            'The learning rate becomes zero',
            'The model reverts to random initialisation',
          ],
          correct: [1],
          explanation: 'm, v and t are part of the training state. A fresh Adam takes sign-like steps of size lr on every parameter (lab 03\'s adam-first-step) where the original run was taking carefully scaled ones — a common cause of a spike right after a restart.',
        },
        {
          q: 'Failures arrive about every 8 hours and a checkpoint stalls training for 4 minutes. Roughly how often should you checkpoint?',
          options: ['every 4 minutes', 'about every 62 minutes', 'every 8 hours', 'once a day'],
          correct: [1],
          explanation: '√(2 × 4 × 480) ≈ 62 minutes. Checkpointing every few minutes would spend much of the time stalled; every 8 hours would lose about 4 hours of work per failure on average.',
        },
        {
          q: 'Why is a resumable 7B checkpoint about 98 GB when the model is 14 GB?',
          options: [
            'Checkpoints are uncompressed',
            'It also stores the fp32 master weights and Adam\'s fp32 m and v — three more 28 GB tensors — which the next step needs',
            'It stores every previous step',
            'It stores the gradients and activations',
          ],
          correct: [1],
          explanation: 'bf16 weights 14 GB + 3 × (7 × 10⁹ × 4 bytes) = 14 + 84 = 98 GB. Gradients and activations are recomputed and are not part of the checkpoint. The 14 GB file is the inference artefact; the 98 GB one is the training artefact.',
        },
      ],
    },
    {
      type: 'deepdive',
      title: 'Going deeper',
      md: `- Young, "A first order approximation to the optimum checkpoint interval" (CACM, 1974); Daly (2006) refines it.
- Grattafiori et al., "The Llama 3 Herd of Models" (2024), section 3.3.4 — reliability and operational challenges at 16,384 GPUs.
- PyTorch Distributed Checkpoint (\`torch.distributed.checkpoint\`) — sharded saving and loading, and resharding to a different number of GPUs.`,
    },
  ],
}

export default lesson

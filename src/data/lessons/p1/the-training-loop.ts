import type { Lesson } from '../types'

const lesson: Lesson = {
  id: 'p1.l2',
  slug: 'the-training-loop',
  trackId: 'p1',
  index: 2,
  title: 'The Training Loop',
  minutes: 21,
  hook: 'T0.L1\'s four lines, grown up: batches drawn from a token stream, AdamW with two parameter groups, a warmup-cosine schedule, gradient accumulation, clipping, periodic evaluation and checkpoints you can resume from bit for bit. On a laptop CPU it trains a GPT in under two minutes.',
  exercise: 'lab+quiz',
  takeaway: {
    number: '4.98 → 1.75 in 1,000 steps',
    claim: 'The reference laptop trained an 828k-parameter GPT from the uniform-guess loss ln 146 = 4.98 to a validation loss of 1.75 in 1,000 steps (about 52 s) — and the loss that matters stopped improving there, while training loss kept falling.',
  },
  blocks: [
    {
      type: 'prose',
      md: `From here on the course uses PyTorch: its autograd is your lab 02 engine at production quality, and \`model.py\` in lab 07 is your lab 06 forward pass in PyTorch syntax. What remains is the loop around it. Here is the whole thing, in the shape of lab 07's \`run.py\`:`,
    },
    {
      type: 'code',
      filename: 'the loop (lab 07)',
      code: `model = GPT(config)
opt = configure_optimizer(model, lr=3e-3, weight_decay=0.1)    # AdamW, two groups
gen = torch.Generator().manual_seed(0)

for step in range(total_steps):
    for group in opt.param_groups:                              # O1.L4
        group["lr"] = lr_at(step, max_lr, warmup, total_steps, max_lr / 10)
    micro = [get_batch(train_data, T, B, gen) for _ in range(accum)]
    loss, grad_norm = train_step(model, opt, micro, grad_clip=1.0)
    if step % eval_every == 0:
        val = evaluate(model, val_data, T, B, n_batches=20, generator=eval_gen)
        log(step, loss, val, grad_norm, tokens_per_second)
        save_checkpoint("runs/ckpt.pt", model, opt, step + 1, gen)`,
      chips: ['PyTorch', 'CPU is enough'],
    },
    {
      type: 'prose',
      md: `## The pieces, and why each exists

**Batches from a stream.** Pretraining data is one long token sequence. A batch is B random windows of T + 1 tokens; the input is the first T, the target is the same window shifted by one, so every position is a training example (M1.L6). One window of 128 tokens is 128 next-token predictions, scored in one forward pass.

**Two parameter groups.** Weight decay pulls parameters towards zero (O1.L3, O1.L6). That is sensible for weight matrices and embeddings, and harmful for biases and LayerNorm gains, whose natural values are 0 and 1. Standard practice: decay tensors with ≥ 2 dimensions, exempt the rest.

**Gradient accumulation.** If the batch you want does not fit in memory, split it into micro-batches, call \`backward()\` on each (gradients add up), and step once. Dividing each micro-batch's loss by their number makes the result exactly the gradient of the full batch — lab 07's \`accumulation\` check requires bit-level agreement. It is how a laptop simulates a large batch, and how clusters reach multi-million-token batches (D1.L2).

**Clipping.** Before the step, rescale all gradients if their global norm exceeds a threshold, usually 1.0. It bounds the damage of one bad batch (P1.L6). Log the pre-clip norm: it is one of the best health signals a run has.

**Evaluation and checkpoints.** Every few hundred steps, compute validation loss on held-out data in \`eval()\` mode with no autograd, and save everything needed to resume: weights, optimiser state, step and data-sampler state (E1.L3).`,
    },
    {
      type: 'prose',
      md: `## A real run, on a laptop

\`python run.py\` with lab 07's defaults — 4 layers, width 128, context 128, batch 32, AdamW at 3·10⁻³, 100 warmup steps, cosine to 3·10⁻⁴ over 2,000 steps — on the reference laptop (Apple M5, PyTorch, float32 on CPU):

| step | train loss | val loss | grad norm |
|---|---|---|---|
| 0 | 4.990 | 4.964 | 5.11 |
| 100 | 2.652 | 2.727 | 0.52 |
| 500 | 1.754 | 1.940 | 2.48 |
| 1,000 | 1.209 | **1.746** | 2.80 |
| 1,999 | 0.495 | 2.333 | 2.92 |

Nineteen steps per second, about 79,000 tokens per second. The first line is the sanity check you will use for every run: the initial loss is ln 146 = 4.98, a uniform guess over the 146 characters. The middle lines are learning. The last line is O1.L6's memorisation: a 140,000-character training set seen 58 times. On real pretraining data the val column would keep falling with the train column.

Sampled at temperature 0.8 after step 2,000, from the prompt "The gradient":

\`\`\`text
The gradient scale: the size of ep902: gradients inputs increation one that lets
you with enumber's the bug its and value) ≡ **add W the dists comentumer weights
\`\`\`

Character-level, 828k parameters, two minutes: it has learned the spelling of the course's vocabulary, its markdown habits, and nothing at all about meaning. That is what a loss of about 1.7 nats per character looks like.`,
    },
    {
      type: 'statline',
      stats: [
        { value: '828,416', label: 'parameters', hint: '4 layers, width 128, tied 146-character vocabulary' },
        { value: '19 steps/s', label: 'fp32 on CPU', hint: '4,096 tokens per step, about 79,000 tokens/s on the reference laptop' },
        { value: '~390 GFLOP/s', label: 'by 6·N', hint: '6 × 828,416 × 79,198 tokens/s — compare T0.L3\'s 2 TFLOP/s matmul peak (P1.L4)' },
        { value: 'step 1,000', label: 'best val loss', hint: '1.746, then rising: 58 epochs over a small corpus' },
      ],
    },
    {
      type: 'callout',
      variant: 'segfault',
      title: 'The three silent bugs of a training loop',
      md: `- **Forgetting \`zero_grad\`.** Gradients accumulate across steps; the run trains, badly (A1.L3).
- **Evaluating in train mode.** With dropout, validation loss is noisier and worse; with batch-dependent layers it is simply wrong. Always \`model.eval()\` and \`torch.no_grad()\`.
- **Accumulation without dividing.** Summing k micro-batch losses multiplies the gradient by k — an invisible k-fold learning-rate increase, until it diverges.`,
    },
    {
      type: 'lab',
      lab: 'train-loop',
      brief: 'Write get_batch, configure_optimizer, train_step (accumulation, clipping, bf16 autocast), evaluate and checkpointing around a given GPT. The checks prove accumulation is exact and resume is bit-for-bit; run.py then trains the model above on your CPU.',
    },
    {
      type: 'quiz',
      questions: [
        {
          q: 'You need an effective batch of 512 sequences but only 64 fit in memory. What do you do, and what must the loss computation include?',
          options: [
            'Use batch 64 and multiply the learning rate by 8',
            'Run 8 micro-batches of 64, calling backward on each loss divided by 8, then one optimiser step — the summed gradients then equal the mean gradient of 512',
            'Run 8 micro-batches and call optimizer.step() after each',
            'Use 8 micro-batches and divide the final gradient by 64',
          ],
          correct: [1],
          explanation: 'Gradients add across backward() calls, so 8 calls on loss/8 give the average gradient over all 512 sequences — identical maths to one big batch, at one eighth of the activation memory. Stepping after each micro-batch is just training at batch 64; scaling the learning rate is not equivalent.',
        },
        {
          q: 'Why are LayerNorm gains and biases usually excluded from weight decay?',
          options: [
            'They have no gradients',
            'Their natural values are 1 and 0, not "small"; decaying a gain towards 0 fights the normalisation and buys no regularisation, because these few parameters are not where memorisation happens',
            'PyTorch cannot decay 1-D tensors',
            'They are frozen during pretraining',
          ],
          correct: [1],
          explanation: 'Weight decay is a prior that weights should be small. A LayerNorm gain\'s job is to set a scale around 1; pulling it towards 0 shrinks every activation downstream. Standard recipes decay matrices and embeddings and exempt 1-D parameters.',
        },
        {
          q: 'A run logs training loss only. Which failure from this lesson\'s table would it miss?',
          options: [
            'Divergence',
            'Memorisation: from step 1,000 training loss kept falling while validation loss rose from 1.75 to 2.33',
            'A wrong initial loss',
            'Slow throughput',
          ],
          correct: [1],
          explanation: 'Training loss alone cannot distinguish learning from memorising. The held-out validation loss is the only signal that the model is improving on text it has not seen, which is why the loop evaluates periodically — and why checkpoints are kept, so the best one can be used.',
        },
      ],
    },
    {
      type: 'deepdive',
      title: 'Going deeper',
      md: `- Andrej Karpathy, *nanoGPT* \`train.py\` — the same loop with distributed data parallelism and \`torch.compile\`; read it after lab 07 and every line will be familiar.
- PyTorch's "Automatic Mixed Precision" recipe and \`torch.compile\` tutorial — the two flags that change GPU throughput most.
- Add a CSV logger to \`run.py\` and plot train and val loss against tokens seen, not steps: tokens are the unit P1.L4 and P1.L5 reason in.`,
    },
  ],
}

export default lesson

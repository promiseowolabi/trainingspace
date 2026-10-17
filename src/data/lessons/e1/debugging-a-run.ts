import type { Lesson } from '../types'

const lesson: Lesson = {
  id: 'e1.l2',
  slug: 'debugging-a-run',
  trackId: 'e1',
  index: 2,
  title: 'Debugging a Run',
  minutes: 19,
  hook: 'Training bugs rarely crash. The loss goes down a bit, the run looks plausible, and the model is quietly worse than it should be. A short, fixed checklist — two numbers to predict and three experiments to run — catches most of them before the expensive run starts.',
  exercise: 'read+quiz',
  takeaway: {
    number: 'first loss = ln 146 = 4.98',
    claim: 'A healthy fresh model starts at ln(vocabulary size) — 4.98 for lab 07\'s 146 characters — and can drive one fixed batch to near-zero loss (4.98 → 0.03 in 100 steps); a run that fails either test has a bug, not a hyper-parameter problem.',
  },
  blocks: [
    {
      type: 'prose',
      md: `The most expensive training bugs are silent. Each lesson in this course has shown one: a broadcast that turns a loss into (B, B) (T0.L2), gradients overwritten instead of accumulated (A1.L3), a missing causal mask that makes the task trivial (M1.L3), accumulation without dividing (P1.L2), bf16 weights that stop learning (P1.L3), a prompt left in the SFT loss (F1.L2). None of them raises an exception. All of them produce a loss curve that goes down.

So debugging a training run is mostly done **before** it runs, with checks whose correct answer you know in advance.`,
    },
    {
      type: 'prose',
      md: `## 1. Predict the first loss

A freshly initialised model with small logits outputs a near-uniform distribution, so its loss is **ln V** (O1.L2). Lab 07's model starts at 4.98 against ln 146 = 4.98. If the first logged loss is far above ln V, the initial logits are too large (an initialisation or scaling bug, or an output layer that is not what you think). If it is far *below* ln V before any training, something is leaking the answer — usually a mask or an off-by-one between inputs and targets.

## 2. Overfit one batch

Take a single small batch and train on it, over and over, with no regularisation. A correct model with enough capacity must memorise it. Measured on lab 07's 828k-parameter model, one batch of 8 × 128 characters, AdamW at 10⁻³:

| step | 0 | 50 | 100 | 200 |
|---|---|---|---|---|
| loss | 4.98 | 0.67 | **0.027** | 0.006 |

If the loss on one repeated batch will not go near zero, nothing will learn on real data either: something between the data and the optimiser is broken — frozen parameters, a detached tensor, a learning rate of zero, a loss computed on the wrong tensor.`,
    },
    {
      type: 'prose',
      md: `## 3. Know what "stuck" values mean

Some plateaus have a recognisable height:

- **ln V** — the model has learned nothing (O1.L4's post-norm runs sat at ln 32).
- **The unigram entropy of the data** — the model has learned token frequencies and nothing else. Measured: with targets deliberately shuffled across the batch, so that inputs carry no information about them, lab 07's model plateaued at **3.34** — and the corpus's character unigram entropy is **3.40**. A real run stuck near its data's unigram entropy usually has misaligned inputs and targets.
- **Just under ln V for a classifier** — often softmax applied twice (O1.L2).

Computing the unigram entropy of your data takes one line and turns a vague "it's not learning" into a specific diagnosis.`,
    },
    {
      type: 'statline',
      stats: [
        { value: '4.98', label: 'first loss', hint: 'ln 146 — lab 07\'s fresh model, exactly as predicted' },
        { value: '0.027', label: 'one batch, step 100', hint: 'overfitting a single batch: the cheapest end-to-end test' },
        { value: '3.34', label: 'misaligned targets', hint: 'plateau after 300 steps with shuffled targets' },
        { value: '3.40', label: 'unigram entropy', hint: 'of lab 07\'s corpus — the floor for a model that ignores its input' },
      ],
    },
    {
      type: 'prose',
      md: `## 4. Log the things that explain the loss

Loss tells you *that* something is wrong; these tell you *where*:

- **Gradient norm**, pre-clip, every step (P1.L6) — and per layer, to find the one that is exploding or dead (O1.L1).
- **Learning rate**, every step — schedules are code, and code has bugs.
- **Parameter and activation norms** per layer, occasionally — slow growth precedes instability.
- **Throughput and MFU** (P1.L4) — a sudden drop is a data-loader or hardware problem.
- **Samples**, decoded, every few thousand steps — and, once, **a decoded training batch**: look at the actual tokens the model is trained on. Many data bugs are visible in seconds that way.

## 5. Change one thing at a time

When a run misbehaves, resist changing the learning rate, the batch size and the data at once. Return to the last configuration that worked, change one variable, compare. It feels slow and is the fastest path.`,
    },
    {
      type: 'callout',
      variant: 'info',
      title: 'The checklist',
      md: `Before any large run: (1) first loss ≈ ln V; (2) one batch overfits to near zero; (3) a decoded batch looks like the data you meant; (4) a short run at the planned settings matches the loss of a smaller trusted run at equal tokens; (5) a resume from checkpoint continues the curve exactly (E1.L3). Each check takes minutes. Each has saved real runs weeks.`,
    },
    {
      type: 'quiz',
      questions: [
        {
          q: 'A 50,257-vocabulary model logs 10.82 at step 0, falls to 7.6, and stays there for thousands of steps. Which check do you run first?',
          options: [
            'Raise the learning rate 10×',
            'Compute the unigram entropy of the training tokens: a plateau near it means the model learns token frequencies but its inputs tell it nothing about its targets — check input/target alignment and masking',
            'Add more layers',
            'Switch optimiser',
          ],
          correct: [1],
          explanation: 'The first loss is exactly ln 50,257, so initialisation is fine. A plateau far below ln V but stuck is the signature of a model that has learned only marginal frequencies. Misaligned targets — an off-by-one, shuffled labels, a wrong shift — are the usual cause.',
        },
        {
          q: 'Your model cannot get the loss on a single repeated batch below 3.0. What does that rule out, and what does it point to?',
          options: [
            'It points to too little data',
            'It rules out "needs more data or tuning": a correct model memorises one batch easily. Look for a pipeline bug — parameters not in the optimiser, requires_grad False, a detached tensor, a zero learning rate, or a loss computed on the wrong tensor',
            'It means the batch is too small',
            'It is expected for transformers',
          ],
          correct: [1],
          explanation: 'Overfitting a single batch tests the entire training path end to end with the data question removed. Lab 07\'s small model reached 0.027 in 100 steps; failing to approach zero means gradients are not reaching the weights correctly.',
        },
        {
          q: 'A run\'s first loss is 2.1 with a 32,000-token vocabulary, before any optimiser step. What is the most likely problem?',
          options: [
            'Excellent initialisation',
            'A leak: ln 32,000 ≈ 10.4 is the expected starting loss, so 2.1 means the model can see the answers — usually a missing causal mask or inputs and targets that are not shifted',
            'The learning rate is too high',
            'The batch is too large',
          ],
          correct: [1],
          explanation: 'An untrained model cannot know anything, so a starting loss far below ln V means information about the targets is reaching it. M1.L3\'s missing mask and an unshifted target sequence are the two classic culprits.',
        },
      ],
    },
    {
      type: 'deepdive',
      title: 'Going deeper',
      md: `- Andrej Karpathy, "A Recipe for Training Neural Networks" (karpathy.github.io, 2019) — the canonical version of this checklist.
- The OPT-175B logbook (Zhang et al., 2022, released with the model) — a real large run's incidents, restarts and fixes, day by day.
- Run each check on lab 07: compute ln V and the unigram entropy, overfit one batch, decode a batch with \`D["decode"]\`.`,
    },
  ],
}

export default lesson

/**
 * Track-level prose: the pitch, the outcomes, the prerequisite line and the
 * side note for each track. Hand-written, and deliberately NOT in the
 * generated manifest — a generator must never own copy.
 *
 * Split out of index.ts so the curriculum and track pages can read it without
 * importing the lesson corpus.
 */

import type { TrackId } from './types'

export interface TrackExtras {
  pitch: string
  outcomes: string[]
  requires: string
  sideNote: string
}

export const TRACK_EXTRAS: Record<TrackId, TrackExtras> = {
  t0: {
    pitch:
      'Training a model is an optimisation loop you could write in twenty lines: run the function, measure how wrong it is, work out which way each knob should move, move it a little. This track names those four steps, then puts a price on them — the FLOPs of a matmul and the bytes of a training run — because every later decision is a trade between the two.',
    outcomes: [
      'Say what a loss, a gradient and an optimiser step are, and write the loop that ties them together.',
      'Predict the shape of any numpy expression from the broadcasting rule, and use broadcasting instead of loops.',
      'Count the FLOPs of a matmul (2·m·k·n) and tell a compute-bound one from a memory-bound one.',
      'Estimate the memory of a training run from its parameter count: 16 bytes per parameter before activations.',
    ],
    requires: 'base of the stack · you write Python',
    sideNote: '// lab 00 runs in your browser — no install',
  },
  a1: {
    pitch:
      'Every framework you will use computes gradients the same way: it records the operations of the forward pass as a graph and walks that graph backwards, multiplying local derivatives. You will build that engine twice — once for single numbers, once for numpy arrays — and then prove it correct the way framework authors do.',
    outcomes: [
      'Read a derivative as sensitivity, and estimate one with a finite difference.',
      'Apply the chain rule as a reverse walk over a computational graph, and say why reverse mode wins for training.',
      'Implement a scalar autograd engine with gradient accumulation and an iterative topological sort.',
      'Back-propagate through broadcasting, sums and matmuls with the right shapes.',
      'Gradient-check any operation with central differences and a relative tolerance.',
    ],
    requires: 'requires T0 · the training contract',
    sideNote: '// labs 01–02: micrograd, then autograd for arrays',
  },
  o1: {
    pitch:
      'Having gradients is not the same as learning. This track covers the knobs that decide whether a network trains: how its weights are initialised, which loss it minimises, how the optimiser turns gradients into steps, how the learning rate changes over the run, and how normalisation and regularisation keep it on the rails.',
    outcomes: [
      'Initialise a deep MLP so activations neither vanish nor explode, by variance arithmetic.',
      'Derive the cross-entropy gradient (softmax − onehot) and implement it stably.',
      'Implement SGD, momentum, Adam and AdamW, and say what each one fixes.',
      'Choose a warmup and decay schedule and explain why warmup exists.',
      'Place LayerNorm or RMSNorm in a residual network and say what pre-norm buys.',
    ],
    requires: 'requires A1 · autograd from scratch',
    sideNote: '// lab 03: optimisers and schedules in numpy',
  },
  m1: {
    pitch:
      'A GPT is a short stack of ideas: split text into tokens, turn tokens into vectors, let every position look at the positions before it, mix, repeat, and predict the next token. You will build each piece in numpy, count every parameter, and run a forward pass that matches a reference to the last decimal.',
    outcomes: [
      'Train a byte-pair tokenizer and explain the vocabulary-size trade-off.',
      'Implement causal multi-head attention and rotary position embeddings.',
      'Assemble a pre-norm transformer block and a full GPT forward pass.',
      'Count a GPT\'s parameters from its config: about 12·L·d² plus the embeddings.',
      'Turn logits into text with temperature, top-k and top-p sampling.',
    ],
    requires: 'requires O1 · networks and optimisation',
    sideNote: '// labs 04–06: tokenizer, attention, a GPT forward pass',
  },
  p1: {
    pitch:
      'Pretraining is the transformer from M1, a pile of text, and a budget. This track moves to PyTorch and makes the budget explicit: where the data comes from, what the training loop must do, how mixed precision halves the bytes, how 6·N·D turns a parameter count and a token count into FLOPs, and how scaling laws decide N and D before the run starts.',
    outcomes: [
      'Describe a pretraining data pipeline: filtering, deduplication, mixtures and token counts.',
      'Write a PyTorch training loop with AdamW, a schedule, clipping, evaluation and resumable checkpoints.',
      'Explain fp32, bf16, fp16 and fp8 by their bits, and when loss scaling is needed.',
      'Compute training FLOPs with 6·N·D and an honest MFU from tokens per second.',
      'Pick a compute-optimal model size and token count from a FLOP budget, and say why teams over-train anyway.',
    ],
    requires: 'requires M1 · the transformer',
    sideNote: '// lab 07 trains a GPT on your CPU; lab 08 does the arithmetic',
  },
  d1: {
    pitch:
      'One GPU holds about 5 billion parameters of training state. Everything bigger is split across devices, and every split has a price in bytes sent over the wire. This track teaches the collectives first, then each form of parallelism as "what is sharded, what is communicated, and when" — with simulated ranks you can inspect and real processes on your CPU.',
    outcomes: [
      'Explain all-reduce, reduce-scatter and all-gather, and the 2(N−1)/N bytes of a ring all-reduce.',
      'Implement data parallelism by all-reducing gradients, and match single-process training exactly.',
      'Shard optimiser state, gradients and parameters (ZeRO-1/2/3) and compute the memory per rank.',
      'Split an MLP across devices with tensor parallelism and count its all-reduces.',
      'Compute a pipeline\'s bubble, and choose a 3D layout for a given model and cluster.',
    ],
    requires: 'requires P1 · pretraining',
    sideNote: '// labs 09–11: collectives, DDP over gloo, ZeRO',
  },
  f1: {
    pitch:
      'A base model continues text; an assistant follows instructions. The distance between them is post-training, and every stage of it is a loss function: supervised fine-tuning on demonstrations, a reward model on preferences, and reinforcement learning — or its closed-form shortcut, DPO — against that reward. You will write each loss down and implement it.',
    outcomes: [
      'Say what pretraining leaves undone and which post-training stage addresses each gap.',
      'Build an SFT batch with a chat template and a loss mask on the prompt tokens.',
      'Implement LoRA, compute the memory it saves, and merge it into the base weights.',
      'Train a Bradley–Terry reward model and recognise reward hacking.',
      'Write the PPO, DPO and GRPO objectives and say when each one is the right tool.',
    ],
    requires: 'requires P1 · pretraining (D1 recommended)',
    sideNote: '// labs 12–13: LoRA, then the preference losses',
  },
  e1: {
    pitch:
      'A training run produces a curve and a checkpoint; neither tells you whether the model is good or whether the run was healthy. This track covers what evaluations actually measure, the checklist for a run that will not learn, what a resume must restore, and how to read any training paper as a table of numbers you can check.',
    outcomes: [
      'Separate perplexity from benchmark scores, and estimate the noise in a benchmark number.',
      'Detect contamination and say why it inflates a score.',
      'Debug a run: initial loss = ln V, overfit one batch, watch the gradient norm.',
      'Checkpoint and resume a run bit-for-bit.',
      'Extract N, D, C, the layout and the recipe from a training paper and check them against each other.',
    ],
    requires: 'requires P1 · F1 recommended',
    sideNote: '// then the capstone: train it, defend it',
  },
}

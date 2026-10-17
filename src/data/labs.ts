/**
 * Python labs — metadata only. The step-by-step guides live in
 * src/data/lab-guides/<id>.ts and are loaded by the lab route alone.
 *
 * Every lab is a folder under labs/<id>/:
 *
 *   <module>.py   the one file the reader edits (TODO(you) markers)
 *   checks.py     the checks — run by pytest, by verify.py, and by the browser
 *   test_lab.py   pytest wrapper, one test per check
 *   verify.py     prints the completion code when every check passes
 *
 * and ships to readers as public/labs/<id>.zip (scripts/pack-labs.py).
 *
 * Two lanes:
 *   browser  numpy / pure Python. The lab page runs checks.py against the
 *            reader's file in Pyodide — drop the file, all green, done.
 *   local    PyTorch. pytest + verify.py on the reader's machine; the site
 *            accepts the completion code verify.py prints.
 *
 * Check ids here MUST match the check_<id> functions in labs/<id>/checks.py,
 * in order — tests/labs.test.ts enforces it.
 */

import type { LabId, TrackId } from '@/data/lessons/types'

export type LabLane = 'browser' | 'local'

export interface LabCheck {
  id: string
  label: string
}

export interface PythonLab {
  id: LabId
  /** 0-based, matches the folder order and the "lab 00" naming in lessons. */
  index: number
  title: string
  hook: string
  lane: LabLane
  trackId: TrackId
  /** The lesson that introduces this lab. */
  lessonId: string
  minutes: number
  /** The one file the reader edits, inside labs/<id>/. */
  module: string
  /** Pyodide packages the checks import (browser lane only). */
  packages: string[]
  /** Relative to the deploy base; resolve with asset(). */
  zip: string
  /** Labs whose code or ideas this one builds on. */
  after: LabId[]
  checks: LabCheck[]
}

export const LANE_LABEL: Record<LabLane, string> = {
  browser: 'graded in your browser',
  local: 'PyTorch · graded on your machine',
}

export const LABS: PythonLab[] = [
  {
    id: 'numpy-shapes',
    index: 0,
    title: 'Shapes, FLOPs and Bytes',
    hook: 'Implement numpy\'s broadcasting rule by hand, vectorise a distance matrix, then put a FLOP count and a byte count on a matmul and a 7B training run.',
    lane: 'browser',
    trackId: 't0',
    lessonId: 't0.l2',
    minutes: 30,
    module: 'shapes.py',
    packages: ['numpy'],
    zip: 'labs/numpy-shapes.zip',
    after: [],
    checks: [
      { id: 'broadcast-rules', label: 'broadcast_shape agrees with numpy on compatible shapes, in either order' },
      { id: 'broadcast-errors', label: 'broadcast_shape raises ValueError where numpy would' },
      { id: 'pairwise-dists', label: 'pairwise_sq_dists builds an (n, m) matrix with no Python loops' },
      { id: 'linear-batched', label: 'linear handles any number of leading batch dimensions' },
      { id: 'matmul-flops', label: 'matmul_flops counts 2·m·k·n' },
      { id: 'arithmetic-intensity', label: 'arithmetic_intensity separates a square matmul from a matrix-vector product' },
      { id: 'memory-16', label: 'mixed-precision Adam costs 16 bytes per parameter — 112 GB for 7B' },
    ],
  },
  {
    id: 'micrograd',
    index: 1,
    title: 'An Autograd Engine in 100 Lines',
    hook: 'A Value that remembers how it was made, a closure per operation, and a backward pass that walks the graph in reverse topological order.',
    lane: 'browser',
    trackId: 'a1',
    lessonId: 'a1.l3',
    minutes: 45,
    module: 'engine.py',
    packages: [],
    zip: 'labs/micrograd.zip',
    after: [],
    checks: [
      { id: 'forward-values', label: 'exp, log, tanh and relu compute the right values' },
      { id: 'add-mul', label: 'backward() seeds 1.0 and + and × push the right local gradients' },
      { id: 'accumulates', label: 'a value used more than once accumulates gradient from every use' },
      { id: 'pow-div', label: 'the power rule, and division built from it' },
      { id: 'nonlinear', label: 'exp, log, tanh and relu multiply their local derivative into out.grad' },
      { id: 'topo-order', label: 'a diamond-shaped graph gets its gradient in reverse topological order' },
      { id: 'deep-graph', label: 'a 5000-node chain back-propagates without recursion' },
      { id: 'matches-numeric', label: 'a tiny two-layer network\'s gradients match central differences' },
    ],
  },
  {
    id: 'tensor-autograd',
    index: 2,
    title: 'Autograd for Arrays',
    hook: 'The same engine over numpy arrays: unbroadcasting, the matmul transpose rule, a fused stable cross-entropy, a gradient checker — and an MLP that trains on it.',
    lane: 'browser',
    trackId: 'a1',
    lessonId: 'a1.l4',
    minutes: 60,
    module: 'tensor.py',
    packages: ['numpy'],
    zip: 'labs/tensor-autograd.zip',
    after: ['micrograd'],
    checks: [
      { id: 'unbroadcast', label: 'unbroadcast sums a gradient back down to the input\'s shape' },
      { id: 'add-broadcast', label: 'a broadcast bias receives the sum of its gradients' },
      { id: 'mul', label: 'elementwise × unbroadcasts both sides' },
      { id: 'matmul', label: 'matmul backward: dA = G·Bᵀ and dB = Aᵀ·G' },
      { id: 'sum-axis', label: 'sum backward, with and without keepdims' },
      { id: 'relu', label: 'relu forward and backward' },
      { id: 'cross-entropy-value', label: 'cross_entropy is the mean negative log-softmax of the target' },
      { id: 'cross-entropy-stable', label: 'cross_entropy survives logits of ±1000 (log-sum-exp)' },
      { id: 'cross-entropy-grad', label: 'its gradient is (softmax − onehot) / N' },
      { id: 'gradcheck', label: 'gradcheck accepts a correct backward and rejects one that is off by 2×' },
      { id: 'trains-mlp', label: 'a 2→16→3 MLP built on your Tensor trains below 0.15 loss' },
    ],
  },
  {
    id: 'optimisers',
    index: 3,
    title: 'Optimisers, Schedules and Clipping',
    hook: 'SGD with momentum, Adam with bias correction, AdamW\'s decoupled weight decay, a warmup-cosine schedule and global-norm clipping — matched to PyTorch\'s update rules, then raced on a badly conditioned bowl.',
    lane: 'browser',
    trackId: 'o1',
    lessonId: 'o1.l3',
    minutes: 45,
    module: 'optim.py',
    packages: ['numpy'],
    zip: 'labs/optimisers.zip',
    after: ['tensor-autograd'],
    checks: [
      { id: 'sgd-step', label: 'SGD updates parameters in place by lr · g' },
      { id: 'momentum', label: 'momentum keeps PyTorch\'s buffer: buf ← μ·buf + g' },
      { id: 'adam-first-step', label: 'bias correction makes Adam\'s first step ≈ lr · sign(g)' },
      { id: 'adam-matches-reference', label: '20 Adam steps match the reference update rule' },
      { id: 'adamw-decoupled', label: 'AdamW decays weights directly; Adam + L2 decays through the gradient' },
      { id: 'warmup-cosine', label: 'linear warmup then cosine decay, hitting max_lr and min_lr exactly' },
      { id: 'clip-norm', label: 'global-norm clipping returns the pre-clip norm and keeps the direction' },
      { id: 'conditioning', label: 'on a condition-1000 bowl SGD stalls above 0.1 and Adam gets below 10⁻⁶' },
    ],
  },
  {
    id: 'bpe-tokenizer',
    index: 4,
    title: 'A Byte-Level BPE Tokenizer',
    hook: 'Count pairs, merge the most frequent, repeat: train the tokenizer every GPT uses, encode and decode any Unicode string losslessly, and measure the compression it buys.',
    lane: 'browser',
    trackId: 'm1',
    lessonId: 'm1.l1',
    minutes: 45,
    module: 'tokenizer.py',
    packages: [],
    zip: 'labs/bpe-tokenizer.zip',
    after: [],
    checks: [
      { id: 'stats-counts', label: 'get_stats counts adjacent pairs in order of first occurrence' },
      { id: 'merge-replaces', label: 'merge replaces a pair left to right without overlap' },
      { id: 'train-classic', label: 'train learns the textbook merges on "aaabdaaabac", ties broken by first occurrence' },
      { id: 'encode-replays', label: 'encode replays merges in the order they were learned' },
      { id: 'roundtrip', label: 'decode(encode(x)) == x for English, unseen words, accents, CJK and emoji' },
      { id: 'compression', label: '144 merges compress their training text to more than 2.5 bytes per token' },
      { id: 'decode-partial', label: 'a token sequence that ends mid-character decodes to U+FFFD instead of raising' },
    ],
  },
  {
    id: 'attention',
    index: 5,
    title: 'Attention, Heads and RoPE',
    hook: 'Stable softmax, the causal mask, scaled dot-product attention over any batch shape, the head split, and rotary position embeddings — with a check that RoPE really makes attention depend only on distance.',
    lane: 'browser',
    trackId: 'm1',
    lessonId: 'm1.l3',
    minutes: 50,
    module: 'attention.py',
    packages: ['numpy'],
    zip: 'labs/attention.zip',
    after: ['numpy-shapes'],
    checks: [
      { id: 'softmax-stable', label: 'softmax survives logits of 1000 and gives masked entries exactly 0' },
      { id: 'causal-mask', label: 'the causal mask lets token i see tokens 0..i' },
      { id: 'attention-matches', label: 'attention matches the reference for (T, D), (B, T, D) and (B, H, T, D)' },
      { id: 'no-peeking', label: 'changing future tokens never changes earlier outputs' },
      { id: 'heads-roundtrip', label: 'split_heads and merge_heads are exact inverses' },
      { id: 'multi-head-matches', label: 'multi-head attention matches the reference' },
      { id: 'rope-rotates', label: 'RoPE rotates each pair by t·θᵢ and preserves its length' },
      { id: 'rope-relative', label: 'after RoPE, q·k depends only on the distance between positions' },
    ],
  },
  {
    id: 'gpt-forward',
    index: 6,
    title: 'A GPT Forward Pass',
    hook: 'LayerNorm, GELU, the pre-norm block and tied embeddings: a GPT-2-architecture forward pass that matches a reference to 10⁻⁸, a parameter count that lands on 124,439,808, greedy decoding, and temperature, top-k and top-p sampling.',
    lane: 'browser',
    trackId: 'm1',
    lessonId: 'm1.l5',
    minutes: 60,
    module: 'gpt.py',
    packages: ['numpy'],
    zip: 'labs/gpt-forward.zip',
    after: ['attention'],
    checks: [
      { id: 'layer-norm', label: 'layer_norm normalises each token and applies g and b' },
      { id: 'gelu', label: 'gelu is GPT-2\'s tanh approximation' },
      { id: 'forward-matches', label: 'logits match an independent reference forward pass to 10⁻⁸' },
      { id: 'causal', label: 'changing later tokens never changes earlier logits' },
      { id: 'param-count', label: 'count_params gives 124,439,808 for GPT-2 small and matches init_params' },
      { id: 'greedy', label: 'greedy generation takes the last position\'s argmax and crops to block_size' },
      { id: 'sampling', label: 'temperature, top-k and top-p sampling draw from the right distribution' },
    ],
  },
  {
    id: 'train-loop',
    index: 7,
    title: 'A Training Loop You Can Resume',
    hook: 'Batches, AdamW parameter groups, gradient accumulation, clipping, bf16 autocast, evaluation and bit-exact checkpoint resume — the loop every pretraining run is built on, training a GPT on this course\'s own text on your CPU.',
    lane: 'local',
    trackId: 'p1',
    lessonId: 'p1.l2',
    minutes: 60,
    module: 'train.py',
    packages: [],
    zip: 'labs/train-loop.zip',
    after: ['gpt-forward', 'optimisers'],
    checks: [
      { id: 'get-batch', label: 'get_batch samples shifted windows using only the generator' },
      { id: 'optimizer-groups', label: 'AdamW decays matrices and embeddings, never biases or norm gains' },
      { id: 'initial-loss', label: 'a fresh model evaluates to ln V and evaluate restores train mode' },
      { id: 'accumulation', label: 'two micro-batches of 4 update the weights exactly like one batch of 8' },
      { id: 'clipping', label: 'clipping bounds the update and returns the pre-clip norm' },
      { id: 'bf16-master', label: 'bf16 autocast leaves fp32 master weights and a close loss' },
      { id: 'learns', label: '300 steps take validation loss from 4.98 to below 2.9' },
      { id: 'resume-exact', label: '10 steps + checkpoint + 10 steps equals 20 straight steps, bit for bit' },
    ],
  },
  {
    id: 'flops-and-memory',
    index: 8,
    title: 'Budgeting a Training Run',
    hook: 'Parameter counts, 6·N·D, MFU, wall-clock days, training-state and activation memory, and the Chinchilla-optimal model for a FLOP budget — checked against GPT-2, GPT-3 and Chinchilla\'s published numbers.',
    lane: 'browser',
    trackId: 'p1',
    lessonId: 'p1.l4',
    minutes: 30,
    module: 'budget.py',
    packages: [],
    zip: 'labs/flops-and-memory.zip',
    after: ['numpy-shapes'],
    checks: [
      { id: 'param-count', label: 'exact counts for GPT-2 small (124,439,808) and XL (1,557,611,200)' },
      { id: 'train-flops', label: '6·N·D gives GPT-3\'s 3.15 × 10²³ FLOPs' },
      { id: 'forward-per-token', label: 'forward FLOPs per token include the attention term' },
      { id: 'mfu', label: 'MFU from tokens per second and the hardware peak' },
      { id: 'training-days', label: 'wall-clock days from FLOPs, GPUs, peak and MFU' },
      { id: 'state-memory', label: '16 bytes per parameter of training state' },
      { id: 'activation-memory', label: 'activation memory with and without FlashAttention' },
      { id: 'chinchilla', label: 'the compute-optimal N and D for Chinchilla\'s budget: 69 B and 1.4 T' },
    ],
  },
  {
    id: 'collectives',
    index: 9,
    title: 'Ring All-Reduce',
    hook: 'Reduce-scatter and all-gather around a simulated ring that only lets each rank talk to its neighbour — then measure that every rank sends exactly 2(N−1)/N of the data, whatever N is.',
    lane: 'browser',
    trackId: 'd1',
    lessonId: 'd1.l1',
    minutes: 40,
    module: 'collectives.py',
    packages: ['numpy'],
    zip: 'labs/collectives.zip',
    after: [],
    checks: [
      { id: 'reduce-scatter', label: 'after N − 1 steps rank r owns the fully summed chunk r' },
      { id: 'all-gather', label: 'after N − 1 more steps every rank holds every chunk' },
      { id: 'all-reduce', label: 'ring all-reduce gives every rank the sum in 2(N − 1) steps' },
      { id: 'ring-traffic', label: 'each rank sends exactly 2(N − 1)/N of the array, measured' },
      { id: 'bytes-formula', label: 'the bytes-per-rank formula, approaching 2× as N grows' },
      { id: 'time-model', label: 'the latency + bandwidth time model, and why tiny messages are latency-bound' },
    ],
  },
  {
    id: 'ddp-from-scratch',
    index: 10,
    title: 'Data Parallelism from Scratch',
    hook: 'Two real processes on your machine, talking over torch.distributed\'s gloo backend: broadcast the weights, shard the batch, average the gradients — bucketed, like DDP — and match single-process training to 10⁻⁶.',
    lane: 'local',
    trackId: 'd1',
    lessonId: 'd1.l2',
    minutes: 50,
    module: 'ddp.py',
    packages: [],
    zip: 'labs/ddp-from-scratch.zip',
    after: ['collectives', 'train-loop'],
    checks: [
      { id: 'shard-batch', label: 'each rank gets its contiguous slice of the global batch' },
      { id: 'plan-buckets', label: 'gradients are grouped in order into buckets of at most bucket_bytes' },
      { id: 'broadcast-sync', label: 'after broadcast both ranks hold rank 0\'s initial weights' },
      { id: 'matches-single-process', label: '2 ranks × 8 examples train exactly like 1 process × 16' },
      { id: 'bucketed-matches', label: 'bucketed all-reduce gives the same parameters' },
      { id: 'bucket-calls', label: 'one all_reduce per bucket, not per tensor' },
    ],
  },
  {
    id: 'zero-sharding',
    index: 11,
    title: 'ZeRO, Sharded',
    hook: 'Shard Adam\'s state across simulated ranks, update only your slice, all-gather the result — and prove it trains identically to unsharded Adam while each rank holds 1/N of the optimiser state.',
    lane: 'browser',
    trackId: 'd1',
    lessonId: 'd1.l3',
    minutes: 40,
    module: 'zero.py',
    packages: ['numpy'],
    zip: 'labs/zero-sharding.zip',
    after: ['collectives', 'optimisers'],
    checks: [
      { id: 'shard-roundtrip', label: 'shard returns copies and unshard inverts it' },
      { id: 'state-is-sharded', label: 'each rank keeps m and v for 1/N of the parameters' },
      { id: 'matches-unsharded', label: '25 ZeRO steps on 4 ranks equal 25 Adam steps on the mean gradient' },
      { id: 'ranks-agree', label: 'after the all-gather every rank holds identical parameters, in place' },
      { id: 'memory-stages', label: 'stages 0–3 reproduce the ZeRO paper\'s 120 / 31.4 / 16.6 / 1.9 GB' },
      { id: 'comm-volume', label: 'stages 0–2 communicate 2Ψ per step, stage 3 3Ψ' },
    ],
  },
  {
    id: 'lora',
    index: 12,
    title: 'LoRA and the SFT Loss',
    hook: 'Wrap frozen linear layers with a zero-initialised low-rank adapter, count what actually trains, merge it back for free inference, mask the prompt out of the loss — and watch a rank-2 adapter learn a rank-1 change exactly.',
    lane: 'local',
    trackId: 'f1',
    lessonId: 'f1.l3',
    minutes: 45,
    module: 'lora.py',
    packages: [],
    zip: 'labs/lora.zip',
    after: ['train-loop'],
    checks: [
      { id: 'zero-init', label: 'with B = 0 the wrapped layer computes exactly the base layer' },
      { id: 'frozen-base', label: 'only A and B train; no gradient reaches the frozen weight' },
      { id: 'scale', label: 'the adapter is scaled by α / r' },
      { id: 'merge-equal', label: 'merging gives a plain nn.Linear with the same outputs' },
      { id: 'apply-counts', label: 'apply_lora wraps the right layers and trains r·(in + out) each' },
      { id: 'sft-masking', label: 'the SFT loss ignores prompt positions completely' },
      { id: 'learns-low-rank', label: 'a rank-2 adapter learns a rank-1 change while the base stays fixed' },
    ],
  },
  {
    id: 'preference-losses',
    index: 13,
    title: 'Preference and RL Losses',
    hook: 'Sequence log-probabilities, the Bradley–Terry reward loss, DPO with its implicit rewards and gradient weights, PPO\'s clipped objective, GRPO\'s group-relative advantages and the k3 KL estimator — each checked against its definition.',
    lane: 'browser',
    trackId: 'f1',
    lessonId: 'f1.l5',
    minutes: 45,
    module: 'prefs.py',
    packages: ['numpy'],
    zip: 'labs/preference-losses.zip',
    after: ['tensor-autograd'],
    checks: [
      { id: 'sequence-logprob', label: 'summed response log-probabilities, prompt masked, numerically stable' },
      { id: 'bradley-terry', label: 'the reward-model loss −log σ(r_chosen − r_rejected)' },
      { id: 'dpo-loss', label: 'the DPO loss starts at ln 2 and matches its definition' },
      { id: 'dpo-gradient', label: 'DPO\'s gradient, checked by finite differences, weights mis-ranked pairs most' },
      { id: 'ppo-clip', label: 'PPO\'s clipped objective caps gains and keeps losses pessimistic' },
      { id: 'grpo-advantages', label: 'GRPO normalises rewards within each group, with no nan for ties' },
      { id: 'kl-k3', label: 'the k3 estimator is non-negative and averages to the true KL' },
    ],
  },
]

export const labById = (id: string | undefined): PythonLab | undefined =>
  id ? LABS.find((l) => l.id === id) : undefined

export const labNumber = (l: PythonLab): string => String(l.index).padStart(2, '0')

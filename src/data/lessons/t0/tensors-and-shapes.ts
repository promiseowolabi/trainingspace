import type { Lesson } from '../types'

const lesson: Lesson = {
  id: 't0.l2',
  slug: 'tensors-and-shapes',
  trackId: 't0',
  index: 2,
  title: 'Tensors and Shapes',
  minutes: 24,
  hook: 'Half the bugs in training code are shape bugs, and most of them do not crash — numpy broadcasts the mistake into a plausible-looking wrong answer. Learn the one rule and the shape becomes the spec.',
  exercise: 'lab+quiz',
  takeaway: {
    number: '(8, 1, 6) + (7, 1) → (8, 7, 6)',
    claim: 'Broadcasting aligns shapes from the right and stretches any dimension of size 1; that one rule explains every shape in a training loop, and every silent shape bug.',
  },
  blocks: [
    {
      type: 'prose',
      md: `A **tensor** is an n-dimensional array of numbers that all have the same type. The word sounds like physics; the thing is a numpy \`ndarray\` (or a \`torch.Tensor\`, which is the same idea plus a device and an autograd tape). Three attributes describe one completely:

- **shape** — a tuple, one entry per dimension: \`(32, 128, 768)\`.
- **dtype** — what each element is: \`float32\` (4 bytes), \`bfloat16\` (2), \`int64\` (8).
- **the data** — \`prod(shape)\` elements, contiguous in memory.

So a \`(32, 128, 768)\` float32 tensor is 32 × 128 × 768 = 3,145,728 numbers and **12.6 MB**. You will compute that product a lot (T0.L4).

## The shape is the specification

In training code almost every tensor has a conventional shape, and reading the shape tells you what the tensor *is*:

| shape | name | meaning |
|---|---|---|
| \`(B, d_in)\` | a batch of inputs | B examples, d_in features each |
| \`(d_in, d_out)\` | a weight matrix | maps d_in features to d_out |
| \`(B, T)\` | a batch of token ids | B sequences of T tokens, integers |
| \`(B, T, C)\` | activations | one C-dimensional vector per token per sequence |
| \`(B, H, T, T)\` | attention scores | per head, every token's score for every other token (M1.L3) |
| \`()\` | the loss | a scalar: zero dimensions, one number |

Write the shape as a comment beside every line you are unsure about. Experienced practitioners do it constantly; it is the type system Python does not give you.`,
    },
    {
      type: 'code',
      filename: 'shapes.py',
      code: `import numpy as np

B, T, C = 4, 8, 16
x = np.random.randn(B, T, C)      # (4, 8, 16)  activations
W = np.random.randn(C, 3 * C)     # (16, 48)    one weight matrix
y = x @ W                         # (4, 8, 48)  @ contracts the LAST axis of x with the FIRST of W
mean = y.mean(axis=-1)            # (4, 8)      reducing an axis removes it...
mean_k = y.mean(axis=-1, keepdims=True)  # (4, 8, 1)  ...unless you keep it
centered = y - mean_k             # (4, 8, 48)  broadcasting stretches the 1`,
      chips: ['numpy', '(B, T, C)'],
    },
    {
      type: 'prose',
      md: `## Broadcasting: the one rule

The last line subtracted a \`(4, 8, 1)\` array from a \`(4, 8, 48)\` one and got \`(4, 8, 48)\`. The shapes are different, yet numpy did not complain. That is **broadcasting**. It has a four-line rule, but the rule only makes sense once you have seen what it is for, so build up to it with small arrays you can check by eye.

### Step 1 — reading a shape

A shape says how many items there are in each direction.

| value | shape | read it as |
|---|---|---|
| \`5\` | \`()\` | one number, no directions |
| \`[1, 2, 3]\` | \`(3,)\` | 3 across |
| \`[[1, 2, 3], [4, 5, 6]]\` | \`(2, 3)\` | 2 down, 3 across |

The **last** number in a shape is always "how many across" — the width of one row. Hold on to that; it is why the rule starts from the right.

### Step 2 — every item needs a partner

Adding two lists of the same length pairs them up by position, first with first, second with second:

\`\`\`
[1,  2,  3]
[10, 20, 30]   +
------------
[11, 22, 33]
\`\`\`

That is all elementwise arithmetic is: every item on one side meets one partner on the other.

### Step 3 — when one side has only one

\`\`\`
[1, 2, 3] + 10
\`\`\`

The 10 has no partner for the 2 or the 3. There is only **one** of it, so the sensible thing is obvious — reuse it:

\`\`\`
[1,  2,  3]
[10, 10, 10]   +     ← the one 10, reused three times
------------
[11, 12, 13]
\`\`\`

That reuse **is** broadcasting. And "reused" is literal: numpy does not write out three 10s, it reads the same 10 three times. Nothing is copied.

### Step 4 — when reuse makes no sense

\`\`\`
[1, 2, 3] + [10, 20]
\`\`\`

Who does the 3 pair with? Repeating 10 and 20 in some pattern would be a guess, so numpy refuses: \`operands could not be broadcast together with shapes (3,) (2,)\`.

The difference between Step 3 and Step 4 is the whole idea: **one** value can fill any number of slots; **two** values cannot fill three.

### Step 5 — a table plus a row

\`\`\`
table = [[1, 2, 3],      # (2, 3)
         [4, 5, 6]]
row   =  [10, 20, 30]    # (3,)
\`\`\`

Going across, the row has 3 items and each table row has 3 items, so they pair up as in Step 2. Going down, the table has 2 rows but there is only **one** \`row\`, so it is reused for each, as in Step 3:

\`\`\`
[[1, 2, 3],     [[10, 20, 30],     [[11, 22, 33],
 [4, 5, 6]]  +   [10, 20, 30]]  =   [14, 25, 36]]
                   ↑ the same row, reused
\`\`\`

This is exactly how a layer adds its bias: one bias row, reused for every example in the batch (fig 1).

### Step 6 — a table plus a column

Now give each **row** its own number instead — a column of shape \`(2, 1)\`: 2 down, 1 across.

\`\`\`
[[1, 2, 3],     [[100],        [[101, 102, 103],
 [4, 5, 6]]  +   [200]]    =    [204, 205, 206]]
\`\`\`

Going down, 2 meets 2: row pairs with row. Going across, there is only **one** number in each row of the column, so it is reused for all 3 items. 100 lands on the whole first row, 200 on the whole second.

### Step 7 — the trap

Write that same column carelessly, as a plain list \`[100, 200]\` of shape \`(2,)\`, and it fails:

\`\`\`
table:  2  3
list:      2     ← lined up against the 3, not the 2
\`\`\`

A plain list is always read as "across" (Step 1), so numpy tries to pair its 2 items with the table's 3 columns — Step 4 again — and raises. To mean "one number per row" you must say so with the shape \`(2, 1)\`. The dangerous version is when the sizes happen to match: with a 3 × 3 table, a length-3 list meant per row is silently applied per column. No error, wrong answer.

### Step 8 — the rule is Steps 2–7 written down

Write the two shapes one above the other, aligned on the right, and compare one column at a time:

\`\`\`
table:  2  3
row:       3
\`\`\`

1. Line the two shapes up **from the right** — the "across" sizes meet first, because pairing items along a row is where you always start (Step 2).
2. A missing dimension on the left counts as **1** — \`row\` has no "down" number; it is a single row, so it counts as 1 row (Step 5).
3. Each aligned pair must be **equal** (items pair up, Step 2), or one of them must be **1** (that one is reused, Step 3). The reuse is virtual — no copy is made.
4. Otherwise: error (Step 4).

### Step 9 — the example from the code

\`\`\`
y:       4  8  48     4 sequences × 8 tokens × 48 numbers per token
mean_k:  4  8   1     one number (the average) per token
\`\`\`

Right to left: **48 vs 1** — each token has one average, reused for all 48 of its numbers (Step 6). **8 vs 8** — token pairs with token. **4 vs 4** — sequence with sequence. The result is \`(4, 8, 48)\`: every token centred on its own average. Without \`keepdims=True\` the mean would be \`(4, 8)\`, which lines up 8 against 48 — Step 7's trap, and an error.

### The rule, compactly

\`\`\`
   (8, 1, 6)        (5, 4)          (3,)
      (7, 1)           (4,)         (4,)
   ---------        -------        -----
   (8, 7, 6)        (5, 4)         error: 3 vs 4
\`\`\`

In the first, both sides have a 1 somewhere: the 1 in \`(8, 1, 6)\` is reused 7 times, the 1 in \`(7, 1)\` is reused 6 times, and the missing left dimension counts as 1 and is reused 8 times.

If you remember one sentence: **every item needs a partner; if one side has only one of something it is reused, and if both sides have several and the counts differ, it is an error.** Lab 00 has you implement this rule without numpy, which is the fastest way to make it automatic.`,
    },
    {
      type: 'diagram',
      caption: 'fig 1 — adding a bias to a batch',
      height: 60,
      nodes: [
        { id: 'x', x: 2, y: 6, w: 22, h: 14, label: 'x', sub: '(B, d_out)' },
        { id: 'b', x: 2, y: 34, w: 22, h: 10, label: 'b', sub: '(d_out,)' },
        { id: 'align', x: 36, y: 20, w: 24, h: 12, label: 'align right', sub: '(1, d_out)' },
        { id: 'stretch', x: 70, y: 6, w: 26, h: 12, label: 'stretch the 1', sub: '(B, d_out), no copy' },
        { id: 'out', x: 70, y: 34, w: 26, h: 12, label: 'x + b', sub: '(B, d_out)' },
      ],
      edges: [
        { from: 'b', to: 'align' },
        { from: 'align', to: 'stretch' },
        { from: 'x', to: 'out' },
        { from: 'stretch', to: 'out' },
      ],
      steps: [
        { caption: 'A layer produces x with one row per example, and has one bias vector b shared by every example. The shapes differ: two dimensions against one.', active: ['x', 'b'] },
        { caption: 'Align from the right. b has no batch dimension, so it is treated as having size 1 there: (1, d_out). The trailing d_out matches.', active: ['b', 'align'], edges: ['b->align'] },
        { caption: 'The size-1 batch dimension is stretched to B. numpy does this with a zero stride: every row reads the same memory, nothing is copied.', active: ['align', 'stretch'], edges: ['align->stretch'] },
        { caption: 'The add is now elementwise between two (B, d_out) views, and the result is (B, d_out). In A1.L4 you will undo this stretch in the backward pass.', active: ['x', 'stretch', 'out'], edges: ['x->out', 'stretch->out'] },
      ],
    },
    {
      type: 'callout',
      variant: 'segfault',
      title: 'The bug that does not crash',
      md: `Targets \`y\` come out of a data loader with shape \`(B,)\`. Predictions come out of the model with shape \`(B, 1)\`. You write \`loss = ((pred - y) ** 2).mean()\`.

Broadcasting turns \`(B, 1) − (B,)\` into **\`(B, B)\`**: every prediction minus every target. The loss is a finite, plausible number; it even goes down a little. The model never learns the task. Lab 00's \`broadcast_shape((5, 1), (5,))\` returns \`(5, 5)\` — that is the bug, stated as a shape. Assert shapes at the boundary where data enters the loss.`,
    },
    {
      type: 'prose',
      md: `## Broadcasting instead of loops

Broadcasting is not only a convenience; it is how you avoid Python loops, which run about **85 million** floating-point operations per second on the reference laptop (an Apple M5, CPython 3.12) against **2 trillion** for a numpy matmul (T0.L3). The canonical example, and lab 00's \`pairwise_sq_dists\`: the squared distance between every row of \`X\` (n, d) and every row of \`Y\` (m, d).

\`\`\`python
diff = X[:, None, :] - Y[None, :, :]    # (n, 1, d) - (1, m, d) -> (n, m, d)
d2 = (diff ** 2).sum(axis=-1)           # (n, m)
\`\`\`

\`None\` (or \`np.newaxis\`) inserts a size-1 axis exactly where you want the stretching to happen. The cost is visible in the shape: the intermediate is **n × m × d** numbers. For n = m = 10,000 and d = 768 in float32 that is 307 GB — a reason the real implementation expands \`|x − y|² = |x|² − 2x·y + |y|²\` and uses a matmul instead. Broadcasting removes the loop, not the memory.`,
    },
    {
      type: 'lab',
      lab: 'numpy-shapes',
      brief: 'Implement the broadcasting rule by hand, vectorise the distance matrix, and write a linear layer that accepts any number of batch dimensions — then the FLOP and byte counts from T0.L3 and T0.L4. Runs in your browser.',
    },
    {
      type: 'quiz',
      questions: [
        {
          q: 'logits has shape (32, 10) and you compute logits - logits.max(axis=1). What happens?',
          options: [
            'Each row has its own max subtracted, as intended',
            'An error: (32, 10) and (32,) do not broadcast, because (32,) aligns with the 10, not the 32',
            'Each column has its own max subtracted',
            'The result has shape (32, 32)',
          ],
          correct: [1],
          explanation: 'max(axis=1) returns (32,). Aligned from the right, 32 meets 10: neither is 1, so numpy raises. With keepdims=True the max is (32, 1), which stretches across the 10 columns — the stable-softmax idiom you will write in lab 02. Had the shape been (10, 10) there would be no error and each column would get the wrong row\'s max: the silent version of the same bug.',
        },
        {
          q: 'A loss is computed as ((pred - y) ** 2).mean() with pred of shape (B, 1) and y of shape (B,). Training runs, the loss falls slightly, the model is useless. Why?',
          options: [
            'The learning rate is too small',
            'pred − y broadcasts to (B, B): every prediction is compared with every target, so the loss rewards predicting the batch mean',
            'mean() should be sum()',
            'Squared error cannot train a regression model',
          ],
          correct: [1],
          explanation: '(B, 1) and (B,) align as (B, 1) against (1, B) and broadcast to (B, B). The minimiser of that loss is a constant near the mean of all targets, so the loss improves a little and then stalls. The fix is to make the shapes equal — y[:, None] or pred.squeeze(-1) — and to assert it.',
        },
        {
          q: 'How much memory does the broadcast intermediate X[:, None, :] - Y[None, :, :] need for n = 50,000, m = 1,000, d = 512 in float32?',
          options: ['about 100 MB', 'about 1 GB', 'about 102 GB', 'none — broadcasting never allocates'],
          correct: [2],
          explanation: 'The stretched inputs are views, but the subtraction materialises its result: n × m × d = 2.56 × 10¹⁰ floats × 4 bytes = 102 GB. Broadcasting avoids copying the inputs; it does not avoid allocating the output.',
        },
      ],
    },
    {
      type: 'deepdive',
      title: 'Going deeper',
      md: `- numpy's own statement of the rule: "Broadcasting" in the numpy user guide (numpy.org/doc/stable/user/basics.broadcasting.html).
- \`np.einsum\` names every axis and makes the contraction explicit: \`np.einsum('btc,cd->btd', x, W)\` is the \`x @ W\` above. M1.L3 writes attention with it.
- Strides: \`x.strides\` shows how many bytes to skip per step along each axis. A broadcast axis has stride 0 — that is the whole trick.`,
    },
  ],
}

export default lesson

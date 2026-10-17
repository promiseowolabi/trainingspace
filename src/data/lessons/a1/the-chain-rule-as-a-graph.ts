import type { Lesson } from '../types'

const lesson: Lesson = {
  id: 'a1.l2',
  slug: 'the-chain-rule-as-a-graph',
  trackId: 'a1',
  index: 2,
  title: 'The Chain Rule as a Graph',
  minutes: 18,
  hook: 'Every gradient a framework computes is the chain rule applied by walking a graph backwards. Once you see the walk, backpropagation stops being an algorithm to memorise and becomes bookkeeping.',
  exercise: 'read+quiz',
  takeaway: {
    number: 'backward ≈ 2× forward',
    claim: 'Reverse mode returns the derivative of one loss with respect to every parameter in a single backward walk costing about twice the forward pass — the reason a training step is roughly 3× a forward pass.',
  },
  blocks: [
    {
      type: 'prose',
      md: `## The chain rule, as a sentence

If \`L\` depends on \`d\`, and \`d\` depends on \`a\`, then

\`\`\`
∂L/∂a = ∂L/∂d · ∂d/∂a
\`\`\`

In words: how much L moves per unit of \`a\` is how much L moves per unit of \`d\`, times how much \`d\` moves per unit of \`a\`. Sensitivities **multiply along a path**. And when \`a\` reaches L by several paths, the contributions **add**. Those two rules — multiply along paths, add across paths — are the whole of backpropagation.

## Write the computation as a graph

Take \`L = (a·b + c)²\` with \`a = 2\`, \`b = −3\`, \`c = 10\`. Break it into one operation per node:

\`\`\`
e = a · b      = −6
d = e + c      =  4
L = d²         = 16
\`\`\`

Every node knows its **local derivatives** — how its output moves with each of its inputs — from its own operation alone, without knowing anything about the rest of the graph:

| node | local derivatives |
|---|---|
| \`e = a·b\` | ∂e/∂a = b = −3, ∂e/∂b = a = 2 |
| \`d = e + c\` | ∂d/∂e = 1, ∂d/∂c = 1 |
| \`L = d²\` | ∂L/∂d = 2d = 8 |`,
    },
    {
      type: 'diagram',
      caption: 'fig 1 — the backward walk for L = (a·b + c)²',
      height: 64,
      nodes: [
        { id: 'a', x: 2, y: 4, w: 14, h: 10, label: 'a = 2', sub: 'grad −24' },
        { id: 'b', x: 2, y: 22, w: 14, h: 10, label: 'b = −3', sub: 'grad 16' },
        { id: 'c', x: 2, y: 44, w: 14, h: 10, label: 'c = 10', sub: 'grad 8' },
        { id: 'e', x: 30, y: 12, w: 16, h: 12, label: 'e = a·b', sub: '−6 · grad 8' },
        { id: 'd', x: 58, y: 26, w: 16, h: 12, label: 'd = e + c', sub: '4 · grad 8' },
        { id: 'L', x: 84, y: 26, w: 14, h: 12, label: 'L = d²', sub: '16 · grad 1' },
      ],
      edges: [
        { from: 'a', to: 'e' },
        { from: 'b', to: 'e' },
        { from: 'e', to: 'd' },
        { from: 'c', to: 'd' },
        { from: 'd', to: 'L' },
      ],
      steps: [
        { caption: 'The forward pass computes values left to right and every node remembers its inputs: e = −6, d = 4, L = 16.', active: ['a', 'b', 'c', 'e', 'd', 'L'], edges: ['a->e', 'b->e', 'e->d', 'c->d', 'd->L'] },
        { caption: 'Seed the walk: the derivative of L with respect to itself is 1. Then L = d² sends 1 × 2d = 8 to d.', active: ['L', 'd'], edges: ['d->L'] },
        { caption: 'd = e + c has local derivative 1 for both inputs, so an addition simply copies its incoming gradient, 8, to e and to c. c is finished: ∂L/∂c = 8.', active: ['d', 'e', 'c'], edges: ['e->d', 'c->d'] },
        { caption: 'e = a·b multiplies the incoming 8 by the OTHER input: a receives 8 × b = −24 and b receives 8 × a = 16. Every leaf now holds ∂L/∂leaf.', active: ['e', 'a', 'b'], edges: ['a->e', 'b->e'] },
      ],
    },
    {
      type: 'prose',
      md: `Check one answer by nudging (A1.L1): \`L(a = 2.001) = (2.001·(−3) + 10)² = 3.997² = 15.976\`, so L fell by 0.024 when \`a\` rose by 0.001 — a derivative of −24. The walk got it without any nudging.

## The rule of the walk

Each node, when its turn comes, receives **one number** — ∂L/∂(its output), called the upstream gradient — and does one thing:

\`\`\`
for each input:  input.grad += upstream_grad × local_derivative(input)
\`\`\`

The \`+=\` is the "add across paths" rule: a value used twice gets gradient from both uses. The order of visits matters: a node may only run once **every** node that consumes its output has run, or its upstream gradient is incomplete. That order is a **reverse topological sort** of the graph — lab 01 has you build it, without recursion.`,
    },
    {
      type: 'isomorphism',
      title: 'You already walk dependency graphs',
      pairs: [
        {
          os: 'a build system (make, Bazel)',
          osLine: 'targets run only after everything they depend on: topological order',
          llm: 'the forward pass',
          llmLine: 'nodes compute only after their inputs: topological order',
        },
        {
          os: 'cost attribution in a call tree',
          osLine: 'a function\'s cost is pushed down to callees in proportion to the calls, and summed when shared',
          llm: 'the backward pass',
          llmLine: 'a node\'s gradient is pushed to its inputs in proportion to local derivatives, and summed when shared',
        },
      ],
    },
    {
      type: 'prose',
      md: `## Why reverse, not forward

You could also walk the graph **forwards**: pick one input, push "∂·/∂a = 1" through the graph, and get ∂L/∂a at the end. That is **forward-mode** differentiation, and it gives one input's sensitivity per pass. **Reverse mode** gives every input's sensitivity for one output per pass.

Training has exactly one output — the scalar loss — and up to billions of inputs. Forward mode would need a pass per parameter (no better than nudging); reverse mode needs one. That asymmetry is why every deep-learning framework is a reverse-mode autograd engine.

**What it costs.** Each node does about as much work in its backward step as in its forward step, and usually more: a matmul \`Y = X·W\` needs two matmuls going back — one for ∂L/∂X and one for ∂L/∂W (A1.L4). So:

\`\`\`
backward ≈ 2 × forward       one training step ≈ 3 × forward
\`\`\`

That "3" becomes the 6 in **6·N·D**, the training-FLOPs formula in P1.L4: a forward pass is 2 FLOPs per parameter per token, and the step is three times that.

**What it stores.** The local derivatives need the forward values — \`e = a·b\` needs \`a\` and \`b\` to send gradient back. Reverse mode therefore keeps every intermediate value alive until the backward walk reaches it. That is the activation memory of T0.L4: the price of reverse mode, paid in bytes.`,
    },
    {
      type: 'quiz',
      questions: [
        {
          q: 'y = x·x. Using the walk (not calculus), what gradient does x receive, and why?',
          options: [
            'x, because one input gets the other input\'s value',
            '2x: the multiply node sends upstream × (other input) to each input slot; both slots are x, and the two contributions add',
            'x², because that is the output',
            '1, because x is a leaf',
          ],
          correct: [1],
          explanation: 'The multiply node sends upstream·b to a and upstream·a to b. Here a and b are the same node, so x receives x + x = 2x. Overwriting instead of accumulating (= instead of +=) gives x — the most common bug in a hand-written engine, and lab 01\'s "accumulates" check.',
        },
        {
          q: 'A model has 10⁹ parameters and one scalar loss. Why does training use reverse mode rather than forward mode?',
          options: [
            'Forward mode is numerically unstable',
            'Reverse mode gives all 10⁹ partial derivatives in one backward pass; forward mode gives one input\'s derivative per pass, so it would need 10⁹ passes',
            'Forward mode cannot handle matrix multiplication',
            'Reverse mode uses less memory',
          ],
          correct: [1],
          explanation: 'The cost of each mode scales with the side of the function it starts from: forward mode with the number of inputs, reverse mode with the number of outputs. Training has one output. Reverse mode does pay in memory — it must store forward values — so "uses less memory" is the opposite of the truth.',
        },
        {
          q: 'Why does a training step cost roughly three forward passes of compute?',
          options: [
            'The optimiser step costs two forward passes',
            'The backward pass computes gradients for both a layer\'s inputs and its weights — about two matmuls for every one in the forward pass',
            'Gradients are computed in float64',
            'The loss is evaluated three times',
          ],
          correct: [1],
          explanation: 'For Y = X·W the backward pass needs ∂L/∂X = G·Wᵀ (to keep propagating) and ∂L/∂W = Xᵀ·G (to update the weight): two matmuls of the same size as the forward one. Forward + backward ≈ 1 + 2 = 3. The optimiser step is elementwise and comparatively negligible in FLOPs.',
        },
      ],
    },
    {
      type: 'deepdive',
      title: 'Going deeper',
      md: `- Christopher Olah, "Calculus on Computational Graphs: Backpropagation" (colah.github.io) — the forward- vs reverse-mode argument with pictures.
- Baydin et al., "Automatic Differentiation in Machine Learning: a Survey" (JMLR, 2018) — the general theory, including forward mode with dual numbers.
- Next: A1.L3 turns this walk into about 100 lines of Python.`,
    },
  ],
}

export default lesson

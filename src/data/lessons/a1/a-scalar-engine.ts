import type { Lesson } from '../types'

const lesson: Lesson = {
  id: 'a1.l3',
  slug: 'a-scalar-engine',
  trackId: 'a1',
  index: 3,
  title: 'A Scalar Engine',
  minutes: 20,
  hook: 'An autograd engine is a number that remembers how it was made. Give every value a list of parents and a closure that knows its local derivative, walk the graph backwards once, and you have the core of PyTorch in about a hundred lines.',
  exercise: 'lab+quiz',
  takeaway: {
    number: '68,140 nodes for 337 parameters',
    claim: 'A scalar engine builds one graph node per arithmetic operation — 68,140 of them for a 337-parameter MLP on 100 examples — which is why it runs at 0.4 MFLOP/s and why real frameworks put a whole array in each node.',
  },
  blocks: [
    {
      type: 'prose',
      md: `A1.L2 walked a graph by hand. To make a program walk it, every intermediate value needs to carry three things beyond its number:

1. **\`grad\`** — a slot for ∂L/∂(this value), starting at 0.
2. **\`_prev\`** — the values it was computed from: its edges in the graph.
3. **\`_backward\`** — a closure that, when called, pushes \`self.grad\` into its parents' \`grad\` using the local derivatives of the operation that made it.

The trick is that you never build the graph explicitly. Overload \`+\`, \`*\` and friends so that every operation creates a new value with those three fields filled in, and **running the forward pass records the graph as a side effect**. This is the design popularised by Andrej Karpathy's *micrograd*, and it is, in miniature, how PyTorch's autograd works.`,
    },
    {
      type: 'code',
      filename: 'engine.py (shape of lab 01)',
      code: `class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None     # leaves have nothing to push
        self._prev = tuple(_children)
        self._op = _op

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():                  # runs later, during the walk
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out`,
      highlightLines: [12, 13, 14, 15],
      chips: ['closure captures self, other, out', '+= not ='],
    },
    {
      type: 'prose',
      md: `Read the closure carefully, because every operation you write in lab 01 has the same three ingredients:

- **\`out.grad\`** is the upstream gradient — by the time this runs, every consumer of \`out\` has already added its share.
- **\`other.data\`** is the local derivative of \`a·b\` with respect to \`a\`. It is a *forward* value, captured by the closure — the stored activation of T0.L4.
- **\`+=\`** accumulates, so a value used twice gets both contributions.

Then \`backward()\` itself is three steps:

\`\`\`python
def backward(self):
    order = topological_order(self)   # every node after all of its inputs
    self.grad = 1.0                   # dL/dL
    for node in reversed(order):      # consumers before producers
        node._backward()
\`\`\``,
    },
    {
      type: 'callout',
      variant: 'segfault',
      title: 'Two bugs every engine has once',
      md: `**The overwrite.** Writing \`self.grad = other.data * out.grad\` passes every simple test — until a value is used twice. \`y = x·x + x\` at x = 3 should give 7; the overwrite gives 1 or 3 depending on visit order. Lab 01's \`accumulates\` check is exactly this expression.

**The recursion.** The obvious topological sort is a recursive depth-first search. Python's default recursion limit is 1,000 frames, and a training graph is thousands of nodes deep: one node per operation per layer per token. Lab 01's \`deep-graph\` check builds a 5,000-node chain. Use an explicit stack: push a node, push its children, emit the node when you come back to it.`,
    },
    {
      type: 'prose',
      md: `## A neuron on top

Once \`Value\` exists, a neuron is ordinary Python — the engine does not know or care that it is part of a neural network:

\`\`\`python
def neuron(weights, bias, xs):
    return sum((w * x for w, x in zip(weights, xs)), bias).tanh()
\`\`\`

Stack neurons into layers and layers into an MLP, compute a loss, call \`loss.backward()\`, and every weight's \`.grad\` holds its partial derivative. Then the step from T0.L1 — and one easily forgotten line:

\`\`\`python
for p in params:
    p.data -= lr * p.grad
    p.grad = 0.0            # gradients accumulate across backward() calls too
\`\`\`

PyTorch has the same semantics, which is why every PyTorch loop calls \`optimizer.zero_grad()\`. Forgetting it is the same bug as the overwrite, inverted: gradients from step 1 leak into step 2.`,
    },
    {
      type: 'statline',
      stats: [
        { value: '337', label: 'parameters', hint: 'an MLP 2 → 16 → 16 → 1 with tanh, built from Value' },
        { value: '68,140', label: 'graph nodes', hint: 'one forward pass over 100 examples — every add, multiply and tanh is a Python object' },
        { value: '148 ms', label: 'forward + backward', hint: '118 ms forward, 30 ms backward on the reference laptop' },
        { value: '0.4 MFLOP/s', label: 'useful throughput', hint: '60,800 matmul FLOPs ÷ 148 ms — about 5 million times slower than numpy\'s matmul (T0.L3)' },
      ],
    },
    {
      type: 'prose',
      md: `## Why this is not how you train a model

The engine is correct and hopeless. Each scalar multiply allocates a Python object, a tuple and a closure; the forward pass spends most of its 118 ms creating 68,140 of them. Useful arithmetic runs at 0.4 MFLOP/s — five million times below what a numpy matmul achieves on the same laptop.

The fix is not a faster scalar engine. It is to make each node hold an **array** and each operation a whole matmul, so that one graph node does a million FLOPs instead of one, and the per-node Python overhead disappears into the noise. Same graph, same walk, same \`+=\` — lab 02 builds it, and A1.L4 works out the local derivatives of array operations, which is where the new difficulty lives: shapes.`,
    },
    {
      type: 'lab',
      lab: 'micrograd',
      brief: 'Write the backward closures for +, ×, ** and four nonlinearities, then an iterative backward(). The checks include the reuse bug, a diamond-shaped graph that needs true reverse topological order, a 5,000-node chain, and a two-layer network checked against central differences.',
    },
    {
      type: 'quiz',
      questions: [
        {
          q: 'Your engine passes every test except one: a node used by two consumers gets the gradient of only one of them. Which line is wrong?',
          options: [
            'self.grad = 1.0 in backward()',
            'A _backward closure assigns (=) to an input\'s .grad instead of accumulating (+=)',
            'The topological sort visits leaves first',
            'The forward value is stored as float instead of int',
          ],
          correct: [1],
          explanation: 'Each consumer\'s closure contributes its share to the shared input; with = the last one to run overwrites the others. Seeding self.grad = 1.0 on the output is correct, and a topological order that lists leaves first is exactly what reversed() needs.',
        },
        {
          q: 'Why must _backward closures run in REVERSE topological order?',
          options: [
            'It is faster',
            'A node may only push gradient to its inputs once its own .grad is complete — i.e. after every node that consumes it has pushed its contribution',
            'Python requires closures to run in reverse creation order',
            'Forward order would compute second derivatives',
          ],
          correct: [1],
          explanation: 'A node\'s _backward multiplies its out.grad into its inputs. If it runs before a later consumer has added its share to out.grad, the inputs receive an incomplete gradient and nothing corrects it afterwards. Reverse topological order guarantees every consumer has run first. Lab 01\'s topo-order check builds a graph where any other order gives the wrong answer.',
        },
        {
          q: 'A training loop calls loss.backward() every step but never resets .grad. What happens?',
          options: [
            'Nothing — backward() overwrites old gradients',
            'Each step\'s gradient is the sum of all previous steps\' gradients, so updates grow and point increasingly in stale directions',
            'The graph is rebuilt, so gradients are fresh',
            'Python raises an error',
          ],
          correct: [1],
          explanation: 'Parameters are long-lived leaves; their .grad slot accumulates by design, across backward() calls as well as within one. Without a reset, step n applies the sum of n gradients. That is why PyTorch loops call optimizer.zero_grad() — and why gradient accumulation (P1.L2) is just "do not zero for k micro-batches" on purpose.',
        },
      ],
    },
    {
      type: 'deepdive',
      title: 'Going deeper',
      md: `- Andrej Karpathy, *micrograd* (github.com/karpathy/micrograd) and the lecture "The spelled-out intro to neural networks and backpropagation" — the engine lab 01 is shaped after.
- PyTorch's autograd works the same way at array granularity: every tensor produced by an operation carries a \`grad_fn\` (its \`_backward\`) whose \`next_functions\` are its \`_prev\`. Print \`loss.grad_fn.next_functions\` on any PyTorch loss and walk it.
- Stretch goal after lab 01: add a \`Neuron\`, \`Layer\` and \`MLP\` class and train on four XOR points until the loss is below 0.01.`,
    },
  ],
}

export default lesson

import type { LabGuide } from './types'

const guide: LabGuide = {
  id: 'micrograd',
  requires: ['a browser — or Python 3.10+ and pytest (no numpy needed)'],
  blocks: [
    {
      type: 'prose',
      md: `## What you will build

The \`Value\` class of A1.L3: a float that remembers which values it was computed from and how to send gradient back to them. The forward arithmetic for \`+\`, \`*\` and \`**\` is given, so the checks on forward values can pass early. You write:

1. the \`_backward\` closures of \`__add__\`, \`__mul__\` and \`__pow__\`;
2. \`exp\`, \`log\`, \`tanh\` and \`relu\` — forward and backward;
3. \`backward()\` — an iterative topological sort and the reverse walk.

\`__neg__\`, \`__sub__\`, \`__truediv__\` and the reflected operators are given; they are built from the operations above, so they work as soon as those do.`,
    },
    {
      type: 'prose',
      md: `## 1. The three closures

Each closure reads \`out.grad\` (the upstream gradient) and adds \`out.grad × local derivative\` into each input's \`.grad\`. The local derivatives are the table in A1.L2: 1 and 1 for addition, the *other* input for multiplication, \`k · x^(k−1)\` for a constant power. Use \`+=\` every time.

Nothing passes yet except \`forward-values\` once step 2 is done: every gradient check needs \`backward()\`.

## 2. The nonlinearities

Each one follows the pattern of the given operations: compute the forward value, create \`out = Value(result, (self,), "name")\`, attach a closure, return \`out\`. Two of them can reuse the forward value in the backward pass — \`exp\` (its derivative is itself) and \`tanh\` (1 − t²) — so compute it once and let the closure capture it.

## 3. backward()

Build the order with an explicit stack. One clean way: push \`(node, False)\`; when you pop a node with \`False\`, mark it seen, push \`(node, True)\`, then push its unseen children with \`False\`; when you pop a node with \`True\`, append it to the order — all its inputs are already there. Then seed \`self.grad = 1.0\` and call \`_backward\` on \`reversed(order)\`.`,
    },
    {
      type: 'code',
      filename: 'try it in a REPL',
      code: `from engine import Value
x = Value(3.0)
y = x * x + x
y.backward()
print(x.grad)   # 7.0 — if you see 1.0 or 3.0, some closure uses = instead of +=`,
    },
    {
      type: 'callout',
      variant: 'segfault',
      title: 'RecursionError on deep-graph',
      md: `If \`deep-graph\` fails with a RecursionError (the browser shows "walk the graph with a loop, not recursion"), your topological sort calls itself. A 5,000-node chain needs 5,000 frames; Python allows about 1,000. The explicit stack above has no depth limit.`,
    },
    {
      type: 'prose',
      md: `## The last check

\`matches-numeric\` builds a small network with a tanh and a relu unit, back-propagates through your engine, and compares every one of its nine parameter gradients with central differences (A1.L1) at ε = 10⁻⁶. If everything else is green and this is red, one of your nonlinearities has a derivative that is right at one point and wrong in general — check it against its formula.`,
    },
  ],
  cleanup: [
    {
      type: 'prose',
      md: `- Add \`Neuron\`, \`Layer\` and \`MLP\` classes and train on the four XOR points (\`[0,0]→0, [0,1]→1, [1,0]→1, [1,1]→0\`) until the loss is below 0.01. Count the graph nodes per step.
- Add a \`sigmoid\` and check it against central differences yourself.
- Lab 02 starts from the same \`backward()\`; keep yours to compare.`,
    },
  ],
}

export default guide

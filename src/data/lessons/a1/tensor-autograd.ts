import type { Lesson } from '../types'

const lesson: Lesson = {
  id: 'a1.l4',
  slug: 'tensor-autograd',
  trackId: 'a1',
  index: 4,
  title: 'Tensor Autograd',
  minutes: 21,
  hook: 'Put a whole array in each graph node and the engine gets 1,700× faster. The price is that every local derivative must now get its shape right — and the rule that does it is: the gradient has the shape of the thing it is the gradient of.',
  exercise: 'lab+quiz',
  takeaway: {
    number: '2 matmuls back per 1 forward',
    claim: 'For Y = X·W the backward pass is two matmuls — dW = Xᵀ·G and dX = G·Wᵀ — whose transposes are forced by shape alone, and a broadcast in the forward pass is always a sum in the backward pass.',
  },
  blocks: [
    {
      type: 'prose',
      md: `Lab 01's engine made one node per scalar operation. Lab 02 keeps the graph, the topological walk and the \`+=\` exactly as they were, and changes one thing: a node's \`data\` and \`grad\` are numpy arrays. The same MLP that took 148 ms per forward-and-backward on the scalar engine takes **0.087 ms** on the array engine on the reference laptop — about 1,700× — because the graph shrinks from 68,140 nodes to 27 and each node is a compiled numpy call.

Everything that is hard about array autograd comes from one invariant:

> **\`x.grad\` always has the same shape as \`x.data\`.**

There is one partial derivative per element (A1.L1), so the gradient of a \`(B, d)\` activation is \`(B, d)\`, and the gradient of a \`(d, d_out)\` weight is \`(d, d_out)\`. Every rule below is the scalar rule from lab 01 plus whatever reshaping that invariant requires.`,
    },
    {
      type: 'prose',
      md: `## Elementwise operations: the scalar rule, applied per element

\`relu\`, \`exp\`, \`tanh\`, and \`+\` or \`*\` between same-shaped arrays act on each element independently, so the local derivative is applied element by element:

| forward | backward (G = out.grad) |
|---|---|
| \`out = x + y\` | \`x.grad += G\`, \`y.grad += G\` |
| \`out = x * y\` | \`x.grad += G * y\`, \`y.grad += G * x\` |
| \`out = relu(x)\` | \`x.grad += G * (x > 0)\` |

## Broadcasting: a stretch forward is a sum backward

The bias add \`x @ W + b\` has \`(B, d)\` on the left and \`(d,)\` on the right. Broadcasting (T0.L2) used the single row \`b\` for all B rows. So each element of \`b\` influenced B outputs, and by the add-across-paths rule its gradient is the **sum** of those B upstream gradients:

\`\`\`python
b.grad += G.sum(axis=0)        # (B, d) -> (d,)
\`\`\`

In general: wherever the forward pass stretched an input — a prepended axis, or a size-1 axis — the backward pass sums the gradient along that axis back down to the input's shape. Lab 02 calls this \`unbroadcast(grad, shape)\` and it is used by every binary operation.`,
    },
    {
      type: 'diagram',
      caption: 'fig 1 — the bias gradient is a column sum',
      height: 62,
      nodes: [
        { id: 'b', x: 2, y: 6, w: 18, h: 12, label: 'b', sub: '(d,)' },
        { id: 'stretch', x: 28, y: 6, w: 22, h: 12, label: 'stretched', sub: 'used by all B rows' },
        { id: 'out', x: 58, y: 6, w: 18, h: 12, label: 'x·W + b', sub: '(B, d)' },
        { id: 'G', x: 58, y: 40, w: 18, h: 12, label: 'G', sub: '(B, d) upstream' },
        { id: 'sum', x: 28, y: 40, w: 22, h: 12, label: 'sum over rows', sub: 'axis=0' },
        { id: 'bg', x: 2, y: 40, w: 18, h: 12, label: 'b.grad', sub: '(d,)' },
      ],
      edges: [
        { from: 'b', to: 'stretch' },
        { from: 'stretch', to: 'out' },
        { from: 'out', to: 'G' },
        { from: 'G', to: 'sum' },
        { from: 'sum', to: 'bg' },
      ],
      steps: [
        { caption: 'Forward: one bias vector of shape (d,) is broadcast across all B rows of the batch, so each bias element contributes to B different outputs.', active: ['b', 'stretch', 'out'], edges: ['b->stretch', 'stretch->out'] },
        { caption: 'Backward: the upstream gradient G has the output\'s shape (B, d) — one sensitivity per output element, including the B copies each bias element fed.', active: ['out', 'G'], edges: ['out->G'] },
        { caption: 'A value used B times receives B contributions, and contributions add. Summing G over the batch axis collects them into one number per bias element.', active: ['G', 'sum'], edges: ['G->sum'] },
        { caption: 'The result has b\'s own shape, (d,). The invariant holds: a gradient always has the shape of the thing it is the gradient of.', active: ['sum', 'bg'], edges: ['sum->bg'] },
      ],
    },
    {
      type: 'prose',
      md: `## Matmul: let the shapes derive it

\`Y = X @ W\` with X of shape \`(B, k)\`, W of shape \`(k, n)\`, so Y and its upstream gradient G are \`(B, n)\`. You need \`X.grad\` of shape \`(B, k)\` and \`W.grad\` of shape \`(k, n)\`, and each must be some product of G with the *other* input — exactly as in the scalar \`a·b\` rule. There is only one way to multiply those shapes together:

\`\`\`
X.grad = G @ W.T      (B, n) @ (n, k) -> (B, k)   ✓
W.grad = X.T @ G      (k, B) @ (B, n) -> (k, n)   ✓
\`\`\`

This "dimension-matching" derivation is how practitioners remember it, and it is rigorous enough: working out ∂L/∂W_ij element by element gives \`Σ_b X_bi · G_bj\`, which is the (i, j) entry of \`Xᵀ G\`.

Note what \`W.grad = X.T @ G\` needs: **X**, the layer's input from the forward pass. That is the stored activation again — and it is also where the sum over the batch happens. W is shared by every example, exactly like the bias, so its gradient sums over the B rows; the matmul does that sum for you.`,
    },
    {
      type: 'code',
      filename: 'tensor.py — the matmul node (lab 02)',
      code: `def __matmul__(self, other):
    out = Tensor(self.data @ other.data, (self, other), "@")

    def _backward():
        self.grad += out.grad @ other.data.T     # dX = G · Wᵀ
        other.grad += self.data.T @ out.grad     # dW = Xᵀ · G

    out._backward = _backward
    return out`,
      chips: ['2 matmuls back for 1 forward', 'A1.L2\'s "backward ≈ 2×"'],
    },
    {
      type: 'prose',
      md: `## Reductions: the opposite of broadcasting

\`sum(axis=0)\` turns \`(B, d)\` into \`(d,)\`; every input element contributed with weight 1 to one output element. So backward **broadcasts** the gradient back: each input element receives the upstream gradient of the output it was summed into. If \`keepdims=False\` removed the axis, put it back first with \`np.expand_dims\` so the broadcast lines up — otherwise a \`(B,)\` gradient aligns with the wrong axis of a \`(B, d)\` input (T0.L2's silent bug, in reverse). \`mean\` is \`sum\` times 1/n.

Broadcast forward ↔ sum backward; sum forward ↔ broadcast backward. Those two dualities are most of the shape logic in every framework.

## Fused operations

Frameworks do not build cross-entropy out of \`exp\`, \`sum\`, \`log\` and indexing nodes. They write one node whose forward is the numerically stable log-softmax and whose backward is the closed form \`(softmax − onehot) / N\`, derived in O1.L2. Fewer nodes, fewer stored intermediates, no overflow. Lab 02 has you write that fused node; it is the loss for every language model in this course.`,
    },
    {
      type: 'lab',
      lab: 'tensor-autograd',
      brief: 'Write unbroadcast, the backward passes for +, ×, @, sum and relu, a fused stable cross-entropy and a gradient checker — then the lab trains a 2 → 16 → 3 MLP on your engine and expects the loss below 0.15.',
    },
    {
      type: 'quiz',
      questions: [
        {
          q: 'In Y = X @ W, X is (64, 512) and W is (512, 128). What are the shapes of X.grad and W.grad, and what is W.grad computed from?',
          options: [
            'X.grad (64, 128), W.grad (512, 512), from G alone',
            'X.grad (64, 512) = G @ Wᵀ; W.grad (512, 128) = Xᵀ @ G — W\'s gradient needs the stored input X',
            'X.grad (512, 64), W.grad (128, 512), both transposed',
            'Both (64, 128), the shape of Y',
          ],
          correct: [1],
          explanation: 'Gradients take the shape of what they differentiate: (64, 512) and (512, 128). G is (64, 128); the only products that produce those shapes are G·Wᵀ and Xᵀ·G. The second needs X from the forward pass, which is why the input to every linear layer is kept until backward.',
        },
        {
          q: 'A hand-written layer adds a bias of shape (1, 256) to activations of shape (32, 256). Its backward does bias.grad += G. What goes wrong?',
          options: [
            'Nothing — numpy handles it',
            'bias.grad += G broadcasts (32, 256) into (1, 256) and numpy raises; the gradient must be summed over the batch axis with keepdims to (1, 256)',
            'The bias gradient is 32× too small',
            'The bias receives the gradient of only the first example',
          ],
          correct: [1],
          explanation: 'An in-place += cannot shrink the left operand: numpy raises "non-broadcastable output operand". The bias was stretched over 32 rows in the forward pass, so its gradient is G.sum(axis=0, keepdims=True). In frameworks without that error the same mistake can surface as a gradient for one row only — which is why unbroadcast is a function with its own tests.',
        },
        {
          q: 'Why does the array engine train the same MLP about 1,700× faster than the scalar engine, with identical gradients?',
          options: [
            'Arrays use lower precision',
            'The graph has 27 nodes instead of 68,140, so per-node Python overhead nearly vanishes and the arithmetic runs inside compiled numpy kernels',
            'Array autograd skips the topological sort',
            'numpy computes gradients symbolically',
          ],
          correct: [1],
          explanation: 'Both engines do the same walk with the same rules. The scalar one spends its time creating Python objects and calling closures, one per multiply. The array one does a whole matmul per node in compiled code. It is the same lesson as T0.L3\'s 24,000× loop-versus-matmul gap, applied to the backward pass.',
        },
      ],
    },
    {
      type: 'deepdive',
      title: 'Going deeper',
      md: `- Justin Johnson, "Backpropagation for a Linear Layer" (Stanford CS231n notes) — the element-by-element derivation of dW = XᵀG.
- PyTorch's \`torch/csrc/autograd/FunctionsManual.cpp\` and \`tools/autograd/derivatives.yaml\` list the backward formula of every operator; \`sum_to_size\` is its \`unbroadcast\`.
- Stretch goal after lab 02: add \`tanh\`, \`exp\` and a batched matmul for 3-D inputs, and gradient-check each with your own \`gradcheck\`.`,
    },
  ],
}

export default lesson

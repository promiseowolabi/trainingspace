import type { Lesson } from '../types'

const lesson: Lesson = {
  id: 'o1.l1',
  slug: 'the-mlp',
  trackId: 'o1',
  index: 1,
  title: 'The MLP',
  minutes: 18,
  hook: 'Stack linear layers with a nonlinearity between them and you have a network that can learn almost any function — if its weights start at the right size. Start them at N(0, 1) and 91% of the units are dead on arrival.',
  exercise: 'read+quiz',
  takeaway: {
    number: 'std = √(2 / fan_in)',
    claim: 'Initialising ReLU layers with standard deviation √(2/fan_in) keeps the activation scale roughly constant through depth — 0.69 after 20 layers — where 1/√fan_in decays it to 0.0008 and N(0, 1) explodes or saturates it.',
  },
  blocks: [
    {
      type: 'prose',
      md: `A **multilayer perceptron** (MLP) is the simplest network worth training:

\`\`\`
h1 = act(x  @ W1 + b1)
h2 = act(h1 @ W2 + b2)
y  =     h2 @ W3 + b3
\`\`\`

Linear layers (T0.L2) with an elementwise **activation function** between them. Every transformer block contains one — the "MLP" or "feed-forward" sub-block that holds two thirds of a GPT's parameters (M1.L5).

## Why the nonlinearity is not optional

Remove \`act\` and the network collapses: \`(x @ W1) @ W2 = x @ (W1 @ W2)\`, a single linear layer no matter how many you stack. Depth buys nothing without a nonlinearity in between. With one, an MLP with enough hidden units can approximate any continuous function on a bounded input — the **universal approximation theorem** — though the theorem says nothing about whether gradient descent will *find* those weights. Everything else in this track is about that second question.

The activations you will meet:

| name | formula | where |
|---|---|---|
| tanh | (eˣ − e⁻ˣ) / (eˣ + e⁻ˣ) | lab 01, older networks |
| ReLU | max(0, x) | lab 02, most CNNs |
| GELU | x · Φ(x) | GPT-2, BERT |
| SwiGLU | (x·W) ⊙ swish(x·V) | Llama and most current LLMs (M1.L4) |`,
    },
    {
      type: 'prose',
      md: `## Initialisation is arithmetic about variance

Feed a batch through 20 layers of width 512 and measure the standard deviation of the activations. The only thing that changes between rows is the scale of the random initial weights. Measured with numpy on the reference laptop:

| init | activation | std, layer 1 | layer 10 | layer 20 | note |
|---|---|---|---|---|---|
| N(0, 1) | tanh | 0.98 | 0.98 | 0.98 | **91% of units saturated** at ±1 |
| N(0, 0.01²) | tanh | 0.22 | 3 × 10⁻⁷ | **1 × 10⁻¹³** | the signal vanishes |
| 1/√n (Xavier) | tanh | 0.63 | 0.23 | 0.16 | slow decay — usable |
| 1/√n (Xavier) | ReLU | 0.58 | 0.027 | **0.0008** | wrong rule for ReLU |
| √(2/n) (Kaiming) | ReLU | 0.83 | 0.77 | **0.69** | stable |

The first row looks healthy — standard deviation 0.98 everywhere — and is the worst of all. Almost every unit sits on tanh's flat tails, where the derivative 1 − tanh² is about zero, so **no gradient flows backward**. A network can have well-scaled activations and dead gradients at the same time.`,
    },
    {
      type: 'prose',
      md: `### Where the √(2/n) comes from

One output of a linear layer is a sum of n terms \`w_i · x_i\`. If the weights and inputs are independent with mean zero, variances add:

\`\`\`
Var(out) = n · Var(w) · Var(x)
\`\`\`

To keep \`Var(out) = Var(x)\` you need \`Var(w) = 1/n\` — std 1/√n, **Xavier/Glorot** initialisation. ReLU then zeroes half of its inputs, halving the variance it passes on, so to compensate \`Var(w) = 2/n\` — std √(2/n), **Kaiming/He** initialisation. That is the whole derivation, and it is why the Xavier-ReLU row decays by a factor of about √2 per layer.

Transformers add one more rule on top: the output projections of each residual branch are often scaled down by a further 1/√(2·layers) (GPT-2), so that summing many branches into the residual stream (O1.L5) does not grow its variance with depth.`,
    },
    {
      type: 'vendor',
      snapshot: '2026-10',
      title: 'What PyTorch does by default',
      md: `\`torch.nn.Linear\` initialises its weight from **U(−1/√fan_in, 1/√fan_in)** (it calls \`kaiming_uniform_\` with \`a = √5\`, which works out to that bound), giving a standard deviation of 1/√(3·fan_in) — smaller than either rule above. Its bias uses the same bound. That default is a reasonable start for shallow models; deep networks and transformers re-initialise explicitly (\`nn.init.normal_(w, std=0.02)\` in GPT-2-style code, or the rules above).`,
      systems: ['pytorch'],
      sources: [
        'https://pytorch.org/docs/stable/generated/torch.nn.Linear.html',
        'https://pytorch.org/docs/stable/nn.init.html',
      ],
    },
    {
      type: 'diagram',
      caption: 'fig 1 — an MLP\'s forward and backward pass',
      height: 64,
      nodes: [
        { id: 'x', x: 2, y: 8, w: 12, h: 12, label: 'x', sub: '(B, 2)' },
        { id: 'l1', x: 20, y: 8, w: 16, h: 12, label: 'x·W1 + b1', sub: 'linear' },
        { id: 'a1', x: 42, y: 8, w: 14, h: 12, label: 'ReLU', sub: 'half zeroed' },
        { id: 'l2', x: 62, y: 8, w: 16, h: 12, label: 'h·W2 + b2', sub: 'logits' },
        { id: 'loss', x: 84, y: 8, w: 14, h: 12, label: 'loss', sub: 'scalar' },
        { id: 'g', x: 42, y: 42, w: 36, h: 12, label: 'gradients flow back', sub: 'through ReLU only where h > 0' },
      ],
      edges: [
        { from: 'x', to: 'l1' },
        { from: 'l1', to: 'a1' },
        { from: 'a1', to: 'l2' },
        { from: 'l2', to: 'loss' },
        { from: 'loss', to: 'g' },
      ],
      steps: [
        { caption: 'The input is multiplied by the first weight matrix and shifted by a bias: a linear map that, on its own, could be folded into any later linear layer.', active: ['x', 'l1'], edges: ['x->l1'] },
        { caption: 'ReLU zeroes every negative pre-activation. This is what makes the network more than one linear map — and it halves the variance, which is why ReLU layers need the factor 2 in √(2/n).', active: ['l1', 'a1'], edges: ['l1->a1'] },
        { caption: 'A second linear layer produces one score per class, and the loss (O1.L2) reduces the batch to one number.', active: ['a1', 'l2', 'loss'], edges: ['a1->l2', 'l2->loss'] },
        { caption: 'Backward, the gradient passes through ReLU only where the unit was active. A unit that is negative for every input never learns again: a dead ReLU, often the result of a bad init or a too-large step.', active: ['loss', 'g', 'a1'], edges: ['loss->g'] },
      ],
    },
    {
      type: 'callout',
      variant: 'segfault',
      title: 'Healthy forward, dead backward',
      md: `The N(0, 1)-tanh network has the most normal-looking activation statistics in the table and trains worst. Watch **gradient norms per layer** as well as activations: a saturated or dead layer shows a gradient norm orders of magnitude below its neighbours from step one. E1.L2's debugging checklist logs both.`,
    },
    {
      type: 'quiz',
      questions: [
        {
          q: 'A 30-layer ReLU MLP initialised with std 1/√fan_in trains extremely slowly; the first layers barely change. What is the most likely cause?',
          options: [
            'The learning rate is too high',
            'Xavier scaling ignores that ReLU halves the variance, so activations — and gradients — shrink by about √2 per layer, roughly 2¹⁵ over 30 layers',
            'ReLU cannot be used in deep networks',
            'The batch is too small',
          ],
          correct: [1],
          explanation: '1/√n preserves variance for a symmetric activation; ReLU throws away half of it per layer. Over 30 layers that compounds to a factor of about 2¹⁵ ≈ 33,000 in variance. Kaiming init (√(2/n)) is the fix; a higher learning rate would only shift the problem around.',
        },
        {
          q: 'Why is an MLP with no activation functions no more powerful than a single linear layer?',
          options: [
            'It has fewer parameters',
            'A composition of linear maps is a linear map: x·W1·W2 = x·(W1·W2), so depth adds no expressiveness',
            'Gradients cannot flow through linear layers',
            'Biases cancel out',
          ],
          correct: [1],
          explanation: 'Matrix multiplication is associative, so any stack of linear layers equals one linear layer with the product matrix (and a combined bias). Only a nonlinearity between them lets depth represent functions a single layer cannot.',
        },
        {
          q: 'A tanh network\'s activations have standard deviation ≈ 1 in every layer, yet it does not learn. What should you check?',
          options: [
            'Nothing — std ≈ 1 means the init is perfect',
            'Whether units are saturated near ±1: there tanh\'s derivative is ≈ 0, so almost no gradient reaches the weights',
            'Whether the loss is cross-entropy',
            'Whether the biases are zero',
          ],
          correct: [1],
          explanation: 'Std ≈ 1 for tanh outputs means most values are close to ±1 — the saturated tails. The measured N(0, 1) network had 91% of units there. Activation scale alone is not a health check; the gradient has to survive the activation\'s derivative too.',
        },
      ],
    },
    {
      type: 'deepdive',
      title: 'Going deeper',
      md: `- Glorot and Bengio, "Understanding the difficulty of training deep feedforward neural networks" (AISTATS 2010) — the Xavier derivation.
- He et al., "Delving Deep into Rectifiers" (2015, arxiv.org/abs/1502.01852) — the ReLU correction.
- Rerun the table yourself: twenty \`x = act(x @ W)\` lines in numpy, printing \`x.std()\` and \`(abs(x) > 0.99).mean()\`.`,
    },
  ],
}

export default lesson

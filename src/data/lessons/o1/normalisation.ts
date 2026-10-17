import type { Lesson } from '../types'

const lesson: Lesson = {
  id: 'o1.l5',
  slug: 'normalisation',
  trackId: 'o1',
  index: 5,
  title: 'Normalisation and Residuals',
  minutes: 19,
  hook: 'Deep networks are trainable because of two cheap tricks: add each layer\'s output to its input instead of replacing it, and rescale what each layer sees to a standard size. Where you put the rescaling decides whether a 6-layer transformer trains at all.',
  exercise: 'read+quiz',
  takeaway: {
    number: '4.6·10¹⁰ vs 5.8',
    claim: 'Stacking 48 residual ReLU layers without normalisation grew the activation scale to 4.6·10¹⁰; normalising each branch\'s input (pre-norm) held it to 5.8 — and the pre-norm transformer learned at every learning rate where the post-norm one failed.',
  },
  blocks: [
    {
      type: 'prose',
      md: `## Residual connections: add, don't replace

A plain deep network computes \`x ← f(x)\` at every layer. A **residual** network computes

\`\`\`
x ← x + f(x)
\`\`\`

Each layer adds a correction to a running vector — the **residual stream** — instead of rewriting it. Two consequences:

- **Gradients have a highway.** The derivative of \`x + f(x)\` with respect to \`x\` is \`1 + f'(x)\`. However badly the branches behave, the identity term carries the gradient from the loss straight back to the first layer. That is why residual networks train at 100+ layers where plain ones stall at about 20.
- **Every layer reads and writes one shared stream.** In a transformer, attention and MLP blocks each read the stream, compute something, and add it back. M1.L4 builds on this view.

The catch: each branch adds variance. Measured with numpy (width 512, Kaiming-initialised ReLU branches), the standard deviation of the stream after L layers:`,
    },
    {
      type: 'prose',
      md: `| layout | update | layer 1 | 12 | 24 | 48 |
|---|---|---|---|---|---|
| plain residual | x + f(x) | 1.3 | 233 | 1.3 × 10⁵ | **4.6 × 10¹⁰** |
| pre-norm | x + f(norm(x)) | 1.3 | 3.0 | 4.2 | **5.8** |
| post-norm | norm(x + f(x)) | 1.0 | 1.0 | 1.0 | **1.0** |

Without normalisation the stream compounds multiplicatively and overflows any float format long before 48 layers (bf16 and fp16 top out at 3.4 × 10³⁸ and 65,504). Normalisation is what makes the residual trick usable at depth.`,
    },
    {
      type: 'prose',
      md: `## LayerNorm and RMSNorm

**LayerNorm** rescales each token's vector to mean 0 and variance 1, then applies a learned per-feature scale γ and shift β:

\`\`\`python
def layer_norm(x, gamma, beta, eps=1e-5):            # x: (..., d)
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return gamma * (x - mu) / np.sqrt(var + eps) + beta
\`\`\`

**RMSNorm** drops the mean subtraction and the shift: \`gamma * x / sqrt(mean(x²) + eps)\`. It is cheaper — one reduction instead of two, no β — and in practice trains as well, which is why Llama-family models use it. The two are not the same function: on inputs whose mean is not zero they differ (by 0.32 on average in a quick test with mean-1 inputs), so a checkpoint trained with one cannot be run with the other.

Both normalise **per token, across features** — unlike batch normalisation, which normalises each feature across the batch and makes one example's output depend on the others in its batch. Language models avoid batch norm for exactly that reason: the batch at training time and the single sequence at inference time would see different statistics.`,
    },
    {
      type: 'prose',
      md: `## Pre-norm versus post-norm

The original transformer (2017) normalised *after* adding: \`x ← norm(x + f(x))\`. The stream is pinned at scale 1 — but the normalisation sits on the gradient highway, so every layer's backward pass goes through a rescaling, and gradients near the output can be much larger than near the input. Post-norm transformers train, but they are fragile: they need warmup and a carefully chosen learning rate.

GPT-2 and nearly every model since normalise the branch's *input* instead: \`x ← x + f(norm(x))\`. The identity path is untouched, the stream grows slowly (√L-ish, 5.8 at 48 layers above), and one final norm before the output layer cleans it up.

The difference is not cosmetic. Rerunning O1.L4's experiment — the same 6-layer transformer, task and seeds — with \`norm_first=True\`:`,
    },
    {
      type: 'prose',
      md: `| peak lr | post-norm, no warmup | post-norm, warmup | **pre-norm, no warmup** | **pre-norm, warmup** |
|---|---|---|---|---|
| 1 × 10⁻³ | 0.55 · 0.76 · 2.10 | 0.77 · 0.03 · 0.02 | 0.004 · 0.012 · 0.005 | 0.034 · 0.025 · 0.003 |
| 3 × 10⁻³ | 3.47 · 3.47 · 3.47 | 0.42 · 0.78 · 2.10 | 0.018 · 0.013 · 0.005 | 0.001 · 0.028 · 0.017 |
| 1 × 10⁻² | 3.47 · 3.47 · 3.47 | 3.47 · 3.47 · 3.47 | 0.025 · 0.007 · 0.033 | 0.015 · 0.021 · 0.029 |

Pre-norm got below 0.04 on all 18 runs, with or without warmup, at every learning rate; post-norm escaped the uniform-guess loss on 9 of 18 and got below 0.1 on only 2. Final losses after 600 steps, three seeds each, PyTorch \`nn.TransformerEncoderLayer\` on the reference laptop.`,
    },
    {
      type: 'diagram',
      caption: 'fig 1 — one pre-norm transformer layer',
      height: 64,
      nodes: [
        { id: 'in', x: 2, y: 26, w: 12, h: 12, label: 'x', sub: 'stream' },
        { id: 'n1', x: 20, y: 4, w: 14, h: 12, label: 'norm', sub: 'RMS/Layer' },
        { id: 'att', x: 40, y: 4, w: 16, h: 12, label: 'attention', sub: 'M1.L3' },
        { id: 'add1', x: 40, y: 26, w: 16, h: 12, label: 'x + …', sub: 'residual add' },
        { id: 'n2', x: 62, y: 4, w: 14, h: 12, label: 'norm', sub: 'RMS/Layer' },
        { id: 'mlp', x: 82, y: 4, w: 16, h: 12, label: 'MLP', sub: 'O1.L1' },
        { id: 'add2', x: 82, y: 46, w: 16, h: 12, label: 'x + …', sub: 'to next layer' },
      ],
      edges: [
        { from: 'in', to: 'n1' },
        { from: 'n1', to: 'att' },
        { from: 'in', to: 'add1' },
        { from: 'att', to: 'add1' },
        { from: 'add1', to: 'n2' },
        { from: 'n2', to: 'mlp' },
        { from: 'add1', to: 'add2' },
        { from: 'mlp', to: 'add2' },
      ],
      steps: [
        { caption: 'The residual stream arrives. The branch reads a normalised copy of it; the stream itself goes around the branch untouched.', active: ['in', 'n1'], edges: ['in->n1'] },
        { caption: 'Attention computes a correction from the normalised input and adds it back onto the stream. The add is the only place the branch touches the stream.', active: ['n1', 'att', 'add1', 'in'], edges: ['n1->att', 'att->add1', 'in->add1'] },
        { caption: 'The MLP branch repeats the pattern: normalise a copy, transform it, add the result.', active: ['add1', 'n2', 'mlp', 'add2'], edges: ['add1->n2', 'n2->mlp', 'mlp->add2'] },
        { caption: 'Backward, the gradient flows along the straight identity path from add2 to add1 to x with no rescaling in between — the property post-norm gives up by putting the norm on that path.', active: ['in', 'add1', 'add2'], edges: ['in->add1', 'add1->add2'] },
      ],
    },
    {
      type: 'vendor',
      snapshot: '2026-10',
      title: 'What current models use',
      md: `GPT-2 moved LayerNorm to the input of each sub-block and added a final LayerNorm after the last block. LLaMA (2023) uses **pre-normalisation with RMSNorm**, together with SwiGLU activations and rotary position embeddings (M1.L2) — a combination most open models since have adopted. PyTorch's \`nn.TransformerEncoderLayer\` defaults to post-norm (\`norm_first=False\`) for compatibility with the 2017 paper.`,
      systems: ['gpt-2', 'llama', 'pytorch'],
      sources: [
        'https://arxiv.org/abs/2302.13971',
        'https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html',
      ],
    },
    {
      type: 'quiz',
      questions: [
        {
          q: 'You build a transformer with nn.TransformerEncoderLayer and default arguments, and it only trains with a long warmup and a small learning rate. What is the first thing to try?',
          options: [
            'Remove the residual connections',
            'Set norm_first=True: the default is the post-norm layout, which puts normalisation on the gradient path and is far less stable',
            'Replace LayerNorm with batch norm',
            'Increase the number of layers',
          ],
          correct: [1],
          explanation: 'PyTorch defaults to post-norm for compatibility with the 2017 paper. In this lesson\'s experiment the pre-norm version trained at every learning rate tested, with or without warmup; the post-norm one failed at 3·10⁻³ without warmup and at 10⁻² with it.',
        },
        {
          q: 'Why do language models use LayerNorm or RMSNorm rather than batch normalisation?',
          options: [
            'Batch norm has more parameters',
            'Batch norm normalises each feature across the examples in a batch, so outputs depend on the other sequences in the batch and its statistics differ between training batches and single-sequence inference; per-token norms have neither problem',
            'Batch norm cannot be differentiated',
            'Batch norm only works for images',
          ],
          correct: [1],
          explanation: 'Per-token normalisation uses only the token\'s own features, so a sequence\'s output does not depend on what else is in the batch and inference behaves exactly like training. Batch norm needs running statistics and couples examples together.',
        },
        {
          q: 'Why does a residual network train at depths where a plain network does not?',
          options: [
            'It has fewer parameters',
            'The identity term in x + f(x) gives the gradient a path from the loss to every layer that does not pass through any branch, so it cannot vanish however many layers there are',
            'Residual connections normalise activations',
            'It uses a larger learning rate',
          ],
          correct: [1],
          explanation: 'd(x + f(x))/dx = 1 + f\'(x): the 1 survives any number of layers. Residuals do NOT control the forward scale — the plain-residual row grew to 4.6·10¹⁰ — which is why they are paired with normalisation.',
        },
      ],
    },
    {
      type: 'deepdive',
      title: 'Going deeper',
      md: `- He et al., "Deep Residual Learning for Image Recognition" (2015) — residual connections.
- Ba, Kiros and Hinton, "Layer Normalization" (2016); Zhang and Sennrich, "Root Mean Square Layer Normalization" (2019).
- Xiong et al., "On Layer Normalization in the Transformer Architecture" (2020) — the gradient analysis behind pre-norm's stability.
- Elhage et al., "A Mathematical Framework for Transformer Circuits" (transformer-circuits.pub, 2021) — the residual stream as a shared communication channel.`,
    },
  ],
}

export default lesson

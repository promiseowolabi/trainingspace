import type { Lesson } from '../types'

const lesson: Lesson = {
  id: 'o1.l3',
  slug: 'optimisers',
  trackId: 'o1',
  index: 3,
  title: 'Optimisers',
  minutes: 21,
  hook: 'Plain gradient descent takes one step size for every parameter. On a loss surface that is a thousand times steeper in one direction than another, that step is either unstable or useless — and the fix, per-parameter step sizes, is why every LLM is trained with Adam.',
  exercise: 'lab+quiz',
  takeaway: {
    number: '0.16 vs 5·10⁻¹⁰',
    claim: 'On a bowl with condition number 1,000, 300 steps leave SGD at its best stable learning rate at loss 0.16 and Adam at 5·10⁻¹⁰, because Adam divides each parameter\'s step by its own gradient scale — at the cost of 8 bytes of state per parameter.',
  },
  blocks: [
    {
      type: 'prose',
      md: `T0.L1's update was \`p −= lr · g\`: one learning rate for every parameter. Real loss surfaces punish that. Think of a long, narrow valley: very steep across, gently sloped along. A step size small enough not to bounce off the steep walls crawls along the floor.

Make it concrete: the loss \`½(1000·x₀² + x₁²)\`, starting at (1, 1). Its curvature is 1,000 in one direction and 1 in the other — a **condition number** of 1,000. Gradient descent is stable only if \`lr < 2 / 1000\`, and at that step size the gentle direction shrinks by at most 0.2% per step. Measured with lab 03's code, 300 steps:`,
    },
    {
      type: 'statline',
      stats: [
        { value: '0.16', label: 'SGD, lr 0.0019', hint: 'the steep direction is solved; x₁ is still 0.57 — 300 steps of 0.19% shrinkage' },
        { value: '3.4·10⁻⁷', label: 'SGD + momentum 0.9', hint: 'same lr; the consistent direction accumulates speed' },
        { value: '4.8·10⁻¹⁰', label: 'Adam, lr 0.05', hint: 'with 10 warmup steps and cosine decay (lab 03\'s lr_at)' },
        { value: '1,000', label: 'condition number', hint: 'ratio of the steepest to the gentlest curvature' },
      ],
    },
    {
      type: 'prose',
      md: `## Momentum: remember the direction

Keep a running sum of past gradients and step along that instead:

\`\`\`
buf ← μ · buf + g          (μ ≈ 0.9)
p   ← p − lr · buf
\`\`\`

Across the valley the gradient flips sign every step, so the contributions cancel. Along the floor it points the same way every step, so the buffer grows towards g / (1 − μ) — ten times the raw gradient at μ = 0.9. The oscillation is damped and the slow direction speeds up: 0.16 becomes 3.4 × 10⁻⁷.

## Adam: a step size per parameter

Adam keeps two running averages per parameter: of the gradient (**m**, momentum) and of the squared gradient (**v**, its typical magnitude). It steps by their ratio:

\`\`\`
m ← β1·m + (1 − β1)·g          β1 = 0.9
v ← β2·v + (1 − β2)·g²         β2 = 0.999 (0.95 for LLMs)
p ← p − lr · m̂ / (√v̂ + ε)      m̂, v̂: bias-corrected (below)
\`\`\`

Dividing by √v normalises each parameter's step by its own gradient scale. In the valley, the steep coordinate has gradients 1,000× larger, and its step is divided by 1,000× more: both coordinates move at roughly \`lr\` per step. Adam is approximately invariant to how each parameter is scaled — and in a transformer, embedding rows, attention projections and LayerNorm gains differ in gradient scale by orders of magnitude.

**Bias correction.** m and v start at zero, so early on they are too small — v especially, because β2 = 0.999 averages over about 1,000 steps. Dividing by (1 − βᵗ) undoes that. Without it, the first update is (1 − β1)/√(1 − β2) ≈ **3.2× too large**, exactly when the model is least settled. With it, step one moves every parameter by exactly lr·sign(g) — lab 03's \`adam-first-step\` check.`,
    },
    {
      type: 'diagram',
      caption: 'fig 1 — one Adam update for one parameter',
      height: 64,
      nodes: [
        { id: 'g', x: 2, y: 22, w: 12, h: 12, label: 'g', sub: 'gradient' },
        { id: 'm', x: 22, y: 6, w: 20, h: 12, label: 'm ← β1·m + …', sub: 'direction' },
        { id: 'v', x: 22, y: 40, w: 20, h: 12, label: 'v ← β2·v + …', sub: 'magnitude²' },
        { id: 'bc', x: 50, y: 22, w: 18, h: 12, label: 'bias-correct', sub: '÷ (1 − βᵗ)' },
        { id: 'step', x: 76, y: 22, w: 22, h: 12, label: 'p −= lr·m̂/√v̂', sub: '≈ lr per param' },
      ],
      edges: [
        { from: 'g', to: 'm' },
        { from: 'g', to: 'v' },
        { from: 'm', to: 'bc' },
        { from: 'v', to: 'bc' },
        { from: 'bc', to: 'step' },
      ],
      steps: [
        { caption: 'The gradient feeds two exponential moving averages: m tracks its direction over about 10 steps, v tracks its squared size over about 1,000.', active: ['g', 'm', 'v'], edges: ['g->m', 'g->v'] },
        { caption: 'Both averages start at zero and would be biased low for the first steps; dividing by 1 − βᵗ corrects them exactly.', active: ['m', 'v', 'bc'], edges: ['m->bc', 'v->bc'] },
        { caption: 'The step is m̂/√v̂: a direction divided by a typical magnitude. A parameter with huge gradients and one with tiny gradients both move about lr per step.', active: ['bc', 'step'], edges: ['bc->step'] },
        { caption: 'The price is state: m and v are one fp32 number each per parameter — the 8 of T0.L4\'s 16 bytes per parameter, and the first thing ZeRO shards (D1.L3).', active: ['m', 'v'] },
      ],
    },
    {
      type: 'prose',
      md: `## AdamW: weight decay done right

**Weight decay** pulls every weight towards zero a little each step — a preference for small weights that helps generalisation (O1.L6). With SGD you can add \`wd · p\` to the gradient (an L2 penalty) and get exactly that. With Adam you cannot: the added term is divided by √v like everything else, so parameters with large gradients barely decay and parameters with tiny gradients decay at full speed. Loshchilov and Hutter's fix, **AdamW**, applies the decay directly to the weights, outside the adaptive step:

\`\`\`
p ← p · (1 − lr · wd)          then the normal Adam update, with g untouched
\`\`\`

Every LLM recipe you will read uses AdamW. In PyTorch, \`torch.optim.Adam(weight_decay=…)\` is the L2 version and \`torch.optim.AdamW\` the decoupled one — same argument name, different maths. Lab 03 implements both behind one flag.`,
    },
    {
      type: 'vendor',
      snapshot: '2026-10',
      title: 'A published LLM optimiser recipe',
      md: `Llama 2's pretraining used **AdamW with β1 = 0.9, β2 = 0.95, ε = 10⁻⁵, weight decay 0.1**, gradient clipping at 1.0, 2,000 warmup steps and cosine decay to 10% of the peak learning rate. The lower β2 (0.95 instead of PyTorch's default 0.999) makes v react faster to sudden gradient growth, which helps stability in large runs (P1.L6). PyTorch's own AdamW defaults are β = (0.9, 0.999), ε = 10⁻⁸, weight decay 0.01.`,
      systems: ['llama-2', 'pytorch'],
      sources: ['https://arxiv.org/abs/2307.09288', 'https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html'],
    },
    {
      type: 'lab',
      lab: 'optimisers',
      brief: 'Implement SGD with momentum, Adam with bias correction, AdamW\'s decoupled decay, the warmup-cosine schedule of O1.L4 and global-norm clipping — matched to PyTorch\'s update rules — then race SGD against Adam on the condition-1,000 bowl from this lesson.',
    },
    {
      type: 'quiz',
      questions: [
        {
          q: 'You switch from torch.optim.Adam(weight_decay=0.1) to torch.optim.AdamW(weight_decay=0.1) with everything else equal. What changed?',
          options: [
            'Nothing — AdamW is an alias',
            'Weight decay is now applied directly to the weights instead of being added to the gradient and rescaled by √v, so every parameter decays at the same rate',
            'AdamW turns off bias correction',
            'AdamW uses SGD for the decay and Adam for the gradient, doubling memory',
          ],
          correct: [1],
          explanation: 'Adam\'s weight_decay is an L2 term added to g, which the adaptive denominator then rescales per parameter. AdamW applies p ← p·(1 − lr·wd) separately. Same argument name, different regulariser — a common source of irreproducible results when porting recipes.',
        },
        {
          q: 'Why is SGD with a single learning rate slow on an ill-conditioned problem?',
          options: [
            'SGD has no momentum by default',
            'The learning rate must stay below 2 / (largest curvature) to be stable, which makes progress along the flattest direction proportional to smallest / largest curvature — tiny when the condition number is large',
            'SGD only works on convex problems',
            'The gradient is computed less accurately',
          ],
          correct: [1],
          explanation: 'Stability is set by the steepest direction, speed by the flattest. On the bowl in this lesson that is a 0.19% improvement per step along x₁. Momentum and per-parameter scaling (Adam) both attack this ratio; a bigger learning rate just diverges.',
        },
        {
          q: 'A 7B model is trained with AdamW. How much memory do Adam\'s m and v take in fp32?',
          options: ['14 GB', '28 GB', '56 GB', '112 GB'],
          correct: [2],
          explanation: 'Two fp32 numbers per parameter: 7 × 10⁹ × 8 bytes = 56 GB — half of T0.L4\'s 112 GB of training state, and more than the bf16 weights and gradients combined. That is why optimiser state is the first thing ZeRO shards and why 8-bit optimisers exist.',
        },
      ],
    },
    {
      type: 'deepdive',
      title: 'Going deeper',
      md: `- Kingma and Ba, "Adam: A Method for Stochastic Optimization" (2014, arxiv.org/abs/1412.6980) — algorithm 1 is lab 03's \`Adam.step\`.
- Loshchilov and Hutter, "Decoupled Weight Decay Regularization" (2017, arxiv.org/abs/1711.05101) — AdamW.
- Goh, "Why Momentum Really Works" (distill.pub/2017/momentum) — the valley picture, interactive.
- Newer optimisers (Lion, Sophia, Muon, Shampoo) trade Adam's per-parameter scaling for other preconditioners; each paper's first table is a comparison against AdamW, which is still the baseline to beat.`,
    },
  ],
}

export default lesson

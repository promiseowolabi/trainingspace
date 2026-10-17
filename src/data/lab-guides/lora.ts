import type { LabGuide } from './types'

const guide: LabGuide = {
  id: 'lora',
  requires: ['Python 3.10+', 'numpy, pytest', 'PyTorch (CPU build is enough)'],
  blocks: [
    {
      type: 'prose',
      md: `## What you will build

The two pieces most fine-tuning runs are made of: a **LoRA adapter** that lets you train a tiny fraction of a model's parameters, and the **SFT loss** that trains a model on responses without training it on prompts. Both are short; the checks are about getting their exact semantics right — zero initialisation, frozen bases, the α/r scale, mergeability, masking.`,
    },
    {
      type: 'prose',
      md: `## 1. LoRALinear

\`__init__\`: freeze the base, create A (r × in, random) and B (out × r, zeros) as \`nn.Parameter\`s. The order of initialisation matters and \`zero-init\` checks both halves: B = 0 makes the adapter a no-op at the start (fine-tuning starts exactly from the pretrained model), and A ≠ 0 makes sure B still receives a gradient on the first step — if both were zero, neither would ever move.

\`forward\`: \`self.base(x) + self.scale * (x @ self.A.T @ self.B.T)\`. Two thin matmuls through an r-dimensional bottleneck. Never build the full out × in product B·A in the forward pass; that would cost as much as the base layer.

\`merged\`: a new \`nn.Linear\` whose weight is \`W + scale · B @ A\` and whose bias is copied. After merging, the adapter has no inference cost at all.

## 2. apply_lora and count_params

Freeze every parameter first, then swap matching \`nn.Linear\` children for \`LoRALinear\` wrappers. Matching by suffix means \`"proj"\` catches both the attention output (\`proj\`) and the MLP output (\`fc_proj\`) in \`model.py\` — the check expects 12 wrapped layers in a 4-block model, and \`r · (in + out)\` trainable parameters for each.`,
    },
    {
      type: 'prose',
      md: `## 3. sft_loss

Per-token cross-entropy with \`reduction="none"\`, multiplied by the float mask, summed, divided by the number of response tokens. The check replaces the logits at every prompt position with huge random values and requires the loss not to change at all.

## 4. The last check

A frozen random \`nn.Linear(32, 32)\` and a target that differs from it by a rank-1 matrix. A rank-2 adapter, trained with Adam on only A and B, must bring the error below 0.1% of where it started — while the base weight stays bit-for-bit unchanged.`,
    },
  ],
  cleanup: [
    {
      type: 'prose',
      md: `- Combine with lab 07: load a trained checkpoint, \`apply_lora\` to \`qkv\` and \`proj\`, and fine-tune on a few hundred prompt/response pairs with \`sft_loss\` — the capstone's fine-tuning step.
- Compare the optimiser memory with and without LoRA: AdamW keeps m and v only for parameters that require gradients.
- Try r = 1, 4 and 16 on the low-rank check with a rank-4 target. Which ranks can represent it exactly?`,
    },
  ],
}

export default guide

import type { LabGuide } from './types'

const guide: LabGuide = {
  id: 'gpt-forward',
  requires: ['a browser — or Python 3.10+ with numpy and pytest', 'lab 05 done (recommended)'],
  blocks: [
    {
      type: 'prose',
      md: `## What you will build

Everything M1 has described, assembled: the GPT-2 architecture as a pure function from token ids to logits. Your weights are random, so the model says nothing sensible — but the checks compare your logits with an independently written reference to within 10⁻⁸, so if they pass, loading real GPT-2 weights into this code would produce real GPT-2 output.

Read \`init_params\` first. Its key names and shapes are the specification for \`block\` and \`forward\`.`,
    },
    {
      type: 'prose',
      md: `## 1. layer_norm and gelu

One line each. LayerNorm normalises the **last** axis — every token separately — using numpy's default (biased) variance. GELU is the tanh formula in the docstring.

## 2. block

Follow the six lines in the docstring. The attention is lab 05's, with two GPT-2 details:

- q, k and v come from **one** matmul with \`w_qkv\` of shape \`(C, 3C)\`; split the result into three along the last axis (\`np.split(qkv, 3, axis=-1)\`) **before** splitting heads.
- Every projection has a bias.

You can paste lab 05's \`split_heads\`/\`merge_heads\` logic, or inline the reshapes.

## 3. forward

Embedding lookup is indexing: \`params["wte"][idx]\` turns \`(B, T)\` ids into \`(B, T, C)\` vectors. Add \`params["wpe"][:T]\`, run the blocks in order, apply \`ln_f\`, and multiply by \`wte.T\` — the same matrix that embedded the tokens now scores them.

## 4. count_params

Arithmetic, not \`sum(v.size …)\`: write the per-block count as a formula in C, multiply by L, and add the embeddings and the final LayerNorm. The check wants **124,439,808** for GPT-2 small (V = 50,257, T = 1,024, L = 12, C = 768). M1.L5 walks through the sum.`,
    },
    {
      type: 'prose',
      md: `## 5. Generation

\`generate_greedy\`: in a loop, crop the sequence to the last \`block_size\` tokens, run \`forward\` on it as a batch of one, take the argmax of the **last** position's logits, append. Each step recomputes the whole prefix — fine here; avoiding that recomputation is what a KV cache does in production inference.

\`sample_next\` follows the six numbered steps in the docstring. The order matters: temperature before top-k before softmax before top-p. The check draws thousands of samples and compares frequencies with the softmax, so a subtle bug (renormalising before masking, say) shows up as a wrong distribution rather than a crash.`,
    },
  ],
  cleanup: [
    {
      type: 'prose',
      md: `- Time \`forward\` on GPT-2 small's shapes with random float32 weights (it needs about 0.5 GB). On the reference laptop a 1,024-token forward pass took 1.56 s — about 160 GFLOP/s by the 2·N-per-token rule (M1.L5).
- If you have the \`transformers\` package: load GPT-2's real weights, map their names onto this lab's keys (Hugging Face stores \`c_attn\`, \`c_proj\`, \`c_fc\` as Conv1D, already in (in, out) layout), and generate text with your own code.
- P1 moves this model to PyTorch so that autograd can train it.`,
    },
  ],
}

export default guide

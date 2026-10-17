import type { Lesson } from '../types'

const lesson: Lesson = {
  id: 'm1.l6',
  slug: 'sampling',
  trackId: 'm1',
  index: 6,
  title: 'Sampling',
  minutes: 15,
  hook: 'Training only ever asks the model for a probability distribution over the next token. Generation is what you do with that distribution — and temperature, top-k and top-p are three small edits to it that change the text completely.',
  exercise: 'read+quiz',
  takeaway: {
    number: '0.62 → 0.86 at T = 0.5',
    claim: 'Halving the temperature raised the top token\'s probability from 0.62 to 0.86 in this lesson\'s example; temperature, top-k and top-p only reshape the distribution the model already computed — they add no knowledge.',
  },
  blocks: [
    {
      type: 'prose',
      md: `## From a loss to a generator

Training minimises cross-entropy at every position: make \`p(next token | previous tokens)\` assign high probability to what actually came next. Nothing in the loss mentions generating text. But a model that has learned \`p(token_t | tokens_<t)\` well can produce text by **using its own predictions as input**:

\`\`\`python
seq = prompt_ids
for _ in range(max_new):
    logits = model(seq[-block_size:])[-1]     # distribution for the NEXT token only
    seq.append(choose(logits))               # greedy or sampled
\`\`\`

This is **autoregressive** generation. The model was trained on every position in parallel (the causal mask made that honest, M1.L3); it generates one token at a time because each new token depends on the one before.

## Greedy: always take the top token

\`choose = argmax\`. Deterministic, and usually bad for open-ended text: it falls into loops ("the the the", or the same sentence repeated), because once a repetition is the most likely continuation it stays the most likely. Greedy is fine for short, constrained outputs — a classification label, a single number.

## Temperature

Divide the logits by T before the softmax. For the logits \`[3, 2, 1, 0.5, −1]\`:

| temperature | probabilities | top token | perplexity |
|---|---|---|---|
| 0.5 | 0.862, 0.117, 0.016, 0.006, 0.000 | **0.86** | 1.61 |
| 1.0 | 0.624, 0.229, 0.084, 0.051, 0.011 | 0.62 | 2.84 |
| 2.0 | 0.417, 0.253, 0.154, 0.120, 0.056 | 0.42 | 4.12 |

Below 1 the distribution sharpens towards greedy (T → 0 *is* greedy); above 1 it flattens towards uniform. Perplexity here is exp(entropy): "as uncertain as choosing among this many options".`,
    },
    {
      type: 'prose',
      md: `## Truncation: top-k and top-p

A 50,000-token vocabulary has a long tail: thousands of tokens each with tiny probability that together can hold a few percent of the mass. Sample a long text and you will hit the tail eventually, and one absurd token derails everything after it. Truncation removes the tail before sampling:

- **top-k**: keep the k most likely tokens, renormalise. Simple, but k is the same whether the model is confident (where k = 40 keeps junk) or uncertain (where k = 40 cuts real options).
- **top-p (nucleus)**: keep the smallest set of top tokens whose probabilities add up to p (say 0.9), renormalise. The kept set shrinks when the model is confident and grows when it is not.

In the T = 1 row above, top-p 0.8 keeps the first two tokens (0.624 alone is short of 0.8; 0.853 reaches it). Lab 06's \`sample_next\` implements all three, in the order temperature → top-k → softmax → top-p, and checks the result statistically over thousands of draws.`,
    },
    {
      type: 'diagram',
      caption: 'fig 1 — one decoding step',
      height: 60,
      nodes: [
        { id: 'lg', x: 2, y: 22, w: 14, h: 12, label: 'logits', sub: '(V,)' },
        { id: 't', x: 22, y: 22, w: 16, h: 12, label: '÷ temperature', sub: 'sharpen/flatten' },
        { id: 'k', x: 44, y: 4, w: 16, h: 12, label: 'top-k', sub: 'keep k' },
        { id: 'sm', x: 44, y: 40, w: 16, h: 12, label: 'softmax', sub: 'probabilities' },
        { id: 'p', x: 66, y: 40, w: 14, h: 12, label: 'top-p', sub: 'keep mass p' },
        { id: 'draw', x: 84, y: 22, w: 14, h: 12, label: 'draw', sub: 'append id' },
      ],
      edges: [
        { from: 'lg', to: 't' },
        { from: 't', to: 'k' },
        { from: 'k', to: 'sm' },
        { from: 'sm', to: 'p' },
        { from: 'p', to: 'draw' },
      ],
      steps: [
        { caption: 'The model produces one logit per vocabulary entry for the last position. Everything after this is post-processing; the model has finished its work.', active: ['lg'] },
        { caption: 'Dividing by the temperature rescales how far apart the logits are: smaller temperature, larger gaps, a more peaked distribution.', active: ['lg', 't'], edges: ['lg->t'] },
        { caption: 'Top-k sets every logit outside the k largest to −∞; the softmax then gives them exactly zero probability.', active: ['t', 'k', 'sm'], edges: ['t->k', 'k->sm'] },
        { caption: 'Top-p sorts the probabilities and keeps the shortest prefix whose mass reaches p, then renormalises. One random draw picks the token, which is appended to the sequence for the next step.', active: ['sm', 'p', 'draw'], edges: ['sm->p', 'p->draw'] },
      ],
    },
    {
      type: 'callout',
      variant: 'warning',
      title: 'Sampling settings are not model quality',
      md: `Temperature and truncation reshape a distribution the model has already computed; they cannot add knowledge. When an evaluation compares two models, fix the decoding settings — or use greedy — or the comparison measures the settings. Likewise, a run's validation **loss** (computed on the full distribution, no sampling) is the clean training signal; sampled text is a sanity check, not a metric.`,
    },
    {
      type: 'quiz',
      questions: [
        {
          q: 'A greedy-decoded story repeats the same sentence forever. What is the most direct fix?',
          options: [
            'Train for longer',
            'Sample instead of taking the argmax — e.g. temperature 0.8 with top-p 0.9 — so a repetition that is merely the most likely continuation is not chosen every time',
            'Increase the context length',
            'Use a larger vocabulary',
          ],
          correct: [1],
          explanation: 'Greedy decoding always takes the mode, and repetitive continuations often are the mode once a loop starts. Sampling breaks the loop by sometimes choosing other likely tokens. Repetition penalties are another option; more training rarely removes greedy loops.',
        },
        {
          q: 'Why is top-p usually preferred to top-k for open-ended text?',
          options: [
            'It is faster to compute',
            'Its cut-off adapts to the model\'s confidence: when one token dominates it keeps very few, when many are plausible it keeps many — a fixed k is too loose in the first case and too tight in the second',
            'It never changes the distribution',
            'It does not need a softmax',
          ],
          correct: [1],
          explanation: 'top-k keeps a fixed count regardless of how the probability is spread. top-p keeps a fixed amount of probability mass, so the number of candidates follows the model\'s uncertainty.',
        },
        {
          q: 'A team reports that model A beats model B on a creative-writing evaluation. A was sampled at temperature 0.7, B at 1.2. What should you ask?',
          options: [
            'Nothing — temperature does not affect quality',
            'For a comparison at matched decoding settings: different temperatures change outputs systematically, so the result may measure the settings rather than the models',
            'Whether B used top-k',
            'Which tokenizer they used',
          ],
          correct: [1],
          explanation: 'Temperature reshapes every distribution: lower is more conservative and coherent, higher more varied and error-prone. Evaluations must fix decoding parameters (or sweep them for both models) before attributing a difference to the weights.',
        },
      ],
    },
    {
      type: 'deepdive',
      title: 'Going deeper',
      md: `- Holtzman et al., "The Curious Case of Neural Text Degeneration" (2019, arxiv.org/abs/1904.09751) — nucleus (top-p) sampling and why greedy text degenerates.
- Generation recomputes the whole prefix every step in lab 06. Production inference caches each layer's keys and values (the KV cache) so each new token costs one position's worth of work — a serving topic, outside this course.
- This closes M1. P1 moves the model to PyTorch and trains it.`,
    },
  ],
}

export default lesson

import type { Lesson } from '../types'

const lesson: Lesson = {
  id: 'e1.l1',
  slug: 'what-evals-measure',
  trackId: 'e1',
  index: 1,
  title: 'What Evals Measure',
  minutes: 17,
  hook: 'A benchmark score is a sample statistic with an error bar, measured under specific prompting choices, on questions that may have leaked into training. Read every score as "70% ± 4, if nothing leaked, with this prompt" — or you will ship decisions on noise.',
  exercise: 'read+quiz',
  takeaway: {
    number: '±4 points on 500 questions',
    claim: 'A 70% score on a 500-question benchmark has a standard error of about 2 points, so its 95% interval spans roughly ±4: two models 2 points apart on that benchmark have not been shown to differ.',
  },
  blocks: [
    {
      type: 'prose',
      md: `Training gives you one honest number for free: **validation loss** on held-out text from the training distribution (O1.L6). It is smooth, cheap, and comparable across checkpoints of the same run — and nearly useless for comparing different models, because it depends on the tokenizer (M1.L1) and says nothing direct about whether a model can do a task. So models are also measured on **benchmarks**: fixed sets of questions with gradable answers.

## Two kinds of measurement

| | validation loss / perplexity | benchmark accuracy |
|---|---|---|
| measures | how well the model predicts held-out text | whether it answers specific questions correctly |
| comparable across tokenizers | no (nats per token differ) | yes |
| noise | tiny (millions of tokens) | large (hundreds of questions) |
| sensitive to prompt format | no | very |
| use it for | monitoring a run, scaling laws | comparing models, tracking capabilities |

Perplexity can be made tokenizer-independent by normalising per byte or per character instead of per token — "bits per byte" — which is how pretraining runs with different tokenizers are compared fairly.`,
    },
    {
      type: 'prose',
      md: `## Every score has an error bar

A benchmark of n questions answered correctly with proportion p is a sample of the model's true accuracy. The standard error is:

\`\`\`
SE = √( p · (1 − p) / n )
\`\`\`

| benchmark size | p = 70% | 95% interval (≈ ±2·SE) |
|---|---|---|
| 100 questions | SE 4.6 pts | **±9.2** |
| 500 questions | SE 2.0 pts | **±4.1** |
| 2,000 questions | SE 1.0 pt | ±2.0 |
| 10,000 questions | SE 0.46 pts | ±0.9 |

A 2-point gap on a 500-question benchmark is inside the noise. For two models evaluated on the **same** questions, a paired comparison — counting only the questions on which they disagree — is tighter than comparing two independent intervals, and is the right test. Either way: report the interval, and do not rank models on differences smaller than it.`,
    },
    {
      type: 'statline',
      stats: [
        { value: '±9 pts', label: '100 questions', hint: '95% interval at 70% accuracy' },
        { value: '±4 pts', label: '500 questions', hint: '2 × √(0.7 · 0.3 / 500) = 4.1' },
        { value: '±0.9 pts', label: '10,000 questions', hint: 'large benchmarks are the only place small gaps mean anything' },
        { value: '√n', label: 'error ∝ 1/√n', hint: 'halving the error needs 4× the questions' },
      ],
    },
    {
      type: 'prose',
      md: `## Sources of error that are not sampling

**Prompt and format sensitivity.** The same model can score several points differently depending on the prompt wording, the number of few-shot examples, whether answer choices are lettered, and how the answer is extracted from free text. A comparison is meaningful only under identical evaluation settings — the same harness, the same prompts, the same decoding (M1.L6).

**Contamination.** If benchmark questions — or pages discussing them — were in the training data, the score partly measures memory. Detection looks for long n-gram overlaps between benchmark items and the training corpus, or tests whether the model can complete a benchmark question verbatim from its first half. Mitigations: decontaminate the training data (P1.L1), and prefer benchmarks with held-out or regularly refreshed questions.

**Saturation and construct validity.** When strong models all score above 90%, the remaining errors are often label noise, and the benchmark has stopped distinguishing them. And every benchmark measures a narrow proxy for the capability its name suggests.`,
    },
    {
      type: 'callout',
      variant: 'isomorphism',
      title: 'A/B tests you already run',
      md: `You would not ship a change because conversion went from 3.1% to 3.2% on 400 visitors. A benchmark is the same statistics: a binomial proportion with a sample size. Same rules — confidence intervals, paired tests where you can, a fixed test set nobody tuned on, and suspicion of any result that only appears under one particular setting.`,
    },
    {
      type: 'quiz',
      questions: [
        {
          q: 'Model A scores 72.4% and model B 70.8% on a 400-question benchmark. What can you conclude?',
          options: [
            'A is better',
            'Not much: the standard error at these rates is about 2.3 points, so the 95% intervals overlap widely; a paired comparison on the questions where they disagree, or a larger benchmark, is needed',
            'B is better because it is more consistent',
            'They are identical',
          ],
          correct: [1],
          explanation: '√(0.71 × 0.29 / 400) ≈ 0.023. A 1.6-point gap is well inside ±4.5 points. Treat it as "no evidence of a difference" until a properly powered or paired test says otherwise.',
        },
        {
          q: 'Why are two pretraining runs with different tokenizers not compared by validation loss per token?',
          options: [
            'Loss is not deterministic',
            'Loss per token depends on how much text a token covers: a tokenizer with longer tokens has fewer, harder predictions. Normalise by bytes or characters (bits per byte) to compare',
            'Validation sets are always different',
            'Loss only works for one epoch',
          ],
          correct: [1],
          explanation: 'Cross-entropy per token measures the cost of predicting each token, and tokens are different units under different tokenizers. Total bits divided by total bytes of text is a tokenizer-independent rate.',
        },
        {
          q: 'A model scores far higher on an old benchmark than on a freshly written one testing the same skill. What is the first hypothesis?',
          options: [
            'The new benchmark is broken',
            'Contamination: the old benchmark\'s questions or answers appear in the training data, so part of the old score is memorisation',
            'The model got worse',
            'Random noise',
          ],
          correct: [1],
          explanation: 'Public benchmarks are discussed and copied across the web, which is what pretraining crawls. A gap between a public test set and fresh, equivalent questions is a classic contamination signature; n-gram overlap searches against the training data can confirm it.',
        },
      ],
    },
    {
      type: 'deepdive',
      title: 'Going deeper',
      md: `- Miller, "Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations" (2024, arxiv.org/abs/2411.00640) — clustered standard errors and paired tests for benchmarks.
- EleutherAI's \`lm-evaluation-harness\` — the de facto open harness; read how one task's prompt and answer extraction are defined.
- Sainz et al., "NLP Evaluation in trouble: On the Need to Measure LLM Data Contamination for each Benchmark" (2023).`,
    },
  ],
}

export default lesson

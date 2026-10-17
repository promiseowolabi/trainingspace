import type { Lesson } from '../types'

const lesson: Lesson = {
  id: 'p1.l1',
  slug: 'data-is-the-model',
  trackId: 'p1',
  index: 1,
  title: 'Data Is the Model',
  minutes: 17,
  hook: 'Two teams with the same architecture and the same compute get different models, and the difference is almost always the data. Pretraining data is a pipeline — crawl, extract, filter, deduplicate, mix — and every stage is measured in tokens.',
  exercise: 'read+quiz',
  takeaway: {
    number: '15 T tokens ≈ 1,875 per parameter',
    claim: 'An 8B model trained on 15 trillion tokens sees about 1,875 tokens per parameter, mostly once each — while lab 07\'s model saw its 140,000 characters 58 times and memorised them.',
  },
  blocks: [
    {
      type: 'prose',
      md: `Lab 07 trained on 140,000 characters of this course. After 2,000 steps the model had seen that text **58 times**, and its validation loss turned upward at step 1,000: it had started memorising (O1.L6). A real pretraining run inverts the situation. There is so much text that each token is usually seen **once**, and the question stops being "how do I avoid overfitting?" and becomes "which trillions of tokens are worth the compute?"

## Tokens are the currency

Everything about pretraining data is counted in tokens of the model's tokenizer (M1.L1), because tokens are what cost compute: 6·N FLOPs each (P1.L4). A rule of thumb for English web text is roughly four bytes per token, so a terabyte of clean text is on the order of 250 billion tokens.`,
    },
    {
      type: 'vendor',
      snapshot: '2026-10',
      title: 'How much data real runs use',
      md: `- Meta reported pretraining **Llama 3 on over 15 trillion tokens** of public data. For the 8B model that is about 15 × 10¹² / 8 × 10⁹ ≈ **1,875 tokens per parameter** — far beyond the ~20 that compute-optimal scaling would suggest (P1.L5).
- Hugging Face's **FineWeb** dataset is about **15 trillion tokens** of English web text, extracted, filtered and deduplicated from 96 Common Crawl snapshots, and published with its full pipeline and ablations.
- GPT-3 (2020) trained on 300 billion tokens.`,
      systems: ['llama-3', 'fineweb', 'gpt-3'],
      sources: ['https://ai.meta.com/blog/meta-llama-3/', 'https://huggingface.co/datasets/HuggingFaceFW/fineweb', 'https://arxiv.org/abs/2005.14165'],
    },
    {
      type: 'prose',
      md: `## The pipeline

**1. Crawl and extract.** Most pretraining text starts as web pages, typically from Common Crawl, a public crawl archived monthly. HTML has to become text: strip navigation, ads, boilerplate. Extraction quality alone measurably changes downstream model quality — a page of menus is tokens that teach nothing.

**2. Filter.** Remove what you do not want to model: other languages (if you are building an English model), machine-generated spam, very short or repetitive pages, personal data, and content that fails quality heuristics (ratio of symbols to words, lines ending in punctuation, and so on). Many pipelines also train a small **quality classifier** to score pages by how much they resemble a reference set of "good" text, and keep the top fraction.

**3. Deduplicate.** The web repeats itself: mirrors, templates, scraped copies, licence boilerplate. Exact duplicates are removed by hashing; near-duplicates by **MinHash**, which estimates the overlap of two documents' word-shingle sets from short signatures, so billions of documents can be compared without comparing every pair. Duplicates waste compute, and heavily repeated passages are what models memorise verbatim.

**4. Mix.** The final dataset is a weighted blend of sources — web, code, books, papers, encyclopedic text, maths, other languages — and the weights are hyper-parameters, chosen by training small models on candidate mixtures and comparing them. Code and maths are often up-weighted beyond their natural share because they improve reasoning benchmarks.`,
    },
    {
      type: 'diagram',
      caption: 'fig 1 — a pretraining data pipeline',
      height: 60,
      nodes: [
        { id: 'crawl', x: 2, y: 22, w: 15, h: 12, label: 'crawl', sub: 'raw HTML' },
        { id: 'extract', x: 21, y: 22, w: 15, h: 12, label: 'extract', sub: 'text only' },
        { id: 'filter', x: 40, y: 22, w: 15, h: 12, label: 'filter', sub: 'lang, quality' },
        { id: 'dedup', x: 59, y: 22, w: 15, h: 12, label: 'dedup', sub: 'exact + MinHash' },
        { id: 'mix', x: 78, y: 22, w: 20, h: 12, label: 'mix + tokenize', sub: 'weighted sources' },
        { id: 'other', x: 78, y: 46, w: 20, h: 12, label: 'code, books, maths', sub: 'other sources' },
      ],
      edges: [
        { from: 'crawl', to: 'extract' },
        { from: 'extract', to: 'filter' },
        { from: 'filter', to: 'dedup' },
        { from: 'dedup', to: 'mix' },
        { from: 'other', to: 'mix' },
      ],
      steps: [
        { caption: 'A crawl is petabytes of raw pages. Most bytes are markup, navigation and boilerplate, not text worth learning from.', active: ['crawl', 'extract'], edges: ['crawl->extract'] },
        { caption: 'Filtering removes the wrong language, spam and low-quality pages; a learned quality classifier often keeps only the best-scoring fraction.', active: ['extract', 'filter'], edges: ['extract->filter'] },
        { caption: 'Deduplication removes exact and near copies so compute is not spent re-reading the same text and the model does not memorise repeated passages.', active: ['filter', 'dedup'], edges: ['filter->dedup'] },
        { caption: 'Sources are blended with tuned weights and tokenized. The mixture is decided by training small models on candidates — the data is a hyper-parameter.', active: ['dedup', 'mix', 'other'], edges: ['dedup->mix', 'other->mix'] },
      ],
    },
    {
      type: 'prose',
      md: `## Epochs

Lab 07's model saw its data 58 times and overfit. Large runs aim for about one epoch over most sources — though high-quality sources are often repeated a few times by design, because more passes over excellent text can beat one pass over mediocre text. Research on data-constrained scaling found that up to about four epochs over the same data is nearly as good as fresh data, with returns falling quickly after that. When teams run out of good text, repetition, synthetic data and stricter filtering are the levers — and all three are active research.

## The held-out set must stay held out

Every pipeline also has to protect evaluation data. If benchmark questions, or web pages quoting them, end up in the training mixture, the model's score measures memory rather than ability. Decontamination — searching the training data for long n-gram overlaps with each benchmark and removing matches — is a pipeline stage of its own (E1.L1).`,
    },
    {
      type: 'callout',
      variant: 'isomorphism',
      title: 'An ETL job with a model at the end',
      md: `You have built this pipeline before: extract, transform, deduplicate, load. The differences are scale — petabytes in, terabytes out — and the acceptance test. A data warehouse is correct when its queries are; a pretraining dataset is correct when a model trained on it is good, which you only learn by spending compute. That is why data teams train many small "ablation" models: they are the integration tests.`,
    },
    {
      type: 'quiz',
      questions: [
        {
          q: 'Two 1B-parameter runs use the same code and compute. Run B\'s data went through near-duplicate removal; run A\'s did not. Which outcome is most likely?',
          options: [
            'Identical models — the token counts are the same',
            'Run B is better on held-out evaluations: run A spent part of its budget re-reading duplicated text and memorised repeated passages',
            'Run A is better because it saw more examples of common text',
            'Run B is worse because deduplication removes useful data',
          ],
          correct: [1],
          explanation: 'Duplicates cost the same compute as fresh tokens but teach less, and heavily repeated passages are memorised verbatim. Deduplication consistently improves models trained on the same token budget; it is a standard stage of every serious pipeline.',
        },
        {
          q: 'Your lab 07 run\'s validation loss rises after step 1,000. A colleague says pretraining runs "always overfit eventually". What is the better explanation?',
          options: [
            'They are right — all language models overfit',
            'The model saw its 140,000-character dataset about 58 times; real pretraining sees most tokens about once, so it rarely reaches the memorisation regime',
            'The learning rate schedule is wrong',
            'Character-level models cannot generalise',
          ],
          correct: [1],
          explanation: 'Overfitting needs repetition. 2,000 steps × 4,096 tokens over 140,467 training characters is about 58 epochs. A 15-trillion-token run over an 8B model is roughly one pass, which is why pretraining train and validation losses usually track each other.',
        },
        {
          q: 'Roughly how many tokens is 4 TB of clean English text?',
          options: ['about 4 billion', 'about 100 billion', 'about 1 trillion', 'about 16 trillion'],
          correct: [2],
          explanation: 'At roughly 4 bytes per token, 4 × 10¹² bytes ÷ 4 ≈ 10¹² tokens. The ratio depends on the tokenizer and the language; non-English text and code usually take more tokens per byte.',
        },
      ],
    },
    {
      type: 'deepdive',
      title: 'Going deeper',
      md: `- Penedo et al., "The FineWeb Datasets" (2024, arxiv.org/abs/2406.17557) — every pipeline stage with ablations showing what it is worth.
- Lee et al., "Deduplicating Training Data Makes Language Models Better" (2021, arxiv.org/abs/2107.06499).
- Muennighoff et al., "Scaling Data-Constrained Language Models" (2023, arxiv.org/abs/2305.16264) — what repeated epochs are worth.
- Broder, "On the resemblance and containment of documents" (1997) — MinHash.`,
    },
  ],
}

export default lesson

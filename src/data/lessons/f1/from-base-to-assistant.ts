import type { Lesson } from '../types'

const lesson: Lesson = {
  id: 'f1.l1',
  slug: 'from-base-to-assistant',
  trackId: 'f1',
  index: 1,
  title: 'From Base Model to Assistant',
  minutes: 15,
  hook: 'A pretrained model continues text. Ask it a question and it may answer — or write three more questions, because that is what the web does after a question. Post-training is the comparatively tiny amount of training that turns a text predictor into something that follows instructions.',
  exercise: 'read+quiz',
  takeaway: {
    number: '1.3 B preferred over 175 B',
    claim: 'Human raters preferred a 1.3B model after instruction tuning and RLHF over the raw 175B GPT-3: post-training costs a small fraction of pretraining compute and changes behaviour more than a 100× larger model does.',
  },
  blocks: [
    {
      type: 'prose',
      md: `Pretraining minimises next-token loss on the web (P1). The result — a **base model** — has learned an enormous amount about language and the world, and exactly one behaviour: continue the document in the most likely way. Prompt it with

\`\`\`text
What is the capital of Australia?
\`\`\`

and the most likely continuation of that string on the web might be the answer, or a list of other quiz questions, or "This question is often asked by…". Nothing in the objective rewards answering, being concise, refusing a harmful request, or saying "I don't know".

**Post-training** is everything after pretraining that shapes behaviour. It uses far less compute — typically a small fraction of a percent of the pretraining FLOPs — and far less data, carefully chosen.`,
    },
    {
      type: 'prose',
      md: `## The stack

| stage | data | what it teaches | lesson |
|---|---|---|---|
| **SFT** (supervised fine-tuning) | thousands to millions of prompt → ideal response demonstrations | the format and style of an assistant: answer the question, then stop | F1.L2 |
| **reward modelling** | pairs of responses with a human (or AI) preference | a learned score for "how good is this response" | F1.L4 |
| **preference optimisation** (RLHF with PPO, or DPO) | the reward model, or the preference pairs directly | prefer better responses over merely plausible ones | F1.L5 |
| **RL with verifiable rewards** | problems whose answers can be checked (maths, code, tests) | reasoning that reaches correct answers | F1.L6 |

Not every model uses every stage; many recipes iterate them several times. LoRA (F1.L3) is how many teams make any of these stages cheap enough to run.`,
    },
    {
      type: 'diagram',
      caption: 'fig 1 — where post-training sits',
      height: 60,
      nodes: [
        { id: 'pre', x: 2, y: 22, w: 20, h: 12, label: 'pretraining', sub: 'trillions of tokens' },
        { id: 'base', x: 26, y: 22, w: 14, h: 12, label: 'base model', sub: 'continues text' },
        { id: 'sft', x: 44, y: 6, w: 14, h: 12, label: 'SFT', sub: 'demonstrations' },
        { id: 'pref', x: 62, y: 6, w: 16, h: 12, label: 'RLHF / DPO', sub: 'preferences' },
        { id: 'rl', x: 62, y: 38, w: 16, h: 12, label: 'RLVR', sub: 'checkable answers' },
        { id: 'asst', x: 82, y: 22, w: 16, h: 12, label: 'assistant', sub: 'follows intent' },
      ],
      edges: [
        { from: 'pre', to: 'base' },
        { from: 'base', to: 'sft' },
        { from: 'sft', to: 'pref' },
        { from: 'sft', to: 'rl' },
        { from: 'pref', to: 'asst' },
        { from: 'rl', to: 'asst' },
      ],
      steps: [
        { caption: 'Pretraining spends nearly all of the compute and produces a base model: broad knowledge, one behaviour — continue the document.', active: ['pre', 'base'], edges: ['pre->base'] },
        { caption: 'SFT shows the model what a good response looks like, by example. The loss is the same cross-entropy, applied only to the response tokens.', active: ['base', 'sft'], edges: ['base->sft'] },
        { caption: 'Preference optimisation teaches it to prefer better responses among plausible ones — something demonstrations alone cannot express. RL with checkable rewards teaches it to reach correct answers.', active: ['sft', 'pref', 'rl'], edges: ['sft->pref', 'sft->rl'] },
        { caption: 'The result answers, refuses, asks for clarification and stops when done — with the knowledge it got from pretraining.', active: ['pref', 'rl', 'asst'], edges: ['pref->asst', 'rl->asst'] },
      ],
    },
    {
      type: 'vendor',
      snapshot: '2026-10',
      title: 'How much post-training buys',
      md: `OpenAI's InstructGPT paper (Ouyang et al., 2022) reported that outputs from a **1.3B-parameter** InstructGPT model — GPT-3 architecture after SFT and RLHF — were **preferred by human labellers over outputs from the 175B GPT-3** base model, despite having over 100× fewer parameters. Meta's Llama 2 paper reports using **27,540** high-quality SFT annotations and over a million human preference comparisons for its chat models — tiny next to its 2-trillion-token pretraining set.`,
      systems: ['instructgpt', 'llama-2'],
      sources: ['https://arxiv.org/abs/2203.02155', 'https://arxiv.org/abs/2307.09288'],
    },
    {
      type: 'callout',
      variant: 'warning',
      title: 'Post-training does not add much knowledge',
      md: `The working assumption behind every stage is that the base model already knows most of what the assistant will need, and post-training mostly selects behaviours. A fact the base model never learned is hard to teach with a few thousand demonstrations — and trying can teach the model to state confident falsehoods in the assistant's style. Knowledge comes from pretraining data (P1.L1); behaviour from post-training.`,
    },
    {
      type: 'quiz',
      questions: [
        {
          q: 'A base model, asked "Write a haiku about rain", produces three more requests for haiku on other topics. What is happening?',
          options: [
            'The model is broken',
            'It is doing exactly what pretraining optimised: continuing the document plausibly. Lists of prompts are common on the web; nothing has taught it that this text is a request to fulfil',
            'The temperature is too high',
            'The tokenizer split the prompt wrongly',
          ],
          correct: [1],
          explanation: 'Next-token prediction rewards plausible continuations, not helpful ones. SFT on prompt → response examples is the stage that teaches "this is an instruction; respond to it, then stop".',
        },
        {
          q: 'Why can a 1.3B instruction-tuned model be preferred over a 175B base model?',
          options: [
            'The smaller model knows more',
            'Raters judge behaviour — answering the question, following the format, being concise — which post-training teaches directly; the base model\'s extra knowledge is wasted if it does not respond the way people want',
            'Smaller models generate faster',
            'The 175B model was undertrained',
          ],
          correct: [1],
          explanation: 'Preference judgements measure whether the response does what was asked. A base model may know the answer and still not give it. Post-training aligns behaviour with intent at a tiny fraction of pretraining cost — which is why every deployed model goes through it.',
        },
        {
          q: 'A team wants their model to know about a proprietary product launched last month. Is SFT on 500 Q&A pairs about it the right tool?',
          options: [
            'Yes — SFT teaches new knowledge efficiently',
            'Mostly not: a few hundred demonstrations teach style far more reliably than facts, and can encourage confident hallucination; retrieval (giving the facts in the prompt) or continued pretraining on the documents is the usual answer',
            'Yes, if they use LoRA',
            'Only with RLHF afterwards',
          ],
          correct: [1],
          explanation: 'Post-training selects and shapes behaviours the base model can already produce. New facts at small scale are better supplied in context or learned through continued pretraining on enough text that mentions them.',
        },
      ],
    },
    {
      type: 'deepdive',
      title: 'Going deeper',
      md: `- Ouyang et al., "Training language models to follow instructions with human feedback" (InstructGPT, 2022, arxiv.org/abs/2203.02155) — the SFT → reward model → PPO pipeline, end to end.
- Touvron et al., "Llama 2" (2023), section 3 — a detailed modern post-training recipe.
- Lambert et al., "Tülu 3" (2024, arxiv.org/abs/2411.15124) — an open post-training recipe with data, code and ablations.`,
    },
  ],
}

export default lesson

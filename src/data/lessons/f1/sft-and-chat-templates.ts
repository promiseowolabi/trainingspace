import type { Lesson } from '../types'

const lesson: Lesson = {
  id: 'f1.l2',
  slug: 'sft-and-chat-templates',
  trackId: 'f1',
  index: 2,
  title: 'SFT and Chat Templates',
  minutes: 17,
  hook: 'Supervised fine-tuning is pretraining\'s loss on a different document: a conversation, written in a fixed template, with the loss switched off on everything the user said. Get the template or the mask wrong and the model learns to impersonate the user.',
  exercise: 'read+quiz',
  takeaway: {
    number: '1,000 examples',
    claim: 'LIMA fine-tuned a 65B base model on just 1,000 carefully chosen prompt–response pairs and got a capable assistant: SFT is about the quality and format of demonstrations, and its loss counts only the response tokens.',
  },
  blocks: [
    {
      type: 'prose',
      md: `## A conversation is a document with a format

A model sees one flat token sequence (M1.L1). A conversation has structure — a system instruction, user turns, assistant turns — so it is serialised with a **chat template**: special tokens that mark where each role starts and ends. A typical shape:

\`\`\`text
<|system|>You are a concise assistant.<|end|>
<|user|>What is 6·N·D?<|end|>
<|assistant|>The FLOPs to train an N-parameter model on D tokens.<|end|>
\`\`\`

The role markers are **special tokens** — single ids added to the vocabulary that can never be produced by encoding ordinary text — so a user cannot fake an assistant turn by typing the string "<|assistant|>". Every model family defines its own template, and the template used in fine-tuning **must** be the template used at inference: the model has only ever seen assistant turns begin with those exact ids.

## The SFT loss: train on the response only

SFT minimises the same cross-entropy as pretraining (O1.L2), over the same model, with one change: a **loss mask**. Positions inside the system and user turns are excluded; only the assistant's tokens — including the closing \`<|end|>\`, so the model learns to stop — contribute:

\`\`\`text
tokens:  <|user|> What is 6·N·D ? <|end|> <|assistant|> The FLOPs to … <|end|>
mask:       0      0    0   0  0    0         0        1    1   1  …   1
\`\`\`

Lab 12's \`sft_loss\` implements this and checks that changing the logits at any masked position does not change the loss.`,
    },
    {
      type: 'callout',
      variant: 'segfault',
      title: 'The model that answers its own questions',
      md: `Train without the mask and the model is also optimised to predict the user's turns. It learns that after an answer comes another question, in the user's voice — and at inference it sometimes keeps going past its \`<|end|>\`, writing the next user message and answering it. A missing end-of-turn token in the training data produces the same failure: the model never learned where its turn stops.`,
    },
    {
      type: 'prose',
      md: `## What matters in SFT data

**Quality over quantity.** SFT teaches format and behaviour on top of knowledge the base model already has (F1.L1). A small set of excellent, diverse demonstrations often beats a large set of mediocre ones — and every flaw in the demonstrations, such as a habit of over-apologising, is learned as faithfully as the strengths.

**Coverage.** The demonstrations define the assistant's range: coding, maths, refusals, clarifying questions, multi-turn follow-ups. Behaviours absent from SFT data appear rarely or not at all.

**Packing.** Many short conversations are concatenated into each training sequence to avoid wasting compute on padding; the attention mask (or document-boundary tokens) must stop one conversation from attending to the previous one.

**Hyper-parameters.** SFT runs are short — a few epochs over thousands to a few million examples — at learning rates around 10× below pretraining's peak. With so few examples, overfitting is real again (O1.L6): watch validation loss on held-out conversations.`,
    },
    {
      type: 'vendor',
      snapshot: '2026-10',
      title: 'How little data SFT can need',
      md: `Zhou et al. ("LIMA: Less Is More for Alignment", 2023) fine-tuned a **65B** LLaMA model on **1,000** curated prompt–response pairs, with no RLHF, and reported that its responses were judged equivalent or preferable to those of several commercial systems of the time in a large fraction of cases. They framed it as the "superficial alignment hypothesis": a model's knowledge comes almost entirely from pretraining, and alignment mostly teaches which format and style to use. Llama 2's chat models used **27,540** SFT annotations, after finding that a smaller set of high-quality vendor annotations outperformed millions of third-party examples.`,
      systems: ['lima', 'llama-2'],
      sources: ['https://arxiv.org/abs/2305.11206', 'https://arxiv.org/abs/2307.09288'],
    },
    {
      type: 'diagram',
      caption: 'fig 1 — one SFT example, end to end',
      height: 60,
      nodes: [
        { id: 'conv', x: 2, y: 22, w: 18, h: 12, label: 'conversation', sub: 'roles + text' },
        { id: 'tmpl', x: 26, y: 22, w: 18, h: 12, label: 'chat template', sub: 'special tokens' },
        { id: 'ids', x: 50, y: 6, w: 20, h: 12, label: 'token ids', sub: '(T,)' },
        { id: 'mask', x: 50, y: 40, w: 20, h: 12, label: 'loss mask', sub: '1 on response' },
        { id: 'loss', x: 78, y: 22, w: 20, h: 12, label: 'masked CE', sub: 'lab 12 sft_loss' },
      ],
      edges: [
        { from: 'conv', to: 'tmpl' },
        { from: 'tmpl', to: 'ids' },
        { from: 'tmpl', to: 'mask' },
        { from: 'ids', to: 'loss' },
        { from: 'mask', to: 'loss' },
      ],
      steps: [
        { caption: 'A conversation is structured data: a list of turns, each with a role and text.', active: ['conv'] },
        { caption: 'The chat template serialises it with special role and end-of-turn tokens — the same template the model will see at inference.', active: ['conv', 'tmpl'], edges: ['conv->tmpl'] },
        { caption: 'Templating produces the token ids and, alongside them, a mask that is 1 exactly on the assistant\'s tokens, including its end-of-turn token.', active: ['tmpl', 'ids', 'mask'], edges: ['tmpl->ids', 'tmpl->mask'] },
        { caption: 'The loss is next-token cross-entropy averaged over masked-in positions only: the model learns to produce responses, not prompts.', active: ['ids', 'mask', 'loss'], edges: ['ids->loss', 'mask->loss'] },
      ],
    },
    {
      type: 'quiz',
      questions: [
        {
          q: 'After SFT, a model sometimes continues past its answer with "User: Thanks! Can you also…" and then answers that too. Which training bug is most likely?',
          options: [
            'The learning rate was too high',
            'The loss mask included user turns (so the model learned to write them) or the end-of-turn token was missing from the response targets (so it never learned to stop)',
            'The model is too small',
            'Too few epochs',
          ],
          correct: [1],
          explanation: 'Both bugs teach the model that text after its answer is its job to produce. Masking user turns removes the incentive to write them; including the end-of-turn token in the response targets teaches where to stop.',
        },
        {
          q: 'A team fine-tunes with their own template ("### Question: … ### Answer:") but serves the model through a library that applies the base model\'s official chat template. What will they see?',
          options: [
            'No difference — templates are cosmetic',
            'Degraded, erratic behaviour: the model learned to respond after "### Answer:" tokens and now sees a different sequence of ids it was never fine-tuned on',
            'Faster inference',
            'An error from the tokenizer',
          ],
          correct: [1],
          explanation: 'The template is part of the input distribution. A model fine-tuned on one format and prompted with another is out of distribution at the exact point where its behaviour was shaped. Train and serve with the same template.',
        },
        {
          q: 'Why are role markers implemented as special tokens rather than ordinary text like "Assistant:"?',
          options: [
            'They are shorter',
            'Special-token ids cannot be produced by encoding user-supplied text, so a user cannot forge a role boundary — and the model gets one unambiguous id per boundary',
            'Ordinary text cannot be masked',
            'Special tokens are faster to embed',
          ],
          correct: [1],
          explanation: 'If "Assistant:" were plain text, any user could type it and inject a fake assistant turn. Reserved ids that the tokenizer never emits for raw text make the structure unforgeable from inside the content.',
        },
      ],
    },
    {
      type: 'deepdive',
      title: 'Going deeper',
      md: `- Zhou et al., "LIMA: Less Is More for Alignment" (2023, arxiv.org/abs/2305.11206).
- Hugging Face's chat templating documentation — templates are stored as Jinja in each model's tokenizer config; read one and find its special tokens.
- Stretch goal with lab 12: build ten prompt/response pairs about this course in a template of your own, fine-tune lab 07's model with LoRA and \`sft_loss\`, and sample from it before and after.`,
    },
  ],
}

export default lesson

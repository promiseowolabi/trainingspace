import type { LabGuide } from './types'

const guide: LabGuide = {
  id: 'bpe-tokenizer',
  requires: ['a browser — or Python 3.10+ and pytest (no numpy needed)'],
  blocks: [
    {
      type: 'prose',
      md: `## What you will build

The tokenizer family GPT-2, GPT-4 and Llama 3 all belong to: **byte-level byte-pair encoding**. Five functions, about forty lines. The first two are utilities, \`train\` learns the merges, \`encode\` and \`decode\` use them.

The training corpus in \`checks.py\` is a short paragraph repeated four times, small enough to train in well under a second in the browser.`,
    },
    {
      type: 'prose',
      md: `## 1. get_stats and merge

\`get_stats\` walks \`zip(ids, ids[1:])\` and counts. Use a plain dict and \`counts.get(pair, 0) + 1\`: dict insertion order then matches first occurrence, which is what \`train\`'s tie-break needs.

\`merge\` walks with an index: if \`ids[i], ids[i+1]\` is the pair, emit \`new_id\` and jump by 2; otherwise emit \`ids[i]\` and step by 1. The jump is what makes \`[5, 5, 5]\` become \`[9, 5]\` and not \`[9, 9]\`.

## 2. train

A loop over \`range(256, vocab_size)\`: stats, \`max(stats, key=stats.get)\`, merge, record. The \`train-classic\` check uses the example from the original BPE description, "aaabdaaabac", and needs exactly four merges in exactly this order: \`aa\`, then \`aa·a\`, then \`aaa·b\`, then \`aaab·d\`. The second and fourth merges are ties; first occurrence decides them.

## 3. encode

Encoding must reproduce what training did, so it applies merges **by rank**: at each step, look at the pairs present and merge the one with the lowest merge id. \`min(stats, key=lambda p: merges.get(p, float("inf")))\` finds it; if that pair is not in \`merges\` at all, nothing more can merge and you are done.`,
    },
    {
      type: 'callout',
      variant: 'warning',
      title: 'Why not merge greedily left to right?',
      md: `Applying "whatever merge fits first in the string" gives different tokens from the ones training produced, so the model would see ids in contexts it never saw during training. Rank order is the contract between the tokenizer that was trained and the one that runs. The \`encode-replays\` check catches the difference.`,
    },
    {
      type: 'prose',
      md: `## 4. decode

\`b"".join(vocab(merges)[i] for i in ids).decode("utf-8", errors="replace")\`. The \`errors\` argument is not optional: a model generating one token at a time will regularly stop in the middle of a multi-byte character, and \`decode-partial\` hands you exactly that — the first byte of "é".`,
    },
  ],
  cleanup: [
    {
      type: 'prose',
      md: `- Train on the source of this lab file itself and look at the first 30 merges with \`vocab(merges)\`: you will see indentation and \`self\` appear early.
- Train on \`CORPUS\` with \`vocab_size=1024\` and encode it: the whole text becomes one token. A tokenizer can overfit its training text too.
- Real tokenizers first split text with a regex (words, numbers, punctuation) and never merge across those boundaries — GPT-2's pattern is in its \`encoder.py\`. Add that and compare the merges.
- Keep \`tokenizer.py\`: the capstone trains one on your own corpus.`,
    },
  ],
}

export default guide

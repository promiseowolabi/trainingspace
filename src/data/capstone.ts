/**
 * Capstone — Train It, Defend It.
 *
 * One small GPT, trained end to end on the reader's laptop with the pieces
 * they built: their tokenizer (lab 04), their training loop (lab 07), their
 * FLOP and memory arithmetic (lab 08) and their LoRA layer (lab 12). The
 * deliverable is a one-page run report in which every number is either
 * measured or computed — and the two are compared.
 *
 * Steps are self-assessed (a checkbox each); the deliverable is the write-up.
 */

export interface CapstoneStep {
  id: string
  title: string
  /** What done looks like, stated so the reader can check it themselves. */
  done: string
  body: string[]
}

export const CAPSTONE_STEPS: CapstoneStep[] = [
  {
    id: 'tokenize',
    title: 'Tokenize a corpus with your own tokenizer',
    done: 'Your lab 04 tokenizer, trained to a vocabulary you chose, encodes and decodes the corpus losslessly, and you have its token count and bytes per token.',
    body: [
      'Pick a corpus of 1–10 MB of plain text — a public-domain book collection, a code repository, your own notes. Train lab 04\'s BPE on it with a vocabulary of 512 to 4096, then encode the whole corpus.',
      'Record the vocabulary size V, the token count D, and bytes per token. Hold out the last 10% as validation before you look at any loss.',
    ],
  },
  {
    id: 'pretrain',
    title: 'Pretrain a small GPT on your CPU',
    done: 'A loss curve for train and validation, starting near ln V and ending well below it, from a run of at most an hour, resumable from its last checkpoint.',
    body: [
      'Use lab 07\'s loop with a model of 1–10M parameters (for example 4 layers, d = 256, context 256). Warm up, decay with cosine, clip gradients at 1.0, evaluate every few hundred steps, checkpoint as you go.',
      'Check the first logged loss against ln V before trusting anything else (E1.L2). Kill the run once and resume it, and show the curve does not jump.',
    ],
  },
  {
    id: 'account',
    title: 'Account for every FLOP and byte',
    done: 'A table of predicted versus measured: parameter count, tokens seen, 6·N·D FLOPs, tokens per second, achieved FLOP/s, MFU against your CPU\'s peak, and peak memory against 16 bytes per parameter plus activations.',
    body: [
      'Predict N from the config with lab 08 and compare with `sum(p.numel() for p in model.parameters())`. Compute 6·N·D for the tokens you actually trained on, divide by wall-clock seconds, and divide again by your CPU\'s peak FLOP/s (cores × clock × FLOPs per cycle — say where your number comes from).',
      'Then the counterfactual: how long would the same run take on one modern GPU at 40% MFU, and how many tokens would Chinchilla have recommended for this N? Show the arithmetic.',
    ],
  },
  {
    id: 'finetune',
    title: 'Fine-tune it with LoRA',
    done: 'A LoRA adapter, trained on a few hundred prompt/response pairs with the loss masked on the prompt, that measurably changes the model\'s outputs — with the trainable-parameter fraction stated.',
    body: [
      'Write a tiny instruction format for your corpus (for example: a question, a separator token, an answer drawn from the text). Apply lab 12\'s LoRA to the attention projections, freeze everything else, and train.',
      'Report the trainable parameter count as a fraction of N, and show three samples before and after, at the same temperature and seed.',
    ],
  },
  {
    id: 'report',
    title: 'Write the run report',
    done: 'One page: the setup, the loss curves, the predicted-versus-measured table, the samples, what went wrong and how you noticed — and the one change you would make with 100× the compute.',
    body: [
      'Use E1.L4\'s paper table as the skeleton: N, D, C, hardware, layout, recipe. A number with no source is a number a reviewer will ask about; label each as measured, computed or cited, the same way the course does.',
    ],
  },
]

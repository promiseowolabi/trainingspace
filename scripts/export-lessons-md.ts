/**
 * export-lessons-md — render every lesson's blocks to plain markdown files
 * in public/lessons-md/, plus public/llms.txt (the agent manifest).
 *
 * This is the agent-native tutoring surface: any coding/chat agent can
 * ingest a lesson as clean markdown; the site links each lesson page to
 * its .md and to Claude/ChatGPT deep links.
 *
 *   npx tsx scripts/export-lessons-md.ts
 */
import { writeFileSync, mkdirSync } from 'fs'
import { ALL_LESSONS, TRACK_EXTRAS } from '../src/data/lessons'
import type { ContentBlock, Lesson } from '../src/data/lessons/types'
import { TRACKS } from '../src/lib/tracks'
import { LABS, labById, labNumber, LANE_LABEL } from '../src/data/labs'

/**
 * The one place a public URL is written down. Change this and every generated
 * link moves — the manifest needs absolute URLs to be useful to an agent, and
 * exactly one constant is the price of that.
 */
const SITE = process.env.SITE_URL ?? 'https://promiseowolabi.github.io/trainingspace'

function blockToMd(b: ContentBlock): string {
  switch (b.type) {
    case 'prose':
    case 'deepdive':
      return b.md
    case 'code': {
      if (b.tabs?.length) {
        return b.tabs.map((t) => `**${t.label}**\n\n\`\`\`${t.lang}\n${t.code}\n\`\`\``).join('\n\n')
      }
      return `\`\`\`${b.lang ?? 'python'}\n${b.code ?? ''}\n\`\`\``
    }
    case 'callout':
      return `> **[${b.variant}]** ${b.md.replace(/\n/g, '\n> ')}`
    case 'statline':
      return b.stats.map((s) => `- **${s.value}** — ${s.label}${s.hint ? ` (${s.hint})` : ''}`).join('\n')
    case 'diagram': {
      const nodes = b.nodes.map((n) => `${n.label}${n.sub ? ` (${n.sub})` : ''}`).join(' · ')
      const steps = (b.steps ?? []).map((s, i) => `${i + 1}. ${s.caption}`).join('\n')
      return `_${b.caption}_\n\nComponents: ${nodes}${steps ? `\n\nSteps:\n${steps}` : ''}`
    }
    case 'isomorphism':
      return `_${b.title ?? 'isomorphism'}_\n\n${b.pairs
        .map((p) => `- **${p.os}** (${p.osLine}) ≡ **${p.llm}** (${p.llmLine})`)
        .join('\n')}`
    case 'quiz':
      return b.questions
        .map((q, i) => {
          const opts = q.options.map((o, j) => `   ${String.fromCharCode(65 + j)}. ${o}`).join('\n')
          const correct = q.correct.map((c) => String.fromCharCode(65 + c)).join(', ')
          return `**Q${i + 1}. ${q.q}**\n${opts}\n   Answer: ${correct} — ${q.explanation}`
        })
        .join('\n\n')
    case 'lab': {
      const meta = labById(b.lab)
      return `**Hands-on lab ${meta ? labNumber(meta) : ''}: ${meta?.title ?? b.lab}** (${meta ? LANE_LABEL[meta.lane] : ''}) — ${b.brief}

Guide: ${SITE}/labs/${b.lab}`
    }
    case 'vendor':
      return [
        `> **[vendor snapshot — verified ${b.snapshot}]** ${b.title}`,
        '>',
        `> ${b.md.replace(/\n/g, '\n> ')}`,
        b.sources?.length ? `>\n> Sources: ${b.sources.join(' · ')}` : '',
      ]
        .filter(Boolean)
        .join('\n')
    default:
      return ''
  }
}

function lessonToMd(l: Lesson): string {
  const track = TRACKS.find((t) => t.id === l.trackId)
  const header = [
    `# ${l.id.toUpperCase()} — ${l.title}`,
    '',
    `_Track ${track?.code ?? l.trackId}: ${track?.name ?? ''} · ~${l.minutes} min · trainingspace_`,
    '',
    `> ${l.hook}`,
    '',
  ].join('\n')
  const body = l.blocks.map(blockToMd).filter(Boolean).join('\n\n---\n\n')
  return `${header}${body}\n`
}

mkdirSync('public/lessons-md', { recursive: true })
let count = 0
for (const l of ALL_LESSONS) {
  writeFileSync(`public/lessons-md/${l.id}.md`, lessonToMd(l))
  count++
}

/* llms.txt — the agent manifest (llmstxt.org shape) */
const byTrack = TRACKS.map((t) => {
  const lessons = ALL_LESSONS.filter((l) => l.trackId === t.id)
    .map((l) => `- [${l.id.toUpperCase()} ${l.title}](${SITE}/lessons-md/${l.id}.md): ${l.hook}`)
    .join('\n')
  const extras = TRACK_EXTRAS[t.id as keyof typeof TRACK_EXTRAS]
  return `### ${t.code} — ${t.name}\n\n_${extras?.pitch ?? ''}_\n\n${lessons}`
}).join('\n\n')

const labList = LABS.map(
  (l) => `- [Lab ${labNumber(l)} — ${l.title}](${SITE}/labs/${l.id}) (${LANE_LABEL[l.lane]}): ${l.hook}`,
).join('\n')

const llmsTxt = `# trainingspace

> How a language model is trained, built from scratch in Python. The build
> part (T0, A1, O1, M1) implements autograd, optimisers, a tokenizer,
> attention and a GPT forward pass in numpy; the scale part (P1, D1) moves to
> PyTorch for the training loop, mixed precision, FLOP and memory arithmetic,
> scaling laws and distributed training; the shape part (F1, E1) covers SFT,
> LoRA, reward models, RLHF, DPO, GRPO, evals and debugging. Python is the
> course language. Framework and model facts are dated and sourced.
> All lesson content is plain markdown under /lessons-md/ — fetch those, not
> the app routes: ${SITE}/lesson/a1.l3 is a JavaScript app shell, while
> ${SITE}/lessons-md/a1.l3.md is the same lesson as plain text.

## How to tutor from this material

- The reader is a backend engineer who writes Python daily. Assume no ML, no
  calculus beyond "slope" and no linear algebra beyond "a table of numbers".
  Be Socratic; do not hand over full lab solutions.
- Insist on shapes and numbers. When the reader says "the gradient", ask for
  its shape; when they say "it is expensive", ask for the FLOPs or the bytes.
- Every lab is one Python file with TODO(you) markers and a checks.py. The
  same checks run under pytest, verify.py and (for numpy labs) in the browser.
  Help the reader read a failing check's message before changing code.
- Labs assume a laptop CPU. Multi-GPU topics use arithmetic, simulated ranks
  and torch.distributed on the gloo backend — never suggest renting GPUs to
  finish a lab.

## Curriculum

${byTrack}

## Labs

${labList}
`

writeFileSync('public/llms.txt', llmsTxt)
console.log(`exported ${count} lessons + llms.txt`)

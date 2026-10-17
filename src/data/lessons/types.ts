/**
 * Trainingspace lesson content model.
 *
 * Forked from the series' shared lesson model. The block system, diagram rules and quiz
 * rules are unchanged. Deltas:
 *
 *   1. `TrackId` covers the build part (t0, a1, o1, m1), the scale part
 *      (p1, d1) and the shape part (f1, e1).
 *   2. `takeaway` is REQUIRED — every lesson hands the reader one number.
 *      Enforced by tests/registry.test.ts.
 *   3. `vendor` blocks quarantine every framework- or model-specific claim
 *      (a PyTorch default, a published token count) with the month verified.
 *   4. `lab` blocks send the reader to a Python lab (src/data/labs.ts),
 *      graded in the browser by Pyodide or locally by verify.py.
 *
 * Only the block types here exist. No new fields, no `any`, no imports beyond
 * types.
 */

import type { QuizQuestion } from '@/components/QuizBlock'
import type { CodeTab } from '@/components/CodeBlock'

/** Track ids: build (t0, a1, o1, m1), scale (p1, d1), shape (f1, e1). */
export type TrackId = 't0' | 'a1' | 'o1' | 'm1' | 'p1' | 'd1' | 'f1' | 'e1'

/** Level bands. */
export type Level = 200 | 300 | 400 | 500

/** Lab ids — fixed; each has a folder under labs/ with a checks.py. */
export type LabId =
  | 'numpy-shapes'
  | 'micrograd'
  | 'tensor-autograd'
  | 'optimisers'
  | 'bpe-tokenizer'
  | 'attention'
  | 'gpt-forward'
  | 'train-loop'
  | 'flops-and-memory'
  | 'collectives'
  | 'ddp-from-scratch'
  | 'zero-sharding'
  | 'lora'
  | 'preference-losses'

/** Exercise type shown as the lesson's chip. */
export type ExerciseKind = 'quiz' | 'read' | 'read+quiz' | 'lab+quiz'

/* ------------------------------ blocks ------------------------------ */

/**
 * `prose` — markdown-lite rendered by the lesson engine:
 *   `## H2` / `### H3`, paragraphs (blank-line separated),
 *   `- ` bullet lists, `1. ` ordered lists, GitHub-style pipe tables,
 *   inline **bold**, *em*, `code`, [label](https://url).
 */
export interface ProseBlock {
  type: 'prose'
  md: string
}

export interface CodeBlockData {
  type: 'code'
  filename?: string
  /** Compare tabs (Python | Java …) — defaults to single `code`/`lang`. Python is the default language. */
  tabs?: CodeTab[]
  code?: string
  lang?: string
  highlightLines?: number[]
  /** Annotation chips under the header, e.g. `numpy` · `O(n²)`. */
  chips?: string[]
}

export type CalloutVariant = 'analogy' | 'info' | 'warning' | 'segfault' | 'isomorphism'

export interface CalloutBlock {
  type: 'callout'
  variant: CalloutVariant
  title?: string
  md: string
}

/** Step-through SVG diagram. */
export interface DiagramNode {
  id: string
  /** Grid coords on a 100×H viewBox canvas. */
  x: number
  y: number
  w?: number
  h?: number
  label: string
  sub?: string
  /** hex color override; defaults to track color for active, line for idle */
  color?: string
}

export interface DiagramEdge {
  from: string
  to: string
  label?: string
}

export interface DiagramStep {
  caption: string
  /** node ids highlighted in this step */
  active?: string[]
  /** edge keys `${from}->${to}` highlighted in this step */
  edges?: string[]
}

export interface DiagramBlock {
  type: 'diagram'
  /** mono figure caption, e.g. `fig 1 — the backward pass` */
  caption: string
  nodes: DiagramNode[]
  edges?: DiagramEdge[]
  steps: DiagramStep[]
  /** viewBox height in arbitrary units (width fixed 100). default 60 */
  height?: number
}

export interface StatChipData {
  value: string
  label: string
  /** plain-English tooltip */
  hint?: string
}

export interface StatlineBlock {
  type: 'statline'
  stats: StatChipData[]
}

export interface QuizBlockData {
  type: 'quiz'
  questions: QuizQuestion[]
}

/**
 * Isomorphism panel. Field names are historical (`os`/`llm` from kernelspace)
 * and kept so the renderer is shared across the series: read them as
 * "left side" and "right side".
 */
export interface IsomorphismPair {
  os: string
  osLine: string
  llm: string
  llmLine: string
}

export interface IsomorphismBlock {
  type: 'isomorphism'
  title?: string
  pairs: IsomorphismPair[]
}

export interface DeepdiveBlock {
  type: 'deepdive'
  title: string
  md: string
}

/** Sends the reader to a hands-on lab (src/data/labs.ts). */
export interface LabBlock {
  type: 'lab'
  lab: LabId
  /** One or two sentences: what this lab makes you observe, in this lesson's terms. */
  brief: string
}

/**
 * `vendor` — the rot rule made structural.
 *
 * No vendor-specific fact may appear loose in prose: not a default, not a
 * limit, not a feature, and NEVER a price. It lives here, with the month it
 * was verified and the source it was verified against, or it does not appear.
 *
 * `snapshot` is the month you CHECKED the claim, not the month you wrote the
 * lesson. If you did not check it, you may not write it as fact.
 */
export interface VendorBlock {
  type: 'vendor'
  /** Month verified, `YYYY-MM`. Rendered to the reader. */
  snapshot: string
  title: string
  md: string
  /** Systems named in this block, e.g. ['pytorch', 'llama-3']. */
  systems?: string[]
  /** Documentation URLs the claim was verified against. Required in practice. */
  sources?: string[]
}

export type ContentBlock =
  | ProseBlock
  | CodeBlockData
  | CalloutBlock
  | DiagramBlock
  | StatlineBlock
  | QuizBlockData
  | IsomorphismBlock
  | DeepdiveBlock
  | LabBlock
  | VendorBlock

/* ------------------------------ lesson ------------------------------ */

/**
 * The one number the reader carries out. REQUIRED and enforced: `number` must
 * contain a digit. This is the machine-checkable form of the course's central
 * discipline — every abstraction gets an arithmetic anchor.
 */
export interface Takeaway {
  /** e.g. `16 B/param`, `6·N·D`, `ln 50257 ≈ 10.8`. Must contain a digit. */
  number: string
  /** One sentence a reader could say out loud in a review. */
  claim: string
}

export interface Lesson {
  /** Canonical id used by the progress store: `a1.l3`. */
  id: string
  /** Human slug — also resolves at /lesson/:lessonId. */
  slug: string
  trackId: TrackId
  /** 1-based position within the track. */
  index: number
  title: string
  minutes: number
  /** One-line hook shown in lesson rows. */
  hook: string
  exercise: ExerciseKind
  /** Exam lesson (amber chip, quiz-gated completion). */
  exam?: boolean
  takeaway: Takeaway
  blocks: ContentBlock[]
}

/** Heading extracted from blocks for the "ON THIS PAGE" rail. */
export interface LessonHeading {
  id: string
  text: string
  level: 2 | 3
}

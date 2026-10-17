/**
 * Lesson registry — trainingspace content source of truth.
 *
 * Build: t0 (The Training Contract), a1 (Autograd from Scratch),
 *        o1 (Networks & Optimisation), m1 (The Transformer).
 * Scale: p1 (Pretraining), d1 (Distributed Training).
 * Shape: f1 (Post-Training), e1 (Evals & Debugging).
 *
 * A track whose array is shorter than its `lessons` count in
 * src/lib/tracks.ts renders the remainder as stubs; `npm run report` prints
 * the gap.
 */

import type { Lesson, TrackId } from './types'
import { ORDERED_LESSON_IDS } from './manifest'

// T0 — The Training Contract
import t0l1 from './t0/what-training-is'
import t0l2 from './t0/tensors-and-shapes'
import t0l3 from './t0/the-cost-of-a-matmul'
import t0l4 from './t0/where-the-memory-goes'

// A1 — Autograd from Scratch
import a1l1 from './a1/derivatives-as-sensitivity'
import a1l2 from './a1/the-chain-rule-as-a-graph'
import a1l3 from './a1/a-scalar-engine'
import a1l4 from './a1/tensor-autograd'
import a1l5 from './a1/gradient-checking'

// O1 — Networks & Optimisation
import o1l1 from './o1/the-mlp'
import o1l2 from './o1/cross-entropy'
import o1l3 from './o1/optimisers'
import o1l4 from './o1/learning-rate-schedules'
import o1l5 from './o1/normalisation'
import o1l6 from './o1/generalisation'

// M1 — The Transformer
import m1l1 from './m1/tokenization'
import m1l2 from './m1/embeddings-and-position'
import m1l3 from './m1/attention'
import m1l4 from './m1/the-block'
import m1l5 from './m1/counting-a-gpt'
import m1l6 from './m1/sampling'

// P1 — Pretraining
import p1l1 from './p1/data-is-the-model'
import p1l2 from './p1/the-training-loop'
import p1l3 from './p1/mixed-precision'
import p1l4 from './p1/flops-and-mfu'
import p1l5 from './p1/scaling-laws'
import p1l6 from './p1/stability'

// D1 — Distributed Training
import d1l1 from './d1/collectives'
import d1l2 from './d1/data-parallel'
import d1l3 from './d1/zero-and-fsdp'
import d1l4 from './d1/tensor-parallel'
import d1l5 from './d1/pipeline-parallel'
import d1l6 from './d1/choosing-a-layout'

// F1 — Post-Training
import f1l1 from './f1/from-base-to-assistant'
import f1l2 from './f1/sft-and-chat-templates'
import f1l3 from './f1/lora'
import f1l4 from './f1/reward-models'
import f1l5 from './f1/rlhf-and-dpo'
import f1l6 from './f1/grpo-and-verifiable-rewards'

// E1 — Evals & Debugging
import e1l1 from './e1/what-evals-measure'
import e1l2 from './e1/debugging-a-run'
import e1l3 from './e1/checkpointing-and-resumption'
import e1l4 from './e1/reading-a-training-paper'

// @lesson-imports

export const LESSONS_BY_TRACK: Record<TrackId, Lesson[]> = {
  t0: [t0l1, t0l2, t0l3, t0l4],
  a1: [a1l1, a1l2, a1l3, a1l4, a1l5],
  o1: [o1l1, o1l2, o1l3, o1l4, o1l5, o1l6],
  m1: [m1l1, m1l2, m1l3, m1l4, m1l5, m1l6],
  p1: [p1l1, p1l2, p1l3, p1l4, p1l5, p1l6],
  d1: [d1l1, d1l2, d1l3, d1l4, d1l5, d1l6],
  f1: [f1l1, f1l2, f1l3, f1l4, f1l5, f1l6],
  e1: [e1l1, e1l2, e1l3, e1l4],
}

export const TRACK_IDS: TrackId[] = ['t0', 'a1', 'o1', 'm1', 'p1', 'd1', 'f1', 'e1']

/** All lessons in curriculum order. */
export const ALL_LESSONS: Lesson[] = TRACK_IDS.flatMap((id) => LESSONS_BY_TRACK[id])

const byIdMap = new Map<string, Lesson>()
for (const l of ALL_LESSONS) {
  byIdMap.set(l.id, l)
  byIdMap.set(l.slug, l)
}

/** Resolve a lesson by canonical id (`a1.l3`) or slug (`a-scalar-engine`). */
export function lessonById(idOrSlug: string | undefined): Lesson | undefined {
  if (!idOrSlug) return undefined
  return byIdMap.get(idOrSlug)
}

export function lessonsForTrack(trackId: string): Lesson[] {
  return LESSONS_BY_TRACK[trackId as TrackId] ?? []
}

/** Next lesson in curriculum order — crosses track boundaries. */
export function nextLesson(lesson: Lesson): Lesson | undefined {
  const i = ORDERED_LESSON_IDS.indexOf(lesson.id)
  return i >= 0 ? lessonById(ORDERED_LESSON_IDS[i + 1]) : undefined
}

/** Previous lesson in curriculum order — crosses track boundaries. */
export function prevLesson(lesson: Lesson): Lesson | undefined {
  const i = ORDERED_LESSON_IDS.indexOf(lesson.id)
  return i > 0 ? lessonById(ORDERED_LESSON_IDS[i - 1]) : undefined
}

/* ---------------------- light re-exports ---------------------- */

/*
 * Metadata and track prose live in modules that do NOT import the corpus, so a
 * list page costs kilobytes. They are re-exported here for callers that already
 * hold a full Lesson.
 */
export * from './manifest'
export * from './track-extras'
export type { Lesson, ContentBlock } from './types'

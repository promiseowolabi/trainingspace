/**
 * gen-manifest — emit src/data/lessons/manifest.ts from the lesson registry.
 *
 * ── Why this exists ────────────────────────────────────────────────────────
 * The lesson files are the source of truth, and importing them all pulls in
 * every word of prose: ~1.35 MB of javascript. The curriculum list, a track
 * page and a lesson row need only METADATA — id, title, hook, minutes, the
 * takeaway — and none of them touch `blocks`. Before this split, opening
 * /curriculum to read a list of titles downloaded the entire corpus.
 *
 * So metadata is generated into a flat, literal module with no imports of the
 * lesson files at all, the list pages import that, and only the lesson route
 * pulls the corpus. tests/bundle.test.ts caught the regression that prompted
 * this, and tests/manifest.test.ts stops the generated copy drifting.
 *
 *   npx tsx scripts/gen-manifest.ts
 */

import { writeFileSync } from 'node:fs'
import { ALL_LESSONS } from '../src/data/lessons'

const q = (s: string): string => JSON.stringify(s)

const rows = ALL_LESSONS.map((l) => {
  const fields = [
    `    id: ${q(l.id)}`,
    `    slug: ${q(l.slug)}`,
    `    trackId: ${q(l.trackId)}`,
    `    index: ${l.index}`,
    `    title: ${q(l.title)}`,
    `    minutes: ${l.minutes}`,
    `    hook: ${q(l.hook)}`,
    `    exercise: ${q(l.exercise)}`,
  ]
  /* Labs this lesson sends the reader to, so a row can show them. */
  const labs = [...new Set(l.blocks.flatMap((b) => (b.type === 'lab' ? [b.lab] : [])))]
  if (labs.length) fields.push(`    labs: [${labs.map(q).join(', ')}]`)
  if (l.exam) fields.push(`    exam: true`)
  fields.push(`    takeaway: { number: ${q(l.takeaway.number)}, claim: ${q(l.takeaway.claim)} }`)
  /* Block-type census, so a list page can show chips without loading blocks. */
  const kinds = [...new Set(l.blocks.map((b) => b.type))].sort()
  fields.push(`    blockKinds: [${kinds.map(q).join(', ')}]`)
  return `  {\n${fields.join(',\n')},\n  },`
}).join('\n')

const out = `/**
 * GENERATED FILE — do not edit by hand.
 *   npx tsx scripts/gen-manifest.ts
 *
 * Lesson METADATA only, as literal data with no imports of the lesson files.
 * This is what the curriculum list, the track pages and the lesson rows read,
 * so those routes cost kilobytes instead of the whole ${(ALL_LESSONS.length)}-lesson corpus.
 * The corpus itself is imported only by the lesson route, through
 * src/data/lessons/index.ts.
 *
 * tests/manifest.test.ts fails if this drifts from the registry.
 */

import type { ContentBlock, ExerciseKind, LabId, Takeaway, TrackId } from './types'

export interface LessonMeta {
  id: string
  slug: string
  trackId: TrackId
  index: number
  title: string
  minutes: number
  hook: string
  exercise: ExerciseKind
  /** Hands-on labs the lesson links to. */
  labs?: LabId[]
  exam?: boolean
  takeaway: Takeaway
  /** Distinct block types present, sorted. Lets a row show chips without the blocks. */
  blockKinds: ContentBlock['type'][]
}

export const LESSON_META: LessonMeta[] = [
${rows}
]

export const TOTAL_LESSON_COUNT = LESSON_META.length

/** Canonical ids in curriculum order. */
export const ORDERED_LESSON_IDS: string[] = LESSON_META.map((l) => l.id)

const byId = new Map<string, LessonMeta>()
for (const l of LESSON_META) {
  byId.set(l.id, l)
  byId.set(l.slug, l)
}

/** Resolve metadata by canonical id (\`k1.l1\`) or slug. */
export const lessonMeta = (idOrSlug: string | undefined): LessonMeta | undefined =>
  idOrSlug ? byId.get(idOrSlug) : undefined

export const metaForTrack = (trackId: string): LessonMeta[] =>
  LESSON_META.filter((l) => l.trackId === trackId)

/** Route helper — canonical lesson URL. */
export const lessonPath = (l: { id: string }) => \`/lesson/\${l.id}\`

/** Every takeaway, in curriculum order. This list is the real syllabus. */
export const takeaways = (): { id: string; title: string; number: string; claim: string }[] =>
  LESSON_META.map((l) => ({
    id: l.id,
    title: l.title,
    number: l.takeaway.number,
    claim: l.takeaway.claim,
  }))

export const nextMeta = (l: LessonMeta): LessonMeta | undefined =>
  LESSON_META[LESSON_META.findIndex((x) => x.id === l.id) + 1]

export const prevMeta = (l: LessonMeta): LessonMeta | undefined => {
  const i = LESSON_META.findIndex((x) => x.id === l.id)
  return i > 0 ? LESSON_META[i - 1] : undefined
}
`

writeFileSync('src/data/lessons/manifest.ts', out)
console.log(`generated manifest: ${ALL_LESSONS.length} lessons`)

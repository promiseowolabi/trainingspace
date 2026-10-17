/**
 * Registry contract tests.
 *
 * The invariants the content is written against — if one of them breaks, some
 * lesson is quietly promising something the machinery does not do.
 */

import { describe, expect, it } from 'vitest'
import {
  ALL_LESSONS,
  LESSONS_BY_TRACK,
  TRACK_EXTRAS,
  TRACK_IDS,
  lessonById,
  takeaways,
} from '@/data/lessons'
import type { ContentBlock, TrackId } from '@/data/lessons/types'
import { TRACKS, CAPSTONE, ORDERED_LESSON_IDS } from '@/lib/tracks'
import { LABS } from '@/data/labs'
import { LAB_GUIDES } from '@/data/lab-guides'

describe('tracks', () => {
  it('every track has extras and a positive lesson count', () => {
    for (const t of TRACKS) {
      expect(TRACK_EXTRAS[t.id as TrackId], `extras for ${t.id}`).toBeDefined()
      expect(t.lessons, `${t.id} lesson count`).toBeGreaterThan(0)
      expect(TRACK_EXTRAS[t.id as TrackId].outcomes.length).toBeGreaterThanOrEqual(3)
    }
  })

  it('TRACK_IDS matches the track registry exactly, in order', () => {
    expect(TRACK_IDS).toEqual(TRACKS.map((t) => t.id))
  })

  it('the capstone is not a content track', () => {
    expect(CAPSTONE.id).toBe('capstone')
    expect(TRACKS.map((t) => t.id)).not.toContain('capstone')
  })

  it('all three parts exist', () => {
    for (const half of ['build', 'scale', 'shape'] as const) {
      expect(TRACKS.filter((t) => t.half === half).length, half).toBeGreaterThan(0)
    }
  })
})

describe('lessons', () => {
  it('id, slug, trackId and index are internally consistent', () => {
    for (const l of ALL_LESSONS) {
      expect(l.id, `${l.slug} id`).toBe(`${l.trackId}.l${l.index}`)
      expect(l.slug, `${l.id} slug is kebab-case`).toMatch(/^[a-z0-9]+(-[a-z0-9]+)*$/)
      expect(lessonById(l.id)).toBe(l)
      expect(lessonById(l.slug)).toBe(l)
    }
  })

  it('lesson indexes within a track are 1..n with no gaps or duplicates', () => {
    for (const id of TRACK_IDS) {
      const idx = LESSONS_BY_TRACK[id].map((l) => l.index).sort((a, b) => a - b)
      expect(idx, `${id} indexes`).toEqual(idx.map((_, i) => i + 1))
    }
  })

  it('no authored track exceeds its declared lesson count', () => {
    for (const t of TRACKS) {
      expect(LESSONS_BY_TRACK[t.id as TrackId].length, `${t.id}`).toBeLessThanOrEqual(t.lessons)
    }
  })

  it('every authored lesson id appears in the ordered id list', () => {
    for (const l of ALL_LESSONS) expect(ORDERED_LESSON_IDS).toContain(l.id)
  })

  /**
   * The takeaway rule, inherited from vectorspace: every lesson hands the
   * reader ONE number they can carry into a meeting. If you cannot name it, the
   * lesson is not finished — so this is a test, not a guideline.
   */
  it('every lesson has a takeaway whose number contains a digit', () => {
    for (const l of ALL_LESSONS) {
      expect(l.takeaway, `${l.id} takeaway`).toBeDefined()
      expect(l.takeaway.number, `${l.id} takeaway.number`).toMatch(/\d/)
      expect(l.takeaway.claim.length, `${l.id} takeaway.claim`).toBeGreaterThan(20)
    }
  })

  it('minutes are an honest reading estimate (9–24)', () => {
    for (const l of ALL_LESSONS) {
      expect(l.minutes, `${l.id} minutes`).toBeGreaterThanOrEqual(9)
      expect(l.minutes, `${l.id} minutes`).toBeLessThanOrEqual(24)
    }
  })

  it('takeaways() returns one entry per lesson', () => {
    expect(takeaways()).toHaveLength(ALL_LESSONS.length)
  })
})

describe('lesson blocks', () => {
  const blocksOf = (): { id: string; b: ContentBlock }[] =>
    ALL_LESSONS.flatMap((l) => l.blocks.map((b) => ({ id: l.id, b })))

  it('exactly one quiz per lesson, and it is not the first block', () => {
    for (const l of ALL_LESSONS) {
      const quizzes = l.blocks.filter((b) => b.type === 'quiz')
      expect(quizzes.length, `${l.id} quiz count`).toBe(1)
      expect(l.blocks[0].type, `${l.id} opens with prose`).toBe('prose')
    }
  })

  it('every quiz question has exactly one correct answer among four options', () => {
    for (const { id, b } of blocksOf()) {
      if (b.type !== 'quiz') continue
      for (const q of b.questions) {
        expect(q.options.length, `${id} options`).toBeGreaterThanOrEqual(2)
        expect(q.correct.length, `${id} single correct answer`).toBe(1)
        expect(q.correct[0]).toBeLessThan(q.options.length)
        /* `explanation` is optional in the shared QuizQuestion type, but this course requires it. */
        expect(q.explanation, `${id} missing explanation`).toBeDefined()
        expect(q.explanation!.length, `${id} explanation`).toBeGreaterThan(40)
      }
    }
  })

  /** The rot rule: a vendor claim without a dated, sourced snapshot is a lie waiting to happen. */
  it('every vendor block is dated YYYY-MM and cites at least one source', () => {
    for (const { id, b } of blocksOf()) {
      if (b.type !== 'vendor') continue
      expect(b.snapshot, `${id} vendor snapshot`).toMatch(/^\d{4}-\d{2}$/)
      expect(b.sources?.length ?? 0, `${id} vendor sources`).toBeGreaterThan(0)
      for (const s of b.sources ?? []) expect(s).toMatch(/^https:\/\//)
    }
  })

  it('no lesson states a price', () => {
    /* Pricing SHAPE is teachable; absolute prices are stale within two quarters. */
    const priceLike = /(\$|£|€)\s?\d|\d+\s?(USD|EUR|GBP)\b|per\s+(TB|GB)\s+(costs|is)\s+\$/i
    for (const l of ALL_LESSONS) {
      for (const b of l.blocks) {
        const text = 'md' in b && typeof b.md === 'string' ? b.md : ''
        expect(priceLike.test(text), `${l.id} contains a price-like string`).toBe(false)
      }
    }
  })

  it('every lab block references a registered lab', () => {
    const labIds = new Set(LABS.map((l) => l.id))
    for (const { id, b } of blocksOf()) {
      if (b.type === 'lab') expect(labIds, `${id} lab ${b.lab}`).toContain(b.lab)
    }
  })

  it('every lab+quiz lesson actually links a lab', () => {
    for (const l of ALL_LESSONS) {
      if (l.exercise !== 'lab+quiz') continue
      expect(l.blocks.some((b) => b.type === 'lab'), `${l.id} is lab+quiz without a lab block`).toBe(true)
    }
  })

  it('diagram nodes do not overlap and stay on the canvas', () => {
    for (const { id, b } of blocksOf()) {
      if (b.type !== 'diagram') continue
      const h = b.height ?? 60
      const boxes = b.nodes.map((n) => ({
        id: n.id,
        x: n.x,
        y: n.y,
        w: n.w ?? 10,
        hh: n.h ?? 8,
      }))
      for (const box of boxes) {
        expect(box.x, `${id}/${box.id} x`).toBeGreaterThanOrEqual(0)
        expect(box.x + box.w, `${id}/${box.id} right edge`).toBeLessThanOrEqual(100)
        expect(box.y + box.hh, `${id}/${box.id} bottom edge`).toBeLessThanOrEqual(h)
      }
      for (let i = 0; i < boxes.length; i++) {
        for (let j = i + 1; j < boxes.length; j++) {
          const a = boxes[i]
          const c = boxes[j]
          const overlap =
            a.x < c.x + c.w && c.x < a.x + a.w && a.y < c.y + c.hh && c.y < a.y + a.hh
          expect(overlap, `${id}: ${a.id} overlaps ${c.id}`).toBe(false)
        }
      }
      for (const e of b.edges ?? []) {
        const ids = new Set(b.nodes.map((n) => n.id))
        expect(ids, `${id} edge from`).toContain(e.from)
        expect(ids, `${id} edge to`).toContain(e.to)
      }
      expect(b.steps.length, `${id} step count`).toBeGreaterThanOrEqual(3)
      for (const s of b.steps) {
        expect(s.caption.length, `${id} step caption teaches`).toBeGreaterThan(40)
      }
    }
  })
})

describe('labs', () => {
  it('ids are distinct, indexes are 0..n in order, and every lab has a guide', () => {
    LABS.forEach((l, i) => expect(l.index, `${l.id} index`).toBe(i))
    expect(new Set(LABS.map((l) => l.id)).size).toBe(LABS.length)
    for (const l of LABS) expect(LAB_GUIDES[l.id], `${l.id} guide loader`).toBeDefined()
  })

  it('point at real tracks, and prerequisites come earlier', () => {
    for (const l of LABS) {
      expect(TRACK_IDS).toContain(l.trackId)
      for (const a of l.after) {
        const dep = LABS.find((x) => x.id === a)
        expect(dep, `${l.id} after ${a}`).toBeDefined()
        expect(dep!.index).toBeLessThan(l.index)
      }
    }
  })

  it('have 4–12 distinct checks', () => {
    for (const l of LABS) {
      const ids = l.checks.map((c) => c.id)
      expect(ids.length, `${l.id} checks`).toBeGreaterThanOrEqual(4)
      expect(ids.length, `${l.id} checks`).toBeLessThanOrEqual(12)
      expect(new Set(ids).size, `${l.id} duplicate check ids`).toBe(ids.length)
    }
  })

  it('zip paths are base-relative', () => {
    for (const l of LABS) expect(l.zip).toBe(`labs/${l.id}.zip`)
  })

  it('browser-lane labs never need torch; local-lane labs list no Pyodide packages', () => {
    for (const l of LABS) {
      if (l.lane === 'browser') expect(l.packages, l.id).not.toContain('torch')
      else expect(l.packages, l.id).toEqual([])
    }
  })
})

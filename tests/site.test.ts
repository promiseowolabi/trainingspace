/**
 * Site integrity gate — inherited from columnspaces, trimmed to this course.
 *
 * Classes of rot it catches, all statically visible:
 *  1. ORPHANED SURFACES — a route nothing links to.
 *  2. DEAD LINKS — a link to a route, lab or lesson that does not exist.
 *  3. HARDCODED COURSE COUNTS — a number in page copy that a registry owns.
 *  4. BASE-PATH BUGS — a root-absolute public asset that 404s under /trainingspace/.
 *  5. MISSING LAB ARTIFACTS — a lab page whose zip is not in public/.
 */

import { existsSync, readFileSync, readdirSync, statSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import { TRACKS, TOTAL_TRACK_LESSONS, ORDERED_LESSON_IDS } from '@/lib/tracks'
import { LABS } from '@/data/labs'

const ROOT = new URL('..', import.meta.url).pathname
const read = (rel: string) => readFileSync(join(ROOT, rel), 'utf8')

const APP = read('src/App.tsx')
const NAVBAR = read('src/components/Navbar.tsx')
const FOOTER = read('src/components/Footer.tsx')
const PALETTE = read('src/components/CommandPalette.tsx')

const AUDITED_PAGES = [
  'src/pages/Home.tsx',
  'src/pages/Curriculum.tsx',
  'src/pages/Labs.tsx',
  'src/pages/LabPage.tsx',
  'src/pages/Capstone.tsx',
  'src/pages/Progress.tsx',
  'src/components/Navbar.tsx',
  'src/components/Footer.tsx',
  'src/components/CommandPalette.tsx',
] as const

/** Strip comments — an explanatory comment may name a number or an old route. */
function prose(src: string): string {
  return src.replace(/\/\*[\s\S]*?\*\//g, ' ').replace(/(^|\s)\/\/[^\n]*/g, ' ')
}

function routePaths(): string[] {
  return [...APP.matchAll(/<Route\s+path="([^"]+)"/g)].map((m) => m[1])
}

function linkTargets(src: string): Set<string> {
  const out = new Set<string>()
  for (const m of src.matchAll(/\bto[=:]\s*["'](\/[^"'`${}]*)["']/g)) out.add(m[1])
  return out
}

function srcFiles(dir = 'src', out: string[] = []): string[] {
  for (const name of readdirSync(join(ROOT, dir))) {
    const rel = `${dir}/${name}`
    if (statSync(join(ROOT, rel)).isDirectory()) srcFiles(rel, out)
    else if (/\.tsx?$/.test(name)) out.push(rel)
  }
  return out
}

const NAV_TARGETS = new Set<string>([...linkTargets(NAVBAR), ...linkTargets(FOOTER), ...linkTargets(PALETTE)])

const DEEP_LINK_ONLY: Record<string, string> = {
  '/tracks/:trackId': 'parameterised — reached from the curriculum stack and track cards',
  '/lesson/:lessonId': 'parameterised — reached from track pages, LessonRow and the up-next card',
  '/labs/:labId': 'parameterised — reached from the labs index and lesson lab blocks',
  '*': 'the NotFound catch-all is not a destination',
}

describe('routes and navigation', () => {
  it('the route table parses', () => {
    const paths = routePaths()
    expect(paths.length).toBeGreaterThan(6)
    expect(paths).toContain('/')
    expect(paths).toContain('*')
  })

  it('every non-parameterised route is reachable from Navbar, Footer or the palette', () => {
    const unreachable = routePaths().filter((p) => !(p in DEEP_LINK_ONLY) && !NAV_TARGETS.has(p))
    expect(unreachable, `built but unreachable: ${unreachable.join(', ')}`).toEqual([])
  })

  it('no nav surface links to a route that does not exist', () => {
    const paths = routePaths()
    const literal = new Set(paths.filter((p) => !p.includes(':') && p !== '*'))
    const prefixes = paths.filter((p) => p.includes(':')).map((p) => p.slice(0, p.indexOf(':')))
    const dead = [...NAV_TARGETS].filter(
      (t) => !literal.has(t) && !prefixes.some((pre) => t.startsWith(pre) && t.length > pre.length),
    )
    expect(dead, `nav links with no route: ${dead.join(', ')}`).toEqual([])
  })

  it('no source file links at a lab id that does not exist', () => {
    const ids = new Set<string>(LABS.map((l) => l.id))
    const offenders: string[] = []
    for (const rel of srcFiles()) {
      for (const m of prose(read(rel)).matchAll(/["'`]\/labs\/([a-z0-9-]+)["'`]/g)) {
        if (!ids.has(m[1])) offenders.push(`${rel} → /labs/${m[1]}`)
      }
    }
    expect(offenders).toEqual([])
  })

  it('no audited page links at a lesson id that does not exist', () => {
    const ids = new Set(ORDERED_LESSON_IDS)
    const offenders: string[] = []
    for (const rel of AUDITED_PAGES) {
      for (const m of prose(read(rel)).matchAll(/["'`]\/lesson\/([a-z0-9]+\.l\d+)["'`]/g)) {
        if (!ids.has(m[1])) offenders.push(`${rel} → /lesson/${m[1]}`)
      }
    }
    expect(offenders).toEqual([])
  })

  it('the footer does not link to the (private) repository', () => {
    expect(FOOTER).not.toMatch(/github\.com\/promiseowolabi/)
  })

  it('no page component is orphaned', () => {
    const orphans = readdirSync(join(ROOT, 'src/pages'))
      .filter((f) => f.endsWith('.tsx'))
      .map((f) => f.replace(/\.tsx$/, ''))
      .filter((name) => !new RegExp(`@/pages/${name}['"]`).test(APP))
      .filter((name) => name !== 'NotFound' || !APP.includes("@/pages/NotFound"))
    expect(orphans, `page components nothing routes to: ${orphans.join(', ')}`).toEqual([])
  })
})

describe('no hardcoded course counts', () => {
  const COUNTS = [
    { n: TOTAL_TRACK_LESSONS, nouns: ['lessons'] },
    { n: TRACKS.length, nouns: ['tracks'] },
    { n: LABS.length, nouns: ['labs', 'hands-on labs'] },
  ]
  it('no audited page states a registry count as a literal', () => {
    const offenders: string[] = []
    for (const rel of AUDITED_PAGES) {
      const text = prose(read(rel))
      for (const { n, nouns } of COUNTS) {
        for (const noun of nouns) {
          if (new RegExp(`\\b${n}\\s+${noun}\\b`, 'i').test(text)) offenders.push(`${rel}: "${n} ${noun}"`)
        }
      }
    }
    expect(offenders).toEqual([])
  })
})

describe('lab artifacts', () => {
  it('every lab folder has its module, checks, pytest wrapper, verify.py and README (and a local solution)', () => {
    for (const l of LABS) {
      for (const f of [l.module, 'checks.py', 'test_lab.py', 'verify.py', 'README.md']) {
        expect(existsSync(join(ROOT, 'labs', l.id, f)), `${l.id}/${f}`).toBe(true)
      }
      /* Reference solutions are local-only (gitignored); check them when present. */
      if (existsSync(join(ROOT, 'labs/_solutions'))) {
        expect(existsSync(join(ROOT, 'labs/_solutions', l.id, l.module)), `_solutions/${l.id}/${l.module}`).toBe(true)
      }
    }
  })

  it('every lab zip exists in public/ and is non-trivial', () => {
    for (const l of LABS) {
      const p = join(ROOT, 'public', l.zip)
      expect(existsSync(p), `${l.zip} missing — run npm run pack-labs`).toBe(true)
      expect(statSync(p).size, `${l.zip} size`).toBeGreaterThan(400)
    }
  })
})

describe('public asset paths and the deploy base', () => {
  it('no source file hardcodes a root-absolute public asset path', () => {
    const re = /["'`]\/[A-Za-z0-9_\-/${}.]*\.(zip|md|svg|png|txt)["'`]/
    const found = srcFiles().filter((rel) => re.test(prose(read(rel))))
    expect(found, 'root-absolute asset paths 404 under VITE_BASE=/trainingspace/; use asset()').toEqual([])
  })

  it('the router basename comes from BASE_URL', () => {
    expect(read('src/main.tsx')).toContain('basename={import.meta.env.BASE_URL}')
  })
})

describe('page hygiene', () => {
  it('no audited page states a currency amount', () => {
    const priceLike = /(\$|£|€)\s?\d|\b\d+(\.\d+)?\s?(USD|EUR|GBP)\b/
    const offenders = AUDITED_PAGES.filter((rel) => priceLike.test(read(rel)))
    expect(offenders).toEqual([])
  })

  it('every <button> declares a type', () => {
    const offenders: string[] = []
    for (const rel of AUDITED_PAGES) {
      for (const m of read(rel).matchAll(/<button\b[^>]*?>/gs)) {
        if (!/\btype=/.test(m[0])) offenders.push(`${rel}: ${m[0].replace(/\s+/g, ' ').slice(0, 70)}`)
      }
    }
    expect(offenders).toEqual([])
  })
})

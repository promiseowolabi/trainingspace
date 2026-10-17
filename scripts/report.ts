/**
 * report — the honest build state.
 *
 * `npm run report` is authoritative for "what is actually built". The gap
 * between the plan and the repo is printed rather than remembered.
 *
 *   npx tsx scripts/report.ts
 */

import { existsSync } from 'node:fs'
import { join } from 'node:path'
import { ALL_LESSONS, LESSONS_BY_TRACK } from '../src/data/lessons'
import { TRACKS } from '../src/lib/tracks'
import { LABS, labNumber } from '../src/data/labs'
import type { TrackId } from '../src/data/lessons/types'

const ROOT = join(import.meta.dirname, '..')

const bar = (done: number, total: number, width = 18): string => {
  const filled = total === 0 ? 0 : Math.round((done / total) * width)
  return `${'█'.repeat(filled)}${'·'.repeat(width - filled)}`
}

console.log('\ntrainingspace — build state\n' + '='.repeat(56) + '\n')
console.log('LESSONS')

let authored = 0
let planned = 0
for (const t of TRACKS) {
  const have = LESSONS_BY_TRACK[t.id as TrackId].length
  authored += have
  planned += t.lessons
  console.log(`  ${t.code.padEnd(3)} ${t.name.padEnd(28)} ${bar(have, t.lessons)} ${have}/${t.lessons}`)
}
console.log(`  ${'all'.padEnd(32)} ${bar(authored, planned)} ${authored}/${planned}\n`)

console.log('LABS')
const outstanding: string[] = []
for (const l of LABS) {
  const dir = join(ROOT, 'labs', l.id)
  const verify = existsSync(join(dir, 'checks.py')) && existsSync(join(ROOT, 'labs/_solutions', l.id))
  const guide = existsSync(join(ROOT, 'src/data/lab-guides', `${l.id}.ts`))
  const zip = existsSync(join(ROOT, 'public', l.zip))
  const mark = (b: boolean) => (b ? '✓' : '·')
  console.log(
    `  ${labNumber(l)} ${l.id.padEnd(28)} checks+solution ${mark(verify)}  guide ${mark(guide)}  zip ${mark(zip)}`,
  )
  if (!verify || !guide || !zip) outstanding.push(`lab ${labNumber(l)} ${l.id}`)
}

if (authored < planned) outstanding.unshift(`${planned - authored} lessons unwritten`)
console.log(`\n${ALL_LESSONS.length} lessons authored.`)
console.log(outstanding.length ? `OUTSTANDING:\n  - ${outstanding.join('\n  - ')}\n` : 'nothing outstanding.\n')

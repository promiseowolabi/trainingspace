/**
 * Branding and anonymisation gate.
 *
 * `naigap` is not our domain. The kernelspace lineage carried it everywhere,
 * and "we removed them" is only true until someone copies a file forward. So
 * it is a test, and it fails the build.
 *
 * The anonymisation list is the other half: the author's VAST cluster lives on
 * an internal, VPN-only lab network, and nothing that identifies it may ship —
 * not a hostname, not the cluster or tenant name, not the shared demo objects.
 * Lessons and labs use placeholders (<VMS_URL>, <TENANT>, …) instead.
 * PLAN.md §"The anonymisation rule" is the policy; this is the enforcement.
 */

import { execFileSync } from 'node:child_process'
import { createHash } from 'node:crypto'
import { lstatSync, readFileSync, readdirSync } from 'node:fs'
import { extname, join } from 'node:path'
import { describe, expect, it } from 'vitest'

const ROOT = new URL('..', import.meta.url).pathname

const SKIP_DIRS = new Set(['node_modules', '.git', 'dist', 'target', '_solutions', '.venv', '.func'])
const TEXT_EXT = new Set([
  '.ts',
  '.tsx',
  '.js',
  '.jsx',
  '.json',
  '.md',
  '.html',
  '.css',
  '.yml',
  '.yaml',
  '.py',
  '.rs',
  '.txt',
  '.toml',
  '.svg',
  '.sh',
  '.env',
  '.example',
])

function walk(dir: string, out: string[] = []): string[] {
  for (const name of readdirSync(dir)) {
    if (SKIP_DIRS.has(name)) continue
    const full = join(dir, name)
    // lstat: a func host build leaves .func/build/f -> ../.., a loop
    if (lstatSync(full).isDirectory()) walk(full, out)
    else if (TEXT_EXT.has(extname(name))) out.push(full)
  }
  return out
}

const FORBIDDEN = new Set([
  'c963d4a7126fcf4fa06920886e67b6c31976d36a53800f4527f7c1d8d857c0fe',
  '07a0a741d65d85fb6098e1ade83602f86944eb43d742e31d7b87cfc059f9fee6',
  'dff7634f12f7de10e10a787843aa1aae21e9790c1234a5c78c690cc91632118e',
  'e5cd9f39c603e01d4b322fdd96c2fc0e385bf4eaebee85e5682879c6ae4f3a57',
  'b050f2ef23a1468ca6fa1689d5742f62253e863f05942834002032bc088378e3',
  'e07d929dac22ff5576bffd4f26d969c418423e2c7ddf82169d35cbcdb7af5412',
  'd32eecdd9cb031f77c4a42fd1fddde04ee0790bf6e34a0a007ae16dffbbf68c1',
  'f5bc72e7cc8d875d00259a62c8e6fe98475026dc9f318b6a583108e56ce63f2b',
  '3105f25d4c25c91efa21c388d904981fd481ac7338b719832eb00ca80b9276a8',
  'adb16799563a00e136cb0cbcf54256552da2d516fc97aae54953f4782a0a6376',
])
const sha = (t: string) => createHash('sha256').update(t).digest('hex')
const candidates = (text: string): Set<string> => {
  const out = new Set<string>()
  for (const tok of text.toLowerCase().split(/[^a-z0-9.-]+/)) {
    if (!tok) continue
    out.add(tok)
    const parts = tok.split(/[.-]/).filter(Boolean)
    for (let i = 0; i < parts.length; i++) {
      out.add(parts[i])
      if (i + 1 < parts.length) {
        out.add(`${parts[i]}-${parts[i + 1]}`)
        out.add(`${parts[i]}.${parts[i + 1]}`)
      }
      if (i + 2 < parts.length) out.add(`${parts[i]}-${parts[i + 1]}-${parts[i + 2]}`)
    }
  }
  return out
}

/* Files git ignores (a reader's filled-in de.env, a local func project) are
   never committed or packed into a lab zip, so they cannot leak. */
function notIgnored(files: string[]): string[] {
  let out = ''
  try {
    out = execFileSync('git', ['-C', ROOT, 'check-ignore', '--stdin', '-z'], {
      input: files.join('\0'),
      encoding: 'utf8',
    })
  } catch (e) {
    const err = e as { status?: number; stdout?: string }
    if (err.status !== 1) throw e // 1 = nothing ignored
    out = err.stdout ?? ''
  }
  const ignored = new Set(out.split('\0').filter(Boolean))
  return files.filter((f) => !ignored.has(f))
}

const FILES = notIgnored(walk(ROOT))
const rel = (f: string) => f.slice(ROOT.length)

describe('de-branding', () => {
  it('finds files to check (the walker is not silently empty)', () => {
    expect(FILES.length).toBeGreaterThan(50)
  })

  it('no file references naigap', () => {
    const offenders: string[] = []
    for (const f of FILES) {
      /* PLAN.md documents the removal itself, so it is allowed to name it. */
      if (rel(f) === 'PLAN.md') continue
      if (rel(f).startsWith('tests/')) continue
      if (readFileSync(f, 'utf8').includes('naigap')) offenders.push(rel(f))
    }
    expect(offenders, `naigap survives in: ${offenders.join(', ')}`).toEqual([])
  })

  it('no file identifies the internal lab environment', () => {
    /*
     * The forbidden names are stored only as SHA-256 digests, so this public
     * file does not itself disclose them. Every file is split into host-like
     * tokens ("a.b-c.d"), their dot/hyphen parts, and adjacent part pairs;
     * each candidate is hashed and compared. To add a name, append
     * sha256(lowercase name) — never the name.
     */
    const offenders: string[] = []
    for (const f of FILES) {
      if (rel(f) === 'package-lock.json') continue
      for (const c of candidates(readFileSync(f, 'utf8'))) {
        if (FORBIDDEN.has(sha(c))) {
          offenders.push(rel(f))
          break
        }
      }
    }
    expect(offenders, `internal environment named in: ${offenders.join(', ')}`).toEqual([])
  })

  it('the forbidden-name check actually fires', () => {
    /* Guard against the tokenizer silently matching nothing. The sample is
       built at runtime so no forbidden name appears in this file. */
    const sample = `host-${'otco'.split('').reverse().join('')}-01.example`
    expect([...candidates(sample)].some((c) => FORBIDDEN.has(sha(c)))).toBe(true)
  })

  it('no file claims to BE a sibling course', () => {
    const identity = ['columnspaces:v1', 'columnspaces-progress', '_columnspaces', 'name": "columnspaces', 'tablespace:v1', 'kernelspace:v1', ']_tablespace', ']_kernelspace', ']_columnspaces', ']_vectorspace', ']_latentspace', 'functionspace:v1', ']_functionspace', 'functionspace-labs']
    const offenders: string[] = []
    for (const f of FILES) {
      if (rel(f) === 'PLAN.md' || rel(f).startsWith('tests/')) continue
      const text = readFileSync(f, 'utf8')
      for (const needle of identity) {
        if (text.includes(needle)) offenders.push(`${rel(f)} (${needle})`)
      }
    }
    expect(offenders, `sibling identity survives in: ${offenders.join(', ')}`).toEqual([])
  })

  it('the progress store uses the trainingspace namespace', () => {
    const progress = readFileSync(join(ROOT, 'src/lib/progress.ts'), 'utf8')
    expect(progress).toContain("name: 'trainingspace:v1'")
  })

  it('package name and html title are trainingspace', () => {
    const pkg = JSON.parse(readFileSync(join(ROOT, 'package.json'), 'utf8'))
    expect(pkg.name).toBe('trainingspace')
    expect(readFileSync(join(ROOT, 'index.html'), 'utf8')).toContain('<title>trainingspace')
  })

  it('ships no binary artwork inherited from a sibling course', () => {
    /* Text checks cannot read a PNG; the inherited og-image said "kernelspace". */
    const pngs = readdirSync(join(ROOT, 'public')).filter((f) => f.endsWith('.png'))
    expect(pngs, 'review any PNG in public/ by eye before allowing it here').toEqual([])
  })

  it('no CNAME is shipped (GitHub Pages project site, not a custom domain)', () => {
    const publicFiles = readdirSync(join(ROOT, 'public'))
    expect(publicFiles).not.toContain('CNAME')
  })

  it('exactly one public URL constant exists, and it points at the personal account', () => {
    const exporter = readFileSync(join(ROOT, 'scripts/export-lessons-md.ts'), 'utf8')
    expect(exporter).toContain('https://promiseowolabi.github.io/trainingspace')
    const defs = exporter.match(/const SITE =/g) ?? []
    expect(defs).toHaveLength(1)
  })
})

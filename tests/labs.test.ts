/**
 * Lab contract tests.
 *
 * Two places must agree that no reviewer would notice drifting apart:
 *
 *  1. The completion code. `completion_code` in labs/common/tscheck.py prints
 *     it; lab-code.ts recomputes it in the browser. If they disagree, every
 *     reader's paste fails.
 *  2. The check ids. The site lists what checks.py checks (src/data/labs.ts);
 *     checks.py is what actually checks — under pytest, verify.py and Pyodide.
 *     A renamed check makes the page lie.
 *
 * Whether the checks are any GOOD — every template fails, every solution
 * passes — is scripts/check-labs.py's job; it needs numpy and torch, which
 * this suite does not.
 */

import { execFileSync } from 'node:child_process'
import { existsSync, readFileSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import { LABS } from '@/data/labs'
import { checkLabCode, isValidPrefix, labCode, normaliseCode } from '@/lib/lab-code'

const ROOT = new URL('..', import.meta.url).pathname
const COMMON = join(ROOT, 'labs/common')

function pythonCode(lab: string, prefix: string): string {
  return execFileSync(
    'python3',
    ['-c', 'import sys; sys.path.insert(0, sys.argv[1]); import tscheck; print(tscheck.completion_code(sys.argv[2], sys.argv[3]))', COMMON, lab, prefix],
    { encoding: 'utf8' },
  ).trim()
}

describe('completion codes', () => {
  it('Python and TypeScript produce the same code', async () => {
    for (const [lab, prefix] of [
      ['micrograd', 'alice'],
      ['tensor-autograd', 'bob-2'],
      ['numpy-shapes', 'zz9'],
    ]) {
      expect(await labCode(lab, prefix), `${lab}/${prefix}`).toBe(pythonCode(lab, prefix))
    }
  })

  it('pins a known vector (so both cannot drift together)', async () => {
    expect(await labCode('micrograd', 'alice')).toBe('TS-MICROGRAD-2df665e2')
  })

  it('accepts sloppy pastes and rejects wrong prefixes', async () => {
    expect(normaliseCode('  ts-micrograd-2DF665E2 ')).toBe('TS-MICROGRAD-2df665e2')
    expect(await checkLabCode('micrograd', 'alice', 'ts-micrograd-2df665e2')).toBe(true)
    expect(await checkLabCode('micrograd', 'bob', 'TS-MICROGRAD-2df665e2')).toBe(false)
  })

  it('validates prefixes the same way tscheck.py does', () => {
    const src = readFileSync(join(COMMON, 'tscheck.py'), 'utf8')
    expect(src).toContain('PREFIX_RE = re.compile(r"^[a-z][a-z0-9-]{1,18}[a-z0-9]$")')
    for (const ok of ['alice', 'bob-2', 'zz9']) expect(isValidPrefix(ok), ok).toBe(true)
    for (const bad of ['a', 'Alice', '9lives', 'trailing-', 'has_underscore', 'x'.repeat(21)]) {
      expect(isValidPrefix(bad), bad).toBe(false)
    }
  })
})

describe('checks.py ⇄ the lab registry', () => {
  for (const lab of LABS) {
    const dir = join(ROOT, 'labs', lab.id)
    it(`${lab.id}: CHECKS lists exactly the ids the site shows, in order`, () => {
      const src = readFileSync(join(dir, 'checks.py'), 'utf8')
      const list = /^CHECKS = \[([\s\S]*?)\]/m.exec(src)
      expect(list, 'checks.py must end with CHECKS = [...]').not.toBeNull()
      const ids = [...list![1].matchAll(/check_([a-z0-9_]+)/g)].map((m) => m[1].replace(/_/g, '-'))
      expect(ids).toEqual(lab.checks.map((c) => c.id))
      for (const id of ids) expect(src).toContain(`def check_${id.replace(/-/g, '_')}(m)`)
    })

    it(`${lab.id}: verify.py and test_lab.py point at this lab and its module`, () => {
      const verify = readFileSync(join(dir, 'verify.py'), 'utf8')
      const mod = lab.module.replace(/\.py$/, '')
      expect(verify).toContain(`LAB = "${lab.id}"`)
      expect(verify).toContain(`MODULE = "${mod}"`)
      expect(readFileSync(join(dir, 'test_lab.py'), 'utf8')).toContain(`import ${mod} as module`)
    })

    it(`${lab.id}: the template marks its work with TODO(you)`, () => {
      expect(existsSync(join(dir, lab.module))).toBe(true)
      expect(readFileSync(join(dir, lab.module), 'utf8')).toContain('TODO(you)')
    })
  }
})

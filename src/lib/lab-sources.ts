/**
 * The lab files the browser lane needs, read straight from labs/ at build
 * time — the same bytes that ship in the zips, so the page and `pytest` can
 * never run different checks.
 *
 * Only `labs/<id>/*.py` and `labs/common/tscheck.py` are reachable: the
 * reference solutions live one level deeper (labs/_solutions/<id>/) and no
 * glob here can match them. tests/bundle.test.ts checks the built site too.
 */

const FILES = import.meta.glob('../../labs/*/*.py', {
  query: '?raw',
  import: 'default',
}) as Record<string, () => Promise<string>>

const path = (lab: string, file: string) => `../../labs/${lab}/${file}`

async function read(lab: string, file: string): Promise<string> {
  const load = FILES[path(lab, file)]
  if (!load) throw new Error(`labs/${lab}/${file} is not in the build`)
  return load()
}

/** The template the reader starts from. */
export const templateFor = (lab: string, module: string) => read(lab, module)

/** Everything a run needs except the reader's own module. */
export async function harnessFor(lab: string): Promise<Record<string, string>> {
  const [tscheck, checks] = await Promise.all([read('common', 'tscheck.py'), read(lab, 'checks.py')])
  return { 'tscheck.py': tscheck, 'checks.py': checks }
}

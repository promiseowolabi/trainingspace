/// <reference lib="webworker" />
/**
 * The browser lane's Python: Pyodide (CPython compiled to WebAssembly) in a
 * module worker, so a learner's infinite loop freezes this thread, never the
 * page — src/lib/pyodide-runner.ts terminates the worker on a timeout.
 *
 * One message in: the lab's files (tscheck.py, checks.py and the learner's
 * module), the module name and the packages the checks import. One message
 * out: tscheck.run's results, exactly what pytest and verify.py would report.
 * Nothing is sent anywhere; the only network traffic is the Pyodide runtime
 * itself, fetched once from the CDN and then served from the browser cache.
 */

const PYODIDE_VERSION = '0.27.7' // keep in step with src/lib/pyodide-runner.ts
const INDEX_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`

interface Pyodide {
  loadPackage(names: string[]): Promise<unknown>
  loadPackagesFromImports(code: string): Promise<unknown>
  runPythonAsync(code: string): Promise<unknown>
  FS: { mkdirTree(path: string): void; writeFile(path: string, data: string): void }
}

export interface RunRequest {
  files: Record<string, string>
  module: string
  packages: string[]
}

export interface CheckResult {
  id: string
  ok: boolean
  msg: string
  ms: number
}

export type WorkerMessage =
  | { type: 'status'; text: string }
  | { type: 'done'; results: CheckResult[] }
  | { type: 'import-error'; error: string }
  | { type: 'error'; error: string }

const post = (m: WorkerMessage) => (self as unknown as DedicatedWorkerGlobalScope).postMessage(m)

let booting: Promise<Pyodide> | null = null

function boot(): Promise<Pyodide> {
  booting ??= (async () => {
    const mod = (await import(/* @vite-ignore */ `${INDEX_URL}pyodide.mjs`)) as {
      loadPyodide(opts: { indexURL: string }): Promise<Pyodide>
    }
    const py = await mod.loadPyodide({ indexURL: INDEX_URL })
    py.FS.mkdirTree('/lab')
    return py
  })()
  return booting
}

const RUN = (module: string) => `
import importlib, json, sys
if '/lab' not in sys.path:
    sys.path.insert(0, '/lab')
for _name in ('tscheck', 'checks', ${JSON.stringify(module)}):
    sys.modules.pop(_name, None)
importlib.invalidate_caches()
import tscheck, checks
try:
    _m = importlib.import_module(${JSON.stringify(module)})
    _out = {'type': 'done', 'results': tscheck.run(checks.CHECKS, _m)}
except BaseException as _e:
    _out = {'type': 'import-error', 'error': f'{type(_e).__name__}: {_e}'}
json.dumps(_out)
`

self.onmessage = async (e: MessageEvent<RunRequest>) => {
  const { files, module, packages } = e.data
  try {
    post({ type: 'status', text: 'starting Python (first run downloads the runtime)…' })
    const py = await boot()
    post({ type: 'status', text: packages.length ? `loading ${packages.join(', ')}…` : 'preparing…' })
    if (packages.length) await py.loadPackage(packages)
    await py.loadPackagesFromImports(Object.values(files).join('\n'))
    for (const [name, src] of Object.entries(files)) py.FS.writeFile(`/lab/${name}`, src)
    post({ type: 'status', text: 'running checks…' })
    const out = (await py.runPythonAsync(RUN(module.replace(/\.py$/, '')))) as string
    post(JSON.parse(out) as WorkerMessage)
  } catch (err) {
    post({ type: 'error', error: err instanceof Error ? err.message : String(err) })
  }
}

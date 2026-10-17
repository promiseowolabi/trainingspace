/**
 * Main-thread side of the browser lane: owns one Pyodide worker, runs a lab's
 * checks in it, and kills it when a run takes too long (a learner's `while
 * True:` is a normal first draft). The next run boots a fresh worker; the
 * runtime download is served from the browser cache, so that costs seconds.
 */

import type { CheckResult, RunRequest, WorkerMessage } from '@/workers/pyodide-worker'

export type { CheckResult }

/** Shown to the reader; keep in step with src/workers/pyodide-worker.ts. */
export const PYODIDE_VERSION = '0.27.7'

export type RunOutcome =
  | { kind: 'results'; results: CheckResult[] }
  | { kind: 'import-error'; error: string }
  | { kind: 'timeout'; seconds: number }
  | { kind: 'error'; error: string }

let worker: Worker | null = null

function getWorker(): Worker {
  worker ??= new Worker(new URL('../workers/pyodide-worker.ts', import.meta.url), { type: 'module' })
  return worker
}

function kill() {
  worker?.terminate()
  worker = null
}

export function runChecks(
  req: RunRequest,
  onStatus: (text: string) => void,
  timeoutMs = 120_000,
): Promise<RunOutcome> {
  return new Promise((resolve) => {
    const w = getWorker()
    let timer: ReturnType<typeof setTimeout> | null = null
    /* The clock starts when checks start, not while the runtime downloads. */
    const arm = () => {
      if (timer) return
      timer = setTimeout(() => {
        kill()
        resolve({ kind: 'timeout', seconds: Math.round(timeoutMs / 1000) })
      }, timeoutMs)
    }
    const finish = (o: RunOutcome) => {
      if (timer) clearTimeout(timer)
      w.onmessage = null
      w.onerror = null
      resolve(o)
    }
    w.onmessage = (e: MessageEvent<WorkerMessage>) => {
      const m = e.data
      if (m.type === 'status') {
        onStatus(m.text)
        if (m.text.startsWith('running')) arm()
      } else if (m.type === 'done') finish({ kind: 'results', results: m.results })
      else if (m.type === 'import-error') finish({ kind: 'import-error', error: m.error })
      else finish({ kind: 'error', error: m.error })
    }
    w.onerror = (e) => {
      kill()
      finish({ kind: 'error', error: e.message || 'the Python worker failed to start' })
    }
    w.postMessage(req)
  })
}

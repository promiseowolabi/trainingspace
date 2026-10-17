import { useEffect, useRef, useState } from 'react'
import { Check, Circle, FileUp, Loader2, Play, RotateCcw, X } from 'lucide-react'
import type { PythonLab } from '@/data/labs'
import { harnessFor, templateFor } from '@/lib/lab-sources'
import { PYODIDE_VERSION, runChecks, type CheckResult, type RunOutcome } from '@/lib/pyodide-runner'
import { useProgress } from '@/lib/progress'
import { cn } from '@/lib/utils'

/**
 * The browser lane: the reader's module, the lab's checks.py and the shared
 * tscheck.py run in Pyodide on this page. All green completes the lab.
 *
 * The draft is kept in localStorage per lab — a convenience, not storage of
 * record: the reader's own file on disk (or git) is the backup, as the Labs
 * page says.
 */

const draftKey = (lab: string) => `trainingspace:draft:${lab}`

function loadDraft(lab: string): string | null {
  try {
    return localStorage.getItem(draftKey(lab))
  } catch {
    return null
  }
}

function saveDraft(lab: string, src: string) {
  try {
    localStorage.setItem(draftKey(lab), src)
  } catch {
    /* private mode or full storage: the editor still works, the draft just is not kept */
  }
}

export default function BrowserGrader({ lab }: { lab: PythonLab }) {
  const done = useProgress((s) => s.labs[lab.id]?.done ?? false)
  const completeLab = useProgress((s) => s.completeLab)
  /* The draft is read synchronously at mount (the component is keyed by lab id
     in LabPage); only the template fallback arrives asynchronously. */
  const [source, setSource] = useState<string | null>(() => loadDraft(lab.id))
  const [status, setStatus] = useState<string | null>(null)
  const [outcome, setOutcome] = useState<RunOutcome | null>(null)
  const [dragging, setDragging] = useState(false)
  const fileInput = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (source !== null) return
    let live = true
    templateFor(lab.id, lab.module).then((t) => {
      if (live) setSource(t)
    })
    return () => {
      live = false
    }
  }, [lab.id, lab.module, source])

  const edit = (src: string) => {
    setSource(src)
    saveDraft(lab.id, src)
  }

  const reset = async () => {
    if (!window.confirm(`Replace the editor with the original ${lab.module}? Your draft here is discarded.`)) return
    edit(await templateFor(lab.id, lab.module))
    setOutcome(null)
  }

  const readFile = async (f: File | undefined) => {
    if (!f) return
    edit(await f.text())
    setOutcome(null)
  }

  const run = async () => {
    if (source === null || status) return
    setOutcome(null)
    setStatus('loading the checks…')
    const files = { ...(await harnessFor(lab.id)), [lab.module]: source }
    const o = await runChecks({ files, module: lab.module, packages: lab.packages }, setStatus)
    setStatus(null)
    setOutcome(o)
    if (o.kind === 'results' && o.results.length === lab.checks.length && o.results.every((r) => r.ok)) {
      completeLab(lab.id, `browser · ${o.results.length}/${o.results.length} checks`)
    }
  }

  const byId = new Map<string, CheckResult>(
    outcome?.kind === 'results' ? outcome.results.map((r) => [r.id, r]) : [],
  )
  const passed = outcome?.kind === 'results' ? outcome.results.filter((r) => r.ok).length : 0

  return (
    <div className="rounded-lg border border-line bg-surface-1 p-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <p className="font-mono text-label uppercase text-text-3">
          run the checks here · {lab.module}
        </p>
        <div className="flex flex-wrap gap-2">
          <button
            type="button"
            onClick={() => fileInput.current?.click()}
            className="flex items-center gap-1.5 rounded-md border border-line px-3 py-1.5 font-mono text-[11px] text-text-2 hover:border-line-bright hover:text-text-1"
          >
            <FileUp size={12} /> open {lab.module}…
          </button>
          <button
            type="button"
            onClick={reset}
            className="flex items-center gap-1.5 rounded-md border border-line px-3 py-1.5 font-mono text-[11px] text-text-2 hover:border-line-bright hover:text-text-1"
          >
            <RotateCcw size={12} /> template
          </button>
          <input
            ref={fileInput}
            type="file"
            accept=".py,text/x-python"
            className="hidden"
            onChange={(e) => readFile(e.target.files?.[0])}
          />
        </div>
      </div>
      <p className="mt-2 text-body-sm text-text-2">
        Edit here, or drop the <code>{lab.module}</code> you wrote locally onto the editor. The page runs the
        lab&apos;s own <code>checks.py</code> in CPython compiled to WebAssembly (Pyodide {PYODIDE_VERSION}),
        inside your browser — the first run downloads the runtime (about 10 MB) once. Your code never
        leaves this tab.
      </p>

      <textarea
        value={source ?? ''}
        onChange={(e) => edit(e.target.value)}
        onKeyDown={(e) => {
          if (e.key !== 'Tab' || e.shiftKey) return
          e.preventDefault()
          const t = e.currentTarget
          const { selectionStart: a, selectionEnd: b } = t
          const next = `${t.value.slice(0, a)}    ${t.value.slice(b)}`
          edit(next)
          requestAnimationFrame(() => t.setSelectionRange(a + 4, a + 4))
        }}
        onDragOver={(e) => {
          e.preventDefault()
          setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault()
          setDragging(false)
          readFile(e.dataTransfer.files?.[0])
        }}
        spellCheck={false}
        aria-label={`${lab.module} editor`}
        placeholder="loading template…"
        className={cn(
          'mt-4 h-[28rem] w-full resize-y rounded-md border bg-surface-2 p-4 font-mono text-[12.5px] leading-relaxed text-text-1 outline-none',
          dragging ? 'border-accent' : 'border-line focus:border-line-bright',
        )}
      />

      <div className="mt-4 flex flex-wrap items-center gap-3">
        <button
          type="button"
          onClick={run}
          disabled={source === null || status !== null}
          className="flex items-center gap-2 rounded-md bg-accent px-4 py-2 font-display text-body-sm font-semibold text-accent-foreground disabled:opacity-50"
        >
          {status ? <Loader2 size={14} className="animate-spin" /> : <Play size={14} />}
          run {lab.checks.length} checks
        </button>
        {status && <p className="font-mono text-[11px] text-text-3">{status}</p>}
        {outcome?.kind === 'results' && (
          <p className={cn('font-mono text-[11px]', passed === lab.checks.length ? 'text-accent' : 'text-text-2')}>
            {passed}/{lab.checks.length} checks passed
            {passed === lab.checks.length && (done ? ' · lab complete' : '')}
          </p>
        )}
      </div>

      {outcome && outcome.kind !== 'results' && (
        <p className="mt-4 rounded-md border border-amber/40 bg-amber/5 p-3 font-mono text-[12px] text-amber">
          {outcome.kind === 'import-error' && <>Your {lab.module} did not import — {outcome.error}</>}
          {outcome.kind === 'timeout' && (
            <>The checks ran for {outcome.seconds} s and were stopped. Look for a loop that never ends.</>
          )}
          {outcome.kind === 'error' && <>Python could not run: {outcome.error}. Are you offline? The local lane (pytest) works without a network.</>}
        </p>
      )}

      <ol className="mt-5 space-y-2">
        {lab.checks.map((c) => {
          const r = byId.get(c.id)
          return (
            <li key={c.id} className="flex gap-3 text-body-sm">
              <span className="mt-0.5 shrink-0">
                {!r ? (
                  <Circle size={14} className="text-text-3" />
                ) : r.ok ? (
                  <Check size={14} className="text-accent" />
                ) : (
                  <X size={14} className="text-danger" />
                )}
              </span>
              <span className="min-w-0">
                <span className={cn(r?.ok ? 'text-text-1' : 'text-text-2')}>{c.label}</span>{' '}
                <span className="font-mono text-[10px] text-text-3">[{c.id}]</span>
                {r && !r.ok && (
                  <span className="mt-1 block break-words font-mono text-[11px] text-text-3">{r.msg}</span>
                )}
              </span>
            </li>
          )
        })}
      </ol>
    </div>
  )
}

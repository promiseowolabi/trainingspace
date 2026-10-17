import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router'
import { motion } from 'framer-motion'
import { ArrowLeft, Check, Download, ListChecks } from 'lucide-react'
import { labById, labNumber, LABS, LANE_LABEL, type PythonLab } from '@/data/labs'
import { LAB_GUIDES, type LabGuide } from '@/data/lab-guides'
import { lessonMeta, lessonPath } from '@/data/lessons/manifest'
import { getTrack } from '@/lib/tracks'
import { useProgress } from '@/lib/progress'
import { checkLabCode, normaliseCode } from '@/lib/lab-code'
import { asset } from '@/lib/asset'
import PrefixSetting from '@/components/PrefixSetting'
import BrowserGrader from '@/components/BrowserGrader'
import { RenderBlock } from '@/pages/lesson/blocks'
import { h2Offsets } from '@/pages/lesson/markdown'
import type { ContentBlock } from '@/data/lessons/types'
import NotFound from '@/pages/NotFound'

function Blocks({ blocks, id, color }: { blocks: ContentBlock[]; id: string; color: string }) {
  const offsets = h2Offsets(blocks)
  return (
    <>
      {blocks.map((b, i) => (
        <RenderBlock key={i} block={b} lesson={{ id }} trackColor={color} h2Start={offsets[i]} />
      ))}
    </>
  )
}

function GetTheFiles({ lab }: { lab: PythonLab }) {
  const deps = lab.lane === 'local' ? 'numpy pytest torch' : 'numpy pytest'
  return (
    <div className="mt-8 rounded-lg border border-line bg-surface-1 p-5">
      <p className="font-mono text-label uppercase text-text-3">get the lab files</p>
      <p className="mt-2 text-body-sm text-text-2">
        Download the zip above and work from the lab&apos;s own folder. You edit one file,{' '}
        <code>{lab.module}</code>; <code>pytest</code> runs the checks, and <code>verify.py</code> prints your
        completion code when they all pass. Every lab shares <code>common/</code>, so unzipping several into the
        same place is fine. Python 3.10 or newer.
        {lab.lane === 'browser' && (
          <> No Python at all? Skip this box — the editor at the bottom of the page runs the same checks.</>
        )}
      </p>
      <pre className="mt-3 overflow-x-auto rounded-md border border-line bg-surface-2 px-4 py-3 font-mono text-[13px] text-text-1">
        {`unzip ${lab.id}.zip
cd trainingspace-labs
python3 -m venv .venv && . .venv/bin/activate   # once, shared by every lab
pip install ${deps}
cd ${lab.id}
pytest                        # red → green
PREFIX=<you> python verify.py # prints the completion code`}
      </pre>
    </div>
  )
}

function Completion({ labId, lane }: { labId: string; lane: PythonLab['lane'] }) {
  const prefix = useProgress((s) => s.settings.prefix)
  const done = useProgress((s) => s.labs[labId]?.done ?? false)
  const saved = useProgress((s) => s.labs[labId]?.code)
  const completeLab = useProgress((s) => s.completeLab)
  const [code, setCode] = useState('')
  const [state, setState] = useState<'idle' | 'wrong'>('idle')

  if (done) {
    return (
      <div className="rounded-lg border border-accent/50 bg-surface-1 p-5">
        <p className="flex items-center gap-2 font-mono text-label uppercase text-accent">
          <Check size={14} /> lab complete
        </p>
        {saved && <p className="mt-2 font-mono text-body-sm text-text-2">{saved}</p>}
      </div>
    )
  }

  return (
    <div className="rounded-lg border border-line bg-surface-1 p-5">
      <p className="font-mono text-label uppercase text-text-3">
        {lane === 'browser' ? 'or paste the code verify.py printed' : 'completion code'}
      </p>
      {!prefix ? (
        <div className="mt-3">
          <p className="mb-3 text-body-sm text-text-2">Set your prefix first — the code is derived from it.</p>
          <PrefixSetting />
        </div>
      ) : (
        <form
          className="mt-3 flex flex-wrap items-center gap-3"
          onSubmit={async (e) => {
            e.preventDefault()
            if (await checkLabCode(labId, prefix, code)) completeLab(labId, normaliseCode(code))
            else setState('wrong')
          }}
        >
          <input
            value={code}
            onChange={(e) => {
              setCode(e.target.value)
              setState('idle')
            }}
            placeholder={`TS-${labId.toUpperCase()}-xxxxxxxx`}
            aria-label="completion code"
            className="min-w-[18rem] flex-1 rounded-md border border-line bg-surface-2 px-3 py-2 font-mono text-body-sm text-text-1 outline-none focus:border-accent"
          />
          <button
            type="submit"
            disabled={!code.trim()}
            className="rounded-md bg-accent px-4 py-2 font-display text-body-sm font-semibold text-accent-foreground disabled:opacity-40"
          >
            mark complete
          </button>
          {state === 'wrong' && (
            <p className="w-full font-mono text-[11px] text-amber">
              That code does not match lab {labId} for prefix “{prefix}”. Did verify.py run with the
              same PREFIX?
            </p>
          )}
        </form>
      )}
    </div>
  )
}

export default function LabPage() {
  const { labId } = useParams()
  const lab = labById(labId)
  const [guide, setGuide] = useState<{ id: string; g: LabGuide } | null>(null)

  useEffect(() => {
    const load = lab ? LAB_GUIDES[lab.id] : undefined
    if (!lab || !load) return
    let live = true
    load().then((m) => {
      if (live) setGuide({ id: lab.id, g: m.default })
    })
    return () => {
      live = false
    }
  }, [lab])

  if (!lab) return <NotFound />
  const track = getTrack(lab.trackId)!
  const lesson = lessonMeta(lab.lessonId)
  const g = guide?.id === lab.id ? guide.g : null
  const next = LABS[lab.index + 1]

  return (
    <div className="mx-auto max-w-[860px] px-6 pb-24 pt-24">
      <Link to="/labs" className="flex items-center gap-1.5 font-mono text-[11px] text-text-3 hover:text-accent">
        <ArrowLeft size={12} /> all labs
      </Link>
      <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}>
        <p className="mt-6 font-mono text-label uppercase" style={{ color: track.color }}>
          lab {labNumber(lab)} · {LANE_LABEL[lab.lane]} · ~{lab.minutes} min
        </p>
        <h1 className="mt-3 font-display text-h1 text-text-1">{lab.title}</h1>
        <p className="mt-3 text-body-lg text-text-2">{lab.hook}</p>
        <div className="mt-5 flex flex-wrap items-center gap-3">
          <a
            href={asset(lab.zip)}
            download
            className="inline-flex items-center gap-2 rounded-md border border-accent/60 bg-accent/10 px-4 py-2 font-mono text-[12px] text-accent hover:bg-accent/20"
          >
            <Download size={13} /> {lab.id}.zip
          </a>
          {lesson && (
            <Link to={lessonPath(lesson)} className="font-mono text-[11px] text-text-3 hover:text-accent">
              introduced in {track.code}.L{lesson.index} — {lesson.title}
            </Link>
          )}
        </div>
      </motion.div>

      <GetTheFiles lab={lab} />

      {g && (
        <div className="mt-4 rounded-lg border border-line bg-surface-1 p-5">
          <p className="font-mono text-label uppercase text-text-3">you need</p>
          <ul className="mt-2 flex flex-wrap gap-2">
            {g.requires.map((r) => (
              <li key={r} className="rounded-full border border-line px-2.5 py-0.5 font-mono text-[11px] text-text-2">
                {r}
              </li>
            ))}
          </ul>
        </div>
      )}

      <div className="mt-4 rounded-lg border border-line bg-surface-1 p-5">
        <p className="flex items-center gap-2 font-mono text-label uppercase text-text-3">
          <ListChecks size={13} /> what checks.py checks
        </p>
        <ol className="mt-3 space-y-1.5">
          {lab.checks.map((c, i) => (
            <li key={c.id} className="flex gap-3 text-body-sm text-text-2">
              <span className="font-mono text-[11px] text-text-3">{String(i + 1).padStart(2, '0')}</span>
              <span>
                {c.label} <span className="font-mono text-[10px] text-text-3">[{c.id}]</span>
              </span>
            </li>
          ))}
        </ol>
      </div>

      <article className="mt-10">
        {g ? (
          <Blocks blocks={g.blocks} id={`lab:${lab.id}`} color={track.color} />
        ) : (
          <p className="animate-pulse font-mono text-[11px] uppercase text-text-3">loading guide…</p>
        )}
      </article>

      <section className="mt-12 space-y-4">
        {lab.lane === 'browser' && <BrowserGrader key={lab.id} lab={lab} />}
        <Completion labId={lab.id} lane={lab.lane} />
      </section>

      {g && g.cleanup.length > 0 && (
        <section className="mt-12">
          <h2 className="font-display text-h3 text-text-1">Going further</h2>
          <Blocks blocks={g.cleanup} id={`lab:${lab.id}:cleanup`} color={track.color} />
        </section>
      )}

      {next && (
        <Link
          to={`/labs/${next.id}`}
          className="mt-12 block rounded-lg border border-line bg-surface-1 p-5 transition-colors hover:border-line-bright"
        >
          <p className="font-mono text-[11px] uppercase text-text-3">next · lab {labNumber(next)}</p>
          <p className="mt-1 font-display text-h4 text-text-1">{next.title}</p>
        </Link>
      )}
    </div>
  )
}

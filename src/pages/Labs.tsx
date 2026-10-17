import { Link } from 'react-router'
import { motion } from 'framer-motion'
import { Check, Globe, Terminal } from 'lucide-react'
import { LABS, labNumber, LANE_LABEL, type LabLane } from '@/data/labs'
import { getTrack, HALF_LABEL, type Half } from '@/lib/tracks'
import { useProgress } from '@/lib/progress'
import PrefixSetting from '@/components/PrefixSetting'
import { cn } from '@/lib/utils'
import { renderInline } from '@/pages/lesson/markdown'

const LANE_COPY: Record<LabLane, { icon: typeof Terminal; needs: string; body: string }> = {
  browser: {
    icon: Globe,
    needs: 'a browser — or Python 3.10+ with numpy and pytest',
    body: 'numpy and pure-Python labs. Write the file in the editor on the lab page, or locally and drop it on the page: the page runs the lab\'s own `checks.py` in Pyodide and marks the lab done when every check is green. Locally, `pytest` runs the identical checks.',
  },
  local: {
    icon: Terminal,
    needs: 'Python 3.10+ · numpy · pytest · torch (CPU is enough)',
    body: 'PyTorch labs. They run on your machine — a laptop CPU finishes each in minutes; multi-process labs use the `gloo` backend, so no GPU is needed. `verify.py` prints a completion code; paste it on the lab page.',
  },
}

const PARTS: Half[] = ['build', 'scale', 'shape']

export default function Labs() {
  const labs = useProgress((s) => s.labs)

  return (
    <div className="mx-auto max-w-app px-6 pb-24 pt-24 lg:px-12">
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
      >
        <p className="section-label">hands-on</p>
        <h1 className="mt-4 font-display text-display-lg text-text-1">Labs</h1>
        <p className="mt-4 max-w-[68ch] text-body text-text-2">
          {LABS.length} Python labs. Each is a small folder: one file you edit (marked{' '}
          <code>TODO(you)</code>), a <code>checks.py</code>, and a <code>verify.py</code>. The same checks
          run under <code>pytest</code>, under <code>verify.py</code>, and — for the numpy labs — in your
          browser.
        </p>
        <p className="mt-3 max-w-[68ch] text-body-sm text-text-3">
          A completion code is a progress marker, not proof: it is derived from the lab and your
          handle, and anyone who reads <code>verify.py</code> can compute it. The checks are there to
          tell <em>you</em> whether it worked.
        </p>
      </motion.div>

      <div className="mt-8">
        <PrefixSetting />
      </div>

      <div className="mt-10 grid gap-4 md:grid-cols-2">
        {(['browser', 'local'] as LabLane[]).map((lane) => {
          const copy = LANE_COPY[lane]
          const Icon = copy.icon
          return (
            <div key={lane} className="rounded-lg border border-line bg-surface-1 p-5">
              <div className="flex items-center gap-2.5">
                <Icon className={cn('h-4 w-4', lane === 'browser' ? 'text-accent' : 'text-amber')} />
                <h2 className="font-display text-h4 text-text-1">{LANE_LABEL[lane]}</h2>
              </div>
              <p className="mt-2 text-body-sm text-text-2">{renderInline(copy.body)}</p>
              <p className="mt-2 font-mono text-[11px] text-text-3">needs: {copy.needs}</p>
            </div>
          )
        })}
      </div>

      {PARTS.map((part) => {
        const list = LABS.filter((l) => getTrack(l.trackId)?.half === part)
        if (list.length === 0) return null
        return (
          <section key={part} className="mt-14">
            <div className="flex flex-wrap items-center gap-3">
              <h2 className="font-display text-h3 text-text-1">{HALF_LABEL[part]}</h2>
              <span className="font-mono text-[11px] text-text-3">
                labs {labNumber(list[0])}–{labNumber(list[list.length - 1])}
              </span>
            </div>
            <div className="mt-5 grid gap-4 md:grid-cols-2">
              {list.map((lab) => {
                const track = getTrack(lab.trackId)
                const done = labs[lab.id]?.done ?? false
                return (
                  <Link
                    key={lab.id}
                    to={`/labs/${lab.id}`}
                    className="group block rounded-lg border border-line bg-surface-1 p-5 transition-colors duration-180 hover:border-line-bright"
                  >
                    <div className="flex items-center justify-between gap-3">
                      <span className="font-mono text-[11px] uppercase" style={{ color: track?.color }}>
                        lab {labNumber(lab)} · {track?.code} · {lab.lane}
                      </span>
                      {done ? (
                        <span className="flex items-center gap-1 font-mono text-[11px] text-accent">
                          <Check size={12} /> done
                        </span>
                      ) : (
                        <span className="font-mono text-[11px] text-text-3">~{lab.minutes} min</span>
                      )}
                    </div>
                    <p className="mt-2 font-display text-h4 text-text-1">{lab.title}</p>
                    <p className="mt-1.5 text-body-sm text-text-2">{lab.hook}</p>
                    <p className="mt-3 font-mono text-[11px] text-text-3">
                      {lab.module} · {lab.checks.length} checks
                      {lab.after.length > 0 && ` · after lab ${lab.after.map((a) => labNumber(LABS.find((x) => x.id === a)!)).join(', ')}`}
                    </p>
                  </Link>
                )
              })}
            </div>
          </section>
        )
      })}
    </div>
  )
}

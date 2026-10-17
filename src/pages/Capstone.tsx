import { Link } from 'react-router'
import { motion } from 'framer-motion'
import { Check, Square } from 'lucide-react'
import { CAPSTONE } from '@/lib/tracks'
import { CAPSTONE_STEPS } from '@/data/capstone'
import { useProgress } from '@/lib/progress'
import { renderInline } from '@/pages/lesson/markdown'
import { cn } from '@/lib/utils'

export default function Capstone() {
  const stepsDone = useProgress((s) => s.capstone.stepsDone)
  const completeStep = useProgress((s) => s.completeCapstoneStep)
  const labs = useProgress((s) => s.labs)
  const ready = labs['tensor-autograd']?.done && labs['train-loop']?.done

  return (
    <div className="mx-auto max-w-[860px] px-6 pb-24 pt-24">
      <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}>
        <p className="section-label">{CAPSTONE.code} — capstone</p>
        <h1 className="mt-4 font-display text-h1 text-text-1">Train It, Defend It</h1>
        <p className="mt-4 text-body-lg text-text-2">{CAPSTONE.promise}</p>
        <p className="mt-4 text-body-sm text-text-3">
          Budget about {CAPSTONE.hours} hours on a laptop CPU. It assumes the engine from{' '}
          <Link className="underline hover:text-accent" to="/labs/tensor-autograd">lab 02</Link> and the
          training loop from <Link className="underline hover:text-accent" to="/labs/train-loop">lab 07</Link>.
          {!ready && ' Those labs are not both marked done yet — the capstone assumes them.'}
        </p>
      </motion.div>

      <ol className="mt-10 space-y-5">
        {CAPSTONE_STEPS.map((step, i) => {
          const done = stepsDone.includes(step.id)
          return (
            <li
              key={step.id}
              className={cn('rounded-lg border bg-surface-1 p-5', done ? 'border-accent/50' : 'border-line')}
            >
              <div className="flex items-start justify-between gap-4">
                <div>
                  <p className="font-mono text-[11px] uppercase text-text-3">step {i + 1}</p>
                  <h2 className="mt-1 font-display text-h4 text-text-1">{step.title}</h2>
                </div>
                <button
                  type="button"
                  onClick={() => completeStep(step.id)}
                  disabled={done}
                  className={cn(
                    'flex shrink-0 items-center gap-1.5 rounded-md border px-3 py-1.5 font-mono text-[11px]',
                    done ? 'border-accent/50 text-accent' : 'border-line text-text-2 hover:border-line-bright hover:text-text-1',
                  )}
                >
                  {done ? <Check size={12} /> : <Square size={12} />}
                  {done ? 'done' : 'mark done'}
                </button>
              </div>
              {step.body.map((p, j) => (
                <p key={j} className="mt-3 text-body-sm text-text-2">
                  {renderInline(p)}
                </p>
              ))}
              <p className="mt-3 border-t border-line pt-3 font-mono text-[11px] text-text-3">
                done when: {renderInline(step.done)}
              </p>
            </li>
          )
        })}
      </ol>
    </div>
  )
}

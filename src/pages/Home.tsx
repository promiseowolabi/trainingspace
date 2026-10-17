import { Link } from 'react-router'
import { motion } from 'framer-motion'
import { ArrowRight, Globe, Terminal } from 'lucide-react'
import {
  TRACKS,
  TOTAL_TRACK_LESSONS,
  ORDERED_LESSON_IDS,
  HALF_LABEL,
  type Half,
} from '@/lib/tracks'
import TrackCard from '@/components/TrackCard'
import { LABS, labNumber } from '@/data/labs'

const HALVES: Half[] = ['build', 'scale', 'shape']

export default function Home() {
  const browserLabs = LABS.filter((l) => l.lane === 'browser')
  const localLabs = LABS.filter((l) => l.lane === 'local')

  return (
    <div className="mx-auto max-w-app px-6 pb-24 pt-24 lg:px-12">
      <motion.div
        initial={{ opacity: 0, y: 24 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
        className="mx-auto max-w-3xl text-center"
      >
        <p className="font-mono text-[11px] uppercase tracking-[0.16em] text-accent">
          trainingspace · how language models are trained
        </p>
        <h1 className="mt-5 text-5xl font-semibold tracking-tight text-text-1 sm:text-6xl">
          From one gradient
          <br />
          <span className="text-accent">to a post-trained GPT.</span>
        </h1>
        <p className="mx-auto mt-6 max-w-xl text-body-lg text-text-2">
          {TOTAL_TRACK_LESSONS} lessons across {TRACKS.length} tracks and {LABS.length} Python labs. Build an
          autograd engine, an optimiser, a tokenizer and a GPT in numpy — then the PyTorch training loop,
          mixed precision, scaling laws, data, tensor and pipeline parallelism, and post-training with SFT,
          LoRA, DPO and GRPO. No GPU required.
        </p>
        <div className="mt-8 flex flex-wrap items-center justify-center gap-3">
          <Link
            to={`/lesson/${ORDERED_LESSON_IDS[0]}`}
            className="inline-flex items-center gap-2 rounded-md border border-accent/60 bg-accent/10 px-5 py-2.5 font-mono text-sm text-accent transition-colors hover:bg-accent/20"
          >
            start the curriculum <ArrowRight className="h-4 w-4" />
          </Link>
          <Link
            to="/labs"
            className="inline-flex items-center gap-2 rounded-md border border-line bg-surface-1 px-5 py-2.5 font-mono text-sm text-text-2 transition-colors hover:text-text-1"
          >
            <Terminal className="h-4 w-4" /> go to the labs
          </Link>
        </div>
        <div className="mt-6 flex flex-wrap justify-center gap-2 font-mono text-[11px] text-text-3">
          {['python', 'numpy', 'pytorch', 'pyodide', 'pytest', 'cpu-only', 'local-first'].map((chip) => (
            <span key={chip} className="rounded-full border border-line px-2.5 py-1">
              {chip}
            </span>
          ))}
        </div>
        <p className="mx-auto mt-5 max-w-xl text-body-sm text-text-3">
          Every number in a lesson is arithmetic you can redo, a cited figure, or a measurement from the
          reference run — and says which. Framework and model facts carry the month they were verified.
        </p>
      </motion.div>

      {/* tracks, grouped by half */}
      {HALVES.map((half, hi) => (
        <section key={half} className="mt-16">
          <p className="mb-4 font-mono text-[11px] uppercase tracking-[0.14em] text-text-3">
            {HALF_LABEL[half]}
          </p>
          <div className="grid gap-4 md:grid-cols-2">
            {TRACKS.filter((t) => t.half === half).map((t, i) => (
              <motion.div
                key={t.id}
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.15 + (hi * 3 + i) * 0.05, ease: [0.16, 1, 0.3, 1] }}
              >
                <TrackCard track={t} />
              </motion.div>
            ))}
          </div>
        </section>
      ))}

      {/* the two lab lanes */}
      <div className="mt-12 grid gap-4 md:grid-cols-2">
        <Link
          to="/labs"
          className="group block rounded-lg border border-line bg-surface-1 p-6 transition-colors hover:border-accent/50"
        >
          <div className="flex items-center gap-2.5">
            <Globe className="h-5 w-5 text-accent" />
            <p className="font-mono text-[11px] uppercase tracking-[0.12em] text-text-3">graded in your browser</p>
          </div>
          <p className="mt-3 text-h4 font-medium text-text-1">{browserLabs.length} numpy labs, zero install</p>
          <p className="mt-2 text-body-sm text-text-2">
            Write the code on the lab page or in your own editor; the page runs the lab&apos;s{' '}
            <code>checks.py</code> in Pyodide — the same file <code>pytest</code> runs locally.
          </p>
          <p className="mt-4 font-mono text-[11px] text-accent">
            start with lab {browserLabs[0] ? labNumber(browserLabs[0]) : '00'} <ArrowRight className="inline h-3 w-3" />
          </p>
        </Link>
        <Link
          to="/labs"
          className="group block rounded-lg border border-amber/40 bg-surface-1 p-6 transition-colors hover:border-amber/70"
        >
          <div className="flex items-center gap-2.5">
            <Terminal className="h-5 w-5 text-amber" />
            <p className="font-mono text-[11px] uppercase tracking-[0.12em] text-text-3">pytorch on your laptop</p>
          </div>
          <p className="mt-3 text-h4 font-medium text-text-1">
            {localLabs.length > 0 ? `${localLabs.length} PyTorch labs on a CPU` : 'PyTorch labs on a CPU'}
          </p>
          <p className="mt-2 text-body-sm text-text-2">
            The training loop, DDP over <code>gloo</code> and LoRA run on a laptop CPU in minutes.{' '}
            <code>verify.py</code> prints a completion code you paste on the lab page.
          </p>
          <p className="mt-4 font-mono text-[11px] text-amber">
            {localLabs[0] ? <>start with lab {labNumber(localLabs[0])} </> : <>see the labs </>}
            <ArrowRight className="inline h-3 w-3" />
          </p>
        </Link>
      </div>

      <Link
        to="/capstone"
        className="group mt-4 block rounded-lg border border-line bg-surface-1 p-5 transition-colors hover:border-accent/50"
      >
        <p className="font-mono text-[11px] uppercase tracking-[0.12em] text-text-3">the capstone</p>
        <p className="mt-2 font-medium text-text-1">Train it, defend it</p>
        <p className="mt-1.5 text-body-sm text-text-2">
          Train a small GPT on your laptop, fine-tune it, and defend every number — loss, tokens, FLOPs,
          MFU and memory — in a one-page run report.
        </p>
      </Link>
    </div>
  )
}

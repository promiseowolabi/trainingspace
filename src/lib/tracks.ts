/**
 * Shared curriculum metadata — trainingspace (how a language model is trained,
 * from a single gradient to a post-trained assistant). Same TrackMeta contract
 * the platform components consume.
 *
 * Three parts, deliberately visible in the registry:
 *   t0, a1, o1, m1  build  — from scratch in numpy, graded in your browser.
 *   p1, d1          scale  — PyTorch on your laptop, simulated ranks for the cluster.
 *   f1, e1          shape  — post-training, evals, and debugging a run.
 */

import type { LucideIcon } from 'lucide-react'
import {
  Bug,
  Flame,
  GitBranch,
  Layers,
  MessageSquare,
  Network,
  Sigma,
  TrendingDown,
  Trophy,
} from 'lucide-react'
import type { Level, TrackId } from '@/data/lessons/types'

export type Half = 'build' | 'scale' | 'shape'

export interface TrackMeta {
  code: string
  /** `capstone` is not a content track — it has no lessons of its own. */
  id: TrackId | 'capstone'
  name: string
  color: string
  glyph: LucideIcon
  promise: string
  lessons: number
  exercises: number
  hours: number
  half: Half
  level: Level
}

export const TRACKS: TrackMeta[] = [
  {
    code: 'T0',
    id: 't0',
    name: 'The Training Contract',
    color: '#A3E635',
    glyph: Sigma,
    promise:
      'A model is a function with knobs; training is nudging every knob downhill. Loss, gradient, step — and the FLOPs and bytes each one costs.',
    lessons: 4,
    exercises: 4,
    hours: 2,
    half: 'build',
    level: 200,
  },
  {
    code: 'A1',
    id: 'a1',
    name: 'Autograd from Scratch',
    color: '#22D3EE',
    glyph: GitBranch,
    promise:
      'The chain rule as a graph walk. Build the engine PyTorch hides — first for scalars, then for tensors — and prove it right with gradient checking.',
    lessons: 5,
    exercises: 5,
    hours: 3.5,
    half: 'build',
    level: 200,
  },
  {
    code: 'O1',
    id: 'o1',
    name: 'Networks & Optimisation',
    color: '#5CA8FF',
    glyph: TrendingDown,
    promise:
      'Layers, initialisation, cross-entropy, AdamW and the learning-rate schedule — the knobs that decide whether a run learns, stalls or explodes.',
    lessons: 6,
    exercises: 6,
    hours: 4,
    half: 'build',
    level: 300,
  },
  {
    code: 'M1',
    id: 'm1',
    name: 'The Transformer',
    color: '#A78BFA',
    glyph: Layers,
    promise:
      'Tokens, embeddings, attention, the block and the sampler: a GPT forward pass you wrote yourself, with every parameter accounted for.',
    lessons: 6,
    exercises: 6,
    hours: 4.5,
    half: 'build',
    level: 300,
  },
  {
    code: 'P1',
    id: 'p1',
    name: 'Pretraining',
    color: '#FB923C',
    glyph: Flame,
    promise:
      'Data, the PyTorch loop, mixed precision, 6·N·D and scaling laws — how a training run is budgeted, measured and kept from diverging.',
    lessons: 6,
    exercises: 6,
    hours: 4.5,
    half: 'scale',
    level: 400,
  },
  {
    code: 'D1',
    id: 'd1',
    name: 'Distributed Training',
    color: '#FB7185',
    glyph: Network,
    promise:
      'Collectives, data parallelism, ZeRO, tensor and pipeline parallelism — what each one shards, what it sends over the wire, and how to choose a layout.',
    lessons: 6,
    exercises: 6,
    hours: 4.5,
    half: 'scale',
    level: 400,
  },
  {
    code: 'F1',
    id: 'f1',
    name: 'Post-Training',
    color: '#FBBF24',
    glyph: MessageSquare,
    promise:
      'From a base model to an assistant: SFT, LoRA, reward models, RLHF, DPO and GRPO — each one as a loss function you can write down.',
    lessons: 6,
    exercises: 6,
    hours: 4.5,
    half: 'shape',
    level: 400,
  },
  {
    code: 'E1',
    id: 'e1',
    name: 'Evals & Debugging',
    color: '#34D399',
    glyph: Bug,
    promise:
      'What a benchmark score can and cannot tell you, the checklist for a run that will not learn, and how to read any training paper as a table of numbers.',
    lessons: 4,
    exercises: 4,
    hours: 2.5,
    half: 'shape',
    level: 400,
  },
]

export const CAPSTONE: TrackMeta = {
  code: 'T*',
  id: 'capstone',
  name: 'Capstone: Train It, Defend It',
  color: '#FDE047',
  glyph: Trophy,
  promise:
    'Train a small GPT end to end on your laptop, fine-tune it, and defend every number — loss, tokens, FLOPs, MFU and memory — in a one-page run report.',
  lessons: 0,
  exercises: 0,
  hours: 5,
  half: 'shape',
  level: 400,
}

export const TOTAL_TRACK_LESSONS = TRACKS.reduce((n, t) => n + t.lessons, 0)

export function getTrack(id: string): TrackMeta | undefined {
  return TRACKS.find((t) => t.id === id)
}

export const BUILD_TRACKS = TRACKS.filter((t) => t.half === 'build')
export const SCALE_TRACKS = TRACKS.filter((t) => t.half === 'scale')
export const SHAPE_TRACKS = TRACKS.filter((t) => t.half === 'shape')

export const HALF_LABEL: Record<Half, string> = {
  build: 'Build it · from scratch in numpy, graded in your browser',
  scale: 'Scale it · PyTorch on your laptop, a cluster in simulation',
  shape: 'Shape it · post-training, evals and debugging',
}

/** Ordered lesson ids across tracks for next-lesson selectors. */
export const ORDERED_LESSON_IDS: string[] = TRACKS.flatMap((t) =>
  Array.from({ length: t.lessons }, (_, i) => `${t.id}.l${i + 1}`),
)

/** TRACKS never contains the capstone, so its ids are always real TrackIds. */
export const CONTENT_TRACK_IDS = TRACKS.map((t) => t.id as TrackId)

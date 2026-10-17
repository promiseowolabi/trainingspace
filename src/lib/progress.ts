import { create } from 'zustand'
import { persist } from 'zustand/middleware'

/**
 * Trainingspace progress store.
 * Single localStorage namespace: `trainingspace:v1`.
 */

export type LessonStatus = 'unstarted' | 'reading' | 'done'

export interface LessonProgress {
  status: LessonStatus
  quizScore?: number // 0..1
  exerciseDone?: boolean
  completedAt?: string // ISO
  lastVisitedAt: string // ISO
  scrollPct?: number // resume position
}

/**
 * One Python lab. A lab is done when every check passes in the browser grader,
 * or when the reader pastes the completion code their `verify.py` printed —
 * derived from the lab id and the reader's handle (src/lib/lab-code.ts), so it
 * is a progress marker, not proof.
 */
export interface LabProgress {
  done: boolean
  completedAt?: string // ISO
  /** The code as pasted (or `browser · n/n checks`), kept so the Progress page can show it back. */
  code?: string
}

export interface CapstoneProgress {
  stepsDone: string[]
}

export interface ProgressSettings {
  reducedMotion?: boolean
  /** The reader's handle: `PREFIX` for verify.py, and the salt of every completion code. */
  prefix?: string
}

export interface ProgressState {
  version: 1
  lessons: Record<string, LessonProgress>
  labs: Record<string, LabProgress>
  capstone: CapstoneProgress
  xp: number
  streakDays: string[] // ISO dates with any activity
  achievements: string[]
  settings: ProgressSettings

  // actions
  markLessonStatus: (lessonId: string, status: LessonStatus) => void
  setLessonScroll: (lessonId: string, scrollPct: number) => void
  recordQuizScore: (lessonId: string, score: number) => void
  markExerciseDone: (lessonId: string) => void
  completeLab: (labId: string, code: string) => void
  completeCapstoneStep: (stepId: string) => void
  unlockAchievement: (id: string) => void
  updateSettings: (patch: Partial<ProgressSettings>) => void
  importProgress: (json: string) => boolean
  resetProgress: () => void
}

export const XP = {
  lesson: 100,
  quiz: 40,
  exercise: 60,
  lab: 250,
  capstoneStep: 150,
} as const

export interface Rank {
  name: string
  minXp: number
}

/** XP → Rank: progress rendered as a training run. */
export const RANKS: Rank[] = [
  { name: 'FRONTIER', minXp: 8000 },
  { name: 'CONVERGED', minXp: 5000 },
  { name: 'PAST WARMUP', minXp: 2000 },
  { name: 'FIRST STEP', minXp: 500 },
  { name: 'RANDOM INIT', minXp: 0 },
]

export function rankForXp(xp: number): Rank {
  return RANKS.find((r) => xp >= r.minXp) ?? RANKS[RANKS.length - 1]
}

export function nextRank(xp: number): Rank | null {
  const sorted = [...RANKS].sort((a, b) => a.minXp - b.minXp)
  return sorted.find((r) => r.minXp > xp) ?? null
}

const todayISO = () => new Date().toISOString().slice(0, 10)

const initialData = {
  version: 1 as const,
  lessons: {} as Record<string, LessonProgress>,
  labs: {} as Record<string, LabProgress>,
  capstone: { stepsDone: [] as string[] },
  xp: 0,
  streakDays: [] as string[],
  achievements: [] as string[],
  settings: {} as ProgressSettings,
}

function touchStreak(streakDays: string[]): string[] {
  const today = todayISO()
  if (streakDays.includes(today)) return streakDays
  return [...streakDays, today]
}

export const useProgress = create<ProgressState>()(
  persist(
    (set) => ({
      ...initialData,

      markLessonStatus: (lessonId, status) =>
        set((s) => {
          const prev = s.lessons[lessonId]
          const wasDone = prev?.status === 'done'
          const nowDone = status === 'done'
          return {
            lessons: {
              ...s.lessons,
              [lessonId]: {
                ...prev,
                status,
                completedAt: nowDone && !wasDone ? new Date().toISOString() : prev?.completedAt,
                lastVisitedAt: new Date().toISOString(),
              },
            },
            xp: s.xp + (nowDone && !wasDone ? XP.lesson : 0),
            streakDays: touchStreak(s.streakDays),
          }
        }),

      setLessonScroll: (lessonId, scrollPct) =>
        set((s) => {
          const prev = s.lessons[lessonId]
          if (!prev) return s
          return { lessons: { ...s.lessons, [lessonId]: { ...prev, scrollPct } } }
        }),

      recordQuizScore: (lessonId, score) =>
        set((s) => {
          const prev = s.lessons[lessonId]
          const best = Math.max(prev?.quizScore ?? 0, score)
          const firstPass = (prev?.quizScore ?? 0) < 0.8 && score >= 0.8
          return {
            lessons: {
              ...s.lessons,
              [lessonId]: {
                ...prev,
                status: prev?.status ?? 'reading',
                quizScore: best,
                lastVisitedAt: new Date().toISOString(),
              },
            },
            xp: s.xp + (firstPass ? XP.quiz : 0),
            streakDays: touchStreak(s.streakDays),
          }
        }),

      markExerciseDone: (lessonId) =>
        set((s) => {
          const prev = s.lessons[lessonId]
          if (prev?.exerciseDone) return s
          return {
            lessons: {
              ...s.lessons,
              [lessonId]: {
                ...prev,
                status: prev?.status ?? 'reading',
                exerciseDone: true,
                lastVisitedAt: new Date().toISOString(),
              },
            },
            xp: s.xp + XP.exercise,
            streakDays: touchStreak(s.streakDays),
          }
        }),

      completeLab: (labId, code) =>
        set((s) => {
          const prev = s.labs[labId]
          if (prev?.done) return s
          return {
            labs: { ...s.labs, [labId]: { done: true, code, completedAt: new Date().toISOString() } },
            xp: s.xp + XP.lab,
            streakDays: touchStreak(s.streakDays),
          }
        }),

      completeCapstoneStep: (stepId) =>
        set((s) => {
          if (s.capstone.stepsDone.includes(stepId)) return s
          return {
            capstone: { stepsDone: [...s.capstone.stepsDone, stepId] },
            xp: s.xp + XP.capstoneStep,
            streakDays: touchStreak(s.streakDays),
          }
        }),

      unlockAchievement: (id) =>
        set((s) => (s.achievements.includes(id) ? s : { achievements: [...s.achievements, id] })),

      updateSettings: (patch) => set((s) => ({ settings: { ...s.settings, ...patch } })),

      importProgress: (json) => {
        try {
          const data = JSON.parse(json)
          if (data?.version !== 1 || typeof data.lessons !== 'object' || typeof data.xp !== 'number') {
            return false
          }
          set({ ...initialData, ...(data as typeof initialData) })
          return true
        } catch {
          return false
        }
      },

      resetProgress: () => set({ ...initialData }),
    }),
    {
      name: 'trainingspace:v1',
    },
  ),
)

/* ---------------- Derived selectors ---------------- */

export const selectDoneLessons = (s: ProgressState) =>
  Object.values(s.lessons).filter((l) => l.status === 'done').length

export const selectLabsDone = (s: ProgressState) =>
  Object.values(s.labs).filter((l) => l.done).length

/** Per-track completion % — lessonIds are prefixed `${trackId}.` (e.g. `k1.l2`). */
export function selectTrackPct(trackId: string, lessonCount: number) {
  return (s: ProgressState) => {
    if (lessonCount <= 0) return 0
    const done = Object.entries(s.lessons).filter(
      ([id, l]) => id.startsWith(`${trackId}.`) && l.status === 'done',
    ).length
    return Math.round((done / lessonCount) * 100)
  }
}

export function selectTrackDone(trackId: string) {
  return (s: ProgressState) =>
    Object.entries(s.lessons).filter(([id, l]) => id.startsWith(`${trackId}.`) && l.status === 'done')
      .length
}

/** First non-done lesson in track order → next recommended lesson id. */
export function selectNextLesson(orderedLessonIds: string[]) {
  return (s: ProgressState) =>
    orderedLessonIds.find((id) => s.lessons[id]?.status !== 'done') ?? null
}

/** Current streak length in consecutive days ending today/yesterday. */
export function selectStreak(s: ProgressState): number {
  if (s.streakDays.length === 0) return 0
  const days = new Set(s.streakDays)
  let streak = 0
  const cursor = new Date()
  if (!days.has(cursor.toISOString().slice(0, 10))) {
    cursor.setDate(cursor.getDate() - 1) // allow streak to end yesterday
  }
  while (days.has(cursor.toISOString().slice(0, 10))) {
    streak += 1
    cursor.setDate(cursor.getDate() - 1)
  }
  return streak
}

/** Activity heatmap data: date → completions (lessons and labs). */
export function selectActivityMap(s: ProgressState): Record<string, number> {
  const map: Record<string, number> = {}
  for (const l of [...Object.values(s.lessons), ...Object.values(s.labs)]) {
    if (l.completedAt) {
      const day = l.completedAt.slice(0, 10)
      map[day] = (map[day] ?? 0) + 1
    }
  }
  return map
}

/** Export the raw store as a JSON download string. */
export function exportProgress(): string {
  const { lessons, labs, capstone, xp, streakDays, achievements, settings } = useProgress.getState()
  return JSON.stringify(
    { version: 1, lessons, labs, capstone, xp, streakDays, achievements, settings },
    null,
    2,
  )
}

// Convenience non-hook getter for one-off reads outside React.
export const getProgress = () => useProgress.getState()

/**
 * Exercise-kind metadata (icon + label) shared by LessonRow and the lesson page.
 */

import { BookOpen, HelpCircle, Terminal } from 'lucide-react'
import type { ExerciseKind } from '@/data/lessons/types'

export const EXERCISE_META: Record<ExerciseKind, { icon: typeof BookOpen; label: string }> = {
  quiz: { icon: HelpCircle, label: 'quiz' },
  read: { icon: BookOpen, label: 'guided read' },
  'read+quiz': { icon: BookOpen, label: 'read + quiz' },
  'lab+quiz': { icon: Terminal, label: 'lab + quiz' },
}

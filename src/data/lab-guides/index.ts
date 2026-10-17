/**
 * Lazy lab-guide registry. Each guide is its own chunk, loaded only by the
 * lab route that shows it. tests/registry.test.ts requires a guide for every
 * lab in src/data/labs.ts.
 */

import type { LabId } from '@/data/lessons/types'
import type { LabGuide } from './types'

type Loader = () => Promise<{ default: LabGuide }>

export const LAB_GUIDES: Partial<Record<LabId, Loader>> = {
  'numpy-shapes': () => import('./numpy-shapes'),
  micrograd: () => import('./micrograd'),
  'tensor-autograd': () => import('./tensor-autograd'),
  optimisers: () => import('./optimisers'),
  'bpe-tokenizer': () => import('./bpe-tokenizer'),
  attention: () => import('./attention'),
  'gpt-forward': () => import('./gpt-forward'),
  'train-loop': () => import('./train-loop'),
  'flops-and-memory': () => import('./flops-and-memory'),
  collectives: () => import('./collectives'),
  'ddp-from-scratch': () => import('./ddp-from-scratch'),
  'zero-sharding': () => import('./zero-sharding'),
  lora: () => import('./lora'),
  'preference-losses': () => import('./preference-losses'),
}

export type { LabGuide }

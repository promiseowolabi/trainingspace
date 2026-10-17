import type { ContentBlock, LabId } from '@/data/lessons/types'

/**
 * A lab guide: the step-by-step body of one hands-on lab. Written with the
 * lesson block system so code blocks, callouts and vendor notes render the
 * same way they do in lessons.
 *
 * `requires` is the tool list the page shows above the steps. `cleanup` is
 * rendered after the completion box — "going further" for a Python lab:
 * the stretch goals, and what to keep for the next lab.
 */
export interface LabGuide {
  id: LabId
  requires: string[]
  blocks: ContentBlock[]
  cleanup: ContentBlock[]
}

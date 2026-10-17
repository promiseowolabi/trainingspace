# CONTENT-SPEC — how a trainingspace lesson and lab are written

Authoring contract for `src/data/lessons/<trackId>/<slug>.ts`, the lab
guides in `src/data/lab-guides/<id>.ts`, and the lab folders in `labs/<id>/`.
Forked from the series' shared spec; the block system, diagram rules and quiz
rules carry over unchanged.

## 1. File shape

```ts
import type { Lesson } from '../types'

const lesson: Lesson = {
  id: 'a1.l3',                     // `${trackId}.l${index}` — the progress store key
  slug: 'a-scalar-engine',         // MUST equal the filename minus .ts
  trackId: 'a1',
  index: 3,
  title: 'A Scalar Engine',
  minutes: 20,                     // honest reading estimate, 9–24
  hook: 'One sentence that makes the lesson unskippable.',
  exercise: 'lab+quiz',            // 'quiz' | 'read' | 'read+quiz' | 'lab+quiz'
  takeaway: { number: '68,140 nodes for 337 parameters', claim: 'One sentence a reader could say in a review.' },
  blocks: [ /* … */ ],
}

export default lesson
```

`takeaway.number` must contain a digit (tested). Register the lesson in
`src/data/lessons/index.ts`, then `npm run export`.

**Escaping:** `md` strings are template literals — write inline code as
`` \` `` and `${` as `\${`.

## 2. Blocks

| block | use it for |
|---|---|
| `prose` | the spine. `## H2`, `### H3`, lists, pipe tables, `**bold**`, `` `code` ``, `[label](url)` |
| `code` | **Python by default** (omit `lang`); `lang: 'text'` for captured output, `bash` for shell |
| `callout` | `analogy` · `info` · `warning` · `segfault` (a real failure mode) · `isomorphism` |
| `diagram` | stepped SVG walkthrough. §3 |
| `statline` | 3–4 numbers that survive the lesson |
| `quiz` | 3 transfer questions. §4 |
| `vendor` | every framework-, paper- or hardware-specific claim. §5 |
| `lab` | `{ lab, brief }` — sends the reader to a lab (`src/data/labs.ts`) |
| `isomorphism` | map onto something a backend engineer owns (fields `os`/`llm` = left/right) |
| `deepdive` | closing "going deeper": papers and source files, by name |

Tested: exactly one quiz, first block is prose, every `lab` id is registered,
`lab+quiz` lessons contain a `lab` block.

## 3. Diagrams

100-wide × `height` (60–80) viewBox; `x`/`y` top-left; boxes ≥ 10×8; no
overlaps; ≥ 3 steps; every caption > 40 characters (captions are narration).

## 4. Quizzes

Transfer, not recall: a training situation the reader must reason about.
Four options, one correct; wrong options are real folklore ("just raise the
learning rate", "the model will learn the scale", "accuracy is fine so
nothing is wrong"). `explanation` says why the right answer is right *and*
why the tempting wrong one fails. Arithmetic questions show the arithmetic.

## 5. Vendor claims — the rot rule

No product-, paper- or hardware-specific fact loose in prose: a PyTorch
default, a published token count, an H100 peak, a paper's hyper-parameter.
It lives in a `vendor` block with `snapshot: 'YYYY-MM'` (the month you
**checked** it) and `https://` sources. Never a price.

## 6. Numbers

- Second person, present tense, no "in this lesson we will".
- **Every mechanism gets a number.** Numbers are (a) arithmetic the reader
  can redo, (b) a cited figure in a vendor block, or (c) **measured on the
  reference machine** — Apple M5 laptop, numpy 2.5, PyTorch 2.12, CPU only —
  and the lesson says so and gives the setup. Never invent a benchmark.
- Name what each technique is bad at, and what it costs.
- Reference lessons as `O1.L3`, labs as `lab 03`.
- No internal hostnames or environment names anywhere (`tests/branding.test.ts`).

## 7. Labs

```
labs/<id>/<module>.py   the one file the reader edits; every gap is
                        `# TODO(you)` + `raise NotImplementedError`
labs/<id>/checks.py     check_<id>(m) functions + CHECKS = [...]; each raises
                        AssertionError with a message that says what was expected
labs/<id>/test_lab.py   pytest wrapper (generated from lab 00's)
labs/<id>/verify.py     prints TS-<LAB>-xxxxxxxx (generated from lab 00's)
labs/<id>/README.md     the brief and the check list
labs/_solutions/<id>/   the reference module — committed, never packed or bundled
src/data/lab-guides/<id>.ts   the page guide; `cleanup` renders as "Going further"
```

- **browser lane** checks import only the Python standard library and
  packages Pyodide ships (numpy). Keep every check under ~2 s in Pyodide.
- **local lane** (PyTorch) checks must pass on a laptop CPU in seconds;
  multi-process checks use `spawn`, `gloo` on 127.0.0.1 and a timeout.
- Check ids in `src/data/labs.ts` equal `CHECKS` in order (tested).
- `python scripts/check-labs.py <id>` must print `ok`: the template fails at
  least one check, the solution passes all of them, `verify.py` prints a code.

## 8. Definition of done

`npm run export && npm run lint && npm run typecheck && npm run build &&
npm run test` clean; `scripts/check-labs.py` clean; browser labs graded once
on the built site; takeaway present; every vendor claim dated and sourced;
every measured number reproducible from the lesson's description; the lesson
names a weakness; it reads in its `minutes`.

/**
 * prepare-pages — give every known SPA route a real index.html in dist/.
 *
 * GitHub Pages has no rewrite rules. Without this, a deep link such as
 * /trainingspace/lesson/a1.l3 is served by 404.html: it renders correctly in a
 * browser, but answers HTTP 404, which link previews, crawlers and strict HTTP
 * clients treat as broken. Writing the app shell at each known route makes
 * those URLs answer 200. Unknown paths still fall through to 404.html and the
 * router's own not-found page.
 *
 * Route ids are derived from build artefacts (lessons-md/*.md, labs/*.zip) so a
 * new lesson or lab is covered without editing this file.
 *
 *   node scripts/prepare-pages.mjs      (after vite build)
 */
import { mkdir, readFile, readdir, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const distRoot = path.join(repoRoot, 'dist')
const indexHtml = await readFile(path.join(distRoot, 'index.html'), 'utf8')

const staticRoutes = ['curriculum', 'labs', 'capstone', 'progress']

async function ids(directory, extension) {
  const entries = await readdir(path.join(distRoot, directory), { withFileTypes: true })
  return entries
    .filter((e) => e.isFile() && e.name.endsWith(extension))
    .map((e) => e.name.slice(0, -extension.length))
    .map((id) => {
      if (!/^[a-z0-9][a-z0-9.-]*$/.test(id)) throw new Error(`unsafe route id: ${id}`)
      return id
    })
}

const lessonIds = await ids('lessons-md', '.md')
const labIds = await ids('labs', '.zip')
const trackIds = [...new Set(lessonIds.map((id) => id.split('.')[0]))]

const routes = new Set([
  ...staticRoutes,
  ...trackIds.map((id) => `tracks/${id}`),
  ...lessonIds.map((id) => `lesson/${id}`),
  ...labIds.map((id) => `labs/${id}`),
])

for (const route of [...routes].sort()) {
  const dir = path.resolve(distRoot, route)
  if (!dir.startsWith(`${distRoot}${path.sep}`)) throw new Error(`route escaped dist: ${route}`)
  await mkdir(dir, { recursive: true })
  await writeFile(path.join(dir, 'index.html'), indexHtml)
}

await writeFile(path.join(distRoot, '404.html'), indexHtml)

console.log(
  `prepared ${routes.size} route shells (${lessonIds.length} lessons, ${labIds.length} labs, ${trackIds.length} tracks)`,
)

#!/usr/bin/env bash
# smoke.sh — render built pages in headless Chrome and fail if any is blank.
#
# Unit tests do not execute the browser bundle. A regex that only breaks at
# runtime once shipped a site where every lesson and lab rendered empty; this
# catches that class of bug. Run after `npm run build` + prepare-pages.
set -euo pipefail
BROWSER="${CHROMIUM_PATH:-$(command -v chromium || command -v google-chrome || command -v chromium-browser)}"
PORT=4173
npx vite preview --base /trainingspace/ --port "$PORT" --strictPort >/dev/null 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null' EXIT
for _ in $(seq 30); do curl -s -o /dev/null "http://localhost:$PORT/trainingspace/" && break; sleep 1; done

fail=0
check() {  # check <path> <text that must appear>
  local text
  text=$("$BROWSER" --headless=new --disable-gpu --no-sandbox --virtual-time-budget=15000 \
    --dump-dom "http://localhost:$PORT/trainingspace/$1" 2>/dev/null \
    | sed -e 's/<script[^>]*>.*<\/script>//g' -e 's/<[^>]*>/ /g')
  if printf '%s' "$text" | grep -q "$2"; then echo "ok    /$1"; else echo "FAIL  /$1 (missing: $2)"; fail=1; fi
}
check "" "to a post-trained GPT"
check "curriculum/" "Autograd from Scratch"
check "lesson/a1.l3/" "topological"
check "labs/" "Labs"
check "labs/micrograd/" "cd micrograd"
check "capstone/" "Pretrain a small GPT"
exit $fail

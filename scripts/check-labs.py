#!/usr/bin/env python3
"""check-labs — prove every lab is gradeable before it ships.

For each labs/<id>/ (ids read from src/data/labs.ts):

  1. the TEMPLATE imports and fails at least one check, with no crash —
     a template that passes means the checks test nothing;
  2. the SOLUTION (labs/_solutions/<id>/) passes every check, and
     verify.py prints the completion code the site expects.

Each lab runs in a fresh temporary copy, in its own interpreter, so labs
never share a ``checks`` module. Run it with the Python the labs need
(numpy, plus torch for the local labs):

    python scripts/check-labs.py            # every lab
    python scripts/check-labs.py micrograd  # just one
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABS = os.path.join(ROOT, "labs")
SOLUTIONS = os.path.join(LABS, "_solutions")

RUNNER = r"""
import importlib, json, sys
sys.path[:0] = [sys.argv[1], sys.argv[2]]
import tscheck, checks
m = importlib.import_module(sys.argv[3])
print(json.dumps(tscheck.run(checks.CHECKS, m)))
"""


def registry() -> list[tuple[str, str]]:
    src = open(os.path.join(ROOT, "src", "data", "labs.ts"), encoding="utf8").read()
    ids = re.findall(r"^\s+id: '([a-z0-9-]+)',\s*$", src, flags=re.M)
    mods = re.findall(r"^\s+module: '([a-z0-9_]+)\.py',\s*$", src, flags=re.M)
    assert len(ids) == len(mods), "every lab in labs.ts needs a module: '<name>.py'"
    return list(zip(ids, mods))


def run_checks(lab_dir: str, module: str) -> list[dict]:
    out = subprocess.run(
        [sys.executable, "-c", RUNNER, lab_dir, os.path.join(LABS, "common"), module],
        capture_output=True, text=True, timeout=900, cwd=lab_dir,
    )
    if out.returncode != 0:
        raise RuntimeError(f"runner crashed:\n{out.stderr}")
    return json.loads(out.stdout.strip().splitlines()[-1])


def main(only: list[str]) -> int:
    failures = 0
    for lab, module in registry():
        if only and lab not in only:
            continue
        with tempfile.TemporaryDirectory() as tmp:
            work = os.path.join(tmp, lab)
            shutil.copytree(os.path.join(LABS, "common"), os.path.join(tmp, "common"))  # the zip's layout
            shutil.copytree(os.path.join(LABS, lab), work,
                            ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pt"))
            template = run_checks(work, module)
            failing = [r["id"] for r in template if not r["ok"]]
            ok_template = bool(failing)

            sol_dir = os.path.join(SOLUTIONS, lab)
            for name in os.listdir(sol_dir):
                if name.endswith(".py"):
                    shutil.copy(os.path.join(sol_dir, name), os.path.join(work, name))
            solution = run_checks(work, module)
            broken = [f"{r['id']}: {r['msg']}" for r in solution if not r["ok"]]

            env = dict(os.environ, PREFIX="alice")
            v = subprocess.run([sys.executable, "verify.py"], cwd=work, env=env,
                               capture_output=True, text=True, timeout=900)
            code = re.search(r"TS-[A-Z0-9-]+-[0-9a-f]{8}", v.stdout)

        status = "ok" if ok_template and not broken and code else "FAIL"
        slow = max(r["ms"] for r in solution)
        print(f"{status:4} {lab:20} template fails {len(failing)}/{len(template)} · "
              f"solution passes {len(solution) - len(broken)}/{len(solution)} · slowest check {slow:.0f} ms"
              + (f" · {code.group(0)}" if code else ""))
        if not ok_template:
            print("     the template passes every check — the checks test nothing")
        for b in broken:
            print(f"     solution: {b}")
        if not code:
            print(f"     verify.py printed no completion code:\n{v.stdout}{v.stderr}")
        failures += status != "ok"
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

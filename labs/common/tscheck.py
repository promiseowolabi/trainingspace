"""trainingspace — the shared check runner.

One file, three callers, identical behaviour:

  * ``pytest``      — each lab's ``test_lab.py`` turns every check into a test.
  * ``verify.py``   — runs the checks, prints a summary and, when every check
                      passes, the completion code the site accepts.
  * the browser     — the lab page loads this file and the lab's ``checks.py``
                      into Pyodide and calls :func:`run` on the file you drop.

A check is a plain function ``check_<id>(m)`` that receives your module and
raises ``AssertionError`` (with a readable message) when something is wrong.
Its id is the function name without ``check_``, underscores as hyphens.

The completion code is ``TS-<LAB>-<8 hex>`` where the hex is
``sha256("trainingspace:<lab>:<prefix>")[:8]``. It marks progress; it does
not prove it. Keep :func:`completion_code` in lock-step with
``src/lib/lab-code.ts`` — ``tests/labs.test.ts`` checks both.
"""

from __future__ import annotations

import hashlib
import importlib
import os
import re
import sys
import time
import traceback

PREFIX_RE = re.compile(r"^[a-z][a-z0-9-]{1,18}[a-z0-9]$")


def check_id(fn) -> str:
    """``check_matmul_flops`` → ``matmul-flops``."""
    name = fn.__name__
    if name.startswith("check_"):
        name = name[len("check_"):]
    return name.replace("_", "-")


def _where(tb) -> str:
    """The last frame inside the learner's module, as ``file:line``."""
    frames = traceback.extract_tb(tb)
    for fr in reversed(frames):
        base = os.path.basename(fr.filename)
        if base not in ("checks.py", "tscheck.py"):
            return f"{base}:{fr.lineno}"
    return ""


def run_one(fn, module) -> dict:
    started = time.perf_counter()
    try:
        fn(module)
        ok, msg = True, ""
    except NotImplementedError as e:
        ok, msg = False, "not implemented yet" + (f" — {e}" if str(e) else "")
    except AssertionError as e:
        ok, msg = False, str(e) or "assertion failed"
    except RecursionError:
        ok, msg = False, "RecursionError — walk the graph with a loop, not recursion"
    except Exception as e:  # noqa: BLE001 — a learner's bug is a failed check, not a crash
        where = _where(e.__traceback__)
        ok, msg = False, f"{type(e).__name__}: {e}" + (f"  ({where})" if where else "")
    return {
        "id": check_id(fn),
        "ok": ok,
        "msg": msg,
        "ms": round((time.perf_counter() - started) * 1000, 1),
    }


def run(checks, module) -> list[dict]:
    """Run every check against ``module``; never raises."""
    return [run_one(fn, module) for fn in checks]


def completion_code(lab: str, prefix: str) -> str:
    digest = hashlib.sha256(f"trainingspace:{lab}:{prefix}".encode()).hexdigest()
    return f"TS-{lab.upper()}-{digest[:8]}"


def _color(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if sys.stdout.isatty() else text


def main(lab: str, module_name: str, here: str) -> int:
    """Entry point for a lab's ``verify.py``. Returns the process exit code."""
    prefix = os.environ.get("PREFIX", "")
    if len(sys.argv) > 1:
        prefix = sys.argv[1]
    if not prefix:
        print(_color("31", "PREFIX is not set.")
              + " Run: PREFIX=<your-handle> python verify.py   (the same value you saved on the site)")
        return 2
    if not PREFIX_RE.match(prefix):
        print(_color("31", f"PREFIX '{prefix}' is not valid.")
              + " 3–20 chars: lowercase letters, digits, hyphens; start with a letter.")
        return 2

    sys.path.insert(0, here)
    checks = importlib.import_module("checks")
    try:
        module = importlib.import_module(module_name)
    except Exception as e:  # noqa: BLE001
        print(_color("31", f"{module_name}.py does not import: {type(e).__name__}: {e}"))
        return 1

    print(f"\n  trainingspace · lab {lab}\n")
    results = run(checks.CHECKS, module)
    for r in results:
        mark = _color("32", "✓") if r["ok"] else _color("31", "✗")
        print(f"  {mark} {r['id']}" + ("" if r["ok"] else _color("2", f"  — {r['msg']}")))
    passed = sum(r["ok"] for r in results)
    print(f"\n  {passed}/{len(results)} checks passed")
    if passed != len(results):
        return 1
    print(f"\n  completion code: {_color('1', completion_code(lab, prefix))}\n")
    return 0

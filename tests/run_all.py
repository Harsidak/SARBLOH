# --- path shim: this runner lives in tests/ but drives the repo root ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, 'eval')):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end shim ---
"""ONE command for every unit suite. Green/red in seconds.

    python tests/run_all.py                    # everything
    python tests/run_all.py --only stategraph  # substring match, repeatable
    python tests/run_all.py --quick            # skip the slow suites
    python tests/run_all.py --json scores/tests_base.json

Each suite is a standalone script that prints `HARD FAILS: n -> [...]` and exits
nonzero on failure. This runner starts them as parallel SUBPROCESSES -- not
imports -- because several of them mutate os.environ and module-level agent
state, and one suite leaking a kill switch into the next is a class of false
green this project cannot afford.

Two environment rules, both load-bearing:

  PYTHONHASHSEED=0   MyAgent seeds its RNG from ARC_AGENT_SEED + hash(gid).
                     Without this, a suite that builds an agent is not
                     reproducible and a flake reads as a regression.
  BLAS threads = 1   Set BEFORE numpy/torch import, in the parent, so children
                     inherit it. bench.py:57 has the measurement: 8 workers each
                     opening a thread per core put ~96 threads on 12 cores, a
                     212s game took 900s+, and 8 of 9 cells were dropped as
                     invalid. The suites are small numpy on small grids.

WARNINGS ARE LISTED BY NAME, never summed away. Two are known-good and expected:

    genuine_coordinate_that_moves_every_step_reads_as_a_clock   (test_goalmodel)
    every_step_blinker_reads_as_hud                             (test_stategraph)

Both are the HUD heuristics correctly refusing to distinguish two things that a
single frame cannot distinguish. A THIRD warning appearing is news.
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "TORCH_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import re
import sys
import glob
import json
import time
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Suites that take minutes rather than seconds. --quick skips them; they are
# still the default, because "quick" is a convenience and green is not.
SLOW = {"test_replay.py", "test_goalmodel.py", "test_components.py"}

KNOWN_WARNINGS = {
    "genuine_coordinate_that_moves_every_step_reads_as_a_clock",
    "every_step_blinker_reads_as_hud",
}

_FAILS = re.compile(r"HARD FAILS:\s*(\d+)")
_WARNS = re.compile(r"WARNINGS\s*:\s*(\d+)\s*->\s*(\[.*\])")
_CHECKS = re.compile(r"CHECKS:\s*(\d+)")
_LIST = re.compile(r"HARD FAILS:\s*\d+\s*->\s*(\[.*\])")


def _names(blob):
    """['a', 'b'] as printed by the suites -> a python list, tolerantly."""
    try:
        return [str(x) for x in json.loads((blob or "[]").replace("'", '"'))]
    except Exception:
        return [s for s in re.findall(r"['\"]([^'\"]+)['\"]", blob or "")]


def run_suite(path, timeout):
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = "0"
    env["PYTHONIOENCODING"] = "utf-8"
    # A stray kill switch inherited from the caller's shell would silently
    # change what these suites test. Start every one from a clean slate.
    for k in [k for k in env if k.startswith("ARC_") and k != "ARC_DATA_ROOT"]:
        env.pop(k, None)
    t0 = time.time()
    try:
        p = subprocess.run([sys.executable, "-u", path], cwd=ROOT, env=env,
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=timeout)
        out, code = (p.stdout or "") + (p.stderr or ""), p.returncode
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "") if isinstance(e.stdout, str) else ""
        out += f"\n** TIMEOUT after {timeout}s **"
        code = 124
    secs = time.time() - t0

    m = _FAILS.search(out)
    fails = int(m.group(1)) if m else (0 if code == 0 else 1)
    fail_names = _names(_LIST.search(out).group(1)) if _LIST.search(out) else []
    mw = _WARNS.search(out)
    warns = _names(mw.group(2)) if mw else []
    mc = _CHECKS.search(out)
    checks = int(mc.group(1)) if mc else out.count("  PASS ") + out.count("  ok ")
    return {"suite": os.path.basename(path), "exit": code, "secs": round(secs, 1),
            "checks": checks, "fails": fails, "fail_names": fail_names,
            "warns": warns, "output": out}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", action="append", default=[],
                    help="substring of a suite name (repeatable)")
    ap.add_argument("--quick", action="store_true", help=f"skip {sorted(SLOW)}")
    ap.add_argument("--jobs", type=int, default=0, help="parallel suites (default: auto)")
    ap.add_argument("--timeout", type=float, default=900.0, help="per suite, seconds")
    ap.add_argument("--json", default="", help="write the table here")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="print every suite's full output, not just failures")
    args = ap.parse_args()

    suites = sorted(glob.glob(os.path.join(HERE, "**", "test_*.py"), recursive=True))
    if args.only:
        suites = [s for s in suites
                  if any(o in os.path.basename(s) for o in args.only)]
    if args.quick:
        suites = [s for s in suites if os.path.basename(s) not in SLOW]
    if not suites:
        print("no suites selected")
        return 1

    jobs = args.jobs or max(1, min(8, (os.cpu_count() or 2) - 1))
    print(f"run_all: {len(suites)} suite(s), {jobs} parallel, "
          f"PYTHONHASHSEED=0, BLAS pinned to 1 thread")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        rows = list(pool.map(lambda s: run_suite(s, args.timeout), suites))
    rows.sort(key=lambda r: r["suite"])

    print(f"\n  {'suite':<26}{'checks':>8}{'fails':>7}{'warns':>7}{'secs':>8}  status")
    for r in rows:
        status = ("OK" if r["exit"] == 0 and not r["fails"]
                  else ("TIMEOUT" if r["exit"] == 124 else "FAIL"))
        print(f"  {r['suite']:<26}{r['checks']:>8}{r['fails']:>7}"
              f"{len(r['warns']):>7}{r['secs']:>8.1f}  {status}")

    hard = [r for r in rows if r["exit"] != 0 or r["fails"]]
    all_warns = [(r["suite"], w) for r in rows for w in r["warns"]]
    new_warns = [(s, w) for s, w in all_warns if w not in KNOWN_WARNINGS]

    if all_warns:
        print(f"\n  WARNINGS ({len(all_warns)}) -- listed, never aggregated away:")
        for s, w in all_warns:
            print(f"    {'KNOWN' if w in KNOWN_WARNINGS else '** NEW **'}  "
                  f"{s}: {w}")
    if new_warns:
        print(f"\n  ** {len(new_warns)} warning(s) this runner has not seen before. "
              f"A new warning is news: read it before trusting the green above. **")

    if hard:
        print(f"\n{'=' * 74}")
        for r in hard:
            print(f"FAILED {r['suite']} (exit {r['exit']}, "
                  f"{r['fails']} hard fail(s)): {r['fail_names']}")
            print("\n".join(r["output"].splitlines()[-40:]))
            print("-" * 74)
    if args.verbose:
        for r in rows:
            print(f"\n{'=' * 74}\n{r['suite']}\n{'=' * 74}\n{r['output']}")

    total_checks = sum(r["checks"] for r in rows)
    print(f"\n  {len(rows) - len(hard)}/{len(rows)} suites green   "
          f"{total_checks} checks   {len(all_warns)} warning(s) "
          f"({len(new_warns)} new)   {time.time() - t0:.0f}s")

    if args.json:
        path = args.json if os.path.isabs(args.json) else os.path.join(ROOT, args.json)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"when": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "wall_s": round(time.time() - t0, 1),
                       "green": len(rows) - len(hard), "total": len(rows),
                       "checks": total_checks,
                       "new_warnings": new_warns,
                       "rows": [{k: v for k, v in r.items() if k != "output"}
                                for r in rows]}, f, indent=1)
        print(f"  wrote {path}")
    return 1 if hard else 0


if __name__ == "__main__":
    sys.exit(main())

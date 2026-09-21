# --- path shim: these suites live in tests/ but import the agent from the repo root ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, 'eval')):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end shim ---
"""test_components.py -- the registry is true, and no kill switch is a decoration.

`eval/components.py` claims, for every component, a class, a line, a unit suite
and a kill switch. A registry that has drifted from the code is worse than none:
it is a map that says the territory is covered. This suite is what makes the
claim checkable.

Four properties, in the order they can fail:

  1. NO ORPHAN SWITCHES. Every ARC_* env var my_agent.py actually READS is either
     a registered component or a registered non-component (config/diagnostic).
     A switch nobody registered is a switch nobody sweeps.
  2. NO PHANTOM ROWS. Every registered switch is still read somewhere in
     my_agent.py, and every named suite exists on disk. Rows that name a deleted
     switch are how a registry rots quietly.
  3. NO SILENT COVERAGE HOLE. A component either names a suite or says
     `no-unit-coverage` with a reason. "Nothing here" must be a decision.
  4. NO INERT SWITCH. On a scripted trace, a switch the registry declares
     `local="moves"` must change the trajectory. This is the direct guard against
     the two the repo has already shipped: `bucket()` ordered an exploration
     frontier with an absolute 0.125 grid against a real spread of 0.036, so it
     ordered nothing; MCTS took 50.5% of think time for zero reward. Both looked
     alive from the outside.

Property 4 turns entirely on one distinction: "the switch did nothing" versus
"the trace never reached what it gates". The first draft of this suite used
"was the env var READ?" as the proxy and reported nine defects, all false -- a
switch is read when its owner is CONSTRUCTED, which says nothing about whether
the gated behaviour was reachable in 18 actions. ProgressModel returns a constant
until its first win; ReplayCache needs a death; `_OBJ_ON` is read at import and
cannot be flipped in-process at all.

So the claim now lives in the registry, per row, as `local=` (see
eval/components.py for the vocabulary). `moves` is the default and the only one
that can hard-fail. Everything else is reported as a WARNING NAMING THE SWITCH
and the reason -- untested, not proven fine, and explicitly bench territory. The
trace still instruments `os.environ.get`, but now as evidence in the message
rather than as the verdict.

Run:  PYTHONIOENCODING=utf-8 PYTHONHASHSEED=0 python tests/test_components.py
Never delete a case.
"""
import os
import sys
import glob
import json
import time
import random

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

FAILS = []
WARNS = []
CHECKS = [0]


def check(name, got, want):
    CHECKS[0] += 1
    if got != want:
        FAILS.append(f"{name}: got {got!r}, want {want!r}")
        print(f"  FAIL {name}: got {got!r}, want {want!r}")
    else:
        print(f"  ok   {name}")


def ok(name, cond, detail=""):
    CHECKS[0] += 1
    if not cond:
        FAILS.append(f"{name}{': ' + detail if detail else ''}")
        print(f"  FAIL {name}  {detail}")
    else:
        print(f"  ok   {name}  {detail}" if detail else f"  ok   {name}")


def warn(name, cond, detail=""):
    CHECKS[0] += 1
    if not cond:
        WARNS.append(name)
        print(f"  WARN {name}  {detail}")
    else:
        print(f"  ok   {name}")


import components as R                                     # noqa: E402

# ============================================================================
print("\n--- 1. the registry covers every switch my_agent.py reads -----------")
# ============================================================================
found = R.scan_switches()
reg = R.by_switch()
print(f"  {len(R.COMPONENTS)} components; {len(reg)} switches registered; "
      f"{len(found)} read in my_agent.py")

orphans = sorted(s for s in found
                 if s not in reg and s not in R.NON_COMPONENT_SWITCHES)
ok("no_orphan_switches", not orphans,
   f"unregistered: {orphans}" if orphans else "")

# The mirror image: a row naming a switch the agent no longer reads. Harmless at
# runtime and lethal to the sweep, which would silently measure nothing.
phantom = sorted(s for s in reg if s not in found)
ok("no_phantom_switches", not phantom,
   f"registered but never read in my_agent.py: {phantom}" if phantom else "")

# ============================================================================
print("\n--- 2. every component names a suite, or says why it does not -------")
# ============================================================================
missing_suite = [c["name"] for c in R.COMPONENTS if not c["suite"]]
ok("every_component_states_its_coverage", not missing_suite,
   f"no suite and no reason: {missing_suite}" if missing_suite else
   "a blank coverage field is an accident; 'no-unit-coverage' is a decision")

absent = [(c["name"], c["suite"]) for c in R.COMPONENTS
          if c["suite"] and c["suite"].endswith(".py")
          and not os.path.isfile(R.suite_path(c["suite"]))]
ok("named_suites_exist", not absent, f"named but absent: {absent}" if absent else "")

uncovered = [c["name"] for c in R.COMPONENTS
             if c["suite"] and not c["suite"].endswith(".py")]
print(f"  {len(uncovered)} component(s) explicitly uncovered: {uncovered}")
ok("uncovered_rows_carry_a_reason",
   all(("gated" in c["suite"] or ":" in c["suite"])
       for c in R.COMPONENTS if c["suite"] and not c["suite"].endswith(".py")),
   "each no-unit-coverage row must say why")

# The class each row points at must exist, and the line must be close to it --
# a stale line number turns the registry into a treasure map with no X.
import re                                                  # noqa: E402
with open(R.AGENT, encoding="utf-8") as f:
    AGENT_LINES = f.readlines()
class_at = {}
for i, ln in enumerate(AGENT_LINES, 1):
    m = re.match(r"class\s+([A-Za-z_][A-Za-z0-9_]*)", ln)
    if m:
        class_at.setdefault(m.group(1), i)
gone = sorted({c["cls"] for c in R.COMPONENTS} - set(class_at))
ok("registered_classes_exist", not gone, f"no such class: {gone}" if gone else "")
drifted = [(c["name"], c["line"], class_at.get(c["cls"]))
           for c in R.COMPONENTS
           if c["cls"] in class_at and not (class_at[c["cls"]] - 5 <= c["line"]
                                            <= class_at[c["cls"]] + 900)]
warn("registry_line_numbers_are_current", not drifted, f"{drifted[:6]}")

# ============================================================================
print("\n--- 3. the sweep arms are the switches worth sweeping ---------------")
# ============================================================================
sweep = R.sweepable()
print(f"  {len(sweep)} sweepable: {[c['switch'] for c in sweep]}")
ok("sweep_excludes_llm_only",
   not [c for c in sweep if c["status"] == "llm-only"],
   "llm-only rows cannot move a local score; sweeping them manufactures "
   "NO-OP verdicts that mean nothing")
ok("every_sweepable_row_has_a_switch", all(c["switch"] for c in sweep))
ok("statuses_are_from_the_vocabulary",
   all(c["status"] in ("live", "opt-in", "llm-only", "inert-by-design",
                       "diagnostic", "config") for c in R.COMPONENTS),
   sorted({c["status"] for c in R.COMPONENTS}))

# ============================================================================
print("\n--- 4. no switch is inert: the scripted trace -----------------------")
# ============================================================================
# One short trace per game, replayed once per switch. Short on purpose: this is
# a unit suite, not a bench run. It cannot and does not measure whether a switch
# HELPS -- only whether flipping it changes anything at all. "Does it help?" is
# `bench.py --sweep`, which needs seeds this suite has no budget for.
TRACE_ACTIONS = int(os.environ.get("COMPTEST_ACTIONS", 18))
TRACE_GAMES = ["lp85", "tr87"]      # a click-only game and a mechanical one

_QUERIED = set()
_real_get = os.environ.get


def _spy_get(key, *a, **kw):
    if isinstance(key, str) and key.startswith("ARC_"):
        _QUERIED.add(key)
    return _real_get(key, *a, **kw)


def _game_file(gid):
    hits = sorted(glob.glob(os.path.join(_ROOT, "eval", "real_games", gid,
                                         "*", f"{gid}.py")))
    return hits[0] if hits else None


def trace(gid, gfile, env_overrides, spy=False):
    """Play TRACE_ACTIONS actions; return the trajectory as a comparable tuple.

    The trajectory -- not the score -- is the observable. A switch that changes
    the score without changing the actions is impossible; a switch that changes
    the actions without changing the score is a real change that a score-only
    check would file as inert. That mistake is how a working component gets
    deleted, so compare the route, and compare it exactly.
    """
    import numpy as np
    from arc_agi.local_wrapper import LocalEnvironmentWrapper
    from scoreboard import make_env_info

    saved = {k: os.environ.get(k) for k in env_overrides}
    os.environ.update({k: v for k, v in env_overrides.items()})
    if spy:
        os.environ.get = _spy_get
    try:
        import my_agent
        info, _ = make_env_info(gfile, gid, {})
        wrapper = LocalEnvironmentWrapper(info, _logger(), seed=0,
                                          scorecard_id="local")
        frame = wrapper.observation_space
        if frame is None:
            return None
        os.environ["ARC_AGENT_SEED"] = "0"
        random.seed(0)
        np.random.seed(0)
        agent = my_agent.MyAgent(game_id=gid)
        frames = [frame]
        out = []
        while (len(out) < TRACE_ACTIONS
               and not agent.is_done(frames, frames[-1])):
            action = agent.choose_action(frames, frames[-1])
            data = {}
            if hasattr(action, "is_complex") and action.is_complex():
                data = {"x": int(action.action_data.x),
                        "y": int(action.action_data.y)}
            out.append((getattr(action, "name", str(action)),
                        data.get("x"), data.get("y"),
                        str(getattr(agent, "_route", "") or "?")))
            resp = wrapper.step(action, data=data)
            if resp is None:
                break
            frames.append(resp)
            agent.action_counter += 1
        return tuple(out)
    finally:
        if spy:
            os.environ.get = _real_get
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _logger():
    import logging
    lg = logging.getLogger("comptest")
    lg.addHandler(logging.NullHandler())
    lg.setLevel(logging.CRITICAL)
    lg.propagate = False
    return lg


files = {g: _game_file(g) for g in TRACE_GAMES}
missing = [g for g, p in files.items() if not p]
if missing:
    warn("trace_games_available", False,
         f"{missing} not under eval/real_games -- property 4 not checked")
else:
    # Clear every switch first: a stray one in the caller's shell would make the
    # control arm something other than the default build.
    for s in list(reg):
        os.environ.pop(s, None)

    control, per_game_queried = {}, {}
    for g in TRACE_GAMES:
        _QUERIED.clear()
        t0 = time.time()
        control[g] = trace(g, files[g], {}, spy=True)
        per_game_queried[g] = set(_QUERIED)
        ok(f"control_trace_{g}", bool(control[g]),
           f"{len(control[g] or ())} actions in {time.time() - t0:.0f}s; "
           f"{len(per_game_queried[g])} ARC_* keys queried")

    queried = set().union(*per_game_queried.values())
    inert, deferred, changed_by = [], [], {}
    for c in sweep:
        s, want = c["switch"], c["on"]
        moved = []
        for g in TRACE_GAMES:
            if control.get(g) is None:
                continue
            t = trace(g, files[g], {s: want})
            if t != control[g]:
                moved.append(g)
        changed_by[s] = moved
        if moved:
            print(f"  ok   switch_changes_behaviour {s:<20} -> {moved}")
            CHECKS[0] += 1
        elif c["local"] == "moves":
            inert.append(s)
        else:
            deferred.append(s)

    ok("no_inert_switch", not inert,
       f"declared local='moves' and changed NOTHING: {inert} -- this is the "
       f"bucket() failure mode: a switch that is consulted and ignored. Either "
       f"the component is dead or the row's `local` is wrong; both are findings."
       if inert else f"{sum(1 for v in changed_by.values() if v)} of "
                     f"{len(sweep)} switches moved the trajectory; "
                     f"{len(deferred)} deferred to the bench by declaration")

    # A row that claims it MOVES and does is settled here. Everything else is
    # named, with its declared reason, so nothing is quietly exempt: a switch the
    # unit suite cannot reach is a switch only `bench.py --sweep` can judge.
    print(f"\n  deferred to bench.py --sweep ({len(deferred)}):")
    for s in sorted(deferred):
        c = reg[s]
        print(f"    {s:<20} local={c['local']:<12} {c['name']} [{c['status']}] "
              f"-- {'read' if s in queried else 'never read'} during the trace")

    # And the partition itself is the assertion. A `moves` row that stops moving
    # lands in `inert` above; a deferred row that STARTS moving shrinks this set,
    # which means the registry was understating the switch and should say so.
    # Pinning the partition catches drift in both directions with no standing
    # warning noise -- the two known warnings stay the only two.
    declared = sorted(c["switch"] for c in sweep if c["local"] != "moves")
    check("deferred_set_matches_the_registry_declaration",
          sorted(deferred), declared)
    ok("local_values_are_from_the_vocabulary",
       all(c["local"] in ("moves", "equivalence", "needs-win", "needs-death",
                          "import-time", "deep") for c in R.COMPONENTS),
       sorted({c["local"] for c in R.COMPONENTS}))

    # The two rows the registry says are provably no-ops offline must in fact be
    # no-ops offline, or the claim in components.py is the thing that is wrong.
    for c in R.COMPONENTS:
        if c["status"] == "llm-only" and c["switch"]:
            t = trace(TRACE_GAMES[0], files[TRACE_GAMES[0]],
                      {c["switch"]: c["on"]})
            ok(f"llm_only_is_a_local_noop__{c['switch']}",
               t == control[TRACE_GAMES[0]],
               "with no LLM loaded this must not move the trajectory; if it "
               "does, the row is mislabelled and the sweep will chase it")

# ============================================================================
print("\n--- 5. the two scorer copies cannot drift --------------------------")
# ============================================================================
# Full equality of the read-outs is tests/test_scorecard.py case 7. Here we pin
# the cheaper invariant that catches the common edit: the same public surface.
try:
    import scoreboard as SB
    import Kaggle_test as KT
    a = {n for n in dir(SB.OfficialScorecard) if not n.startswith("_")}
    b = {n for n in dir(KT.OfficialScorecard) if not n.startswith("_")}
    ok("scorecard_copies_have_the_same_surface", a - {"wrapper_kwargs"} == b,
       f"eval-only {sorted(a - b)}, kaggle-only {sorted(b - a)} "
       f"(wrapper_kwargs is bench-side wiring and is expected to be eval-only)")
    ok("kaggle_copy_does_not_import_eval",
       "from scoreboard" not in open(os.path.join(_ROOT, "Kaggle_test.py"),
                                     encoding="utf-8").read(),
       "there is no eval/ directory on Kaggle; the copy must stay standalone")
except Exception as e:
    warn("scorer_copies_importable", False, f"{type(e).__name__}: {e}")

print("\n================ SUMMARY ================")
print(f"CHECKS: {CHECKS[0]}   HARD FAILS: {len(FAILS)}")
print(f"HARD FAILS: {len(FAILS)} -> {FAILS}")
print(f"WARNINGS  : {len(WARNS)} -> {WARNS}")
sys.exit(1 if FAILS else 0)

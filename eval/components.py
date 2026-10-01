"""The registry: component -> kill switch -> unit suite -> bench ablation.

One table, so three questions have one answer each:

  * "what covers this class?"          -> `suite`
  * "how do I turn it off?"            -> `switch`
  * "what does it earn?"               -> `ablation`, run by `bench.py --sweep`

`tests/test_components.py` asserts against my_agent.py on every run: no switch
exists that is not registered, no component claims a suite that does not exist,
and no switch is silently inert. That last one is not hypothetical -- this repo
has shipped two of them. `bucket()` spent months ordering an exploration frontier
with an absolute 0.125 grid against a real spread of 0.036, so it ordered
nothing; MCTS took 50.5% of think time for zero rewards. Both looked alive from
the outside.

STATUS vocabulary
-----------------
  live             default ON; ablating it must change something locally
  opt-in           default OFF; the ablation turns it ON
  llm-only         only observable with the LLM loaded. Verified inert locally
                   BY DESIGN and exercised on Kaggle only -- so a NO-OP verdict
                   here is the expected result, not a bucket()-style defect.
  inert-by-design  intentionally a no-op, here and on Kaggle
  diagnostic       dumps/tags; never scored, never swept
  config           a tunable number, not a kill switch

LOCAL vocabulary -- what a SHORT scripted trace may conclude
-----------------------------------------------------------
`tests/test_components.py` plays ~18 actions and flips each switch. Whether an
identical trajectory is a defect depends entirely on the switch, so each row
declares it. Reading the env var proves the line executed; it does NOT prove the
behaviour it gates was reachable, and conflating the two turns the check into
noise loud enough that someone eventually deletes it.

  moves            must change a short local trace. Identical trajectory = HARD
                   FAIL. This is the bucket() guard, and it is the default.
  equivalence      a cache: identical output IS the success criterion (fitmemo).
  needs-win        provably constant until a level is completed -- ProgressModel
                   returns a constant before its first win, and GoalModel's alpha
                   channel cannot fill without a level-up. 18 actions never
                   completes a level, so a local NO-OP here says nothing.
  needs-death      only reachable once a death re-issues a banked route.
  import-time      read while my_agent.py is imported (a class attribute), so an
                   in-process flip cannot reach it at all. Saying so beats
                   pretending to test it.
  deep             gates machinery a short trace does not drive far enough to
                   distinguish (synthesis, model promotion, patch rules).

Everything that is not `moves` is measured by `bench.py --sweep`, not here, and
the suite names each one in a warning so the gap stays visible.
"""
from __future__ import annotations

import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
AGENT = os.path.join(ROOT, "sarbloh", "legacy", "my_agent.py")
TESTS = os.path.join(ROOT, "tests", "component")

# The gate for D2/D3/D4 in Docs/feedback_loop_plan.txt: a suite is written when a
# planned change implicates the component, not because the row is empty. Coverage
# for something nobody is about to touch is a detour wearing a test costume.
GATED = "no-unit-coverage (gated: write when a change implicates it)"


def C(name, cls, line, suite=None, switch=None, on="1", status="live",
      note="", route=None, local="moves"):
    """One row. `on` is the value the ablation sets the switch to."""
    return {"name": name, "cls": cls, "line": line, "suite": suite,
            "switch": switch, "on": on, "status": status, "note": note,
            "route": route, "local": local}


# ---------------------------------------------------------------- the table
COMPONENTS = [
    # --- perception ------------------------------------------------------
    C("StateEncoder", "StateEncoder", 321, "test_eyes.py",
      note="objects, panels, HUD; the 'phantom objects' were real panels"),
    C("GridObject", "GridObject", 262, "test_eyes.py"),
    C("GridStats", "GridStats", 5798, "test_goalmodel.py"),
    C("Readout", "Readout", 5885, "test_goalmodel.py"),

    # --- world models ----------------------------------------------------
    C("Transition", "Transition", 215, "test_synth.py"),
    C("WorldModel", "WorldModel", 254, "test_synth.py"),
    C("GridDSL", "GridDSL", 529, "test_griddsl.py",
      note="vacate closed the vocabulary gap; 204/204"),
    C("EnumerativeSynthesizer", "EnumerativeSynthesizer", 1351, "test_synth.py"),
    C("  .fit-memo", "EnumerativeSynthesizer", 1366, "test_synth.py",
      switch="ARC_NO_FITMEMO", local="equivalence",
      note="an equivalence cache: 85/85 identical pairs is the SUCCESS "
           "criterion, so NO-OP is expected. Speed comes from the profiler, "
           "never from an A/B."),
    C("  .stride", "EnumerativeSynthesizer", 1614, "test_griddsl.py",
      switch="ARC_NO_STRIDE", local="deep",
      note="observed shift candidates; fixed +-1/+-2 fit 0/1535 transitions"),
    C("WorldModelManager", "WorldModelManager", 2242, "test_synth.py"),
    C("CandidateVerifier", "CandidateVerifier", 1194, "test_verifier.py",
      switch="ARC_NO_VERIFIER", local="deep",
      note="ablation restores the last-5 gate that promoted recency-overfit "
           "models at 1.00 and demoted them at 0.34"),
    C("Verdict", "Verdict", 1173, "test_verifier.py"),
    C("PatchWorldModel", "PatchWorldModel", 5394, GATED, switch="ARC_NO_PWM", local="deep",
      note="learned 5x5 rules; prunes certified no-ops from the frontier"),

    # --- planners --------------------------------------------------------
    C("ReactivePlanner", "ReactivePlanner", 2651, "test_goalmodel.py",
      route="reactive", note="movement games; toys keydoor 2/2, reachgoal 3/3"),
    C("  .mech-gate", "ReactivePlanner", 2844, "test_goalmodel.py",
      switch="ARC_NO_MECH_GATE",
      note="looks_mechanical(): unexplained-ratio OR displacement sign-flips"),
    C("ExecPlanner", "ExecPlanner", 3094, "test_execplanner.py",
      route="exec", note="RHAE-critical: plan UNSCORED through a verified model"),
    C("  .certify", "ExecPlanner", 3279, "test_execplanner.py",
      switch="ARC_NO_CERT", note="ablation makes every premise read green"),
    C("ClickPlanner", "ClickPlanner", 3981, "test_clickplanner.py",
      route="click", note="10 of 25 games are available_actions=[6]"),
    C("MCTSPlanner", "MCTSPlanner", 2548, GATED, switch="ARC_MCTS", local="deep",
      status="opt-in",
      note="50.5% of think time for 0 rewards; removal cost no levels"),
    C("MCTSNode", "MCTSNode", 2520, GATED, status="opt-in"),
    C("StateCounter", "StateCounter", 2614, GATED),
    C("ExploitGate", "ExploitGate", 2493, GATED),

    # --- graph / exploration --------------------------------------------
    C("StateGraph", "StateGraph", 5014, "test_stategraph.py",
      switch="ARC_NO_GRAPH", route="graph",
      note="exact per-level transition graph + periodicity HUD mask"),
    C("DeadActionTracker", "DeadActionTracker", 4948, "test_stategraph.py"),
    C("LivelockBreaker", "LivelockBreaker", 5521, "test_livelock.py",
      switch="ARC_NO_BREAKER", route="breaker"),
    C("  .least-spent", "MyAgent", 7532, GATED, switch="ARC_LEASTSPENT", local="deep",
      status="opt-in", note="did not prove out; kept opt-in"),

    # --- progress / goals ------------------------------------------------
    C("ProgressModel", "ProgressModel", 3661, "test_progress.py"),
    C("  .obj-features", "ProgressModel", 3747, "test_progress.py",
      switch="ARC_NO_OBJFEAT", local="import-time", note="C1 kill switch"),
    C("O3 ordering", "ClickPlanner", 4099, "test_progress.py",
      switch="ARC_NO_O3", local="needs-win",
      note="bucket() was an absolute 0.125 grid against a 0.036 spread -- it "
           "ordered nothing. The identical trace WAS the evidence."),
    C("  .o3-prior", "ClickPlanner", 4100, "test_progress.py",
      switch="ARC_O3_PRIOR", local="deep", status="opt-in"),
    C("  .death-fit", "ClickPlanner", 4101, "test_progress.py",
      switch="ARC_NO_DEATH", local="needs-win",
      note="deaths are the control group; fit_death subtracts only"),
    C("GoalModel", "GoalModel", 6043, "test_goalmodel.py", switch="ARC_NO_GOALS", local="needs-win",
      route="goals",
      note="alpha is structurally unreachable without a level-up"),
    C("GoalPredicate", "GoalPredicate", 5970, "test_goalmodel.py"),
    C("Feature", "Feature", 5995, "test_goalmodel.py"),
    C("Timeline", "Timeline", 5696, "test_goalmodel.py",
      note="append-only ground truth; the np.False_ trap lives here"),
    C("TimelineEntry", "TimelineEntry", 5659, "test_goalmodel.py"),

    # --- memory / replay -------------------------------------------------
    C("ReplayCache", "ReplayCache", 6831, "test_replay.py",
      switch="ARC_NO_REPLAY", local="needs-death", route="replay",
      note="re-issue a banked route after a death; one wasted action worst case"),
    C("MemoryManager", "MemoryManager", 2160, GATED),

    # --- LLM (Kaggle only) ----------------------------------------------
    C("LocalLLM", "LocalLLM", 1812, "test_llm_repair.py", status="llm-only",
      note="27B Qwen cannot load on the dev box; the READER contract "
           "(extract_candidate + repair loop) is what is testable locally"),
    C("LLMProposalStats", "LLMProposalStats", 1107, "test_llm_repair.py",
      status="llm-only"),
    C("theorizer", "MyAgent", 7857, GATED, switch="ARC_NO_THEORIZE",
      status="llm-only",
      note="Phase-2 budgeted Qwen goal hints. Provably no-op with no LLM "
           "loaded, so a local NO-OP verdict confirms the design."),

    # --- the ladder ------------------------------------------------------
    C("MyAgent routing", "MyAgent", 7003, GATED,
      note="replay -> ExecPlanner -> graph -> livelock precedence; "
           "attribution via _route/_route_ms"),
]

# Env vars that are configuration or diagnostics, not components. Registered so
# the orphan-switch check has somewhere to put them instead of failing.
NON_COMPONENT_SWITCHES = {
    "ARC_LLM_PATH": "config: LLM weights dir (Kaggle dataset path)",
    "ARC_EMBED_PATH": "config: embedder weights dir",
    "ARC_LLM_CALLS_PER_LEVEL": "config: LLM budget per level",
    "ARC_LLM_MAX_TOKENS": "config: generation cap",
    "ARC_LLM_GEN_MAX_S": "config: generation wall cap",
    "ARC_LLM_REPAIR_ROUNDS": "config: verifier-driven repair rounds",
    "ARC_LLM_SESSION_BUDGET_S": "config: whole-session LLM budget",
    "ARC_SYNTH_INFLIGHT_S": "config: synth watchdog",
    "ARC_CONTEXT_TOKENS": "config: context budget",
    "ARC_O3_DUMP": "diagnostic: dump raw episode boards to a dir",
    "ARC_O3_DUMP_TAG": "diagnostic: label for the dump",
    "ARC_DATA_ROOT": "config: Kaggle dataset root",
    "ARC_AGENT_SEED": "harness: set by bench.py per cell",
}


# ---------------------------------------------------------------- helpers
def by_switch() -> dict:
    return {c["switch"]: c for c in COMPONENTS if c["switch"]}


def sweepable() -> list:
    """Rows bench.py --sweep turns into arms.

    `llm-only`, `diagnostic` and `config` rows are excluded: the sweep asks "did
    the score move?", and for those the honest answer is known in advance and is
    not a defect.
    """
    return [c for c in COMPONENTS
            if c["switch"] and c["status"] in ("live", "opt-in")]


def scan_switches(path=AGENT) -> dict:
    """Every ARC_* env var my_agent.py actually READS -> [(line, expr)].

    Reads, not mentions: a switch named only in a comment is documentation, and
    counting it as a switch is how the registry fills up with rows that cannot
    be ablated.
    """
    import re
    out = {}
    pat = re.compile(r'(?:os\.environ\.get|os\.getenv|_envf)\(\s*["\'](ARC_[A-Z0-9_]+)["\']')
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            for m in pat.finditer(line):
                out.setdefault(m.group(1), []).append((i, line.strip()[:90]))
    return out


def suite_path(name):
    return os.path.join(TESTS, name) if name and name.endswith(".py") else None


if __name__ == "__main__":
    found = scan_switches()
    reg = by_switch()
    print(f"{len(COMPONENTS)} components, {len(reg)} switches registered, "
          f"{len(found)} read in my_agent.py")
    for s in sorted(found):
        c = reg.get(s)
        tag = (f"{c['name']} [{c['status']}]" if c
               else NON_COMPONENT_SWITCHES.get(s, "** UNREGISTERED **"))
        print(f"  {s:<26} {tag}")
    miss = [c["name"] for c in COMPONENTS
            if c["suite"] and c["suite"].endswith(".py")
            and not os.path.isfile(suite_path(c["suite"]))]
    if miss:
        print(f"  ** suites named but absent: {sorted(set(miss))}")

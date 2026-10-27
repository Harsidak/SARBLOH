# --- path shim: these suites live in tests/ but import from the repo root ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_PRIME = _os.environ.get("PRIME_DIR", "Sarbloh-Experimentation")   # these changes are in the experimentation copy only
for _p in (_ROOT, _os.path.join(_ROOT, _PRIME)):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end shim ---
"""The code graph, ``recall()`` and SKILLS in the kernel, and the curator that feeds them (Sarbloh-Experimentation).
It replaces the suite of the old ``hide_code`` stubs, which the code graph replaced.
- config and prompt: one tool (ipython), ``code_graph`` on, ``hide_code`` gone, the prompt explains ``recall()``,
  ``graph()``, SKILLS and the graph lines, and has no recall tool and no open questions;
- in a real kernel: ``graph()`` and the change lines follow the live namespace: a new, changed, deleted or rebound
  function, a class with methods, a decorated function, and a call to a name that no longer exists; ``show_source``
  still reads a function back;
- ``recall()`` in the kernel: the timeline by step, the skills by name (loaded into SKILLS, at most five, the oldest
  dropped first), the skill list, a refused scope, and no memory outside the harness;
- working memory: SKILLS is copied like the other variables, a wrong type is refused, a level-up keeps it;
- the curator: no open questions, it reads the loaded skills and the code graph, and picks at most five known skills;
- a real session on ls20 with a scripted model that defines, uses and deletes functions: each output ends with the
  change line, a drain puts the whole map back once, and the trace counts a call to a removed function.

Run: uv run python tests/unit/test_code_graph.py
"""
import contextlib
import io
import json
import linecache
import sys
import tempfile
import threading
import time
from pathlib import Path

from harness import trace
from harness.agent import agent as A, context, curator, prompts
from harness.agent.tools import game_tools
from harness.config import build_config
from harness.llm.client import Reply
from harness.memory.lessons import SkillBook
from harness.memory.lifecycle import GameMemory, LOADED_SKILLS
from harness.memory.timeline import Timeline
from harness.run import fit_context
from harness.runtime.kernel import Kernel
from harness.runtime.skills import codegraph

sys.path.insert(0, str(Path(_ROOT) / _PRIME / "harness" / "runtime" / "skills"))
from source import show_source  # noqa: E402

FAILS, WARNS, CHECKS = [], [], [0]


def ok(name, cond, detail=""):
    CHECKS[0] += 1
    print(("PASS " if cond else "FAIL ") + name + (f"  {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


# --- 1. config and prompt --------------------------------------------------------------------------------------
cfg = build_config({})["agent"]
ok("config", cfg["code_graph"] is True and "hide_code" not in cfg and cfg["memory"]["curator_every_turns"] == 0
   and "questions" not in cfg["memory"]["caps"] and "lessons" not in cfg["memory"]["caps"])
ok("one_tool", [t["function"]["name"] for t in game_tools()] == ["ipython"])
on, off = prompts.game_system(act_max=100), prompts.game_system(act_max=100, code_graph=False)
ok("prompt_kernel_functions", "recall():" in on and "`graph()`" in on and "SKILLS" in on and "[code graph: ...]" in on
   and "recall tool" not in on and "open question" not in on.lower() and "lessons" not in on.lower())
ok("prompt_graph_note_only_when_on", prompts.GRAPH_NOTE.strip() in on and prompts.GRAPH_NOTE.strip() not in off)
vis = prompts.game_system(act_max=100, vision=True)
ok("prompt_picture_last", prompts.PICTURE_LAST.strip() in vis
   and prompts.PICTURE_LAST.strip() not in prompts.game_system(act_max=100, vision=True, picture_last=False))
lu = prompts.event_message("level_up", levels_done=1, win_levels=7, level=2, new_skills=["a_b"], loaded_skills=["a_b"])
ok("level_up_names_skills", "New skills from the level you just won: a_b." in lu and "now in SKILLS: a_b" in lu
   and "lessons" not in lu)

# --- 2. the code graph in a real kernel ---------------------------------------------------------------------------
root = Path(tempfile.mkdtemp())
Timeline(root / "g1" / "timeline.jsonl").append({"kind": "step", "i": 3, "level": 0, "action": "A1(up)",
                                                 "change": "obj 1 moved up"})
k = Kernel(root / "work", lambda req: {}, env={"SARBLOH_MEMORY_DIR": str(root), "SARBLOH_GAME": "g1",
                                               "SARBLOH_RECALL_TOKENS": "1000"})
k.start()
prev: dict = {}


def cell(code):
    global prev
    r = k.execute(code)
    now = k.code_graph()
    line = codegraph.delta(prev, now)
    prev = now
    return r, line


try:
    _, d1 = cell("def step(s, m):\n    '''The next state.'''\n    return s\n"
                 "def bfs(start, goal):\n    return step(start, 1)\n"
                 "def solve(g):\n    return bfs(g, 0)")
    ok("graph_added", d1 == "[code graph: +step(s, m) | +bfs(start, goal) -> step | +solve(g) -> bfs]", d1)
    _, d2 = cell("x = 5")
    ok("graph_quiet_without_change", d2 == "")
    _, d3 = cell("def bfs(start, goal, limit=5):\n    return step(start, 2)")
    ok("graph_changed", d3 == "[code graph: ~bfs(start, goal, limit=5) -> step]", d3)
    _, d4 = cell("del step")
    ok("graph_removed_and_missing", d4 == "[code graph: -step | ! bfs calls missing step]", d4)
    _, d5 = cell("import functools\n@functools.lru_cache(None)\ndef cost(n):\n    return solve(n)\n"
                 "class Model:\n    def run(self):\n        return self.go() + cost(1)\n    def go(self):\n        return 1")
    ok("graph_decorated_and_class", "+cost(n) -> solve" in d5 and "+class Model [go, run] -> cost" in d5, d5)
    r6, _ = cell("graph()")
    ok("graph_tree", r6.stdout.splitlines()[0].startswith("code graph: 3 functions, 1 classes")
       and "class Model [go, run]\n  cost(n)\n    solve(g)\n      bfs(start, goal, limit=5)" in r6.stdout
       and "! bfs calls missing step" in r6.stdout, r6.stdout)
    r7, _ = cell("graph('solve')")
    ok("graph_one", "calls: bfs" in r7.stdout and "called by: cost" in r7.stdout and 'show_source("solve")' in r7.stdout,
       r7.stdout)
    _, d8 = cell("solve = 3")
    ok("graph_rebound_is_removed", d8 == "[code graph: -solve]", d8)
    r9, _ = cell("graph('nope')")
    ok("graph_unknown", "is not a function or class in your kernel now" in r9.stdout)
    r10, _ = cell("show_source('cost')")
    ok("show_source_still_works", "def cost(n):\n    return solve(n)" in r10.stdout, r10.stdout)
    ok("render_full", codegraph.render(prev).startswith("code graph:") and codegraph.render({}).endswith("yet"))
    big = {f"f{i}": {"kind": "function", "sig": f"f{i}()", "calls": [], "missing": [], "hash": "h", "lines": 1,
                     "doc": ""} for i in range(200)}
    ok("delta_capped", len(codegraph.delta({}, big)) <= codegraph.DELTA_CHARS + 60
       and "graph() shows all" in codegraph.delta({}, big))

    # --- 3. recall() and SKILLS in the kernel ------------------------------------------------------------------
    r, _ = cell("recall('#3')")
    ok("recall_timeline", "recall '#3' in all: 1 hit" in r.stdout and "#3 L1 A1(up): " in r.stdout,
       r.stdout)
    r, _ = cell("recall('bfs')\nprint(list(SKILLS))")
    ok("recall_loads_named_skill", "### code_bfs" in r.stdout and r.stdout.strip().endswith("['code_bfs']")
       and any(codegraph and d.get("application/vnd.sarbloh.recall+json") for d in r.displays), r.stdout[-300:])
    r, _ = cell("for q in ['probe', 'scoring', 'frame diff', 'step model', 'flood fill']:\n"
                "    recall(q, 'skills')\nprint(list(SKILLS))")
    names = json.loads(r.stdout.strip().splitlines()[-1].replace("'", '"'))
    ok("skills_hold_five_oldest_dropped", len(names) == LOADED_SKILLS and "code_bfs" not in names
       and names[-1] == "code_flood_fill", str(names))
    r, _ = cell("recall('', 'skills')")
    ok("recall_lists_skills", "recall '' in skills:" in r.stdout and "- code_bfs: breadth-first search." in r.stdout,
       r.stdout[:300])
    r, _ = cell("recall('x', 'hypotheses')")
    ok("recall_refuses_old_scope", "recall refused: scope must be one of ['all', 'timeline', 'skills']" in r.stdout)
    r, _ = cell("import json as _j\nprint(_j.dumps(sorted(_memory_values())))")
    ok("skills_is_a_memory_variable", json.loads(r.stdout.strip()) == ["FINDINGS", "GOAL", "HYPOTHESES", "PLAN",
                                                                       "SKILLS"])
    ok("host_loads_skills", k.load_skills({"my_note": "a note"})
       and k.memory_values()["SKILLS"].get("my_note") == "a note" and len(k.memory_values()["SKILLS"]) == LOADED_SKILLS)
finally:
    k.close()
bare = Kernel(Path(tempfile.mkdtemp()) / "work", lambda req: {})
bare.start()
try:
    ok("recall_without_memory", "this kernel has no game memory" in bare.execute("recall('x')").stdout)
finally:
    bare.close()

# --- 4. working memory ---------------------------------------------------------------------------------------------
mcfg = build_config({})["agent"]["memory"]
m = GameMemory(Path(tempfile.mkdtemp()), "g2", mcfg)
out = m.sync({"SKILLS": {"code_bfs": "What it is: breadth-first search. More."}, "PLAN": {"L1": "go"}}, turn=1, level=0)
ok("sync_skills", out["skills"] == ["code_bfs"] and m.working.skills == {"code_bfs": "What it is: breadth-first search. More."}
   and m.blocks()["skills"] == "- code_bfs: breadth-first search.", str(m.blocks()))
bad = m.sync({"SKILLS": ["code_bfs"]}, turn=2, level=0)
ok("sync_skills_wrong_type", bad["errors"] == ["SKILLS must be a dict of skill name -> text, not list"]
   and m.working.skills == {"code_bfs": "What it is: breadth-first search. More."})
ok("blocks_no_lessons_or_questions", set(m.blocks()) == {"goal", "plan", "skills", "hypotheses", "findings"})
m.on_level_up(level_won=1, step=4, objects=[])
ok("level_up_keeps_skills_and_plan", m.working.skills and m.working.plan_for(0) == "go"
   and "SKILLS" in m.working.kernel_values())

# --- 5. the curator --------------------------------------------------------------------------------------------------
cm = GameMemory(Path(tempfile.mkdtemp()), "g3", mcfg)
cm.sync({"SKILLS": {"code_bfs": "x"}, "HYPOTHESES": {"L1": {"1 moves up": "kept"}}}, turn=1, level=0)
text_in = curator.curator_input(cm, [], "level_up", code_graph="code graph: 1 functions\nbfs(start, goal)")
ok("curator_reads_skills_and_graph", "<loaded_skills>\ncode_bfs\n</loaded_skills>" in text_in
   and "bfs(start, goal)" in text_in and "open_questions" not in text_in)
ok("curator_prompt_no_questions", "open_questions" not in curator.CURATOR_SYSTEM
   and "load_skills: the names of at most 5" in curator.CURATOR_SYSTEM_LEVEL_UP)
rec = curator.apply(cm, {"open_questions": ["ignored"], "skills": [{"name": "push_then_search", "content": "Do it."}],
                         "load_skills": ["push_then_search", "nope", "code_bfs", "code_bfs", "planning",
                                         "world_model", "code_probe", "code_scoring"]}, [], 5)
ok("curator_loads_known_five", rec["loaded"] == ["push_then_search", "code_bfs", "planning", "world_model",
                                                  "code_probe"] and "questions" not in rec, str(rec))
ok("skill_book_has_no_loaded_state", not hasattr(SkillBook(Path(tempfile.mkdtemp()) / "s.json"), "loaded"))


# --- 6. a real session: change lines, the map after a drain, a call to a removed function ----------------------------
CODE = {1: "def step(s, m):\n    return s\ndef bfs(a, b):\n    return step(a, 1)",
        3: "del step",
        5: "print(bfs(1, 2) if 'step' in dir() else step(1, 1))"}   # calls the removed step: a NameError


class ScriptedLLM:
    def __init__(self):
        self.requests, self.turn = [], 0

    def chat(self, messages, tools=None, max_tokens=None, timeout_s=None, template_kwargs=None, thinking=None):
        if tools is None:
            raise AssertionError("no other model call is expected")
        size = sum(len(json.dumps(m)) for m in messages) // 4
        self.requests.append(messages)
        self.turn += 1
        n = self.turn
        code = CODE.get(n) or (f"x{n} = {n}" if n % 2 else "r = await play(['1'])")
        tc = [{"id": f"call{n}", "type": "function", "function": {"name": "ipython",
                                                                  "arguments": json.dumps({"code": code})}}]
        return Reply("", f"thinking {n} " + "z" * 2400, tc, "tool_calls", size, 700)


try:
    from harness.game.arc_host import ArcHost
    from harness.game.games import build_games, make_arcade

    game = build_games(make_arcade("offline", environments_dir=str(Path(_ROOT) / "eval" / "real_games")), "offline",
                       only=["ls20"])[0]
    game.start()
    run = Path(tempfile.mkdtemp())
    # every thinking kept (keep_reasoning on): it fills the context to the drain under test
    c = build_config({"agent": {"vision": False, "limits": {"max_turns": 30}, "memory": {"curator": False},
                                "keep_reasoning": True,
                                "compaction": {"reserve_tokens": 4096, "keep_recent_tokens": 3000,
                                               "trigger_tokens": 16000, "cap_tokens": 24000}}})
    fit_context(c["agent"], 40000)
    llm = ScriptedLLM()
    s = A.AgentSession(cfg=c["agent"], llm=llm, name="ls20", session_dir=run / "games" / "ls20",
                       task=prompts.task_message(game_id=game.game_id, max_actions=400, minutes=60),
                       arc=ArcHost(game, 400, should_stop=lambda: False, tokens_spent=lambda: 0),
                       deadline=time.time() + 3600, stop_event=threading.Event(), memory_root=run / "memory")
    s.run()
    game.finish()
    events = [json.loads(x) for x in (run / "games" / "ls20" / "transcript.jsonl").read_text(encoding="utf-8")
              .splitlines()]
    outs = {e["tool_call_id"]: e["content"] for e in events if e.get("event") == "message" and e.get("role") == "tool"}
    ok("e2e_ran", len(llm.requests) >= 20 and not s.end_reason.startswith("crash") and s.stats["drains"] >= 1,
       f"{len(llm.requests)} {s.end_reason} {s.stats}")
    ok("e2e_change_lines", "[code graph: +step(s, m) | +bfs(a, b) -> step]" in outs["call1"]
       and "[code graph: -step | ! bfs calls missing step]" in outs["call3"] and "[code graph" not in outs["call2"]
       and s.stats["graph_deltas"] == 2, str({k: v[-160:] for k, v in outs.items() if k in ("call1", "call3")}))
    ok("e2e_name_error_kept", "NameError" in outs["call5"])
    graph_msgs = [m for m in s.messages if m.get("_kind") == context.GRAPH_KIND]
    after_drain = next((r for r in llm.requests if any("[code graph] Your functions" in str(m.get("content"))
                                                       for m in r)), None)
    ok("e2e_map_after_drain", len(graph_msgs) == 1 and "bfs(a, b)" in graph_msgs[0]["content"]
       and "! bfs calls missing step" in graph_msgs[0]["content"] and after_drain is not None
       and s.stats["graph_snapshots"] == s.stats["drains"] + s.stats["context_trims"], str(s.stats))
    ok("e2e_drain_masks_graph_lines", after_drain is not None and not any(
        "[code graph: +step" in str(m.get("content")) for m in after_drain[:-6]))
    report = trace.code_graph_report(events)
    ok("trace_calls_to_removed", report["calls_to_removed"] == 1 and report["calls_to_removed_at"][0]["names"] == ["step"]
       and report["removed"] == ["step"] and report["deltas"] == 2 and report["missing_warnings"] == 1, str(report))
except ImportError as e:   # no arc-agi SDK or games in this checkout
    WARNS.append(f"e2e skipped: {e!r}")

print("\n================ SUMMARY ================")
print(f"CHECKS: {CHECKS[0]}")
print(f"HARD FAILS: {len(FAILS)} -> {FAILS}")
print(f"WARNINGS  : {len(WARNS)} -> {WARNS}")
sys.exit(1 if FAILS else 0)

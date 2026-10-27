# --- path shim: these suites live in tests/ but import from the repo root ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_PRIME = _os.environ.get("PRIME_DIR", "Sarbloh")   # PRIME_DIR=Sarbloh-Experimentation tests the copy
for _p in (_ROOT, _os.path.join(_ROOT, _PRIME)):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end shim ---
"""E008 memory without a model: the agent's writes (plan required, hypothesis status changes, all-or-nothing),
the caps (a block over its cap loses its oldest lines, which recall still finds), the level-up lifecycle (promotion
of verified items into the lessons graph with links to the shapes they name, erasure of plan/hypotheses/findings/
questions, goal confirmation), the new-game lifecycle (timeline archived, lessons kept), the shared lessons graph and
skill book, and recall by step, range, level and words.
"""
import sys
import tempfile
from pathlib import Path

from harness.config import DEFAULT
try:
    from harness.agent.intuition import Obj
except ImportError:   # E116 copy: Obj lives in perception.py
    from harness.agent.perception import Obj
from harness.memory.lessons import LessonsGraph, SkillBook, shape_id
from harness.memory.lifecycle import GameMemory
from harness.memory.store import tokens
from harness.memory.working import parse_hypothesis

FAILS, CHECKS = [], [0]


def ok(name, cond, detail=""):
    CHECKS[0] += 1
    print(("PASS " if cond else "FAIL ") + name + (f"  {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


def obj(i, colour, box, h):
    cells = frozenset((r, c) for r in range(box[0], box[2] + 1) for c in range(box[1], box[3] + 1))
    return Obj(i, colour, cells, box, h, 4)


root = Path(tempfile.mkdtemp(prefix="e008_mem_"))
cfg = dict(DEFAULT["agent"]["memory"])
m = GameMemory(root, "ls20", cfg)

# --- 1. parsing ----------------------------------------------------------------------------------------------
ok("parse_string_with_status", parse_hypothesis("h2 verified: A1 moves obj 4 up")
   == {"id": "h2", "text": "A1 moves obj 4 up", "status": "verified", "evidence": []})
ok("parse_plain_string", parse_hypothesis("the bar counts steps")["status"] == "proposed")
ok("parse_dict", parse_hypothesis({"text": "x", "status": "refuted", "evidence": ["#3", 5]})["evidence"] == [3, 5])
try:
    parse_hypothesis({"text": "x", "status": "maybe"})
    ok("parse_rejects_bad_status", False)
except ValueError:
    ok("parse_rejects_bad_status", True)

# --- 2. act writes -------------------------------------------------------------------------------------------
# Sarbloh requires `plan`; Sarbloh-Experimentation makes it optional (a missing plan keeps the old one). A scratch
# memory is used so the checks below start from the same state on both.
m0 = GameMemory(Path(tempfile.mkdtemp(prefix="mem_plan_")), "ls20", cfg)
try:
    w0 = m0.write_act({"hypotheses": ["a"]}, turn=1, level=0)
    m0.write_act({"plan": "keep me"}, turn=2, level=0)
    w1 = m0.write_act({"findings": ["f"]}, turn=3, level=0)
    ok("plan_optional_keeps_old", w0["plan"] is None and [h["text"] for h in m0.working.hypotheses] == ["a"]
       and w1["plan"] is None and m0.working.plan == "keep me", str((w0, w1, m0.working.plan)))
except ValueError:
    ok("plan_required", m0.working.hypotheses == [])
w = m.write_act({"plan": "probe A1", "hypotheses": ["A1 moves obj 1 up", {"text": "obj 2 is a wall"}],
                 "goal": "reach obj 3"}, turn=1, level=0)
ok("write_new_hypotheses", [h["id"] for h in w["hypotheses"]] == ["h1", "h2"] and w["goal"] == "reach obj 3")
w = m.write_act({"plan": "go", "hypotheses": [{"id": "h1", "status": "verified", "evidence": [0, 1]},
                                              "h2 refuted: obj 2 is a wall"]}, turn=2, level=0)
ok("status_changes_logged", [(e["id"], e["prev_status"], e["status"]) for e in w["hypotheses"]]
   == [("h1", "proposed", "verified"), ("h2", "proposed", "refuted")], str(w["hypotheses"]))
ok("goal_unchanged_is_none", w["goal"] is None and m.working.goal == "reach obj 3")
before = [dict(h) for h in m.working.hypotheses]
try:
    m.write_act({"plan": "x", "hypotheses": ["fine one", {"text": "bad", "status": "perhaps"}]}, turn=3, level=0)
    ok("all_or_nothing", False)
except ValueError:
    ok("all_or_nothing", [h["id"] for h in m.working.hypotheses] == [h["id"] for h in before]
       and m.working.plan == "go")
m.write_act({"plan": "go", "findings": ["the HUD bar loses 2 cells per action", "the HUD bar loses 2 cells per action"]},
            turn=3, level=0)
ok("findings_deduped", len(m.working.findings) == 1)

# --- 3. caps -------------------------------------------------------------------------------------------------
for k in range(40):
    m.write_act({"plan": "p", "hypotheses": [f"filler hypothesis number {k} about the long corridor"]}, turn=4 + k,
                level=0)
blocks = m.blocks([])
ok("hypotheses_under_cap", tokens(blocks["hypotheses"]) <= cfg["caps"]["hypotheses"], str(tokens(blocks["hypotheses"])))
ok("hypotheses_trim_oldest", blocks["hypotheses"].startswith("(+") and "number 39" in blocks["hypotheses"]
   and "h1 [verified]" not in blocks["hypotheses"])
m.write_act({"plan": "P" * 5000}, turn=50, level=0)
ok("plan_clipped", tokens(m.blocks([])["plan"]) <= cfg["caps"]["plan"])
ok("skills_default_three_base", m.blocks([])["skills"].count("### ") == 3
   and tokens(m.blocks([])["skills"]) <= cfg["caps"]["skills"])
txt, hits = m.recall("verified", "hypotheses")
ok("recall_finds_trimmed", "h1 [verified] A1 moves obj 1 up" in txt, txt[:300])

# --- 4. timeline and recall ----------------------------------------------------------------------------------
for i in range(6):
    m.record_step({"i": i, "level": 0, "action": "A1", "change": f"obj 1 moved up {i + 1}", "state": "NOT_FINISHED"})
ok("recall_step", "#3 L0 A1: obj 1 moved up 4" in m.recall("#3", "timeline")[0])
ok("recall_range", m.recall("2-4", "timeline")[1] == 3)
ok("recall_level", m.recall("level 0", "timeline")[1] == 6)
ok("recall_words", m.recall("up 6", "timeline")[1] == 1)
ok("recall_cap", tokens(m.recall("", "all", cap_tokens=200)[0]) <= 230)
try:
    m.recall("x", "nowhere")
    ok("recall_bad_scope", False)
except ValueError:
    ok("recall_bad_scope", True)

# --- 5. level up ---------------------------------------------------------------------------------------------
m.write_act({"plan": "win", "findings": [{"text": "obj 3 is the exit", "evidence": [5]}]}, turn=60, level=0)
objs = [obj(1, 8, (5, 5, 6, 6), "aaa111"), obj(3, 14, (1, 1, 2, 2), "bbb222")]
res = m.on_level_up(level_won=1, step=5, objects=objs)
types = sorted(p["type"] for p in res["promoted"])
ok("promote_verified_refuted_findings_goal", types.count("rule") >= 3 and "refuted" in types and "goal" in types,
   str(types))
ok("promoted_linked_to_shapes", any(shape_id("R", "aaa111") in p["shapes"] for p in res["promoted"])
   and any(shape_id("N", "bbb222") in p["shapes"] for p in res["promoted"]), str(res["promoted"][:3]))
ok("level_erases_2_3_5_6", m.working.plan == "" and m.working.hypotheses == [] and m.working.findings == []
   and m.working.questions == [])
ok("goal_kept_and_confirmed", m.working.goal == "reach obj 3" and m.working.goal_log[-1].startswith(
    "confirmed: won level 1"))
ok("goal_block_shows_confirmation", "confirmed: won level 1" in m.blocks([])["goal"])
lessons_now = m.blocks(objs)["lessons"]
ok("lessons_subgraph_for_shapes_on_screen", "A1 moves obj 1 up" in lessons_now
   and tokens(lessons_now) <= cfg["caps"]["lessons"], lessons_now)
ok("recall_after_erase", "(level 0) h1 [verified]" in m.recall("A1", "hypotheses")[0])
ok("hypothesis_ids_continue", m.write_act({"plan": "p", "hypotheses": ["new"]}, turn=61, level=1)["hypotheses"][0]
   ["id"].startswith("h"))

# --- 6. shared graph, new game, lessons switch ---------------------------------------------------------------
m2 = GameMemory(root, "ft09x", cfg)
ok("lessons_shared_across_games", m2.lessons is m.lessons and "A1 moves obj 1 up" in m2.blocks(objs)["lessons"])
ok("working_is_per_game", m2.working.goal == "")
m.on_new_game()
ok("new_game_archives_timeline", m.timeline.rows == [] and any(p.name.startswith("timeline.")
                                                               for p in (root / "ls20").iterdir()))
ok("new_game_keeps_lessons", m.lessons.counts().get("rule", 0) >= 3 and m.working.goal == "")
m3 = GameMemory(root / "c", "g", {**cfg, "lessons": False})
ok("lessons_switch_off", m3.lessons is None and m3.blocks(objs)["lessons"] == "")
g = LessonsGraph(root / "lessons.json")
ok("lessons_persisted", g.counts().get("rule", 0) >= 3 and any(e["type"] == "about" for e in g.edges))
ok("node_dedupe", g.add_node("rule", "A1 moves obj 1 up") == g.add_node("rule", "a1  moves obj 1 up"))

sk = SkillBook(root / "skills.json")
try:
    sk.upsert("planning", "x")
    ok("base_skills_read_only", False)
except ValueError:
    ok("base_skills_read_only", True)
name = sk.upsert("Push Blocks!", "Walk into a block to push it one cell. " * 60, game="ls20")
ok("skill_capped_and_named", name == "push_blocks" and tokens(sk.text(name)) <= 200)
ok("skill_load_max_three_known", sk.set_loaded("ls20", ["push_blocks", "nope", "planning", "world_model",
                                                        "first_principles"]) == ["push_blocks", "planning",
                                                                                 "world_model"])
ok("skill_render_loaded", sk.render("ls20", 600).startswith("### push_blocks"))

print("\n================ SUMMARY ================")
print(f"CHECKS: {CHECKS[0]}")
print(f"HARD FAILS: {len(FAILS)} -> {FAILS}")
sys.exit(1 if FAILS else 0)

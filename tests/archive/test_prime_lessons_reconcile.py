# --- path shim: these suites live in tests/ but import from the repo root ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_PRIME = _os.environ.get("PRIME_DIR", "Sarbloh-Experimentation")   # E039 exists only in the copy so far
for _p in (_ROOT, _os.path.join(_ROOT, _PRIME)):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end shim ---
"""E039 lessons reconcile, without a model (experiments/E039_lessons_reconcile).

After a level won, one hidden pass reads the lessons graph and the game's wrong rulebook (E018) and drops, rewrites
or merges lessons. Checks: the input (this game's lessons first, ids, the wrong rules, the newest review), the graph
operations (edges and games kept), the guards (unseen ids, a lesson of other games cannot be dropped, the change
limit, bad items skipped), an unparseable reply raises, and nothing runs with the switch off.
"""
import json
import sys
import tempfile
from pathlib import Path

from harness.agent import lessons_reconcile
from harness.agent.intuition import Obj
from harness.config import DEFAULT
from harness.memory.lessons import LessonsGraph
from harness.memory.lifecycle import GameMemory

FAILS, CHECKS = [], [0]


def ok(name, cond, detail=""):
    CHECKS[0] += 1
    print(("PASS " if cond else "FAIL ") + name + (f"  {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


def raises(fn, exc=ValueError):
    try:
        fn()
        return False
    except exc:
        return True


class R:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def chat(self, messages, tools=None, max_tokens=None, timeout_s=None, thinking=None):
        self.calls.append(messages)
        return R(self.replies.pop(0))


ON = {**DEFAULT["agent"]["memory"], "wrong_rulebook": True, "lessons_reconcile": True}
red = Obj(4, 8, frozenset({(1, 1), (1, 2), (2, 1), (2, 2)}), (1, 1, 2, 2), "abc123", 4)
root = Path(tempfile.mkdtemp(prefix="e039_"))

# --- 1. graph operations -----------------------------------------------------------------------------------------
g = LessonsGraph(root / "g.json")
r1 = g.add_node("rule", "red square moves up on A1", game="ls20", level=0)
r2 = g.add_node("rule", "A1 moves the red square up", game="tr87", level=2)
s = g.add_shape("R", "abc123", "red square", game="ls20")
g.add_edge(r1, s, "about")
g.add_edge(r2, s, "moves")
g.nodes[r1]["uses"], g.nodes[r2]["uses"] = 3, 2
nid = g.merge([r1, r2], "A1 moves the red square one cell up", source="reconcile")
n = g.nodes[nid]
ok("merge_keeps_games_levels_uses", r1 not in g.nodes and r2 not in g.nodes and n["games"] == ["ls20", "tr87"]
   and n["levels"] == [0, 2] and n["uses"] == 5 and n["source"] == "reconcile", n)
ok("merge_moves_edges", sorted((e["src"], e["dst"], e["type"]) for e in g.edges)
   == sorted([(nid, s, "about"), (nid, s, "moves")]), g.edges)
same = g.merge([nid], "a1 moves the  red square one cell up", source="reconcile")
ok("rewrite_same_text_same_node", same == nid and g.nodes[nid]["uses"] == 5)
goal = g.add_node("goal", "won when all red squares are on blue", game="ls20")
ok("merge_rejects_mixed_types", raises(lambda: g.merge([nid, goal], "x", source="reconcile")))
ok("merge_rejects_shapes", raises(lambda: g.merge([s], "x", source="reconcile")))
g.remove(nid)
ok("remove_drops_edges", nid not in g.nodes and not g.edges and s in g.nodes)
ok("remove_rejects_shape", raises(lambda: g.remove(s)))

# --- 2. the input ------------------------------------------------------------------------------------------------
m = GameMemory(root / "on", "ls20", ON)
other = m.lessons.add_node("rule", "clicking a blue tile wins", game="tr87", level=0)
both = m.lessons.add_node("rule", "yellow bar shrinks on every move", game="tr87", level=0)
m.lessons.add_node("rule", "yellow bar shrinks on every move", game="ls20", level=0)
m.write_act({"plan": "p", "hypotheses": ["h1 proposed: obj 4 moves up on A1", "h2 proposed: obj 4 can pass walls",
                                         "h3 proposed: A5 swaps colours"]}, turn=1, level=0)
m.write_act({"plan": "p", "hypotheses": [{"id": "h1", "status": "verified"}, {"id": "h2", "status": "verified"},
                                         {"id": "h3", "status": "refuted"}]}, turn=2, level=0)
m.on_level_up(level_won=1, step=5, objects=[red])
m.write_act({"plan": "p", "hypotheses": [{"text": "obj 4 can pass walls", "status": "refuted"}]}, turn=3, level=1)
mine = [i for i, x in m.lessons.nodes.items() if x["type"] == "rule" and x["games"] == ["ls20"]]
text, shown = lessons_reconcile.reconcile_input(m, 1, 2000)
ok("input_has_lessons_with_ids", all(i in text for i in mine) and other in text and set(mine) | {other, both} <= shown)
ok("input_this_game_first", text.index("obj 4 can pass walls") < text.index("clicking a blue tile wins"), text)
ok("input_marks_where", "(this game and 1 other game)" in text and "clicking a blue tile wins (1 other game)" in text,
   text)
ok("input_has_wrong_rules", "<wrong_rulebook>" in text and "- obj 4 can pass walls (level 2)" in text
   and "A5 swaps colours (level 1)" in text, text)
_, few = lessons_reconcile.reconcile_input(m, 1, 30)
ok("input_capped", 0 < len(few) < len(shown))

# --- 3. a pass: drop, rewrite, merge, and the guards ------------------------------------------------------------------
wall = next(i for i in mine if "pass walls" in m.lessons.nodes[i]["text"])
up = next(i for i in mine if "moves up" in m.lessons.nodes[i]["text"])
reply = {"drop": [{"id": wall, "why": "wrong rule: obj 4 can pass walls"},
                  {"id": other, "why": "not shown here"},
                  {"id": "rule:ffffffff", "why": "unknown"}],
         "rewrite": [{"id": up, "text": "A1 moves the red square one cell up", "why": "vague"},
                     {"id": both, "text": ""}],
         "merge": [{"ids": [both], "text": "x"}]}
llm = FakeLLM([json.dumps(reply)])
rec = lessons_reconcile.reconcile(llm=llm, memory=m, level=1, max_tokens=100, lessons_tokens=2000, overflow=Exception)
ok("system_and_input_sent", llm.calls[0][0]["content"] == lessons_reconcile.RECONCILE_SYSTEM
   and "<wrong_rulebook>" in llm.calls[0][1]["content"])
ok("drop_applied", [d["id"] for d in rec["dropped"]] == [wall] and wall not in m.lessons.nodes, rec)
ok("drop_other_game_refused", other in m.lessons.nodes and any("also from" in e for e in rec["errors"]), rec)
ok("unknown_id_refused", any("rule:ffffffff" in e for e in rec["errors"]))
new = rec["rewritten"][0]["node"] if rec["rewritten"] else None
ok("rewrite_applied", new in m.lessons.nodes and up not in m.lessons.nodes
   and m.lessons.nodes[new]["text"] == "A1 moves the red square one cell up"
   and rec["rewritten"][0]["before"] == ["obj 4 moves up on A1"], rec["rewritten"])
ok("rewrite_keeps_shape_edge", any(e["src"] == new and e["type"] == "about" for e in m.lessons.edges))
ok("bad_items_skipped", len(rec["errors"]) == 4 and both in m.lessons.nodes, rec["errors"])
ok("saved", wall not in json.loads((root / "on" / "lessons.json").read_text(encoding="utf-8"))["nodes"])
ok("shown_count", rec["shown"] == len(shown))

merge = {"merge": [{"ids": [other, both], "text": "the bar and the tile", "why": "test"}]}
rec = m.reconcile_lessons(merge, {other, both})
mid = rec["merged"][0]["node"] if rec["merged"] else None
ok("merge_applied", mid in m.lessons.nodes and other not in m.lessons.nodes and both not in m.lessons.nodes
   and m.lessons.nodes[mid]["games"] == ["tr87", "ls20"], rec)

many = {"rewrite": [{"id": i, "text": f"lesson {k}"} for k, i in enumerate(sorted(set(m.lessons.nodes) - {s for s in m.lessons.nodes if s.startswith("shape:")}))] * 6}
rec = m.reconcile_lessons(many, set(m.lessons.nodes), max_changes=10)
ok("change_limit", any("over the limit" in e for e in rec["errors"]))

llm = FakeLLM(["not json", "still not json"])
ok("unparseable_raises", raises(lambda: lessons_reconcile.reconcile(llm=llm, memory=m, level=1, max_tokens=100,
                                                                    lessons_tokens=2000, overflow=Exception)))

# --- 4. without lessons or rulebook ---------------------------------------------------------------------------------
m0 = GameMemory(root / "nolessons", "ls20", {**ON, "lessons": False})
ok("skipped_without_lessons", lessons_reconcile.reconcile(llm=FakeLLM([]), memory=m0, level=0, max_tokens=10,
                                                          lessons_tokens=100, overflow=Exception)
   == {"skipped": "lessons graph off"})
m1 = GameMemory(root / "empty", "ls20", {**ON, "wrong_rulebook": False})
ok("skipped_without_lessons_yet", lessons_reconcile.reconcile(llm=FakeLLM([]), memory=m1, level=0, max_tokens=10,
                                                              lessons_tokens=100, overflow=Exception)
   == {"skipped": "no lessons"})
m1.lessons.add_node("rule", "A2 moves down", game="ls20")
ok("works_without_rulebook", "(empty)" in lessons_reconcile.reconcile_input(m1, 0, 500)[0])
ok("off_by_default", DEFAULT["agent"]["memory"]["lessons_reconcile"] is False)

print("\n================ SUMMARY ================")
print(f"CHECKS: {CHECKS[0]}")
print(f"HARD FAILS: {len(FAILS)} -> {FAILS}")
sys.exit(1 if FAILS else 0)

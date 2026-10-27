# --- path shim: these suites live in tests/ but import from the repo root ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_PRIME = _os.environ.get("PRIME_DIR", "Sarbloh")   # PRIME_DIR=Sarbloh-Experimentation tests the copy
for _p in (_ROOT, _os.path.join(_ROOT, _PRIME)):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end shim ---
"""E008 curator with a fake LLM: its input carries the last steps, the agent's memory, the shapes on screen and the
skills; a good reply sets the open questions (at most 3), adds lessons linked to shapes, writes capped skills and picks
at most 3 known skills to load; bad items are dropped one by one; an unparseable reply raises (the agent logs it) and
changes nothing; a context overflow halves the input and retries.
On Sarbloh-Experimentation (E118) the level-up pass also writes up to 5 skills, sees more steps and says "Do not think
or reason much".
"""
import json
import sys
import tempfile
from pathlib import Path

from harness.agent import curator
try:
    from harness.agent.intuition import Obj
except ImportError:   # E116 copy: Obj lives in perception.py
    from harness.agent.perception import Obj
from harness.config import DEFAULT
from harness.memory.lessons import shape_id
from harness.memory.lifecycle import GameMemory

FAILS, CHECKS = [], [0]


def ok(name, cond, detail=""):
    CHECKS[0] += 1
    print(("PASS " if cond else "FAIL ") + name + (f"  {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


class Overflow(Exception):
    pass


class R:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def chat(self, messages, tools=None, max_tokens=None, timeout_s=None, thinking=None):
        self.calls.append({"messages": messages, "thinking": thinking, "tools": tools})
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return R(r)


root = Path(tempfile.mkdtemp(prefix="e008_cur_"))
m = GameMemory(root, "g1", dict(DEFAULT["agent"]["memory"]))
m.write_act({"plan": "probe", "hypotheses": [{"text": "1 moves obj 4 up 5", "status": "verified"}],
             "goal": "touch obj 7"}, turn=1, level=0)
for i in range(30):
    m.record_step({"i": i, "level": 0, "action": "A1(up)", "change": f"obj 4 moved up {i}", "state": "NOT_FINISHED"})
red = Obj(4, 8, frozenset({(1, 1), (1, 2), (2, 1), (2, 2)}), (1, 1, 2, 2), "abc123", 4)

inp = curator.curator_input(m, [red], "turn_interval")
ok("input_has_last_25_steps", "#29 L0" in inp and "#5 L0" in inp and "#4 L0" not in inp)
ok("input_has_memory_and_shapes", "1 moves obj 4 up 5" in inp and "touch obj 7" in inp and "R#abc123" in inp)
ok("input_lists_skills", "planning (base)" in inp)

good = {"open_questions": ["q1", "q2", "q3", "q4"],
        "lessons": [{"type": "rule", "text": "Action 1 moves the red square up 5", "shapes": ["R#abc123"],
                     "edge": "moves"},
                    {"type": "nonsense", "text": "bad type"},
                    {"type": "rule", "text": "   "}],
        "skills": [{"name": "Red Squares", "content": "word " * 400}, {"name": "planning", "content": "overwrite"},
                   {"name": "third", "content": "only two are read"}],
        "load_skills": ["red_squares", "ghost", "planning", "world_model", "first_principles"]}
llm = FakeLLM(["<think>hm</think>" + json.dumps(good)])
# Sarbloh-Experimentation writes more skills at a level-up (checked below); the 2-skill cap is the other passes'.
LEVEL_UP_SKILLS = hasattr(curator, "MAX_SKILLS_LEVEL_UP")
rec = curator.curate(llm=llm, memory=m, objects=[red], reason="turn_interval" if LEVEL_UP_SKILLS else "level_up",
                     max_tokens=100, overflow=Overflow)
ok("thinking_off_no_tools", llm.calls[0]["thinking"] is False and llm.calls[0]["tools"] is None)
ok("questions_capped_3", m.working.questions == ["q1", "q2", "q3"] and rec["questions"] == ["q1", "q2", "q3"])
ok("lesson_linked_to_shape", len(rec["lessons"]) == 1 and rec["lessons"][0]["shapes"] == [shape_id("R", "abc123")]
   and any(e["type"] == "moves" for e in m.lessons.edges), str(rec))
ok("bad_lessons_dropped_and_logged", len(rec["errors"]) >= 2, str(rec["errors"]))
ok("skill_written_capped", rec["skills"] == ["red_squares"] and len(m.skills.text("red_squares")) <= 800)
ok("base_skill_not_overwritten", "overwrite" not in m.skills.text("planning"))
ok("load_three_known", rec["loaded"] == ["red_squares", "planning", "world_model"])
ok("pinned_shows_curated", "### red_squares" in m.blocks([red])["skills"] and "q1" in m.blocks([red])["questions"])
ok("lessons_on_screen", "red square up 5" in m.blocks([red])["lessons"])

if LEVEL_UP_SKILLS:   # the level-up pass: up to 5 specific skills, more steps, "do not think much" stated
    many = {"skills": [{"name": f"skill_{k}", "content": f"rule {k}"} for k in range(7)]}
    llm3 = FakeLLM([json.dumps(many)])
    rec3 = curator.curate(llm=llm3, memory=m, objects=[red], reason="level_up", max_tokens=100, overflow=Overflow)
    sys_text, user_text = llm3.calls[0]["messages"][0]["content"], llm3.calls[0]["messages"][1]["content"]
    ok("level_up_writes_up_to_5_skills", rec3["skills"] == [f"skill_{k}" for k in range(5)], str(rec3["skills"]))
    ok("level_up_prompt", "Write 2 to 5 skills" in sys_text and "Do not think or reason much" in sys_text
       and llm3.calls[0]["thinking"] is False)
    ok("level_up_sees_more_steps", "#5 L0" in user_text and "#0 L0" in user_text)
    ok("other_passes_say_do_not_think_much", "Do not think or reason much" in llm.calls[0]["messages"][0]["content"]
       and "Write 2 to 5 skills" not in llm.calls[0]["messages"][0]["content"])

before = (list(m.working.questions), json.dumps(m.lessons.nodes, sort_keys=True))
try:
    curator.curate(llm=FakeLLM(["no json here", "still none"]), memory=m, objects=[red], reason="x", max_tokens=10,
                   overflow=Overflow)
    ok("unparseable_raises", False)
except ValueError:
    ok("unparseable_raises", (list(m.working.questions), json.dumps(m.lessons.nodes, sort_keys=True)) == before)
llm2 = FakeLLM([Overflow("too long"), json.dumps({"open_questions": ["after overflow"]})])
curator.curate(llm=llm2, memory=m, objects=[red], reason="x", max_tokens=10, overflow=Overflow)
ok("overflow_halves_and_retries", len(llm2.calls) == 2 and m.working.questions == ["after overflow"]
   and len(llm2.calls[1]["messages"][1]["content"]) < len(llm2.calls[0]["messages"][1]["content"]))
m2 = GameMemory(root / "off", "g2", {**DEFAULT["agent"]["memory"], "lessons": False})
rec2 = curator.apply(m2, good, [red])
ok("lessons_off_still_questions_and_skills", rec2["lessons"] == [] and rec2["questions"] and rec2["skills"])

print("\n================ SUMMARY ================")
print(f"CHECKS: {CHECKS[0]}")
print(f"HARD FAILS: {len(FAILS)} -> {FAILS}")
sys.exit(1 if FAILS else 0)

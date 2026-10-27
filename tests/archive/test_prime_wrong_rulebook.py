# --- path shim: these suites live in tests/ but import from the repo root ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_PRIME = _os.environ.get("PRIME_DIR", "Sarbloh-Experimentation")   # E018 exists only in the copy so far
for _p in (_ROOT, _os.path.join(_ROOT, _PRIME)):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end shim ---
"""E018 wrong rulebook and use-case skill names, without a model (experiments/E018_wrong_rulebook).

With ``memory.wrong_rulebook`` on:
- a hypothesis the agent marks refuted goes into ``<game>/wrong_rules.json`` at once, and the same text again is one
  entry; each game has its own rulebook;
- a new hypothesis, or one taken back from refuted, that restates a wrong rule is kept and counted (log only);
- refuted rules from the level-up, the E022 review and the curator go to the rulebook, not to the lessons graph;
- the agent never sees it: no pinned block, no recall hits, no prompt line, no act note; the curator does not read it.

With ``memory.skill_names: "use_case"`` a skill is named by its use case in at most 10 words and is found by that
name in any case or punctuation. With both switches off nothing changes.
"""
import inspect
import json
import sys
import tempfile
from pathlib import Path

from harness.agent import context, curator, prompts
from harness.agent.intuition import Obj
from harness.config import DEFAULT
from harness.memory.lessons import BASE_USE_CASES, SkillBook
from harness.memory.lifecycle import GameMemory
from harness.memory.wrong_rules import WrongRulebook, similar

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


ON = {**DEFAULT["agent"]["memory"], "wrong_rulebook": True, "skill_names": "use_case"}
OFF = dict(DEFAULT["agent"]["memory"])
red = Obj(4, 8, frozenset({(1, 1), (1, 2), (2, 1), (2, 2)}), (1, 1, 2, 2), "abc123", 4)

# --- 1. the store ------------------------------------------------------------------------------------------------
ok("similar_reworded", similar("Action 1 moves obj 4 up", "action 1 moves OBJ 4 up!")
   and not similar("A1 moves obj 4 up", "A2 opens the green door"))
root = Path(tempfile.mkdtemp(prefix="e018_"))
wb = WrongRulebook(root / "solo.json")
a = wb.add("Clicking the blue tile wins", level=0)
b = wb.add("clicking  the BLUE tile wins", level=1, source="curator")
ok("same_text_one_entry", a == b and len(wb.rules) == 1 and wb.rules[a]["levels"] == [0, 1]
   and wb.rules[a]["sources"] == ["agent", "curator"], wb.rules)
wb.add("Space resets the board")
blk = wb.render(200)
ok("render_lines", "- Clicking the blue tile wins (level 1, 2)" in blk and "- Space resets the board" in blk, blk)
ok("render_cap", "more)" in wb.render(5))
ok("store_rejects_empty", raises(lambda: wb.add("  ")))
wb.save()
ok("store_persisted", len(WrongRulebook(root / "solo.json").rules) == 2)

# --- 2. act: refuted -> rulebook at once; a re-proposal is kept and counted ----------------------------------------
m = GameMemory(root / "on", "ls20", ON)
m.write_act({"plan": "p", "hypotheses": ["h1 proposed: A1 moves obj 4 up", "h2 proposed: clicking obj 9 wins"]},
            turn=1, level=0)
ok("proposed_not_in_rulebook", not m.wrong.rules)
r = m.write_act({"plan": "p", "hypotheses": [{"id": "h2", "status": "refuted", "evidence": [3]}]}, turn=2, level=0)
ok("refuted_in_rulebook_at_once", [x["text"] for x in m.wrong.rules.values()] == ["clicking obj 9 wins"]
   and r["wrong"] == [], m.wrong.rules)
ok("refuted_saved_per_game", "clicking obj 9 wins" in (root / "on" / "ls20" / "wrong_rules.json").read_text(
    encoding="utf-8") and not (root / "on" / "wrong_rules.json").exists())
r = m.write_act({"plan": "p", "hypotheses": ["proposed: Clicking obj 9 wins!"]}, turn=3, level=0)
ok("reproposal_kept_and_logged", r["wrong"] and r["wrong"][0]["id"] == "h3"
   and r["wrong"][0]["rule"] == "clicking obj 9 wins" and r["wrong"][0]["wrong_id"].startswith("wrong:")
   and any(h["id"] == "h3" for h in m.working.hypotheses), r["wrong"])
ok("reproposal_counted", m.wrong.counts() == {"rules": 1, "reproposed": 1})
r = m.write_act({"plan": "p", "hypotheses": [{"id": "h3", "status": "verified"}]}, turn=4, level=0)
ok("status_change_not_counted_twice", r["wrong"] == [] and m.wrong.counts()["reproposed"] == 1)
r = m.write_act({"plan": "p", "hypotheses": [{"id": "h2", "status": "proposed"}]}, turn=5, level=0)
ok("taken_back_from_refuted_counted", [x["id"] for x in r["wrong"]] == ["h2"])
ok("bad_act_writes_nothing", raises(lambda: m.write_act({"plan": "p", "hypotheses": [
    {"text": "obj 1 is a wall", "status": "refuted"}, {"text": "x", "status": "maybe"}]}, turn=6, level=0))
   and len(m.wrong.rules) == 1)
other = GameMemory(root / "on", "tr87", ON)
ok("other_game_own_rulebook", other.wrong is not m.wrong and not other.wrong.rules
   and other.write_act({"plan": "p", "hypotheses": ["proposed: clicking obj 9 wins"]}, turn=1, level=0)["wrong"] == [])

# --- 3. level-up and review: refuted -> rulebook, not graph -----------------------------------------------------
m.write_act({"plan": "p", "hypotheses": [{"id": "h1", "status": "verified"},
                                         {"text": "A3 pushes obj 4 left", "status": "refuted"}]}, turn=7, level=0)
promo = m.on_level_up(level_won=1, step=9, objects=[])
ok("level_up_refuted_to_rulebook", m.lessons.counts().get("refuted", 0) == 0 and m.lessons.counts().get("rule", 0) >= 1
   and any(p.get("wrong") for p in promo["promoted"]) and len(m.wrong.rules) == 2, (m.lessons.counts(), promo))
m.add_level_review(0, {"summary": "won", "rules": [{"text": "A5 does nothing here", "status": "refuted"},
                                                   {"text": "A2 moves obj 4 down", "status": "verified"}]})
ok("review_refuted_to_rulebook", m.lessons.counts().get("refuted", 0) == 0
   and any(x["text"] == "A5 does nothing here" and "level_review" in x["sources"] for x in m.wrong.rules.values()))

# --- 4. the agent never sees it ------------------------------------------------------------------------------------
blocks = m.blocks([red])
msg = context.pinned_message(blocks, "status")["content"]
ok("no_pinned_block", "wrong" not in blocks and "rulebook" not in msg.lower() and "clicking obj 9 wins" not in msg, msg)
text, hits = m.recall("obj 9", "lessons")
ok("no_recall_hits", "clicking obj 9 wins" not in text, text)
ok("no_prompt_line", "rulebook" not in prompts.e008_system(game_id="ls20", win_levels=7, act_max=5, cwd="/w",
                                                           level_review=True).lower()
   and "wrong_rulebook" not in inspect.signature(prompts.e008_system).parameters)
ok("no_act_note", "restates a rule" not in (Path(prompts.__file__).parent / "agent.py").read_text(encoding="utf-8"))

# --- 5. curator: does not read it, refuted lessons go there, use-case skill names ----------------------------------
inp = curator.curator_input(m, [red], "turn_interval")
ok("curator_input_has_no_rulebook", "wrong_rulebook" not in inp and "clicking obj 9 wins" not in inp)
sysp = curator.system_prompt(m)
ok("curator_prompt_edits", "rulebook" not in sysp and "at most 10 words" in sysp and "snake_case" not in sysp)
reply = {"lessons": [{"type": "refuted", "text": "Red squares can be pushed through walls", "shapes": ["R#abc123"]},
                     {"type": "rule", "text": "Red squares stop at walls", "shapes": ["R#abc123"]}],
         "skills": [{"name": "When blocks must be pushed onto matching targets", "content": "Push from behind."},
                    {"name": "one two three four five six seven eight nine ten eleven", "content": "too long a name"}],
         "load_skills": ["when blocks must be pushed onto matching targets!", "Planning",
                         BASE_USE_CASES["planning"].upper()]}
llm = FakeLLM([json.dumps(reply)])
rec = curator.curate(llm=llm, memory=m, objects=[red], reason="level_up", max_tokens=100, overflow=Exception)
ok("curator_system_sent", llm.calls[0][0]["content"] == sysp)
ok("curator_refuted_to_rulebook", len(rec["wrong"]) == 1 and m.lessons.counts().get("refuted", 0) == 0
   and len(rec["lessons"]) == 1 and len(m.wrong.rules) == 4, rec)
ok("use_case_skill_stored", rec["skills"] == ["when blocks must be pushed onto matching targets"]
   and any("10 words" in e for e in rec["errors"]), rec)
ok("use_case_loaded_by_name", rec["loaded"] == ["when blocks must be pushed onto matching targets",
                                                BASE_USE_CASES["planning"]], rec["loaded"])
ok("use_case_render", m.skills.render("ls20", 600).startswith("### when blocks must be pushed onto matching targets"))

# --- 6. SkillBook in both modes ---------------------------------------------------------------------------------------
sk = SkillBook(root / "uc.json", use_case_names=True)
ok("base_use_case_names", sorted(sk.base) == sorted(BASE_USE_CASES.values())
   and all(len(n.split()) <= 10 for n in sk.base))
ok("base_read_only_use_case", raises(lambda: sk.upsert(BASE_USE_CASES["planning"], "x")))
n = sk.upsert("When a HUD bar shrinks on every move", "It is the action budget.", game="g")
ok("upsert_reuse_name_updates", sk.upsert("when a hud bar shrinks, on every move", "v2") == n
   and sk.curated[n]["version"] == 2 and sk.text("WHEN A HUD BAR SHRINKS ON EVERY MOVE") == "v2")
snake = SkillBook(root / "snake.json")
ok("snake_mode_unchanged", snake.upsert("Push Blocks!", "x") == "push_blocks" and "planning" in snake.base)

# --- 7. switches off: no rulebook, E008 prompts ------------------------------------------------------------------------
m0 = GameMemory(root / "off", "ls20", OFF)
m0.write_act({"plan": "p", "hypotheses": ["h1 refuted: obj 2 is a wall"]}, turn=1, level=0)
promo = m0.on_level_up(level_won=1, step=2, objects=[])
ok("off_no_rulebook", m0.wrong is None and not (root / "off" / "ls20" / "wrong_rules.json").exists()
   and m0.lessons.counts().get("refuted", 0) == 1)
ok("off_curator_prompt_e008", curator.system_prompt(m0) == curator.CURATOR_SYSTEM)
ok("bad_skill_names_value", raises(lambda: GameMemory(root / "bad", "g", {**OFF, "skill_names": "words"})))

print("\n================ SUMMARY ================")
print(f"CHECKS: {CHECKS[0]}")
print(f"HARD FAILS: {len(FAILS)} -> {FAILS}")
sys.exit(1 if FAILS else 0)

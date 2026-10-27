# --- path shim: these suites live in tests/ but import from the repo root ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_PRIME = _os.environ.get("PRIME_DIR", "Sarbloh-Experimentation")   # E021 exists only in the copy so far
for _p in (_ROOT, _os.path.join(_ROOT, _PRIME)):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end shim ---
"""E021 versioned goals without a model (experiments/E021_versioned_goal): with ``memory.goal_versioning`` on, act's
`goal` takes a string (a new version of the active goal) or a list (rivals, switch, refute); at most one goal is
active and at most 3 are open; a bad list changes nothing (all or nothing, with the hypotheses); a refuted goal
proposed again is flagged; the level-up marks the active goal won, keeps it and the refuted ones and clears the
rivals; the block, recall, save/load, new game, the act schema and the prompt lines. With the switch off, `goal` is
E008's single string. Amendment 1: each history entry keeps its reason and the block shows the last 5 changes; once
level 1 is won with an active goal, the goal is locked (act's `goal` ignored, the act still runs).
"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

from harness.agent import prompts
from harness.agent.tools import toolset
from harness.config import DEFAULT
from harness.memory.lifecycle import GameMemory
from harness.memory.working import parse_goal

FAILS, CHECKS = [], [0]


def ok(name, cond, detail=""):
    CHECKS[0] += 1
    print(("PASS " if cond else "FAIL ") + name + (f"  {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


def raises(fn):
    try:
        fn()
        return False
    except ValueError:
        return True


def status(m):
    return {g["id"]: (g["status"], g["version"]) for g in m.working.goals}


root = Path(tempfile.mkdtemp(prefix="e021_goal_"))
cfg = dict(DEFAULT["agent"]["memory"], goal_versioning=True)
ok("default_is_off", DEFAULT["agent"]["memory"]["goal_versioning"] is False)
m = GameMemory(root, "ls20", cfg)


def act(goal, **kw):
    return m.write_act({"plan": "p", "goal": goal, **kw}, turn=1, level=0)



# --- 1. parsing ----------------------------------------------------------------------------------------------
ok("parse_string", parse_goal("g2 refuted: touch obj 4") == {"id": "g2", "text": "touch obj 4", "status": "refuted",
                                                              "evidence": [], "why": ""})
ok("parse_plain", parse_goal("reach the door")["status"] is None)
ok("parse_dict", parse_goal({"id": "G1", "status": "active", "evidence": ["#3", 5], "why": "w"})
   == {"id": "g1", "text": "", "status": "active", "evidence": [3, 5], "why": "w"})
ok("parse_rejects_status", raises(lambda: parse_goal({"text": "x", "status": "won"})))
ok("parse_rejects_empty", raises(lambda: parse_goal({"status": "active"})))

# --- 2. a string: E008's call shape, now versioned -----------------------------------------------------------
w = act("reach the door")
ok("string_first_goal_is_active", status(m) == {"g1": ("active", 1)} and m.working.goal == "reach the door"
   and w["goal"] == "reach the door" and [e["kind"] for e in w["goals"]] == ["new"])
w = act("reach the door")
ok("string_same_text_no_event", w["goals"] == [] and w["goal"] is None and status(m) == {"g1": ("active", 1)})
w = act("reach the yellow door")
ok("string_new_version", status(m) == {"g1": ("active", 2)} and w["goals"][0]["kind"] == "version"
   and m.working.goal == "reach the yellow door")

# --- 3. a list: rivals, switch, refute -----------------------------------------------------------------------
w = act([{"text": "collect all 3 keys"}, "push obj 5 onto obj 9"])
ok("list_adds_candidates", status(m) == {"g1": ("active", 2), "g2": ("candidate", 1), "g3": ("candidate", 1)}
   and [e["kind"] for e in w["goals"]] == ["new", "new"])
before = json.dumps(m.working.goals)
ok("max_open_goals", raises(lambda: act([{"text": "a fourth goal"}])) and json.dumps(m.working.goals) == before)
ok("one_active_only", raises(lambda: act([{"id": "g2", "status": "active"}, {"id": "g3", "status": "active"}]))
   and json.dumps(m.working.goals) == before)
ok("unknown_id_rejected", raises(lambda: act([{"id": "g9", "status": "refuted"}])))
hyps_before = list(m.working.hypotheses)
ok("all_or_nothing_with_hypotheses", raises(lambda: act([{"id": "g9"}], hypotheses=["a new rule"]))
   and m.working.hypotheses == hyps_before)
w = act([{"id": "g2", "status": "active", "why": "obj 3 counts keys"}])
ok("switch_demotes_old_active", status(m)["g2"] == ("active", 1) and status(m)["g1"] == ("candidate", 2)
   and sorted(e["kind"] for e in w["goals"]) == ["status", "switch"] and m.working.goal == "collect all 3 keys")
w = act([{"id": "g3", "status": "refuted", "evidence": [14], "why": "obj 5 reached obj 9, no win"}])
ok("refute", status(m)["g3"] == ("refuted", 1) and w["goals"][0]["kind"] == "refuted"
   and w["goals"][0]["evidence"] == [14] and not w["goals"][0]["reproposed"])
w = act([{"text": "a third open goal"}])
ok("refuted_frees_a_slot", status(m)["g4"] == ("candidate", 1))
w = act([{"id": "g4", "status": "refuted"}, {"text": "Push obj 5 onto obj 9", "status": "candidate"}])
ok("reproposal_by_text_flagged", any(e["id"] == "g3" and e["reproposed"] for e in w["goals"])
   and status(m)["g3"] == ("candidate", 1))      # same text up to case: no new version
w = act("collect all 3 keys")
ok("string_matching_active_no_event", w["goals"] == [])

# --- 4. block, recall, history -------------------------------------------------------------------------------
blk = m.blocks([])["goal"]
lines = blk.splitlines()
ok("block_active_first", lines[0].startswith("g2 v1 [active] collect all 3 keys"), blk)
ok("block_shows_rivals_and_refuted", "[candidate]" in blk and "g4 v1 [refuted]" in blk)
ok("block_under_cap", len(blk) <= cfg["caps"]["goal_versioned"] * 4 + 40, len(blk))
text, hits = m.recall("", "goal")
ok("recall_goal_history", "g1 turn 1 v1 [active] reach the door" in text
   and "g1 turn 1 v2 [active] reach the yellow door" in text and "g3 turn 1 v1 [refuted]" in text, text)
ok("history_keeps_why", any("(why: " in x for x in m.working.goal_history()), m.working.goal_history())
ok("history_why_on_demotion", any("became active" in x for x in m.working.goal_history()))
ok("block_shows_last_5_changes", "Goal changes, last 5" in blk and "older: recall scope goal" in blk
   and sum(1 for x in blk.splitlines() if x.startswith("- g")) == 5, blk)
before = m.working.goal_history()
w = act([{"id": "g4", "text": "touch obj 7 twice", "why": "obj 7 flashed once"}])
after = m.working.goal_history()
ok("version_keeps_old_text_and_reason", after[:len(before)] == before
   and after[-1].endswith("v2 [refuted] touch obj 7 twice (why: obj 7 flashed once)"), after[-2:])
ok("block_last_change_is_newest", m.blocks([])["goal"].splitlines()[-1].startswith("- g4 turn 1 v2"),
   m.blocks([])["goal"])
act([{"id": "g3", "status": "refuted"}])
for i in range(5, 9):          # refute more: the block keeps only the last 2 refuted
    act([{"text": f"goal {i}"}])
    act([{"id": f"g{i}", "status": "refuted"}])
blk = m.blocks([])["goal"]
ok("block_keeps_last_2_refuted", "g8 v1 [refuted]" in blk and "g4 v2 [refuted]" not in blk.split("Goal changes")[0]
   and "older refuted: recall scope goal" in blk, blk)
ok("history_in_write_order", blk.splitlines()[-1] == "- g8 turn 1 v1 [refuted] goal 8", blk)

# --- 5. save and load ----------------------------------------------------------------------------------------
m.save()
m2 = GameMemory(root, "ls20", cfg)
ok("save_load_roundtrip", m2.working.goals == m.working.goals and m2.working.next_g == m.working.next_g
   and m2.working.goal == m.working.goal)

# --- 6. level-up ---------------------------------------------------------------------------------------------
open_before = [g["id"] for g in m.working.goals if g["status"] == "candidate"]
info = m.on_level_up(level_won=1, step=20, objects=[])
st = status(m)
ok("levelup_confirms_active", info["confirmation"] == "confirmed: won level 1 with goal g2 v1: collect all 3 keys",
   info["confirmation"])
ok("levelup_keeps_active_won", st["g2"] == ("active", 1) and m.working.goals[[g["id"] for g in m.working.goals]
                                                                           .index("g2")]["won"] == [1])
ok("levelup_clears_rivals", open_before and not any(s == "candidate" for s, _ in st.values()))
ok("levelup_keeps_refuted", st["g3"] == ("refuted", 1) and st["g8"] == ("refuted", 1), st)
hist = m.working.goal_history()
ok("cleared_rivals_keep_history", all(any(x.startswith(f"{gid} ") and "[cleared]" in x and "won with another goal" in x
                                          for x in hist) for gid in open_before)
   and hist[-1].split(" [")[1].startswith("cleared"), hist[-4:])
lv = [json.loads(x) for x in (root / "ls20" / "levels.jsonl").read_text(encoding="utf-8").splitlines()]
ok("levels_jsonl_has_goals", lv[-1]["goal"] == "collect all 3 keys"
   and any(g["id"] in open_before for g in lv[-1]["goals"]))
ok("lesson_goal_node", m.lessons is None or "goal" in m.lessons.counts())
blk = m.blocks([])["goal"]
ok("block_after_levelup", "(won level 1)" in blk and "confirmed: won level 1 with goal g2" in blk, blk)

# --- 6b. lock after level 1 (amendment 1; default goal_lock_after_level = 1) ----------------------------------
ok("lock_default_is_1", DEFAULT["agent"]["memory"]["goal_lock_after_level"] == 1)
ok("locked_after_level_1", m.working.goal_locked and "(locked: level 1 is won" in blk, blk)
snap = json.dumps(m.working.goals)
w = act([{"text": "new level rival"}], hypotheses=["h: obj 2 is a key"])
ok("locked_list_ignored", w["goal_locked"] is True and w["goals"] == [] and json.dumps(m.working.goals) == snap)
ok("locked_act_still_writes", m.working.plan == "p" and any(h["text"] == "obj 2 is a key" or "obj 2" in h["text"]
                                                             for h in m.working.hypotheses))
w = act("a different goal")
ok("locked_string_ignored", w["goal_locked"] and m.working.goal == "collect all 3 keys" and w["goal"] is None)
w = act([{"id": "g99", "status": "refuted"}, {"text": "x", "status": "active"}, {"text": "y", "status": "active"}])
ok("locked_bad_list_not_an_error", w["goal_locked"] and json.dumps(m.working.goals) == snap)
w = m.write_act({"plan": "p"}, turn=2, level=1)
ok("locked_no_goal_arg_no_flag", w["goal_locked"] is False)
ok("locked_update_goals_raises", raises(lambda: m.working.update_goals("z", turn=3, level=1)))
m.on_level_up(level_won=2, step=40, objects=[])
ok("locked_goal_marked_won_again", next(g for g in m.working.goals if g["id"] == "g2")["won"] == [1, 2]
   and m.working.goal_locked)
m.save()
ok("lock_survives_save_load", GameMemory(root, "ls20", cfg).working.goal_locked)

# --- 7. no active goal (lock off) and the lock when level 1 is won without one --------------------------------
u = GameMemory(Path(tempfile.mkdtemp(prefix="e021_nolock_")), "ls20", dict(cfg, goal_lock_after_level=0))
u.write_act({"plan": "p", "goal": "collect all 3 keys"}, turn=1, level=0)
u.on_level_up(level_won=1, step=5, objects=[])
ok("lock_off_not_locked", not u.working.goal_locked)
w = u.write_act({"plan": "p", "goal": [{"text": "new level rival"}]}, turn=2, level=1)
ok("rivals_again_after_levelup", [(e["id"], e["status"]) for e in w["goals"]] == [("g2", "candidate")])
u.write_act({"plan": "p", "goal": [{"id": "g1", "status": "refuted"}]}, turn=3, level=1)
ok("no_active_goal_shown", u.working.goal == "" and u.blocks([])["goal"].startswith("(no active goal"))
ok("levelup_without_active", u.on_level_up(level_won=2, step=30, objects=[])["confirmation"]
   == "confirmed: won level 2 (no active goal)")
n = GameMemory(Path(tempfile.mkdtemp(prefix="e021_noactive_")), "ls20", cfg)
n.write_act({"plan": "p", "goal": "reach the door"}, turn=1, level=0)
n.write_act({"plan": "p", "goal": [{"id": "g1", "status": "refuted", "why": "no"}]}, turn=2, level=0)
n.on_level_up(level_won=1, step=9, objects=[])
ok("won_without_active_not_locked", not n.working.goal_locked)
w = n.write_act({"plan": "p", "goal": "touch obj 3"}, turn=3, level=1)
ok("first_goal_after_lock_level_allowed_then_locks", w["goal_locked"] is False and n.working.goal == "touch obj 3"
   and n.working.goal_locked)
w = n.write_act({"plan": "p", "goal": "touch obj 4"}, turn=4, level=1)
ok("then_locked", w["goal_locked"] and n.working.goal == "touch obj 3")

# --- 8. new game ---------------------------------------------------------------------------------------------
m.on_new_game()
ok("new_game_clears_goals", m.working.goals == [] and m.working.next_g == 1 and m.working.goal == ""
   and m.blocks([])["goal"] == "")
ok("new_game_unlocks", not m.working.goal_locked and m.working.levels_won == 0
   and act("reach the door")["goal_locked"] is False)

# --- 9. switch off = E008 --------------------------------------------------------------------------------------
off = GameMemory(Path(tempfile.mkdtemp(prefix="e021_off_")), "ls20", dict(DEFAULT["agent"]["memory"]))
w = off.write_act({"plan": "p", "goal": "reach the door"}, turn=1, level=0)
ok("off_string_goal", w["goal"] == "reach the door" and w["goals"] == [] and off.working.goals == [])
ok("off_block_is_e008", off.blocks([])["goal"] == "reach the door")
ok("off_levelup_line", off.on_level_up(level_won=1, step=3, objects=[])["confirmation"]
   == "confirmed: won level 1 with goal: reach the door")

# --- 10. schema and prompt -----------------------------------------------------------------------------------
g_on = toolset("e008", depth=0, max_depth=0, act_max=5, goal_versioning=True)[1]["function"]["parameters"]
g_off = toolset("e008", depth=0, max_depth=0, act_max=5)[1]["function"]["parameters"]
ok("schema_on_takes_list", "anyOf" in g_on["properties"]["goal"]
   and g_on["properties"]["goal"]["anyOf"][1]["items"]["properties"]["status"]["enum"]
   == ["active", "candidate", "refuted"] and g_on["required"] == ["actions", "plan"])
ok("schema_off_is_string", g_off["properties"]["goal"]["type"] == "string")
ok("schema_json", json.loads(json.dumps(g_on)) == g_on)
p_on = prompts.e008_system(game_id="ls20", win_levels=7, act_max=5, cwd="/x", goal_versioning=True)
p_off = prompts.e008_system(game_id="ls20", win_levels=7, act_max=5, cwd="/x")
ok("prompt_on_replaces_goal_lines", all(old not in p_on and new in p_on for old, new in prompts.E021_GOAL_LINES))
ok("prompt_off_unchanged", all(old in p_off for old, _ in prompts.E021_GOAL_LINES) and p_on != p_off)
p_lock = prompts.e008_system(game_id="ls20", win_levels=7, act_max=5, cwd="/x", goal_versioning=True,
                             goal_lock_after=1)
lock_line = prompts.E021_LOCK_LINE.format(n=1)
ok("prompt_lock_line", lock_line in p_lock and lock_line not in p_on and "last 5 goal changes" in p_on)
ok("prompt_lock_needs_versioning", lock_line not in prompts.e008_system(game_id="ls20", win_levels=7, act_max=5,
                                                                        cwd="/x", goal_lock_after=1))


# --- 11. scripted end to end: a fake model plays the real ls20 with the switch on ----------------------------
class Reply:
    def __init__(self, calls=None, content=""):
        self.content, self.reasoning = content, "scripted"
        self.tool_calls = [{"id": f"c{i}_{time.time_ns()}", "type": "function",
                            "function": {"name": n, "arguments": json.dumps(a)}} for i, (n, a) in enumerate(calls or [])]
        self.finish_reason, self.prompt_tokens, self.completion_tokens = "stop", 1000, 10


class Usage:
    def to_json(self):
        return {}


SCRIPT = [
    [("act", {"actions": ["1"], "plan": "probe", "goal": "reach the exit"})],
    [("act", {"actions": ["1"], "plan": "probe", "goal": [{"text": "collect the key", "status": "candidate"}]})],
    [("act", {"actions": ["3"], "plan": "switch", "goal": [
        {"id": "g1", "status": "refuted", "evidence": [0], "why": "no exit"}, {"id": "g2", "status": "active"}]})],
    [("act", {"actions": ["4"], "plan": "back", "goal": [{"text": "Reach the exit", "status": "candidate"}]})],
    [("act", {"actions": ["4"], "plan": "bad", "goal": [{"id": "g9", "status": "refuted"}]})],      # refused
    [],
]


class FakeLLM:
    model = "fake"

    def __init__(self):
        self.usage, self.script, self.requests, self.tools = Usage(), list(SCRIPT), [], None

    def chat(self, messages, tools=None, max_tokens=None, timeout_s=None, thinking=None):
        if "You review the memory of an agent" in messages[0]["content"]:
            return Reply(content=json.dumps({"open_questions": [], "lessons": [], "skills": [], "load_skills": []}))
        self.requests.append(messages)
        self.tools = tools
        calls = self.script.pop(0) if self.script else []
        return Reply(calls, content="" if calls else "done")


from harness import run as RUN
from harness.config import build_config
from harness.game.games import build_games, make_arcade

out = Path(tempfile.mkdtemp(prefix="e021_e2e_"))
cfg = build_config({"experiment": "E021_goals_unit", "games": ["ls20"], "concurrency": 1, "game_wall_s": 300,
                    "max_actions_per_game": 40, "notebook_budget_s": 360, "teardown_reserve_s": 0,
                    "agent": {"tool_mode": "native", "toolset": "e008", "vision": False,
                              "memory": {"goal_versioning": True, "curator": False},
                              "limits": {"max_turns": len(SCRIPT) + 1, "max_output_tokens": 10**6}}})
os.environ["ONLY_RESET_LEVELS"] = "true"
arcade = make_arcade("offline", environments_dir=str(Path(_ROOT) / "eval" / "real_games"),
                     recordings_dir=str(out / "recordings"))
llm = FakeLLM()
RUN.run_games(build_games(arcade, "offline", only=["ls20"], record=False), cfg, llm, out, time.time() + 300)
gdir = next((out / "games").iterdir())
ev = [json.loads(x) for x in (gdir / "transcript.jsonl").read_text(encoding="utf-8").splitlines()]
goals = [e for e in ev if e.get("event") == "goal"]
acts = [e for e in ev if e.get("event") == "act"]
errs = [e for e in ev if e.get("event") == "tool_error"]
tool_msgs = [e for e in ev if e.get("event") == "message" and e.get("role") == "tool"]
ok("e2e_goal_events", [(e["goal_id"], e["kind"]) for e in goals]
   == [("g1", "new"), ("g2", "new"), ("g1", "refuted"), ("g2", "switch"), ("g1", "status")], goals)
ok("e2e_goal_event_fields", goals[1]["open"] == 2 and goals[-1]["reproposed"] is True
   and all("version" in e and "action_count" in e for e in goals))
ok("e2e_reproposal_note", any("note: goal g1 was refuted before" in m["content"] for m in tool_msgs))
ok("e2e_bad_goal_refused_no_spend", any(e["tool"] == "act" and "no goal 'g9'" in e["error"] for e in errs)
   and len(acts) == 4 and acts[-1]["actions"][1] == 4, [e.get("error") for e in errs])
sysmsg = llm.requests[-1][0]["content"]
ok("e2e_system_prompt_e021", all(new in sysmsg for _, new in prompts.E021_GOAL_LINES)
   and prompts.E021_LOCK_LINE.format(n=1) in sysmsg)
ok("e2e_tool_schema_e021", "anyOf" in next(t for t in llm.tools if t["function"]["name"] == "act")
   ["function"]["parameters"]["properties"]["goal"])
pinned = next(m["content"] for m in llm.requests[-1] if m["role"] == "user" and "[memory]" in str(m["content"]))
ok("e2e_memory_block", "g2 v1 [active] collect the key" in pinned and "g1 v1 [candidate] reach the exit" in pinned
   and "Goal changes, last 5" in pinned and "(why: no exit)" in pinned, pinned[:900])

print("\n================ SUMMARY ================")
print(f"CHECKS: {CHECKS[0]}")
print(f"HARD FAILS: {len(FAILS)} -> {FAILS}")
sys.exit(1 if FAILS else 0)

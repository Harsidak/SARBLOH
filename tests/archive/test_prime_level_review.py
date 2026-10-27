# --- path shim: these suites live in tests/ but import from the repo root ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_PRIME = _os.environ.get("PRIME_DIR", "Sarbloh-Experimentation")   # E022 exists only in the copy so far
for _p in (_ROOT, _os.path.join(_ROOT, _PRIME)):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end shim ---
"""E022 level review without a model (experiments/E022_level_review): with ``memory.level_review`` on, after every
level-up one hidden JSON call reads the whole level (steps, plans, memory at the end) and its reply is written into
the working memory (block "levels"), the lessons graph and levels.jsonl (recall). A bad reply is logged and skipped.
The goal is never touched. With the switch off nothing changes. Then scripted end-to-end runs on the real ls20 with
E021 + E022 on: a forced level-up, the review before the next turn, the E021 lock after level 1, and a bad review.
"""
import json
import os
import queue
import sys
import tempfile
import time
from pathlib import Path

from harness.agent import context, level_review, prompts
from harness.config import DEFAULT, build_config
from harness.game.arc_host import ArcHost
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


class Obj:
    def __init__(self, oid, letter, h):
        self.id, self.hash, self._label = oid, h, f"{letter} 3x3"

    def label(self):
        return self._label


class Reply:
    def __init__(self, calls=None, content=""):
        self.content, self.reasoning = content, "scripted"
        self.tool_calls = [{"id": f"c{i}_{time.time_ns()}", "type": "function",
                            "function": {"name": n, "arguments": json.dumps(a)}} for i, (n, a) in enumerate(calls or [])]
        self.finish_reason, self.prompt_tokens, self.completion_tokens = "stop", 1000, 10


class Overflow(Exception):
    pass


REVIEW = {"summary": "Moved obj 4 up twice (#1, #2) onto obj 9; the level ended.",
          "win_condition": "the red block reaches the blue cell",
          "rules": [{"text": "action 1 moves obj 4 up by 1", "status": "verified"},
                    {"text": "the bar is the goal", "status": "refuted"},
                    {"text": "maybe obj 2 matters", "status": "proposed"}],
          "mistakes": ["#3 repeated #2 with no change"],
          "carry_over": ["try action 1 toward the blue cell first", "ignore the bar"]}

# --- 1. config and the switch ---------------------------------------------------------------------------------
mc = DEFAULT["agent"]["memory"]
ok("default_off", mc["level_review"] is False and mc["level_review_max_tokens"] == 2048
   and mc["level_review_steps_tokens"] == 3000 and mc["caps"]["levels"] == 250)
ok("context_order_has_levels", context.ORDER[:2] == ("goal", "levels") and "levels" in context.TITLES)

# --- 2. one level in memory, then a level-up -----------------------------------------------------------------
root = Path(tempfile.mkdtemp(prefix="e022_"))
cfg = dict(mc, level_review=True, goal_versioning=True)
m = GameMemory(root, "ls20", cfg)
m.write_act({"plan": "probe up", "goal": "reach the blue cell",
             "hypotheses": [{"text": "action 1 moves obj 4 up", "status": "verified", "evidence": [1]}],
             "findings": ["the bar shrinks each step"]}, turn=1, level=0)
for i, (a, ch) in enumerate([("A1(up)", "obj 4 moved up 1"), ("A1(up)", "obj 4 moved up 1"),
                             ("A1(up)", "no change"), ("A1(up)", "LEVEL UP")]):
    m.record_step({"i": i, "level": 0, "action": a, "change": ch, "state": "NOT_FINISHED", "level_up": ch == "LEVEL UP",
                   "turn": 1})
m.record_act({"turn": 1, "level": 0, "steps": [0, 1, 2, 3], "plan": "probe up", "stop": "level up"})
objs = [Obj(4, "R", "aaaaaa"), Obj(9, "B", "bbbbbb")]
goals_before = json.dumps([{k: v for k, v in g.items() if k != "won"} for g in m.working.goals])
m.on_level_up(level_won=1, step=3, objects=objs)
rec = m.level_record(0)
ok("record_has_steps_acts_end", len(rec["steps"]) == 4 and len(rec["acts"]) == 1
   and rec["ended"]["hypotheses"][0]["text"] == "action 1 moves obj 4 up" and rec["ended"]["confirmation"]
   .startswith("confirmed: won level 1"), rec["ended"])
text = level_review.review_input(m, 0, 3000)
ok("input_has_whole_level", "<level>1 won after 4 steps</level>" in text and "#0 L0 A1(up): obj 4 moved up 1" in text
   and "#3 L0 A1(up): LEVEL UP" in text and "turn 1 steps [0, 1, 2, 3]: probe up" in text
   and "h1 [verified] action 1 moves obj 4 up" in text and "- the bar shrinks each step" in text
   and "reach the blue cell" in text, text)

# --- 3. clipping the steps ---------------------------------------------------------------------------------------
lines = [f"#{i} L0 A1(up): obj 4 moved up 1 cell and something long happened here" for i in range(400)]
cl = level_review.clip_steps(lines, 600)
ok("clip_keeps_first_and_last", cl[0] == lines[0] and cl[-1] == lines[-1] and any("steps omitted" in x for x in cl)
   and len("\n".join(cl)) <= 600 * 4, len("\n".join(cl)))
ok("clip_short_unchanged", level_review.clip_steps(lines[:5], 600) == lines[:5])

# --- 4. writing a review into the memories ------------------------------------------------------------------------
out = m.add_level_review(0, REVIEW)
r = out["review"]
ok("review_stored", m.working.reviews[-1] == r and r["level"] == 1 and r["steps"] == 4
   and [x["status"] for x in r["rules"]] == ["verified", "refuted"], r)
blk = m.blocks(objs)["levels"]
ok("levels_block", blk.startswith("Level 1 won in 4 steps.") and "Won by: the red block reaches the blue cell" in blk
   and "Next: try action 1 toward the blue cell first; ignore the bar" in blk and "Wasted: #3 repeated" in blk, blk)
types = sorted(p["type"] for p in out["promoted"])
ok("lessons_written", types == ["goal", "refuted", "rule"]
   and all(m.lessons.nodes[p["node"]]["source"] in ("level_review", "level_up") for p in out["promoted"]), types)
ok("lesson_links_obj", any(p["shapes"] == ["shape:R#aaaaaa"] for p in out["promoted"]), out["promoted"])
lv = [json.loads(x) for x in (root / "ls20" / "levels.jsonl").read_text(encoding="utf-8").splitlines()]
ok("levels_jsonl_review_row", lv[-1]["kind"] == "review" and lv[-1]["level"] == 0 and lv[-1]["win_condition"])
t, hits = m.recall("", "findings")
ok("recall_finds_review", "(level 1 review) Level 1 won in 4 steps." in t and "Summary: Moved obj 4" in t, t)
t, _ = m.recall("blue cell", "all")
ok("recall_all_words", "(level 1 review)" in t, t)
ok("recall_hypotheses_ok", m.recall("", "hypotheses")[1] >= 1)
ok("goal_untouched", json.dumps([{k: v for k, v in g.items() if k != "won"} for g in m.working.goals]) == goals_before)
ok("review_saved", GameMemory(root, "ls20", cfg).working.reviews == m.working.reviews)
ok("empty_review_refused", raises(lambda: m.add_level_review(0, {"rules": [{"text": "x", "status": "maybe"}]})))
for k in (2, 3):
    m.add_level_review(k - 1, {"summary": f"level {k}", "win_condition": f"win {k}"})
blk = m.blocks([])["levels"]
ok("block_last_2_only", "Won by: win 2" in blk and "Won by: win 3" in blk and "red block" not in blk
   and len(blk) <= mc["caps"]["levels"] * 4, blk)
pin = context.pinned_message(m.blocks([]), "status")["content"]
ok("pinned_has_levels_after_goal", pin.index("## Goal") < pin.index("## Level reviews") < pin.index("## Plan"))
m.on_new_game()
ok("new_game_clears_reviews", m.working.reviews == [] and m.blocks([])["levels"] == "")

# --- 5. the switch off ----------------------------------------------------------------------------------------------
off = GameMemory(Path(tempfile.mkdtemp(prefix="e022_off_")), "ls20", dict(mc))
off.write_act({"plan": "p", "goal": "g"}, turn=1, level=0)
off.on_level_up(level_won=1, step=1, objects=[])
ok("off_no_levels_block", off.blocks([])["levels"] == ""
   and "Level reviews" not in context.pinned_message(off.blocks([]), "s")["content"])
p_off = prompts.e008_system(game_id="ls20", win_levels=7, act_max=5, cwd="/x")
p_on = prompts.e008_system(game_id="ls20", win_levels=7, act_max=5, cwd="/x", level_review=True)
ok("prompt", prompts.E022_REVIEW_TEXT not in p_off and prompts.E022_REVIEW_TEXT in p_on
   and p_on.replace(prompts.E022_REVIEW_TEXT, "") == p_off)


# --- 6. the review call --------------------------------------------------------------------------------------------
class OneShot:
    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    def chat(self, messages, tools=None, max_tokens=None, thinking=None, timeout_s=None):
        self.calls.append({"system": messages[0]["content"], "user": messages[1]["content"], "thinking": thinking,
                           "tools": tools, "max_tokens": max_tokens})
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return Reply(content=r)


m = GameMemory(Path(tempfile.mkdtemp(prefix="e022_call_")), "ls20", cfg)
m.record_step({"i": 0, "level": 0, "action": "A1(up)", "change": "LEVEL UP", "level_up": True, "turn": 1})
m.on_level_up(level_won=1, step=0, objects=[])
llm = OneShot("thinking... " + json.dumps(REVIEW) + " done")
rec = level_review.review(llm=llm, memory=m, level=0, max_tokens=99, steps_tokens=3000, overflow=Overflow)
ok("review_call_shape", llm.calls[0]["thinking"] is False and llm.calls[0]["tools"] is None
   and llm.calls[0]["max_tokens"] == 99 and "You review one level" in llm.calls[0]["system"])
ok("review_call_writes", rec["review"]["win_condition"] and len(m.working.reviews) == 1)
llm = OneShot("no json", "still none")
ok("bad_reply_raises", raises(lambda: level_review.review(llm=llm, memory=m, level=0, max_tokens=9,
                                                          steps_tokens=3000, overflow=Overflow))
   and len(llm.calls) == 2 and len(m.working.reviews) == 1)
llm = OneShot(Overflow("too long"), json.dumps(REVIEW))
level_review.review(llm=llm, memory=m, level=0, max_tokens=9, steps_tokens=3000, overflow=Overflow)
ok("overflow_halves_input", len(llm.calls[1]["user"]) < len(llm.calls[0]["user"]) and len(m.working.reviews) == 2)


# --- 7. scripted end to end on the real ls20: E021 + E022 on, a forced level-up -----------------------------------
class Usage:
    def to_json(self):
        return {}


FORCE = {"at": None}
SCRIPT = [
    [("act", {"actions": ["1"], "plan": "probe", "goal": "reach the exit",
              "hypotheses": [{"text": "1 moves obj 4 up", "status": "verified", "evidence": [0]}]})],
    [("act", {"actions": ["1", "1", "1"], "plan": "force a level up on the 2nd step",
              "goal": [{"text": "collect the key", "status": "candidate", "why": "a key is on screen"}]})],
    [("act", {"actions": ["1"], "plan": "new level", "goal": "a new goal after level 1"})],
    [],
]


class FakeLLM:
    model = "fake"

    def __init__(self, review_reply):
        self.usage, self.review_reply = Usage(), review_reply
        self.script = queue.Queue()
        for s in SCRIPT:
            self.script.put(s)
        self.requests, self.order = [], []

    def chat(self, messages, tools=None, max_tokens=None, timeout_s=None, thinking=None):
        sys_msg = messages[0]["content"]
        if "You review one level" in sys_msg:
            self.order.append("review")
            self.review_input = messages[1]["content"]
            return Reply(content=self.review_reply)
        if "You review the memory of an agent" in sys_msg:
            self.order.append("curator")
            return Reply(content=json.dumps({"open_questions": [], "lessons": [], "skills": [], "load_skills": []}))
        self.order.append("agent")
        self.requests.append(messages)
        try:
            calls = self.script.get_nowait()
        except queue.Empty:
            calls = []
        if calls and "force a level up" in calls[0][1].get("plan", ""):
            FORCE["at"] = "next+1"
        return Reply(calls, content="" if calls else "done")


_real_handle = ArcHost.handle
_real_level_up = GameMemory.on_level_up


def _patched(self, req, depth, source="repl"):
    if FORCE["at"] == "next+1":
        FORCE["at"] = len(self.transitions) + 1
    res = _real_handle(self, req, depth, source)
    if FORCE["at"] is not None and len(self.transitions) - 1 == FORCE["at"]:
        res = {**res, "level_up": True}
        FORCE["at"] = None
    return res


def _level_up(self, *, level_won, step, objects):
    # the forced level-up leaves the game's counter at 0; a real level-up raises it to 1
    return _real_level_up(self, level_won=level_won + 1, step=step, objects=objects)


def e2e(review_reply):
    from harness import run as RUN
    from harness.game.games import build_games, make_arcade

    out = Path(tempfile.mkdtemp(prefix="e022_e2e_"))
    cfg = build_config({"experiment": "E022_review_unit", "games": ["ls20"], "concurrency": 1, "game_wall_s": 300,
                        "max_actions_per_game": 40, "notebook_budget_s": 360, "teardown_reserve_s": 0,
                        "agent": {"tool_mode": "native", "toolset": "e008", "vision": False,
                                  "memory": {"goal_versioning": True, "level_review": True},
                                  "limits": {"max_turns": len(SCRIPT) + 1, "max_output_tokens": 10**6}}})
    os.environ["ONLY_RESET_LEVELS"] = "true"
    arcade = make_arcade("offline", environments_dir=str(Path(_ROOT) / "eval" / "real_games"),
                         recordings_dir=str(out / "recordings"))
    llm = FakeLLM(review_reply)
    RUN.run_games(build_games(arcade, "offline", only=["ls20"], record=False), cfg, llm, out, time.time() + 300)
    gdir = next((out / "games").iterdir())
    ev = [json.loads(x) for x in (gdir / "transcript.jsonl").read_text(encoding="utf-8").splitlines()]
    return llm, ev, out


try:
    ArcHost.handle, GameMemory.on_level_up = _patched, _level_up
    llm, ev, out = e2e(json.dumps(REVIEW))
    tool_msgs = [e["content"] for e in ev if e.get("event") == "message" and e.get("role") == "tool"]
    acts = [e for e in ev if e.get("event") == "act"]
    revs = [e for e in ev if e.get("event") == "level_review"]
    ok("e2e_level_up_happened", any(e.get("event") == "level_up" for e in ev) and acts[1]["stop"] == "level up")
    ok("e2e_one_review", len(revs) == 1 and revs[0]["level"] == 1 and "error" not in revs[0]
       and revs[0]["review"]["steps"] >= 2, revs)
    ok("e2e_review_before_curator_and_next_turn", llm.order.index("review") < llm.order.index("curator")
       and llm.order[llm.order.index("review"):].index("agent") > 0, llm.order)
    ok("e2e_review_input_has_level", "#0 L0" in llm.review_input and "force a level up" in llm.review_input
       and "1 moves obj 4 up" in llm.review_input, llm.review_input[:800])
    ok("e2e_levelup_message", any("A review of the level you won is added" in t and "now locked" in t
                                  for t in tool_msgs), [t[-300:] for t in tool_msgs if "LEVEL UP" in t])
    pinned_next = next(m["content"] for m in llm.requests[2] if m["role"] == "user" and "[memory]" in str(m["content"]))
    ok("e2e_pinned_has_review", "## Level reviews" in pinned_next and "Won by: the red block reaches the blue cell"
       in pinned_next, pinned_next[:1200])
    ok("e2e_lock_note", any("note: the goal is locked after level 1" in t for t in tool_msgs)
       and acts[-1]["plan"] == "new level" and acts[-1]["done"] == 1)
    gl = [e for e in ev if e.get("event") == "goal_locked"]
    ok("e2e_goal_locked_event", len(gl) == 1 and gl[0]["goal"] == "reach the exit", gl)
    ok("e2e_goal_unchanged", not any(e.get("event") == "goal" and "a new goal" in str(e.get("goal")) for e in ev))
    ok("e2e_pinned_goal_locked", "(locked: level 1 is won" in pinned_next and "(why: a key is on screen)"
       in pinned_next, pinned_next[:1200])
    sysmsg = llm.requests[-1][0]["content"]
    ok("e2e_system_prompt", prompts.E022_REVIEW_TEXT in sysmsg and prompts.E021_LOCK_LINE.format(n=1) in sysmsg)
    summ = json.loads((out / "results.json").read_text(encoding="utf-8")) if (out / "results.json").exists() else {}
    ok("e2e_run_finished", bool(summ), list(out.iterdir()))

    llm, ev, _ = e2e("not json at all")
    revs = [e for e in ev if e.get("event") == "level_review"]
    acts = [e for e in ev if e.get("event") == "act"]
    ok("e2e_bad_review_logged", len(revs) == 1 and "unparseable" in revs[0].get("error", ""), revs)
    ok("e2e_bad_review_session_goes_on", acts[-1]["plan"] == "new level" and llm.order.count("review") == 2)
finally:
    ArcHost.handle, GameMemory.on_level_up = _real_handle, _real_level_up

print("\n================ SUMMARY ================")
print(f"CHECKS: {CHECKS[0]}")
print(f"HARD FAILS: {len(FAILS)} -> {FAILS}")
sys.exit(1 if FAILS else 0)

# --- path shim: these suites live in tests/ but import from the repo root ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_PRIME = _os.environ.get("PRIME_DIR", "Sarbloh-Experimentation")
for _p in (_ROOT, _os.path.join(_ROOT, _PRIME)):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end shim ---
"""The stuck review: working memory keeps the goals and hypotheses the agent deleted (not the won ones); the review's
input covers the whole level (early moves kept, runs of the same move folded, the middle cut above the cap), the beliefs
held and dropped, and what the host counts (moves never used, objects never clicked, GAME_OVERs, repeats); one call
with thinking on and the run's reasoning effort; the reply is cut to its caps and bad items dropped; a reply with
nothing usable raises; an overflow halves the move lines; the message names every part. In the session it fires
once per game, at the first turn end after review_at_s (4 h) of run time, never on a level-up turn or near the
deadline, and a failure is logged without a message (and not retried).
"""
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

from harness.agent import curator
from harness.agent.agent import AgentSession
from harness.agent.perception import Obj
from harness.config import build_config
from harness.memory.lifecycle import GameMemory

FAILS, CHECKS = [], [0]


def ok(name, cond, detail=""):
    CHECKS[0] += 1
    print(("PASS " if cond else "FAIL ") + name + (f"  {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


class Overflow(Exception):
    pass


class FakeLLM:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def chat(self, messages, tools=None, max_tokens=None, timeout_s=None, thinking=None, template_kwargs=None):
        self.calls.append({"messages": messages, "thinking": thinking, "template_kwargs": template_kwargs,
                           "max_tokens": max_tokens, "tools": tools})
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return SimpleNamespace(content=r, completion_tokens=1234)


mcfg = build_config({})["agent"]["memory"]
ok("config_defaults", mcfg["review_max_tokens"] == 24000,
   str({k: v for k, v in mcfg.items() if k.startswith("review")}))

# --- working memory keeps what the agent dropped ------------------------------------------------------------------
m = GameMemory(Path(tempfile.mkdtemp()), "g1", mcfg)
w = m.working
m.sync({"GOAL": {"L1": ["reach the flag"]}, "HYPOTHESES": {"L1": ["UP moves the red block", "A5 rotates"]}},
       turn=1, level=0)
m.sync({"GOAL": {"L1": ["match the colours"]}, "HYPOTHESES": {"L1": ["UP moves the red block"]}}, turn=4, level=0)
ok("dropped_goal_and_hypothesis",
   [(d["kind"], d["text"], d["level"], d["turn"]) for d in w.dropped]
   == [("goal", "reach the flag", 0, 4), ("hypothesis", "A5 rotates", 0, 4)], str(w.dropped))
w.win_level(0)
m.sync({"GOAL": {"L2": ["fill the bar"]}}, turn=9, level=1)
ok("won_goal_not_dropped", all(d["text"] != "match the colours" for d in w.dropped), str(w.dropped))
w.save()
ok("dropped_persists", GameMemory(m.root, "g1", mcfg).working.dropped == w.dropped)
w.clear_game()
ok("new_game_clears_dropped", w.dropped == [])

# --- host facts and move lines ------------------------------------------------------------------------------------
red = Obj(4, 8, frozenset({(10, 10), (10, 11)}), (10, 10, 10, 11), "abc123", 4)
blue = Obj(5, 9, frozenset({(20, 20)}), (20, 20, 20, 20), "def456", 4)
bar = Obj(6, 2, frozenset({(63, 0)}), (63, 0, 63, 0), "aaa111", 4, hud=True)
steps = [{"i": 0, "level": 1, "action": "UP", "change": "no change", "state": "NOT_FINISHED"},
         {"i": 1, "level": 1, "action": "UP", "change": "no change", "state": "NOT_FINISHED"},
         {"i": 2, "level": 1, "action": "A6(r10,c11)", "change": "R moved", "state": "NOT_FINISHED"},
         {"i": 3, "level": 1, "action": "LEFT", "change": "bar ran out", "state": "GAME_OVER"},
         {"i": 4, "level": 1, "action": "UP", "change": "no change", "state": "NOT_FINISHED", "repeat_of": 0}]
facts = curator.host_facts(steps, [1, 2, 3, 6], [red, blue, bar])
text = "\n".join(facts)
ok("facts_counts", "UP 3 times (3 changed nothing)" in text and "A6 (click) 1 times" in text, text)
ok("facts_never_used", "legal moves never used on this level: DOWN" in text, text)
ok("facts_game_over_and_repeats", "GAME_OVER 1 times" in text and "1 of 5 moves repeated" in text, text)
ok("facts_never_clicked", "b 1x1 at r20 c20" in text and "R 1x2" not in text and "r63" not in text, text)
ok("facts_names_as_timeline", "never used on this level: FIRE" in "\n".join(curator.host_facts(steps, [1, 5], []))
   and not any("never used" in f for f in curator.host_facts([{"i": 0, "action": "FIRE", "change": "x"}], [5], [])))
ok("facts_no_click_line_without_A6", not any("click on this level" in f for f in curator.host_facts(steps, [1, 2], [blue])))

lines = curator._step_lines(steps, 200)
ok("lines_fold_runs", lines[0] == "#0-#1 UP x2: no change" and len(lines) == 4, str(lines))
ok("lines_keep_game_over", "GAME_OVER" in lines[2], str(lines))
many = [{"i": i, "level": 1, "action": "RIGHT" if i % 2 else "LEFT", "change": f"moved {i}", "state": "NOT_FINISHED"}
        for i in range(500)]
cut = curator._step_lines(many, 200)
ok("lines_cut_middle", len(cut) == 201 and cut[0].startswith("#0 ") and cut[-1].startswith("#499 ")
   and "lines left out" in cut[curator.REVIEW_HEAD], f"{len(cut)} {cut[curator.REVIEW_HEAD]}")

# --- the input ------------------------------------------------------------------------------------------------------
m = GameMemory(Path(tempfile.mkdtemp()), "g2", mcfg)
m.sync({"GOAL": {"L1": ["touch the flag"]}, "FINDINGS": {"L1": ["the bar shrinks per move"]}}, turn=1, level=0)
m.working.win_level(0)
m.sync({"GOAL": {"L1": ["touch the flag"], "L2": ["stack the blocks"]}, "PLAN": {"L2": "push left"},
        "HYPOTHESES": {"L2": ["LEFT pushes the stack"]}}, turn=5, level=1)
bad = m.sync({"GOAL": {"L1": ["touch the flag"], "L2": ["the level is unwinnable"]}}, turn=7, level=1)
ok("unwinnable_goal_refused", bad["rejected"] == ["GOAL"], str(bad))
m.sync({"GOAL": {"L1": ["touch the flag"], "L2": ["open the door"]}, "PLAN": {}, "HYPOTHESES": {"L2": []}},
       turn=8, level=1)
for r in [{"i": 0, "level": 0, "action": "UP", "change": "old level", "state": "NOT_FINISHED"}] + many:
    m.record_step(r)
inp = curator.review_input(m, level=1, legal=[3, 4, 6], objects=[red, blue], tokens=150000, moves_left=900)
ok("input_header", "Level 2. The agent has made 500 moves" in inp and "150k tokens" in inp and "900 moves left" in inp
   and "Legal moves: LEFT, RIGHT, A6 (click)" in inp, inp[:400])
ok("input_earlier_levels", "confirmed: won level 1 with goal: touch the flag" in inp
   and "L1: the bar shrinks per move" in inp)
ok("input_beliefs_now", "GOAL:\n- open the door\nPLAN:\n(none)\nHYPOTHESES:\n(none)" in inp, inp[:1500])
ok("input_dropped", "goal (dropped at turn 8): stack the blocks" in inp
   and "hypothesis (dropped at turn 8): LEFT pushes the stack" in inp, inp)
ok("input_only_this_level", "old level" not in inp and "#0 L2 LEFT" in inp and "#499 L2 RIGHT" in inp)
ok("input_no_unwinnable_goal_saved", "unwinnable" not in inp)

# --- the call -------------------------------------------------------------------------------------------------------
reply = ('<think>long thoughts</think>{"beliefs": [{"belief": "open the door", "status": "untested", "evidence": ""},'
         ' {"belief": ""}, "junk"], "mistakes": ["#3: lost the bar twice", 7, {"x": 1}],'
         ' "untried": ["click the blue dot at r20 c20"], "goals": ["a", "b", "c", "d"],'
         ' "next_experiment": "click the blue dot once"}')
llm = FakeLLM([reply])
rev = curator.review(llm=llm, memory=m, level=1, legal=[3, 4, 6], objects=[red, blue], tokens=150000, moves_left=900,
                     max_tokens=24000, overflow=Overflow, template_kwargs={"reasoning_effort": "xhigh"})
c = llm.calls[0]
ok("call_thinking_on_with_effort", c["thinking"] is True and c["template_kwargs"] == {"reasoning_effort": "xhigh"}
   and c["max_tokens"] == 24000 and c["tools"] is None and c["messages"][0]["content"] == curator.REVIEW_SYSTEM)
ok("reply_cleaned", [b["belief"] for b in rev["beliefs"]] == ["open the door"]
   and rev["mistakes"] == ["#3: lost the bar twice", "7"] and rev["goals"] == ["a", "b", "c"]
   and rev["next_experiment"] == "click the blue dot once" and rev["reply_tokens"] == 1234, str(rev))
msg = curator.review_message(rev, moves=500)
ok("message_parts", "all 500 moves" in msg and "update GOAL, HYPOTHESES and PLAN" in msg
   and "- open the door: untested" in msg and "Mistakes so far:\n- #3" in msg and "Not tried yet:\n- click" in msg
   and "1. a\n2. b\n3. c" in msg and msg.endswith("Next experiment: click the blue dot once"), msg)

try:
    curator.review(llm=FakeLLM(['{"beliefs": [], "mistakes": []}']), memory=m, level=1, legal=[3], objects=[],
                   tokens=1, moves_left=None, max_tokens=10, overflow=Overflow)
    ok("empty_reply_raises", False)
except ValueError:
    ok("empty_reply_raises", True)
try:
    curator.review(llm=FakeLLM(["I could not decide."]), memory=m, level=1, legal=[3], objects=[], tokens=1,
                   moves_left=None, max_tokens=10, overflow=Overflow)
    ok("no_json_raises", False)
except ValueError:
    ok("no_json_raises", True)
llm = FakeLLM([Overflow(), '{"goals": ["x"]}'])
curator.review(llm=llm, memory=m, level=1, legal=[3], objects=[], tokens=1, moves_left=None, max_tokens=10,
               overflow=Overflow)
ok("overflow_halves_lines", len(llm.calls) == 2
   and len(llm.calls[1]["messages"][1]["content"]) < len(llm.calls[0]["messages"][1]["content"]))

# --- the trigger in the session: once per game, after 4 h of run time --------------------------------------------
H4 = 4 * 3600


def session(replies, *, run_age_s, level=1, stuck_level=1, reviewed=False, left_s=3600.0, at=H4):
    events, pushed = [], []
    state = SimpleNamespace(levels_completed=level, available_actions=[0, 3, 4, 6])
    s = SimpleNamespace(
        cfg={"memory": {"review_at_s": at, "review_max_tokens": 24000}, "request_timeout_s": 900.0},
        arc=SimpleNamespace(game=SimpleNamespace(state=state), finished=False, budget_left=900),
        _stuck_level=stuck_level, _reviewed=reviewed, _level_tokens=150000, deadline=time.time() + left_s,
        run_start=time.time() - run_age_s,
        llm=FakeLLM(replies), memory=m, _objects=[blue], _effort=lambda: {"reasoning_effort": "xhigh"}, _graph={},
        stats={"reviews": 0, "review_errors": 0, "review_tokens": 0},
        _log_event=events.append, _push=pushed.append)
    AgentSession._maybe_review(s)
    return s, events, pushed


good = '{"goals": ["open the door"], "next_experiment": "click the door"}'
ok("config_review_at_4h", mcfg["review_at_s"] == H4 and "review_every_tokens" not in mcfg)
s, ev, pu = session([good], run_age_s=H4 - 60)
ok("before_4h_no_review", not s.llm.calls and not pu and not s._reviewed)
s, ev, pu = session([good], run_age_s=H4 + 1)
ok("review_at_4h", len(s.llm.calls) == 1 and s._reviewed and s.stats["reviews"] == 1
   and s.stats["review_tokens"] == 1234 and len(pu) == 1 and pu[0]["role"] == "user"
   and "all 500 moves" in pu[0]["content"] and ev[0]["event"] == "review" and ev[0]["level"] == 2
   and ev[0]["moves"] == 500, str(ev))
AgentSession._maybe_review(s)
ok("only_once_per_game", len(s.llm.calls) == 1)
s, ev, pu = session([good], run_age_s=H4 * 2, reviewed=True)
ok("never_again_later", not s.llm.calls)
s, ev, pu = session([good], run_age_s=H4 + 1, level=2, stuck_level=1)
ok("not_on_level_up_turn", not s.llm.calls and not s._reviewed)
s, ev, pu = session([good], run_age_s=H4 + 1, left_s=200)
ok("not_near_deadline", not s.llm.calls)
s, ev, pu = session([good], run_age_s=H4 * 2, at=0)
ok("off_when_zero", not s.llm.calls)
s, ev, pu = session(["no json here"], run_age_s=H4 + 1)
ok("failure_logged_no_message_no_retry", not pu and s.stats["review_errors"] == 1 and s._reviewed
   and "error" in ev[0], str(ev))

print("================ SUMMARY ================")
print(f"CHECKS: {CHECKS[0]}")
print(f"HARD FAILS: {len(FAILS)} -> {FAILS}")
sys_exit = 1 if FAILS else 0
raise SystemExit(sys_exit)

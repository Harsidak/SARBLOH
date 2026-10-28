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
nothing usable raises; an overflow halves the move lines; the message names every part. With the host's record of
moves the input starts with the boards (level start, now, the last won level before its winning move), each move says
what it did as objects (a scrolled view named as such), and the beliefs, with the hypotheses' ids, come after the
evidence. The review rewrites the memory of its level (goals, plan, judged hypotheses to findings, rules), never with an
"unwinnable" goal. In the session a review is due after review_at_s (3 h) of run time on a level with review_min_tokens
since its last review, never on a level-up turn, near the deadline or when the lane reviews the game; it is applied by
the game's own thread (memory, kernel, message), a review of a level already won is only logged, and a failure is
logged without a message.
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
        if isinstance(r, tuple):   # (content, thinking): a reply cut off while thinking
            return SimpleNamespace(content=r[0], reasoning=r[1], completion_tokens=24000)
        return SimpleNamespace(content=r, completion_tokens=1234)


mcfg = build_config({})["agent"]["memory"]
ok("config_defaults", mcfg["review_max_tokens"] == 40000 and mcfg["review_at_s"] == 3 * 3600
   and mcfg["review_min_tokens"] == 100000 and mcfg["review_lanes"] == 1 and mcfg["review_borrow"] == 2
   and mcfg["caps"]["findings"] == 500, str({k: v for k, v in mcfg.items() if k.startswith("review")}))

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
ok("input_beliefs_now", "GOAL (id goal):\n- open the door\nPLAN:\n(none)\nHYPOTHESES (with their ids):\n(none)" in inp,
   inp[:1500])
ok("input_no_boards_without_record", "<board_now>" not in inp)
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
    llm = FakeLLM(["I could not decide.", "Still no idea."])
    curator.review(llm=llm, memory=m, level=1, legal=[3], objects=[], tokens=1,
                   moves_left=None, max_tokens=10, overflow=Overflow)
    ok("no_json_raises", False)
except ValueError:
    ok("no_json_raises", len(llm.calls) == 2)

# A reply cut off at the cap while thinking (run 10-27: 10 of 15): one more call, thinking off, writes the JSON from
# the tail of the notes.
notes = "x" * 30000 + "the agent never clicked the blue dot; goal: open the door"
llm = FakeLLM([("", notes), '{"goals": ["open the door"], "untried": ["click the blue dot"]}'])
rev = curator.review(llm=llm, memory=m, level=1, legal=[3], objects=[], tokens=1, moves_left=None, max_tokens=24000,
                     overflow=Overflow, template_kwargs={"reasoning_effort": "xhigh"})
c2 = llm.calls[1]
ok("cut_off_finished", rev["goals"] == ["open the door"] and rev.get("finished") is True
   and rev["reply_tokens"] == 24000 + 1234, str(rev))
ok("finish_call_thinking_off", c2["thinking"] is False and c2["max_tokens"] == curator.REVIEW_FINISH_TOKENS
   and c2["template_kwargs"] is None and c2["messages"][0]["content"] == curator.REVIEW_SYSTEM)
user2 = c2["messages"][1]["content"]
ok("finish_call_has_notes_tail", user2.startswith(llm.calls[0]["messages"][1]["content"])
   and "goal: open the door\n</your_notes>" in user2 and user2.endswith(curator.REVIEW_FINISH)
   and user2.count("x") < 25000, str(len(user2)))

# The short form kept in memory.
short = curator.review_memory({"mistakes": ["m1", "m2", "m3", "m4"], "untried": ["u1"], "goals": ["g1", "g2"],
                               "next_experiment": "click the door", "beliefs": [{"belief": "b", "status": "x"}]})
ok("memory_form", short == "Mistakes:\n- m1\n- m2\n- m3\nNot tried:\n- u1", short)
ok("memory_form_capped", len(curator.review_memory({"mistakes": ["y" * 390] * 5, "untried": ["z" * 390] * 5,
                                                     "goals": ["w" * 390] * 3})) <= curator.REVIEW_MEMORY_CHARS)
llm = FakeLLM([Overflow(), '{"goals": ["x"]}'])
curator.review(llm=llm, memory=m, level=1, legal=[3], objects=[], tokens=1, moves_left=None, max_tokens=10,
               overflow=Overflow)
ok("overflow_halves_lines", len(llm.calls) == 2
   and len(llm.calls[1]["messages"][1]["content"]) < len(llm.calls[0]["messages"][1]["content"]))

# --- what one move did, as objects ------------------------------------------------------------------------------------
from harness.runtime.skills import geometry as geo

scene = ["W" * 20 for _ in range(20)]   # W: colour 0, the background
scene = geo.paste(scene, ["RR", "RR"], 3, 4)
scene = geo.paste(scene, ["BBBB"], 12, 9)
scene = geo.paste(scene, ["GGG", "GWG"], 8, 14)
right = geo.paste(geo.paste(scene, ["WW", "WW"], 3, 4), ["RR", "RR"], 3, 6)
ok("change_none", curator.move_change(scene, scene) == "no change")
c = curator.move_change(scene, right)
ok("change_moved", c == "R 4px moved +0,+2 to r3-4 c6-7", c)
c = curator.move_change(scene, geo.paste(scene, ["WWWW"], 12, 9))
ok("change_gone", c == "B 4px gone from r12-12 c9-12", c)
c = curator.move_change(scene, geo.shift(scene, -2, 3))
ok("change_scroll", c == "the whole view scrolled -2 rows +3 columns", c)
c = curator.move_change(scene, geo.paste(geo.shift(scene, -2, 3), ["WWWW"], 10, 12))
ok("change_scroll_then_objects", c == "the whole view scrolled -2 rows +3 columns; B 4px gone from r10-10 c12-15", c)
push = geo.paste(geo.paste(scene, ["WW", "WW"], 3, 4), ["RRB", "RRB"], 3, 5)
c = curator.move_change(geo.paste(scene, ["B", "B"], 3, 6), push)
ok("change_push_not_scroll", "scrolled" not in c and "R 4px moved +0,+1" in c and "B 2px moved +0,+1" in c, c)

# --- the input with the host's record: boards first, moves as objects, beliefs with ids last ----------------------
def grid(board):   # letters -> the ints the host records (perception.letter maps them back)
    from harness.agent import perception
    back = {perception.letter(v): v for v in range(16)}
    return [[back[x] for x in row] for row in board]


m3 = GameMemory(Path(tempfile.mkdtemp()), "g3", mcfg)
m3.sync({"GOAL": {"L1": ["touch the flag"]}}, turn=1, level=0)
m3.working.win_level(0)
m3.sync({"GOAL": {"L1": ["touch the flag"], "L2": ["open the door"]},
         "HYPOTHESES": {"L2": ["UP moves the red square", "the blue bar is a timer"]}}, turn=3, level=1)
ids = {h["id"]: h["text"] for h in m3.working.hypotheses}
trans =[{"i": 0, "level": 0, "action": 4, "x": None, "y": None, "before": grid(scene), "frames": [grid(right)],
          "level_up": True, "state": "NOT_FINISHED"},
         {"i": 1, "level": 1, "action": 4, "x": None, "y": None, "before": grid(scene), "frames": [grid(right)],
          "level_up": False, "state": "NOT_FINISHED"},
         {"i": 2, "level": 1, "action": 1, "x": None, "y": None, "before": grid(right),
          "frames": [grid(geo.shift(right, -2, 3))], "level_up": False, "state": "NOT_FINISHED"}]
for t in trans:
    m3.record_step({"i": t["i"], "level": t["level"], "action": "RIGHT" if t["action"] == 4 else "UP",
                    "change": "29 cells changed at r3-4 c4-7", "state": t["state"], "level_up": t["level_up"]})
inp = curator.review_input(m3, level=1, legal=[1, 4], objects=[], tokens=150000, moves_left=None, transitions=trans)
ok("input_has_boards", "<board_at_level_start>" in inp and "<board_now>" in inp
   and "Level 1 was won by RIGHT from this board:" in inp, inp[:600])
ok("input_board_labelled", "  3 WWWWRRWWWWWWWWWWWWWW" in inp and "    01234567890123456789" in inp, inp[:900])
ok("input_moves_as_objects", "#1 L2 RIGHT: R 4px moved +0,+2 to r3-4 c6-7" in inp
   and "#2 L2 UP: the whole view scrolled -2 rows +3 columns" in inp and "29 cells changed" not in inp, inp)
ok("input_evidence_before_beliefs", inp.index("<board_now>") < inp.index("<moves_on_this_level>")
   < inp.index("<what_the_agent_believes_now>"))
ok("input_hypothesis_ids", all(f"- {i}: {t}" in inp for i, t in ids.items()), str(ids))
ok("system_asks_rules_and_ids", '"rules"' in curator.REVIEW_SYSTEM and '"id"' in curator.REVIEW_SYSTEM
   and "Read the boards and the moves before you read what the agent believes" in curator.REVIEW_SYSTEM)

# --- the review rewrites the memory of its level -------------------------------------------------------------------
h_up, h_bar = sorted(ids)   # h1 UP moves the red square, h2 the blue bar is a timer
values = m3.working.kernel_values()
rev = {"goals": ["the level is unwinnable", "put the red square on the door"], "next_experiment": "press UP twice",
       "rules": ["RIGHT moves the red square 2 cells (#1)"],
       "beliefs": [{"id": h_up.upper(), "belief": "UP moves it", "status": "contradicted", "evidence": "#2 scrolled"},
                   {"id": h_bar, "belief": "timer", "status": "Supported", "evidence": "#1"},
                   {"id": "h99", "belief": "?", "status": "contradicted", "evidence": ""},
                   {"id": "goal", "belief": "open the door", "status": "untested", "evidence": ""}]}
new, changed = curator.memory_from_review(values, 1, rev, ids)
ok("rewrite_goals_no_unwinnable", new["GOAL"]["L2"] == ["put the red square on the door"]
   and new["GOAL"]["L1"] == ["touch the flag"] and changed["old_goals"] == ["open the door"], str(new["GOAL"]))
ok("rewrite_plan", new["PLAN"]["L2"] == "press UP twice")
ok("rewrite_hypotheses_judged", new["HYPOTHESES"]["L2"] == [] and changed["refuted"] == [h_up]
   and changed["confirmed"] == [h_bar], str(changed))
f = new["FINDINGS"]["L2"]
ok("rewrite_findings", f[0] == "Refuted by the reviewer (#2 scrolled): UP moves the red square"
   and f[1] == "Confirmed by the reviewer (#1): the blue bar is a timer"
   and f[2] == "The reviewer found: RIGHT moves the red square 2 cells (#1)" and changed["rules"] == 1, str(f))
ok("rewrite_input_untouched", values["HYPOTHESES"]["L2"] == ["UP moves the red square", "the blue bar is a timer"])
new2, ch2 = curator.memory_from_review(values, 1, {"goals": ["it cannot be won"]}, ids)
ok("only_unwinnable_goals_keep_old", new2["GOAL"]["L2"] == ["open the door"] and ch2["goals"] == [])
wrote = m3.sync(new, turn=9, level=1)
ok("rewrite_saves", not wrote["errors"] and m3.working.plan_for(1) == "press UP twice"
   and any(d["text"] == "open the door" for d in m3.working.dropped), str(wrote))
msg = curator.review_message({**rev, "goals": new["GOAL"]["L2"]}, moves=2, level=1, changed=changed)
ok("message_says_what_changed", 'GOAL["L2"] now holds its goals' in msg and 'PLAN["L2"] is its next experiment' in msg
   and f"{h_up} moved from HYPOTHESES to FINDINGS as refuted" in msg and "1 rules it found added" in msg
   and "Your goals before the review were:\n- open the door" in msg and "change your memory back" in msg
   and "What the moves show the game does:\n- RIGHT moves" in msg, msg)

# --- in the session: due after 3 h on a stuck level, applied by the game's own thread -------------------------------
import threading

H3 = 3 * 3600
good = ('{"goals": ["open the door"], "next_experiment": "click the door", "untried": ["the blue dot"],'
        ' "beliefs": [{"id": "%s", "belief": "x", "status": "contradicted", "evidence": "#3"}]}' % h_up)


class FakeKernel:
    def __init__(self):
        self.sets = []

    def set_memory(self, values):
        self.sets.append(values)
        return True


def session(replies, *, run_age_s, level=1, stuck_level=1, left_s=3600.0, at=H3, tokens=150000, lane=False):
    events, pushed = [], []
    state = SimpleNamespace(levels_completed=level, available_actions=[0, 3, 4, 6])
    mem = GameMemory(Path(tempfile.mkdtemp()), "g4", mcfg)
    mem.sync({"GOAL": {"L2": ["reach the flag"]}, "HYPOTHESES": {"L2": ["UP moves the red square"]}}, turn=1, level=1)
    s = SimpleNamespace(
        cfg={"memory": {"review_at_s": at, "review_max_tokens": 40000, "review_min_tokens": 100000},
             "request_timeout_s": 900.0, "reasoning_effort": {"early": "xhigh", "later": "xhigh"}},
        arc=SimpleNamespace(game=SimpleNamespace(state=state), finished=False, budget_left=None, transitions=trans),
        _stuck_level=stuck_level, _level_tokens=tokens, _reviewed_at={}, _review_ready=None,
        _review_lock=threading.Lock(), review_lane=lane, end_reason="", _review=None, _shown={},
        deadline=time.time() + left_s, run_start=time.time() - run_age_s,
        llm=FakeLLM(replies), memory=mem, _objects=[blue], _graph={}, kernel=FakeKernel(),
        stats={"reviews": 0, "review_errors": 0, "review_tokens": 0, "turns": 7},
        _log_event=events.append, _push=pushed.append)
    for name in ("review_due", "review_job", "offer_review", "_apply_review", "_write_review"):
        setattr(s, name, getattr(AgentSession, name).__get__(s))
    AgentSession._maybe_review(s)
    return s, events, pushed


s, ev, pu = session([good], run_age_s=H3 - 60)
ok("before_3h_no_review", not s.llm.calls and not pu)
s, ev, pu = session([good], run_age_s=H3 + 1)
w4 = s.memory.working
ok("review_at_3h", len(s.llm.calls) == 1 and s.stats["reviews"] == 1 and s.stats["review_tokens"] == 1234
   and len(pu) == 1 and pu[0]["role"] == "user" and ev[0]["event"] == "review" and ev[0]["level"] == 2, str(ev))
ok("review_effort_and_boards", s.llm.calls[0]["template_kwargs"] == {"reasoning_effort": "xhigh"}
   and "<board_now>" in s.llm.calls[0]["messages"][1]["content"])
ok("review_wrote_memory", w4.plan_for(1) == "click the door" and [g["text"] for g in w4.goals] == ["open the door"]
   and not w4.hypotheses and w4.findings[0]["text"].startswith("Refuted by the reviewer (#3)"),
   str(w4.kernel_values()))
ok("review_wrote_kernel", s.kernel.sets and s.kernel.sets[-1]["PLAN"] == {"L2": "click the door"}
   and "SKILLS" not in s.kernel.sets[-1] and ev[0]["memory"]["kernel"] is True, str(s.kernel.sets))
ok("review_message_after_write", "It has already updated your memory" in pu[0]["content"]
   and "Your goals before the review were:\n- reach the flag" in pu[0]["content"], pu[0]["content"][:600])
AgentSession._maybe_review(s)
ok("not_again_without_new_tokens", len(s.llm.calls) == 1)
s.llm.replies.append(good)
s._level_tokens += 100000
AgentSession._maybe_review(s)
ok("again_after_more_tokens", len(s.llm.calls) == 2 and s.stats["reviews"] == 2)
s, ev, pu = session([good], run_age_s=H3 + 1, tokens=50000)
ok("not_below_min_tokens", not s.llm.calls)
s, ev, pu = session([good], run_age_s=H3 + 1, level=2, stuck_level=1)
ok("not_on_level_up_turn", not s.llm.calls)
s, ev, pu = session([good], run_age_s=H3 + 1, left_s=600)
ok("not_near_deadline", not s.llm.calls)
s, ev, pu = session([good], run_age_s=H3 * 2, at=0)
ok("off_when_zero", not s.llm.calls)
s, ev, pu = session([good], run_age_s=H3 + 1, lane=True)
ok("lane_reviews_not_turn_end", not s.llm.calls and s.review_due() == 150000)
s, ev, pu = session(["no json here", "nor here"], run_age_s=H3 + 1)
ok("failure_logged_no_message", not pu and s.stats["review_errors"] == 1 and "error" in ev[0]
   and len(s.llm.calls) == 2 and s._review is None, str(ev))

# A review that finishes after its level was won is only logged.
s, ev, pu = session([good], run_age_s=H3 + 1, lane=True)
res = s.review_job(FakeLLM([good]))
ok("job_marks_level", s._reviewed_at == {1: 150000} and "rev" in res and s.review_due() is None)
s.offer_review(res)
ok("due_waits_for_pending", s._review_ready is not None and s.review_due() is None)
s.arc.game.state.levels_completed = 2
s._apply_review()
ok("stale_review_only_logged", not pu and ev[-1].get("stale") is True and not s.kernel.sets
   and s._review_ready is None, str(ev[-1])[:300])

# --- the review stays in memory until its level is won -----------------------------------------------------------
s, ev, pu = session([good], run_age_s=H3 + 1)
ok("review_kept_as_memory", s._review == (1, "Not tried:\n- the blue dot"), str(s._review))
ok("review_marked_shown", s._shown["review"] == s._review[1])
s.memory = SimpleNamespace(blocks=lambda objects: {"goal": "L2: open the door"})
ok("blocks_have_review", AgentSession._blocks(s).get("review") == s._review[1])
s.arc.game.state.levels_completed = 2
ok("blocks_drop_review_after_win", "review" not in AgentSession._blocks(s))
from harness.agent import context
s.arc.game.state.levels_completed = 1
pinned = context.pinned_message(AgentSession._blocks(s), "")["content"]
ok("pinned_shows_review", "## The reviewer's advice on this level (kept until you win it)\nNot tried" in pinned
   and pinned.index("Hypotheses") < pinned.index("reviewer's advice"), pinned)

print("================ SUMMARY ================")
print(f"CHECKS: {CHECKS[0]}")
print(f"HARD FAILS: {len(FAILS)} -> {FAILS}")
sys_exit = 1 if FAILS else 0
raise SystemExit(sys_exit)

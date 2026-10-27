# --- path shim: these suites live in tests/ but import from the repo root ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_PRIME = _os.environ.get("PRIME_DIR", "Sarbloh-Experimentation")   # these changes are in the experimentation copy only
for _p in (_ROOT, _os.path.join(_ROOT, _PRIME)):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end shim ---
"""The turn's thinking budget, the memory kept out of the prompt, and no board text (Sarbloh-Experimentation):
- config defaults: max_inner_requests 12, turn_yield_tokens 16384, a 12000-token reply cap that survives fit_context, memory.in_context off,
  perception.ascii on, code_graph on; with ascii off the system prompt says the board is not shown, and the memory
  is not shown;
- a real session on ls20 with a scripted model (turn_yield_tokens 2048, ascii off, stuck note off): code that runs
  without moving keeps the turn going with no note and the same scheduler slot; past 2048 tokens the turn ends
  silently; a reply with no tool call gets the continuation, one cut off at the length limit the "call the tool now"
  note; a move ends the turn; each turn end hands the scheduler slot over, no [memory] message and no board text
  reach any request, the memory variables set in the code (PLAN, HYPOTHESES) are saved before the moves and named in
  the output, a value of the wrong type is named and kept out while the moves still run, a deleted variable is put
  back, and `recall()` in a cell finds the timeline;
- a session that never moves: the turn ends after max_inner_requests requests, with no note;
- ScheduledLLM.yield_turn gives the slot to a waiting game, and does nothing before the game has used its slot; a
  turn-driven ScheduledLLM keeps its slot past quantum_calls and after a cold prompt.

Run: uv run python tests/unit/test_turn_budget.py
"""
import json
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from harness.agent import agent as A, prompts
from harness.config import build_config
from harness.llm.client import Reply
from harness.run import fit_context
from harness.scheduler import PriorityScheduler, ScheduledLLM

FAILS, WARNS, CHECKS = [], [], [0]


def ok(name, cond, detail=""):
    CHECKS[0] += 1
    print(("PASS " if cond else "FAIL ") + name + (f"  {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


# --- 1. config and prompt --------------------------------------------------------------------------------------
cfg = build_config({})["agent"]
ok("defaults", cfg["max_inner_requests"] == 12 and cfg["turn_yield_tokens"] == 16384 and cfg["max_tokens_per_turn"] == 12000
   and cfg["memory"]["in_context"] is False and cfg["perception"]["ascii"] is False and cfg["code_graph"] is True
   and "hide_code" not in cfg, str({k: cfg.get(k) for k in ("turn_yield_tokens", "max_tokens_per_turn", "code_graph")}))
fit = build_config({})["agent"]
fit_context(fit, 139264)
ok("cap_survives_fit_context", fit["max_tokens_per_turn"] == 12000, str(fit["max_tokens_per_turn"]))
hidden = prompts.game_system(act_max=100)
shown = prompts.game_system(act_max=100, board_text=True, memory_shown=True)
ok("prompt_board_and_memory_hidden", "The board itself is not written" in hidden
   and prompts.MEMORY_HIDDEN in hidden and "[memory]" not in hidden)
ok("prompt_board_and_memory_shown", "whole board written as letters" in shown and "[memory] block" in shown
   and "not written" not in shown)
ok("prompt_no_slow_down_words", not any(w in hidden.lower() for w in ("rush", "take your time", "carefully")))
ok("budget_notes", "await play" in prompts.event_message("act_now") and "without calling a tool"
   in prompts.event_message("call_tool_now") and "Only 3 moves" in prompts.event_message("act_now", moves_left=3))


# --- 2. a real session on ls20 with a scripted model ------------------------------------------------------------
def code_call(n, code, name="ipython", args=None):
    return [{"id": f"call{n}", "type": "function",
             "function": {"name": name, "arguments": json.dumps(args if args is not None else {"code": code})}}]


SCRIPT = {   # turn -> (tool calls, finish, completion tokens)
    1: (code_call(1, "x = 1"), "tool_calls", 1000),
    2: (code_call(2, "y = 2"), "tool_calls", 1000),
    3: (code_call(3, "z = 3"), "tool_calls", 1000),                     # 3000 since the last move: turn ends
    4: ([], "stop", 2500),                                               # no tool: continuation, turn ends
    5: ([], "length", 12288),                                            # cut off: call the tool now, turn ends
    6: (code_call(6, 'PLAN["L1"] = "go up to the yellow bar"\nHYPOTHESES["L1"] = ["A1 moves the block up"]\n'
                     'GOAL = 5\nr = await play(["1"])'), "tool_calls", 1000),
    7: (code_call(7, "del HYPOTHESES\nq = 1"), "tool_calls", 1500),     # 1500 since the move: no note
    8: (code_call(8, 'r = await play(["1"])'), "tool_calls", 1000),
    9: (code_call(9, "recall('', 'timeline')"), "tool_calls", 500),
}


class ScriptLLM:
    def __init__(self):
        self.requests, self.max_tokens, self.turn, self.yields = [], [], 0, []

    def chat(self, messages, tools=None, max_tokens=None, timeout_s=None, template_kwargs=None, thinking=None):
        self.requests.append(messages)
        self.max_tokens.append(max_tokens)
        self.turn += 1
        calls, finish, tokens, *say = SCRIPT.get(self.turn, (code_call(self.turn, 'r = await play(["1"])'), "tool_calls", 300))
        return Reply(say[0] if say else "", f"thinking {self.turn}", calls, finish, 5000, tokens)

    def yield_turn(self):
        self.yields.append(self.turn)


def text(request):
    out = []
    for m in request:
        c = m.get("content") or ""
        out.append(c if isinstance(c, str) else " ".join(p.get("text", "") for p in c if isinstance(p, dict)))
    return "\n".join(out)


try:
    from harness.game.arc_host import ArcHost
    from harness.game.games import build_games, make_arcade

    game = build_games(make_arcade("offline", environments_dir=str(Path(_ROOT) / "eval" / "real_games")), "offline",
                       only=["ls20"])[0]
    game.start()
    run = Path(tempfile.mkdtemp())
    c = build_config({"agent": {"vision": True, "limits": {"max_turns": 11}, "memory": {"curator": False},
                                "turn_yield_tokens": 2048, "stuck_note_tokens": 0,
                                "perception": {"ascii": False}}})
    fit_context(c["agent"], 131072)
    llm = ScriptLLM()
    host = ArcHost(game, 400, should_stop=lambda: False, tokens_spent=lambda: 0)
    s = A.AgentSession(cfg=c["agent"], llm=llm, name="ls20", session_dir=run / "games" / "ls20",
                       task=prompts.task_message(game_id=game.game_id, max_actions=400, minutes=45), arc=host,
                       deadline=time.time() + 3600, stop_event=threading.Event(), memory_root=run / "memory")
    s.run()
    game.finish()
    events = [json.loads(x) for x in (run / "games" / "ls20" / "transcript.jsonl").read_text(encoding="utf-8")
              .splitlines()]
    st = s.stats
    R = llm.requests
    ok("e2e_ran", len(R) == 11 and not s.end_reason.startswith("crash") and st["act_actions"] >= 2,
       f"{len(R)} requests, {s.end_reason}")
    ends = [(e["reason"], e["requests"]) for e in events if e.get("event") == "turn_end"]
    ok("turns", ends == [("token_cap", 3), ("token_cap", 1), ("token_cap", 1), ("moved", 1), ("moved", 2),
                         ("moved", 2), ("moved", 1), ("stop", 0)] and st["turn_yields"] == 3 and st["outer_turns"] == 8,
       str(ends))
    ok("no_move_now_note", not any(prompts.ACT_NOW in text(r) for r in R))
    # code that only looks keeps the turn going: the board did not change, so no new picture follows the cell output
    ok("code_keeps_turn", all(R[k][-1]["role"] == "tool" for k in (1, 2, 3)), str([R[k][-1]["role"] for k in (1, 2, 3)]))
    ok("continuation_after_no_tool", R[4][-1]["role"] == "user" and prompts.CALL_TOOL_NOW not in text(R[4][-1:]))
    ok("call_tool_after_cut_off", prompts.CALL_TOOL_NOW in text(R[5][-1:]) and st.get("cut_off_notes") == 1)
    ok("one_note_in_all", text(R[-1]).count(prompts.CALL_TOOL_NOW) == 1)
    ok("slot_handed_over_at_turn_ends", llm.yields == [3, 4, 5, 6, 8, 10, 11], str(llm.yields))
    ok("reply_cap_sent", set(llm.max_tokens) == {12000}, str(set(llm.max_tokens)))
    ok("no_memory_in_any_request", not any("[memory]" in text(r) or "[memory update]" in text(r) for r in R)
       and st["memory_full"] == 0 and st["memory_updates"] == 0)
    obs = [text([m]) for r in R for m in r if m["role"] == "user" and "[state" in text([m])]
    ok("no_board_text", obs and not any("board:" in o or "changed region r" in o for o in obs), obs[0][:300] if obs else "")
    ok("system_prompt_matches", "The board itself is not written" in R[0][0]["content"]
       and prompts.MEMORY_HIDDEN in R[0][0]["content"] and "`HYPOTHESES`" in R[0][0]["content"])
    recall_out = [m["content"] for m in R[9] if m["role"] == "tool"][-1]
    ok("recall_in_kernel", "recall '' in timeline" in recall_out and "## timeline" in recall_out
       and st["recalls"] == 1 and any(e.get("event") == "recall" and e.get("scope") == "timeline" for e in events),
       recall_out[:300])
    hyps = s.memory.working.hypotheses
    ok("kernel_memory_hypothesis", len(hyps) == 1 and hyps[0]["text"] == "A1 moves the block up"
       and hyps[0]["status"] == "kept" and hyps[0]["level"] == 0 and s.memory.working.goals == [], str(hyps))
    out6 = [m["content"] for m in R[6] if m["role"] == "tool"][-1]
    ok("kernel_memory_receipt", "[memory saved: plan L1, h1 added]" in out6 and "GOAL must be a dict" in out6
       and "play: 1 of 1 moves done" in out6, out6[-500:])
    acts = [e for e in events if e.get("event") == "act"]
    ok("kernel_memory_saved_before_moves", acts and acts[0]["plan"] == "go up to the yellow bar", str(acts[:1]))
    ok("kernel_memory_deleted_is_put_back", any(e.get("event") == "memory_restored" and e["names"] == ["HYPOTHESES"]
                                                for e in events) and len(hyps) == 1)
    ok("kernel_memory_refused_is_put_back", any(e.get("event") == "memory_restored" and e["names"] == ["GOAL"]
                                                for e in events))

    # a session that only looks: one turn is max_inner_requests requests, then the slot goes, with no note
    class LookLLM(ScriptLLM):
        def chat(self, messages, tools=None, max_tokens=None, timeout_s=None, template_kwargs=None, thinking=None):
            self.requests.append(messages)
            self.turn += 1
            return Reply("", "", code_call(self.turn, f"v{self.turn} = {self.turn}"), "tool_calls", 5000, 100)

    game.start()
    run3 = Path(tempfile.mkdtemp())
    c3 = build_config({"agent": {"vision": False, "limits": {"max_turns": 14}, "memory": {"curator": False},
                                 "max_inner_requests": 5, "stuck_note_tokens": 0}})
    fit_context(c3["agent"], 131072)
    look = LookLLM()
    s3 = A.AgentSession(cfg=c3["agent"], llm=look, name="ls20", session_dir=run3 / "games" / "ls20",
                        task=prompts.task_message(game_id=game.game_id, max_actions=400, minutes=45),
                        arc=ArcHost(game, 400, should_stop=lambda: False, tokens_spent=lambda: 0),
                        deadline=time.time() + 3600, stop_event=threading.Event(), memory_root=run3 / "memory")
    s3.run()
    game.finish()
    ev3 = [json.loads(x) for x in (run3 / "games" / "ls20" / "transcript.jsonl").read_text(encoding="utf-8")
           .splitlines()]
    ends3 = [(e["reason"], e["requests"]) for e in ev3 if e.get("event") == "turn_end"]
    ok("request_cap", ends3 == [("request_cap", 5), ("request_cap", 5), ("stop", 4)] and look.yields == [5, 10],
       f"{ends3} {look.yields}")
    ok("request_cap_no_note", all(r[-1]["role"] == "tool" for r in look.requests[1:])
       and not any(prompts.ACT_NOW in text(r) or prompts.CALL_TOOL_NOW in text(r) for r in look.requests))
except ImportError as e:   # no arc-agi SDK or games in this checkout
    WARNS.append(f"e2e skipped: {e!r}")


# --- 3. ScheduledLLM.yield_turn ---------------------------------------------------------------------------------
class FakeLLM:
    def chat(self, *a, **k):
        return SimpleNamespace(completion_tokens=100)


sch = PriorityScheduler(1, quantum_calls=10)
a = ScheduledLLM(FakeLLM(), sch, "a", levels_won=lambda: 0, n_levels=7, should_stop=lambda: False)
b = ScheduledLLM(FakeLLM(), sch, "b", levels_won=lambda: 0, n_levels=7, should_stop=lambda: False)
a.yield_turn()
ok("yield_before_any_call_is_noop", not sch._holders and sch._games["a"].grants == 0)
a.chat([])
ok("a_holds_after_call", sch._holders == {"a"})
done = threading.Event()
threading.Thread(target=lambda: (b.chat([]), done.set()), daemon=True).start()
for _ in range(100):
    if sch._games["b"].waiting_since is not None:
        break
    time.sleep(0.01)
ok("b_waits", sch._games["b"].waiting_since is not None and sch._holders == {"a"})
a.yield_turn()
ok("yield_gives_slot_to_waiter", done.wait(2.0) and sch._holders == {"b"} and sch._games["a"].waiting_since is not None)

# a turn-driven game keeps its slot past quantum_calls and after a cold prompt, and gives it at yield_turn
sch2 = PriorityScheduler(1, quantum_calls=2)
t = ScheduledLLM(FakeLLM(), sch2, "t", levels_won=lambda: 0, n_levels=7, should_stop=lambda: False, turn_driven=True)
u = ScheduledLLM(FakeLLM(), sch2, "u", levels_won=lambda: 0, n_levels=7, should_stop=lambda: False)
t.chat([])
done2 = threading.Event()
threading.Thread(target=lambda: (u.chat([]), done2.set()), daemon=True).start()
for _ in range(100):
    if sch2._games["u"].waiting_since is not None:
        break
    time.sleep(0.01)
t.chat([])
t.chat([])
t.mark_cold()
t.chat([])
ok("turn_driven_keeps_slot", sch2._holders == {"t"} and sch2._games["t"].grants == 1 and not done2.is_set())
t.yield_turn()
ok("turn_driven_yield_hands_over", done2.wait(2.0) and sch2._holders == {"u"})

print("\n================ SUMMARY ================")
print(f"CHECKS: {CHECKS[0]}")
print(f"HARD FAILS: {len(FAILS)} -> {FAILS}")
print(f"WARNINGS  : {len(WARNS)} -> {WARNS}")
raise SystemExit(1 if FAILS else 0)

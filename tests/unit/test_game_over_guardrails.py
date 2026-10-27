# --- path shim: these suites live in tests/ but import from the repo root ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_PRIME = _os.environ.get("PRIME_DIR", "Sarbloh-Experimentation")
for _p in (_ROOT, _os.path.join(_ROOT, _PRIME)):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end shim ---
"""The guardrails taken from the Duck (TAAF) harness on 10-27:
- config replay_stop on (the default): a play() path that would repeat, from the start of the level or its last
  reset, the exact moves of a try that ended in GAME_OVER is refused whole before any move is spent; a later
  one-move play() that completes the series is refused too; a different path runs;
- the GAME_OVER note asks why the level was lost (the [GAME_OVER] move, moves(), a bar running out) and says the
  same moves are refused;
- a reply with no tool call gets a note to call ipython, not "continue";
- a <tool_call> written as text (JSON or <function=...> form) runs when the server did not parse it: all of them in
  the reply's text, only the last in its thinking;
- a click outside rows and columns 0 to 63 is refused; tool_output_chars is 4096.

Run: uv run python tests/unit/test_game_over_guardrails.py
"""
import json
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from harness.agent import agent as A, guardrails, prompts
from harness.config import build_config
from harness.llm.client import Reply

FAILS, CHECKS = [], [0]


def ok(name, cond, detail=""):
    CHECKS[0] += 1
    print(("PASS " if cond else "FAIL ") + name + (f"  {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


# --- defaults and prompts -------------------------------------------------------------------------------------------
cfg = build_config({})["agent"]
ok("defaults", cfg["replay_stop"] is True and cfg["recover_tool_calls"] is True and cfg["tool_output_chars"] == 4096)
system = A.system_prompt(cfg)
ok("prompt_replay_refusal", prompts.REPLAY_REFUSAL in system)
ok("prompt_no_replay_refusal_off", prompts.REPLAY_REFUSAL not in A.system_prompt({**cfg, "replay_stop": False}))
ok("prompt_click_range", "rows and columns are 0 to 63" in system)
cont = prompts.continuation(moves_left=300, minutes_left=60)
ok("continue_note", "without calling a tool" in cont and "await play(" in cont, cont)
over = prompts.event_message("game_over", replay_note=prompts.REPLAY_NOTE, moves_left=300, minutes_left=60)
ok("game_over_note", "[GAME_OVER]" in over and "moves()" in over and "bar" in over and prompts.REPLAY_NOTE in over,
   over)
ok("game_over_note_off", prompts.REPLAY_NOTE not in prompts.event_message("game_over", replay_note=""))

# --- click range ----------------------------------------------------------------------------------------------------
ok("click_in_range", guardrails.parse_action("6 0 63") == (6, 0, 63))
for bad in ("6 64 0", "6 0 64", "6 -1 5"):
    try:
        guardrails.parse_action(bad)
        ok(f"click_refused {bad}", False)
    except ValueError as exc:
        ok(f"click_refused {bad}", "0 to 63" in str(exc), str(exc))

# --- tool calls written as text -------------------------------------------------------------------------------------
JSON_CALL = '<tool_call>\n{"name": "ipython", "arguments": {"code": "print(1)"}}\n</tool_call>'
STR_ARGS = '<tool_call>{"name": "ipython", "arguments": "{\\"code\\": \\"print(2)\\"}"}</tool_call>'
XML_CALL = "<tool_call>\n<function=ipython>\n<parameter=code>\nprint(3)\n</parameter>\n</function>\n</tool_call>"
OTHER = '<tool_call>{"name": "web", "arguments": {"code": "x"}}</tool_call>'
ok("text_json", guardrails.text_tool_calls(JSON_CALL) == ["print(1)"])
ok("text_json_string_args", guardrails.text_tool_calls(STR_ARGS) == ["print(2)"])
ok("text_xml", guardrails.text_tool_calls(XML_CALL) == ["print(3)"], str(guardrails.text_tool_calls(XML_CALL)))
ok("text_other_tool_skipped", guardrails.text_tool_calls(OTHER + "<tool_call>not json</tool_call>") == [])


def calls(content, reasoning="", **knobs):
    s = SimpleNamespace(tools=[{"function": {"name": "ipython"}}], stats={"native_calls": 0, "fenced_calls": 0},
                        cfg={"allow_fenced_code": True, "recover_tool_calls": True, **knobs})
    return guardrails.read_calls(s, Reply(content, reasoning, [], "stop", 0, 0)), s.stats


got, st = calls("text " + JSON_CALL + " more " + XML_CALL)
ok("content_all_calls", [c["code"] for c in got] == ["print(1)\n\nprint(3)"] and st.get("recovered_calls") == 1)
got, _ = calls("```python\nprint(9)\n```", reasoning=JSON_CALL)
ok("fenced_before_thinking", [c["code"].strip() for c in got] == ["print(9)"])
got, _ = calls("", reasoning=JSON_CALL + " then " + XML_CALL)
ok("thinking_last_only", [c["code"] for c in got] == ["print(3)"])
got, _ = calls(JSON_CALL, recover_tool_calls=False)
ok("recover_off", got == [])


# --- the replay stop on a real game: sp80 level 1 is lost by FIRE x5 (run 10-26) -------------------------
class ScriptLLM:
    def __init__(self, script):
        self.requests, self.turn, self.script = [], 0, script

    def chat(self, messages, tools=None, max_tokens=None, timeout_s=None, template_kwargs=None, thinking=None):
        self.requests.append(messages)
        self.turn += 1
        code = self.script.get(self.turn, "z = 0")
        call = [{"id": f"call{self.turn}", "type": "function",
                 "function": {"name": "ipython", "arguments": json.dumps({"code": code})}}]
        return Reply("", f"thinking {self.turn}", call, "tool_calls", 5000, 100)

    def yield_turn(self):
        pass


def session(name, script, **agent):
    game = build_games(make_arcade("offline", environments_dir=str(Path(_ROOT) / "eval" / "real_games")),
                       "offline", only=[name])[0]
    game.start()
    run = Path(tempfile.mkdtemp())
    c = build_config({"agent": {"vision": False, "memory": {"curator": False},
                                "limits": {"max_turns": len(script) + 1}, **agent}})
    fit_context(c["agent"], 131072)
    llm = ScriptLLM(script)
    s = A.AgentSession(cfg=c["agent"], llm=llm, name=name, session_dir=run / "games" / name,
                       task=prompts.task_message(game_id=game.game_id, max_actions=400, minutes=45),
                       arc=ArcHost(game, 400, should_stop=lambda: False, tokens_spent=lambda: 0),
                       deadline=time.time() + 3600, stop_event=threading.Event(), memory_root=run / "memory")
    s.run()
    return s, game, llm


def outputs(llm):
    """The tool output of each scripted turn, in order (the request after turn n holds turn n's output last)."""
    out = []
    for req in llm.requests[1:]:
        tools = [m for m in req if m["role"] == "tool"]
        out.append(str(tools[-1].get("content")) if tools else "")
    return out


def try_play(path, tag):
    return (f'try:\n    r = await play({path!r})\n    print("{tag}", r.done, r.stopped)\n'
            f'except RuntimeError as e:\n    print("{tag} REFUSED", e)')


try:
    from harness.game.arc_host import ArcHost
    from harness.game.games import build_games, make_arcade
    from harness.run import fit_context

    LOSE = ["5"] * 5
    LOOP = ('for m in {moves!r}:\n    r = await play([m], quiet=True)\n    if r.state == "GAME_OVER":\n        break\n'
            'print("{tag}", r.last.text)')
    script = {
        1: LOOP.format(moves=LOSE, tag="LOST"),       # one move per play(), so the wall stop never cuts it
        2: try_play(LOSE + ["1"], "SAME"),          # the lost series inside a longer path: refused whole
        3: try_play(["5", "5"], "START"),           # a part of it: runs
        4: try_play(["5", "5", "5"], "REST"),       # completes the series across play() calls: refused whole
        5: try_play(["5", "5"], "TWO"),             # 4 of the 5: runs
        6: try_play(["5"], "LAST"),                 # the losing move by itself: refused
        7: try_play(["1"], "OTHER"),                # a different move: runs
    }
    s, game, llm = session("sp80", script)
    out = outputs(llm)
    ok("e2e_lost", "GAME_OVER" in out[0] and "Find out why" in out[0]
       and prompts.REPLAY_NOTE.strip() in out[0], out[0][:600])
    ok("e2e_same_refused", "SAME REFUSED" in out[1] and "Move 5 of this path" in out[1], out[1][:400])
    ok("e2e_start_runs", "START 2 None" in out[2], out[2][:300])
    ok("e2e_rest_refused", "REST REFUSED" in out[3] and "Move 3 of this path" in out[3], out[3][:400])
    ok("e2e_two_runs", "TWO 2 None" in out[4], out[4][:300])
    ok("e2e_last_refused", "LAST REFUSED" in out[5] and "exact 5 moves" in out[5], out[5][:400])
    ok("e2e_other_runs", "OTHER 1" in out[6], out[6][:300])
    # Moves made: 5 lost + START 2 + TWO 2 + OTHER 1; the refused paths spent nothing.
    ok("e2e_moves_spent", s.stats["act_actions"] == 10 and s.stats.get("replay_stops") == 3,
       f"{s.stats['act_actions']} {s.stats.get('replay_stops')}")
    game.finish()

    # Off: the same series is sent again and loses again.
    s, game, llm = session("sp80", {1: LOOP.format(moves=LOSE, tag="LOST"), 2: LOOP.format(moves=LOSE, tag="AGAIN")},
                           replay_stop=False)
    out = outputs(llm)
    ok("e2e_off_replays", "GAME_OVER" in out[1] and s.stats.get("replay_stops") is None
       and prompts.REPLAY_NOTE.strip() not in out[0], out[1][:300])
    game.finish()
except ImportError as exc:
    print("SKIP e2e:", exc)

print(f"\n{CHECKS[0] - len(FAILS)}/{CHECKS[0]} passed")
_sys.exit(1 if FAILS else 0)

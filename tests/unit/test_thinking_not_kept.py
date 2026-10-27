# --- path shim: these suites live in tests/ but import from the repo root ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_PRIME = _os.environ.get("PRIME_DIR", "Sarbloh")
for _p in (_ROOT, _os.path.join(_ROOT, _PRIME)):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end shim ---
"""Sarbloh defaults: config keep_code and keep_reasoning both on. With keep_reasoning off, stored replies keep their
code but not their thinking:
no old thinking is sent back, the code is, the transcript still logs the thinking, every request is rebuilt from the
transcript (``trace.sft_wire``) with the thinking as the training target, and the system prompt says the thinking is
not kept and where to write what the model needs later. keep_reasoning on sends every thinking back.
"""
import json
import sys
import tempfile
import threading
import time
from pathlib import Path

from harness import trace
from harness.agent import agent as A, prompts
from harness.config import build_config
from harness.llm.client import Reply
from harness.run import fit_context

FAILS, CHECKS = [], [0]


def ok(name, cond, detail=""):
    CHECKS[0] += 1
    print(("PASS " if cond else "FAIL ") + name + (f"  {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


cfg = build_config({})["agent"]
ok("defaults_code_on_thinking_on", cfg["keep_code"] is True and cfg["keep_reasoning"] is True,
   f"{cfg.get('keep_code')} {cfg.get('keep_reasoning')}")

# --- the system prompt ------------------------------------------------------------------------------------------
off = prompts.game_system(act_max=100, code_kept=True, reasoning_kept=False)
on = prompts.game_system(act_max=100, code_kept=True, reasoning_kept=True)
ok("prompt_says_thinking_not_kept", "Your thinking is not kept." in off
   and "write it into GOAL, PLAN, HYPOTHESES or FINDINGS" in off and "Your earlier code and tool outputs stay" in off)
ok("prompt_keeps_old_text_when_on", "Your thinking is not kept." not in on
   and "Your earlier thinking, code and tool outputs stay" in on)

# --- a scripted game: the requests the model received -----------------------------------------------------------
# The moves that won ls20 level 1 in run 20261023_115157 (steps.jsonl, actions 0-34).
LS20_L1 = list("11113332341144411122333212121444111")


def code_call(n, code):
    return [{"id": f"call{n}", "type": "function", "function": {"name": "ipython", "arguments": json.dumps({"code": code})}}]


SCRIPT = {1: code_call(1, 'PLAN["L1"] = "walk"\n' f"r = await play({LS20_L1[:4]!r})"),
          2: code_call(2, "print(repr(PLAN))"),
          3: code_call(3, "x = 3")}


class ScriptLLM:
    def __init__(self):
        self.requests, self.turn = [], 0

    def chat(self, messages, tools=None, max_tokens=None, timeout_s=None, template_kwargs=None, thinking=None):
        self.requests.append(messages)
        self.turn += 1
        if self.turn in SCRIPT:
            return Reply("", f"secret thinking {self.turn}", SCRIPT[self.turn], "tool_calls", 5000, 300)
        return Reply("done", "secret thinking end", [], "stop", 5000, 300)

    def yield_turn(self):
        pass


def play(keep_reasoning):
    from harness.game.arc_host import ArcHost
    from harness.game.games import build_games, make_arcade

    game = build_games(make_arcade("offline", environments_dir=str(Path(_ROOT) / "eval" / "real_games")), "offline",
                       only=["ls20"])[0]
    game.start()
    run = Path(tempfile.mkdtemp())
    c = build_config({"agent": {"limits": {"max_turns": 4}, "memory": {"curator": False},
                                "keep_reasoning": keep_reasoning}})
    fit_context(c["agent"], 131072)
    llm = ScriptLLM()
    s = A.AgentSession(cfg=c["agent"], llm=llm, name="ls20", session_dir=run / "games" / "ls20",
                       task=prompts.task_message(game_id=game.game_id, max_actions=400, minutes=45),
                       arc=ArcHost(game, 400, should_stop=lambda: False, tokens_spent=lambda: 0),
                       deadline=time.time() + 3600, stop_event=threading.Event(), memory_root=run / "memory",
                       tracing=True)
    s.run()
    game.finish()
    path = run / "games" / "ls20" / "transcript.jsonl"
    return llm, [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]


def sent_replies(llm):
    return [m for r in llm.requests[1:] for m in r if m["role"] == "assistant"]


try:
    llm, events = play(keep_reasoning=False)
    sent = sent_replies(llm)
    wire = json.dumps(llm.requests)
    ok("ran", len(llm.requests) >= 3 and sent, f"{len(llm.requests)} requests")
    ok("no_thinking_sent_back", "secret thinking" not in wire
       and not any(m.get("reasoning") or m.get("reasoning_content") or m.get("_reasoning") for m in sent))
    codes = [json.loads(tc["function"]["arguments"])["code"] for m in sent for tc in m.get("tool_calls") or []]
    ok("code_sent_back", any("await play(" in c for c in codes) and not any("code not kept" in c for c in codes),
       str(codes)[:300])
    ok("prompt_in_request", "Your thinking is not kept." in llm.requests[0][0]["content"])
    replies = [e for e in events if e.get("event") == "message" and e.get("role") == "assistant"]
    ok("transcript_keeps_thinking", replies and all((e.get("reasoning") or "").startswith("secret thinking")
                                                    and e.get("reasoning_unsent") for e in replies))
    run_info = {"game_id": "ls20", "levels_completed": 0, "number_of_levels": 7, "actions_per_level": [4],
                "baseline_actions": [21]}
    rows, counts = trace.sft_wire(events, run_info, {}, "games/ls20")
    ok("every_request_rebuilt", counts["turns"] == len(llm.requests) and counts["verified"] == counts["turns"],
       str(dict(counts)))
    targets = [m for r in rows for m in r["messages"] if m["role"] == "assistant" and m.get("train")]
    ok("sft_target_has_thinking", targets and all((m.get("reasoning_content") or "").startswith("secret thinking")
                                                  for m in targets), str(targets[:1])[:300])
    history = [m for r in rows for m in r["messages"] if m["role"] == "assistant" and not m.get("train")]
    ok("sft_history_without_thinking", not any(m.get("reasoning_content") for m in history))

    llm_on, _ = play(keep_reasoning=True)
    ok("on_sends_thinking_back", all(m.get("reasoning_content", "").startswith("secret thinking")
                                     for m in sent_replies(llm_on)) and sent_replies(llm_on))
except Exception as exc:  # noqa: BLE001
    import traceback
    traceback.print_exc()
    ok("e2e_ran", False, repr(exc))

print("================ SUMMARY ================")
print(f"CHECKS: {CHECKS[0]}")
print(f"HARD FAILS: {len(FAILS)} -> {FAILS}")
sys.exit(1 if FAILS else 0)

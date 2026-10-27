# --- path shim: these suites live in tests/ but import from the repo root ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_PRIME = _os.environ.get("PRIME_DIR", "Sarbloh-Experimentation")   # these fixes are in the experimentation copy only
for _p in (_ROOT, _os.path.join(_ROOT, _PRIME)):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end shim ---
"""Sarbloh-Experimentation session details, without a model or a game:
- the continuation is the one word "continue" (the low-budget and low-time notes are still added);
- the reasoning effort is "early" on levels 1 and 2 and "later" after, sent as a chat template argument, and only
  the values the Qwen3.8 template takes are allowed (it has no "high");
- after a trim the full view of the current step is put back once, before the current tool-call chain (it
  replaces the short view of the same step, or goes just after the newest user message), never twice;
- a drain in the middle of a tool-call chain adds no message and writes no summary: the chain stays word for word
  with its thinking, and the newest picture stays.
"""
import sys
import time
from types import SimpleNamespace

from harness.agent import agent as A, compaction, context, prompts
from harness.config import build_config
from harness.llm.client import LLM

FAILS, CHECKS = [], [0]


def ok(name, cond, detail=""):
    CHECKS[0] += 1
    print(("PASS " if cond else "FAIL ") + name + (f"  {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


# --- 1. continuation text -----------------------------------------------------------------------------------
# Since 10-27 a reply with no tool call gets the Duck harness's push to call the tool, not "continue".
ok("continue_calls_tool", prompts.continuation(moves_left=300, minutes_left=60) == prompts.CONTINUE_GAME
   and "without calling a tool" in prompts.CONTINUE_GAME)
low = prompts.continuation(moves_left=5, minutes_left=60)
ok("continue_keeps_low_budget_note", low.startswith(prompts.CONTINUE_GAME + "\n\n") and "Only 5 moves" in low, low)

# --- 2. the client sends the template arguments -------------------------------------------------------------
sent = []
llm = LLM({"base_url": "http://x/v1", "model": "m", "chat_template_kwargs": {"enable_thinking": True}})
llm._post = lambda path, body, timeout: sent.append(body) or {
    "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}], "usage": {}}
llm.chat([{"role": "user", "content": "a"}], template_kwargs={"reasoning_effort": "medium"})
llm.chat([{"role": "user", "content": "a"}])
llm.chat([{"role": "user", "content": "a"}], thinking=False, template_kwargs={"reasoning_effort": "low"})
ok("client_adds_effort", sent[0]["chat_template_kwargs"] == {"enable_thinking": True, "reasoning_effort": "medium"},
   str(sent[0].get("chat_template_kwargs")))
ok("client_default_unchanged", sent[1]["chat_template_kwargs"] == {"enable_thinking": True})
ok("client_thinking_off_wins", sent[2]["chat_template_kwargs"]["enable_thinking"] is False)
ok("client_cfg_not_mutated", llm.cfg["chat_template_kwargs"] == {"enable_thinking": True})


# --- 3. a session stub ---------------------------------------------------------------------------------------
def session(level_done=0, last_step=4, effort="default"):
    cfg = build_config({})["agent"]
    if effort != "default":
        cfg["reasoning_effort"] = effort
    s = A.AgentSession.__new__(A.AgentSession)
    s.cfg, s.events, s.messages = cfg, [], []
    s.append_only = False   # sections 4 and 5 test the old builder (tests/unit/test_append_only_context.py: the new one)
    s._log_event = s.events.append
    state = SimpleNamespace(levels_completed=level_done)
    s.arc = SimpleNamespace(game=SimpleNamespace(state=state, action_count=10), budget_left=300)
    s.deadline = time.time() + 3600
    s._effort_sent, s._last_step, s.llm = None, last_step, None
    s.refreshes = 0

    def refresh():
        s.refreshes += 1
        return {"role": "user", "_kind": "observation", "_full": True, "_step": s._last_step, "content": "FULL"}
    s._observation_refresh = refresh
    return s


s = session()
ok("config_default_effort", s.cfg["reasoning_effort"] == {"early": "medium", "later": "medium"})
s = session(effort={"early": "xhigh", "later": "medium"})   # the split still works when asked for
seen = []
for done in (0, 1, 2, 3):
    s.arc.game.state.levels_completed = done
    seen.append(s._effort())
ok("effort_by_level", seen == [{"reasoning_effort": "xhigh"}] * 2 + [{"reasoning_effort": "medium"}] * 2, str(seen))
ok("effort_logged_on_change", [e["value"] for e in s.events if e["event"] == "reasoning_effort"] == ["xhigh", "medium"])
ok("effort_none_sends_nothing", session(effort=None)._effort() is None
   and session(effort={"early": None, "later": "low"})._effort() is None)
try:
    session(effort={"early": "high", "later": "medium"})._effort()
    ok("effort_rejects_high", False)
except ValueError:
    ok("effort_rejects_high", True)


# --- 4. the full view after a compaction --------------------------------------------------------------------
def obs(step, full, image=False):
    m = {"role": "user", "_kind": "observation", "_full": full, "_step": step, "content": f"obs {step} {full}"}
    if image:
        m["content"] = [{"type": "text", "text": m["content"]}, {"type": "image_url", "image_url": {"url": "data:x"}}]
        m["_image_step"] = step
    return m


def act(i, reasoning="think"):
    return {"role": "assistant", "content": "", "_reasoning": reasoning,
            "tool_calls": [{"id": f"t{i}", "type": "function", "function": {"name": "ipython", "arguments": "{}"}}]}


def tool(i):
    return {"role": "tool", "tool_call_id": f"t{i}", "content": "out"}


s = session()
kept = [obs(4, False, image=True), act(1), tool(1)]
s._refresh_after_compaction(kept)
ok("short_view_replaced_in_place", len(kept) == 3 and kept[0]["content"] == "FULL" and kept[-1]["role"] == "tool"
   and s.refreshes == 1, str(kept))
kept = [obs(4, True), act(1), tool(1)]
s._refresh_after_compaction(kept)
ok("full_view_not_repeated", s.refreshes == 1 and kept[0]["content"] == "obs 4 True")
kept = [act(1), tool(1)]
s._refresh_after_compaction(kept)
ok("cut_observation_put_before_chain", kept[0]["content"] == "FULL" and kept[1]["role"] == "assistant")
kept = [obs(4, False), act(1), tool(1), {"role": "user", "content": "continue"}]
s._refresh_after_compaction(kept)
ok("after_newest_user_message", kept[-1]["content"] == "FULL" and kept[0]["content"] == "obs 4 False")


# --- 5. a drain in the middle of a chain adds nothing and keeps the chain ------------------------------------
s = session()
s.messages = [{"role": "user", "content": "task"}, obs(3, True, image=True), act(0), tool(0),
              obs(4, False, image=True), act(1), tool(1)]
chain = [dict(m) for m in s.messages[4:]]
s.stats = {"drains": 0, "context_trims": 0}
s._pinned, s._system, s.tracing, s._objects = None, "SYS", False, []
s._usage_tokens, s._usage_at, s._token_scale = None, 0, 1.0
s._drain(110_000, "threshold")
ok("drain_mid_chain_adds_nothing", s.messages[-3:] == chain and s.stats["drains"] == 1
   and not any(m.get("_kind") == compaction.COMPACTION_KIND for m in s.messages), str([m["role"] for m in s.messages]))
wire = context.build("SYS", s.messages, None, vision=True)
ok("drain_mid_chain_keeps_thinking", wire[-2].get("reasoning_content") == "think" and wire[-1]["role"] == "tool",
   str(wire[-2])[:200])
ok("drain_keeps_newest_picture_only", sum(1 for m in wire if isinstance(m["content"], list)
                                          for p in m["content"] if p.get("type") == "image_url") == 1)

print("\n================ SUMMARY ================")
print(f"CHECKS: {CHECKS[0]}")
print(f"HARD FAILS: {len(FAILS)} -> {FAILS}")
sys.exit(1 if FAILS else 0)

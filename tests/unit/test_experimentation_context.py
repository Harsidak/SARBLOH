# --- path shim: these suites live in tests/ but import from the repo root ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_PRIME = _os.environ.get("PRIME_DIR", "Sarbloh-Experimentation")   # these changes are in the experimentation copy only
for _p in (_ROOT, _os.path.join(_ROOT, _PRIME)):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end shim ---
"""Sarbloh-Experimentation context window, without a model or a game:
- the SGLang NVFP4 profiles serve 131072 tokens; the agent drains above 118k (cap 131072), or window - reserve
  when lower;
- the old builder (append_only off) for sections 2 to 8:
- the thinking re-sent in the current chain is newest first, up to about 16k tokens (the newest reply's always goes);
  older replies keep their tool calls and outputs; the agent's estimate counts exactly what is sent;
- the compaction estimate counts the stored thinking, so the cut point sees it;
- the summarizer in compaction.py (no longer called by the agent) is capped and replaces the previous summary;
- the drain (no summary) drops the oldest whole turns, as the Duck harness does, until the context is under its 55k
  target: the task stays first and every kept message is unchanged; it trims when the newest turn alone is too big; fit_context refuses a trigger too near the cap;
- overflows never end the game: drain, then trim in code, then reset;
- a level-up resets the context in code: the level-up message (with the kernel's names, or with the code graph
  after it when the graph is on) and the new full state;
- the kernel lists the names the agent defined, from its globals.
"""
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

from harness.agent import agent as A, compaction, context
from harness.config import build_config
from harness.llm import qwen
from harness.run import fit_context
from harness.runtime.kernel import Kernel

FAILS, CHECKS = [], [0]


def ok(name, cond, detail=""):
    CHECKS[0] += 1
    print(("PASS " if cond else "FAIL ") + name + (f"  {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


# --- 1. window and trigger ------------------------------------------------------------------------------------
sgl = [p for p in qwen.PROFILES if p.startswith("sglang_")]
ok("sglang_window_131072", sgl and all(qwen.SPEC.max_model_len(p) == 131072 for p in sgl),
   str({p: qwen.SPEC.max_model_len(p) for p in sgl}))
cfg = build_config({})["agent"]
fit_context(cfg, 131072)
c = cfg["compaction"]
ok("trigger_118k_cap_131k", c["trigger_tokens"] == 118000 and c["cap_tokens"] == 131072
   and c["drain_target_tokens"] == 70000 and "keep_recent_tokens" not in c
   and not any(k in c for k in ("summary_tokens", "in_context", "floor_gap_tokens")), str(c))
ok("no_drain_fraction", "drain_max_fraction" not in cfg)
ok("drains_above_118k_at_139264", compaction.should_compact(118001, 139264, c["reserve_tokens"], c["trigger_tokens"])
   and not compaction.should_compact(117999, 139264, c["reserve_tokens"], c["trigger_tokens"]))
ok("trigger_decides_at_131072", compaction.should_compact(118001, 131072, c["reserve_tokens"], c["trigger_tokens"])
   and not compaction.should_compact(117999, 131072, c["reserve_tokens"], c["trigger_tokens"]))
near = build_config({"agent": {"compaction": {"cap_tokens": 125000}}})["agent"]
try:
    fit_context(near, 131072)
    refused = False
except ValueError:
    refused = True
ok("fit_refuses_trigger_near_cap", refused)
small = build_config({})["agent"]
fit_context(small, 40000)
ok("small_window_target_below_trigger", small["compaction"]["drain_target_tokens"] < 40000 - small["compaction"]
   ["reserve_tokens"], str(small["compaction"]))
ok("chain_budget_config", cfg["chain_reasoning_tokens"] == 2000)


# --- 2. the thinking budget -----------------------------------------------------------------------------------
def act(i, tokens=0):
    m = {"role": "assistant", "content": "",
         "tool_calls": [{"id": f"t{i}", "type": "function", "function": {"name": "ipython", "arguments": "{}"}}]}
    if tokens:
        m["_reasoning"] = "x" * (4 * tokens)
    return m


def tool(i, text="out"):
    return {"role": "tool", "tool_call_id": f"t{i}", "content": text}


def obs(step, full):
    return {"role": "user", "_kind": "observation", "_full": full, "_step": step, "content": f"obs {step} {full}"}


msgs = [{"role": "user", "content": "task"}, act(0, 3000), tool(0), obs(1, True),
        act(1, 7000), tool(1), act(2, 6000), tool(2), act(3, 9000), tool(3)]
sent = context.reasoning_sent(msgs, 16000)
ok("newest_first_within_budget", sent == [8, 6], str(sent))
wire = context.build("SYS", msgs, None, vision=False, reasoning_budget=16000)
with_thinking = [i for i, m in enumerate(wire) if m.get("reasoning_content")]
ok("older_lose_thinking_keep_calls", all(m.get("tool_calls") for m in wire if m["role"] == "assistant")
   and len(with_thinking) == 2 and not any("_reasoning" in m for m in wire), str(with_thinking))
ok("before_chain_never_sent", 1 not in sent)
big = [obs(1, True), act(1, 3000), tool(1), act(2, 20000), tool(2)]
ok("newest_always_sent", context.reasoning_sent(big, 16000) == [3])

s = A.AgentSession.__new__(A.AgentSession)
s.cfg, s.messages, s.append_only = {**cfg, "chain_reasoning_tokens": 16000}, msgs, False   # the budget of the cases above
wire_tokens = sum(-(-len(m["reasoning_content"]) // 4) for m in wire if m.get("reasoning_content"))
ok("estimate_matches_request", s._chain_reasoning_tokens(0) == wire_tokens == 15000,
   f"{s._chain_reasoning_tokens(0)} vs {wire_tokens}")
ok("estimate_from_start", s._chain_reasoning_tokens(7) == 9000)

# --- 3. the compaction estimate counts thinking ---------------------------------------------------------------
m = act(9, 1000)
ok("estimate_counts_reasoning", compaction.estimate_tokens(m) - compaction.estimate_tokens(m, reasoning=False) == 1000)
long_think = [obs(1, True), act(1), tool(1), obs(2, False), act(2, 20000), tool(2), obs(3, False), act(3), tool(3)]
cut, _, _ = compaction.find_cut_point(long_think, 0, 16000)
ok("cut_point_sees_thinking", cut >= 4, str(cut))


# --- 4. the summary: capped, replacing, skipped when nothing is new -------------------------------------------
class FakeLLM:
    def __init__(self):
        self.calls = []

    def chat(self, messages, tools=None, max_tokens=None, thinking=None):
        self.calls.append((messages[-1]["content"], max_tokens))
        return SimpleNamespace(content="NOTE")


def prep(to_summarize, prefix=(), previous=None):
    return compaction.Preparation(first_kept=0, to_summarize=list(to_summarize), turn_prefix=list(prefix),
                                  is_split=bool(prefix), previous_summary=previous, anchor=None, tokens_before=0)


llm = FakeLLM()
out = compaction.summarize(llm, prep([obs(1, True)], previous="OLD"), 2048, RuntimeError)
text, cap = llm.calls[-1]
ok("summary_capped_2k", cap == 2048 and "at most 1000 words" in text, str(cap))
ok("summary_replaces_previous", "replaces it" in text and "<previous-summary>\nOLD" in text
   and "PRESERVE" not in text and "Python kernel keeps running" in text)
llm = FakeLLM()
out = compaction.summarize(llm, prep([obs(1, True)], prefix=[act(1), tool(1)]), 2048, RuntimeError)
ok("split_shares_cap", [c for _, c in llm.calls] == [1536, 512] and "Turn Context" in out, str(llm.calls))
llm = FakeLLM()
out = compaction.summarize(llm, prep([], previous="OLD"), 2048, RuntimeError)
ok("nothing_new_keeps_previous", out == "OLD" and not llm.calls)


# --- 5. a session stub ----------------------------------------------------------------------------------------
class FakeKernel:
    def user_names(self):
        return {"functions": ["walls(grid)"], "classes": [], "values": ["board: list[64]"], "modules": []}


def session(messages=(), last_step=4):
    s = A.AgentSession.__new__(A.AgentSession)
    s.cfg = build_config({})["agent"]
    fit_context(s.cfg, 139264)
    s.append_only = False
    s.events, s.messages = [], list(messages)
    s._log_event = s.events.append
    s.arc = SimpleNamespace(game=SimpleNamespace(state=SimpleNamespace(levels_completed=0), action_count=10),
                            budget_left=300, finished=False)
    s.deadline = time.time() + 3600
    s.kernel, s.llm = FakeKernel(), None
    s.stats = {k: 0 for k in ("drains", "continuations", "llm_failures", "context_overflows", "context_trims",
                              "context_resets", "graph_snapshots")}
    s._pinned, s._system, s.tracing = None, "SYS", False
    s._usage_tokens, s._usage_at, s._token_scale = None, 0, 1.0
    s._overflows, s._level_up_text = 0, None
    s._last_step = last_step
    s.refreshes = 0

    def refresh():
        s.refreshes += 1
        return {"role": "user", "_kind": "observation", "_full": True, "_step": s._last_step, "content": "FULL"}
    s._observation_refresh = refresh
    return s


# --- 6. the drain: no summary, under the cap, the newest messages word for word ---------------------------------
s = session()
ok("drain_gate_below_trigger", not s._wants_drain(117_000))
ok("drain_gate_above_trigger", s._wants_drain(119_000))


def history(turns, think=1500, out_chars=12000):
    """A level's history: a full board, then per turn a reply with thinking, a long ipython output and a short
    board."""
    h = [{"role": "user", "content": "task"}, obs(0, True)]
    for i in range(turns):
        h += [act(i, think), tool(i, f"out {i} " + "y" * out_chars), obs(i + 1, False)]
    return h


s = session(history(60), last_step=60)
before = s._context_tokens()
tail = [dict(m) for m in s.messages[-6:]]
s._drain(before, "threshold")
after = s._context_tokens()
ev = [e for e in s.events if e.get("event") == "drain"]
ok("drain_started_above_trigger", before > 100_000, str(before))
ok("drain_under_target", after <= s.cfg["compaction"]["drain_target_tokens"], f"{before} -> {after}")
ok("drain_no_summary_no_model", s.llm is None and not any(m.get("_kind") == compaction.COMPACTION_KIND
                                                           for m in s.messages))
ok("drain_keeps_newest_word_for_word", s.messages[-6:] == tail)
# as the Duck harness drains: the task stays first, old turns go whole, every kept message is unchanged
originals = history(60)
ok("drain_drops_whole_turns", s.messages[0]["content"] == "task" and len(s.messages) < len(originals)
   and all(m in originals for m in s.messages) and after > 30_000, f"{len(s.messages)} messages, {after} tokens")
ok("drain_logged", ev and ev[0]["reason"] == "threshold" and ev[0]["keep_tokens"] <= 70000
   and s.stats["drains"] == 1 and s.stats["context_trims"] == 0, str(ev))

# a single huge newest reply: even the smallest tail is above the target, so the drain trims as well
s = session(history(20) + [act(99, 70_000), tool(99, "newest")], last_step=20)
s._drain(s._context_tokens(), "threshold")
ok("drain_trims_when_tail_too_big", s.stats["drains"] == 1 and s.stats["context_trims"] == 1
   and s.messages[-1]["content"] == "newest")

# --- 7. overflows: drain, trim, reset -----------------------------------------------------------------------
head_msg = compaction.head_message("S", "")
msgs = [obs(2, False), act(0, 50), tool(0), obs(3, True), act(1, 50), tool(1, "first line\n" + "y" * 5000),
        obs(4, False), act(2, 50), tool(2, "z" * 3000), act(3, 60), tool(3, "newest output")]
s = session(msgs, last_step=4)
s._trim_context()
m = s.messages
ok("trim_back_to_full_state", m[0]["content"] == "obs 3 True" and len(m) == 8, str([x.get("content", "")[:20] for x in m]))
ok("trim_thinking_newest_only", [x.get("_reasoning") is not None for x in m if x["role"] == "assistant"]
   == [False, False, True])
tools_ = [x["content"] for x in m if x["role"] == "tool"]
ok("trim_tool_placeholders", tools_[0].startswith("[output removed") and "first line" in tools_[0]
   and "\n" not in tools_[0] and tools_[1].startswith("[output removed") and tools_[2] == "newest output", str(tools_))
ok("trim_refreshes_current_full_view", s.refreshes == 1 and any(x.get("content") == "FULL" for x in m))
ok("trim_no_model_call", s.llm is None and s.stats["context_trims"] == 1)

s = session([obs(4, True), act(1, 10), tool(1)])
calls = []
s._drain = lambda tokens, reason: calls.append(("drain", reason))
s._trim_context = lambda: calls.append(("trim",))
for _ in range(4):
    s._on_overflow(RuntimeError("too long"), 140_000)
ok("overflow_sequence", calls == [("drain", "overflow"), ("trim",)] and s.stats["context_resets"] == 2, str(calls))
ok("overflow_never_ends", not getattr(s, "end_reason", "") and s.stats["llm_failures"] == 1)

# --- 8. level-up reset ----------------------------------------------------------------------------------------
full = obs(5, True)
before = [head_msg, obs(4, False), act(1, 10), tool(1), act(2, 10), tool(2, "act: 1 of 1 done; level up"), full]
s = session(list(before), last_step=5)
s._level_up_text = "act: 1 of 1 actions done; stopped: level up\nLEVEL UP"
s._reset_if_level_up()
ok("level_up_default_appends", s.messages == before and s._level_up_text is None
   and s.stats["context_resets"] == 0 and build_config({})["agent"]["level_up_reset"] is False)
s = session(list(before), last_step=5)
s.cfg["level_up_reset"] = True
s.code_graph = False   # the kernel's names in the reset message
s._level_up_text = "act: 1 of 1 actions done; stopped: level up\nLEVEL UP"
s._reset_if_level_up()
m = s.messages
ok("level_up_reset", len(m) == 2 and m[0]["role"] == "user" and m[0]["content"].startswith("act: 1 of 1")
   and m[1] is full and s.refreshes == 0, str(m)[:300])
ok("level_up_names", "walls(grid)" in m[0]["content"])
ok("level_up_state_cleared", s._pinned is None and s._usage_tokens is None
   and s._level_up_text is None and s.stats["context_resets"] == 1)
# with the code graph on, the functions are in the graph message after the reset message, not in its text
s = session(list(before), last_step=5)
s.cfg["level_up_reset"] = True
s._graph = {"walls": {"kind": "function", "sig": "walls(grid)", "calls": [], "missing": [], "hash": "h",
                      "lines": 3, "doc": ""}}
s._level_up_text = "act: 1 of 1 actions done; stopped: level up\nLEVEL UP"
s._reset_if_level_up()
m = s.messages
ok("level_up_graph_message", len(m) == 3 and m[1]["_kind"] == context.GRAPH_KIND and "walls(grid)" in m[1]["content"]
   and "walls(grid)" not in m[0]["content"] and m[2] is full and s.stats["graph_snapshots"] == 1, str(m)[:400])
wire = context.build("SYS", m, {"role": "user", "content": "[memory] m"}, vision=False)
ok("level_up_wire", [x["role"] for x in wire] == ["system", "user"] and "[memory]" in wire[1]["content"]
   and "LEVEL UP" in wire[1]["content"] and "obs 5 True" in wire[1]["content"])
s = session([full], last_step=5)
s.arc.finished = True
s._level_up_text = "won"
s._reset_if_level_up()
ok("no_reset_when_finished", s.messages == [full] and s._level_up_text is None)

# --- 9. the kernel lists the agent's names --------------------------------------------------------------------
k = Kernel(Path(tempfile.mkdtemp()) / "work", lambda r: {})
try:
    k.start()
    empty = k.user_names()
    k.execute("def walls(grid, color=5):\n    return 1\nclass Model: pass\nimport collections as C2\n"
              "board = [[0] * 3] * 4\nn = 7")
    names = k.user_names()
    left = k.execute("print(sorted(x for x in globals() if x in ('_names', '_k', '_v', '_sig', '_inspect', '_n')))")
finally:
    k.close()
ok("kernel_names_empty_at_boot", empty == {"functions": [], "classes": [], "values": [], "modules": []}, str(empty))
ok("kernel_names", names == {"functions": ["walls(grid, color=5)"], "classes": ["Model"],
                             "values": ["board: list[4]", "n: int"], "modules": ["C2"]}, str(names))
ok("kernel_names_leave_nothing", left.stdout.strip() == "[]", left.stdout)

# --- 10. prefix-cache settings: old pictures, short view, observe(), pre-warm -----------------------------------
from harness.agent import perception, prompts  # noqa: E402


def pic(step):
    return {"role": "user", "_kind": "observation", "_image_step": step,
            "content": [{"type": "text", "text": f"obs {step}"}, {"type": "image_url", "image_url": {"url": "x"}}]}


hist = [pic(1), {"role": "assistant", "content": "a"}, pic(2)]
kept = context.build("SYS", hist, append_only=True, old_images=True)
newest = context.build("SYS", hist, append_only=True, old_images=False)
n_img = lambda w: sum(1 for m in w if isinstance(m["content"], list) for p in m["content"]
                      if p.get("type") == "image_url")
ok("old_images_on_sends_all", n_img(kept) == 2)
ok("old_images_off_sends_newest", n_img(newest) == 1 and "image of step 1" in str(newest[1]["content"])
   and build_config({})["agent"]["perception"]["old_images"] is False)

grid = [[0] * 8 for _ in range(8)]
moved = [r[:] for r in grid]
moved[2][2] = moved[2][3] = 4
objs = perception.segment(moved)[0]
sc = perception.Scene(3, moved, objs, change="#3 A1(up): moved", prev=grid, status="game x")
ok("short_view_objects_on", "objects in the changed region" in sc.text(show_ascii=False))
ok("short_view_objects_off", "objects in the changed region" not in sc.text(show_ascii=False, short_objects=False)
   and "last step: #3" in sc.text(show_ascii=False, short_objects=False))
off = prompts.game_system(act_max=100, short_objects=False, observe_view=False)
on = prompts.game_system(act_max=100)
ok("prompt_follows_views", "in your code only" in off and "the area that changed" not in off
   and "in the output of a call that runs `observe()`, you get the full state" in on)
dflt = build_config({})["agent"]["perception"]
ok("view_defaults", dflt["short_objects"] is False and dflt["observe_view"] is False)


class WarmLLM:
    def __init__(self):
        self.calls = []

    def chat(self, messages, **kw):
        self.calls.append((messages, kw))


warm = WarmLLM()
wcfg = build_config({"agent": {"reasoning_effort": {"early": "low", "later": "medium"}}})["agent"]
sent = A.prewarm(wcfg, warm)
efforts = [(kw["template_kwargs"] or {}).get("reasoning_effort") for _, kw in warm.calls]
ok("prewarm_each_effort", sent == 2 and efforts == ["low", "medium"]
   and all(m[0]["content"] == A.system_prompt(wcfg) and kw["max_tokens"] == 1 for m, kw in warm.calls), str(efforts))
ok("prewarm_default_on", build_config({})["prewarm_system"] is True)

print("\n================ SUMMARY ================")
print(f"CHECKS: {CHECKS[0]}")
print(f"HARD FAILS: {len(FAILS)} -> {FAILS}")
sys.exit(1 if FAILS else 0)

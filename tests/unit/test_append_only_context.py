# --- path shim: these suites live in tests/ but import from the repo root ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_PRIME = _os.environ.get("PRIME_DIR", "Sarbloh-Experimentation")   # these changes are in the experimentation copy only
for _p in (_ROOT, _os.path.join(_ROOT, _PRIME)):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end shim ---
"""The append-only context (Sarbloh-Experimentation), the cold hand-over and the host-memory guard, without a model:
- a real session on ls20 with a scripted model: every request starts with the whole previous request word for word,
  its last message included, then that request's reply, except right after a rewrite (drain, trim, reset); the
  hybrid model's prefix cache needs this, as the server keeps its recurrent state only at the end of an earlier
  request; the picture of the board is a stored message after each new state, so a request after a move ends with
  it; the old builder breaks that on almost every new board;
- every reply's thinking is sent; the memory goes in full first, then only its changed sections;
- at the trigger a drain runs (thinking before the tail, old pictures and memory messages out, one full memory back
  in); there is no summary, and every request stays under the cap;
- ``context.drain``, ``context.memory_update``, the masking of old outputs (printed boards and shown states become
  stubs, code and errors stay, the code-graph line goes) and the tail picture on hand-built messages;
- the scheduler hands a slot over when the game's prompt was rewritten, not before its first call;
- ``sysmon`` against a fake /proc: process tree, OOM marks, memory snapshot; the server watchdog writes ``ram`` rows.

Run: uv run python tests/unit/test_append_only_context.py
"""
import json
import re
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from harness import sysmon
from harness.agent import agent as A, compaction, context, prompts
from harness.config import build_config
from harness.llm.client import Reply
from harness.llm.server import WATCHDOG, LlmServer
from harness.run import fit_context
from harness.scheduler import PriorityScheduler, ScheduledLLM

FAILS, WARNS, CHECKS = [], [], [0]


def ok(name, cond, detail=""):
    CHECKS[0] += 1
    print(("PASS " if cond else "FAIL ") + name + (f"  {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


# --- 1. config -------------------------------------------------------------------------------------------------
cfg = build_config({})["agent"]
ok("default_append_only", cfg["append_only"] is True and cfg["memory_full_every"] == 10
   and "drain_output_chars" not in cfg and cfg["compaction"]["trigger_tokens"] == 118000
   and cfg["compaction"]["cap_tokens"] == 131072 and "keep_recent_tokens" not in cfg["compaction"]
   and "in_context" not in cfg["compaction"] and cfg["max_tokens_per_turn"] == 12000
   and cfg["compaction"]["reserve_tokens"] == 12288 and cfg["memory"]["in_context"] is False)
ok("watchdog_ram_every_minute", WATCHDOG["ram_every_s"] == 60.0)
# the reply cap: one int on every level (the default), or "early" on levels 1 and 2 and "later" after
cap_at = lambda cap, level: A.AgentSession._max_tokens(SimpleNamespace(cfg={"max_tokens_per_turn": cap},
                                                                       _level=lambda: level))
ok("max_tokens_by_level", [cap_at({"early": 8192, "later": 4096}, lv) for lv in (1, 2, 3, 7)] == [8192, 8192, 4096, 4096]
   and cap_at(4096, 5) == 4096)
small = build_config({})["agent"]
fit_context(small, 16384)
ok("fit_context_caps_both_levels", small["max_tokens_per_turn"] == small["compaction"]["reserve_tokens"] < 12000,
   str(small["max_tokens_per_turn"]))


# --- 2. pieces on hand-built messages --------------------------------------------------------------------------
def pic(step, text):
    return {"role": "user", "_kind": "observation", "_image_step": step, "_step": step, "_full": False,
            "content": [{"type": "text", "text": text}, {"type": "image_url", "image_url": {"url": "data:x"}}]}


def reply(i, think, call=True):
    m = {"role": "assistant", "content": "", "_reasoning": think}
    if call:
        m["tool_calls"] = [{"id": f"c{i}", "type": "function", "function": {"name": "ipython", "arguments": "{}"}}]
    return m


def tool(i):
    return {"role": "tool", "tool_call_id": f"c{i}", "content": f"out {i}"}


blocks = {"goal": "reach the door", "plan": "go up"}
mem = context.pinned_message(blocks, "status")
upd = context.memory_update({**blocks, "plan": "go left"}, context.changed_sections({**blocks, "plan": "go left"}, blocks))
ok("changed_sections", context.changed_sections({**blocks, "plan": "go left"}, blocks) == ["plan"]
   and context.changed_sections(dict(blocks), blocks) == [])
ok("memory_update_text", upd["_kind"] == context.MEMORY_UPDATE_KIND and "go left" in upd["content"]
   and "reach the door" not in upd["content"] and context.is_memory(upd) and context.is_memory(mem))
ok("memory_cleared_shows_empty", "(none yet)" in context.memory_update({"goal": "g"}, ["plan"])["content"])

hist = [{"role": "user", "content": "task"}, mem, pic(0, "board 0"), reply(1, "t1"), tool(1), upd, pic(1, "board 1"),
        reply(2, "t2"), tool(2), pic(2, "board 2"), reply(3, "t3"), tool(3)]
wire = context.build("SYS", hist, append_only=True, vision=True)
ok("build_sends_every_thinking", [m.get("reasoning_content") for m in wire if m["role"] == "assistant"]
   == ["t1", "t2", "t3"])
ok("build_sends_every_picture", sum(1 for m in wire if isinstance(m["content"], list)
                                    for p in m["content"] if p.get("type") == "image_url") == 3)
ok("build_no_private_keys", all(not k.startswith("_") for m in wire for k in m))
ok("build_text_only_stubs_pictures", all(p.get("type") != "image_url" for m in
                                         context.build("SYS", hist, append_only=True, vision=False)
                                         if isinstance(m["content"], list) for p in m["content"]))
d = context.drain(hist)
ok("drain_thinking_only_in_chain", [m.get("_reasoning") for m in d if m["role"] == "assistant"] == [None, None, "t3"])
ok("drain_one_picture", sum(1 for m in d if context._has_image(m)) == 1 and context._has_image(d[-3]))
ok("drain_drops_memory", not any(context.is_memory(m) for m in d))
ok("drain_keeps_calls_and_outputs", sum(1 for m in d if m["role"] == "tool") == 3
   and all(m.get("tool_calls") for m in d if m["role"] == "assistant"))
ok("drain_leaves_input", hist[3].get("_reasoning") == "t1" and context._has_image(hist[2]))
# the deep drain: a word-for-word tail from keep_from, old boards as stubs, ipython outputs cut, act results as a
# short move log (change lines cut with their marks kept, event notes dropped)
big = {"role": "tool", "tool_call_id": "c1", "content": "y" * 3000 + "\n[game: level 0/7]"}
act_out = {"role": "tool", "tool_call_id": "c2", "content": "act: 1 of 1 actions done\n#0 A1(up): " + "m" * 900
           + " [repeat: same state and action as #3] [GAME_OVER]\nGAME_OVER: this level is lost.\n1. Find the step"
           + "\n[game: level 0/7]"}
board = lambda step: {**pic(step, f"[state after step #{step}] game ls20\n" + "0" * 4000), "_step": step}
deep = [{"role": "user", "content": "task"}, mem, board(0), reply(1, "t1"), big, reply(2, "t2"), act_out, upd,
        board(1), reply(3, "t3"), tool(3), reply(4, "t4"), tool(4), board(2), reply(5, "t5"), tool(5)]
dd = context.drain(deep, keep_from=11, output_chars=400)
texts = [compaction._text(m) for m in dd]
ok("deep_drain_tail_keeps_thinking", [m.get("_reasoning") for m in dd if m["role"] == "assistant"]
   == [None, None, None, "t4", "t5"])
ok("deep_drain_stubs_old_boards", sum("the board is removed to save space" in t for t in texts) == 2
   and texts[1].startswith("[state after step #0: the board") and all(m.get("_kind") != "observation" or m["_step"] == 2
                                                                         for m in dd))
ok("deep_drain_keeps_newest_board", any(m.get("_step") == 2 and "0" * 4000 in compaction._text(m) for m in dd)
   and sum(1 for m in dd if context._has_image(m)) == 1)
ok("deep_drain_cuts_ipython_keeps_status", len(texts[3]) < 600 and "more characters removed" in texts[3]
   and texts[3].endswith("[game: level 0/7]"))
short = texts[5].splitlines()
ok("deep_drain_act_result_move_log", len(short) == 3 and short[0] == "act: 1 of 1 actions done"
   and short[1].startswith("#0 A1(up): mmm") and short[1].endswith("... [repeat: same state and action as #3] [GAME_OVER]")
   and len(short[1]) <= context.ACT_LINE_CHARS + 4 and short[2] == "[game: level 0/7]", texts[5][:400])
ok("deep_drain_no_memory", not any(context.is_memory(m) for m in dd))
ok("deep_drain_newest_board_never_stubbed", any(m.get("_step") == 2 and "0" * 4000 in compaction._text(m)
                                               for m in context.drain(deep, keep_from=len(deep), output_chars=400)))
ok("estimate_counts_pictures", compaction.estimate_tokens(pic(0, "x" * 40), image_tokens=500) == 510
   and compaction.estimate_tokens(pic(0, "x" * 40)) == 10)

# masking: an ipython output stored with its pieces; a drain masks each kind its own way
rows = "\n".join(f"{r:>3} " + "BBWWRRGG" * 8 for r in range(20))
arr = "\n".join(" [" + " ".join(["0"] * 12) + "]" for _ in range(6))
ok("mask_boards", context.mask_boards(f"head\n{rows}\nmid\n{arr}\ntail") ==
   "head\n[a printed board of 20 rows, removed to save space]\nmid\n[a printed board of 6 rows, removed to save space]\ntail"
   and context.mask_boards("a\nTraceback\nb") == "a\nTraceback\nb"
   and context.mask_boards("[" + ", ".join(["'" + "BW" * 32 + "'"] * 20) + "]").startswith("[a printed board of"))
err = "Traceback (most recent call last):\n  File \"<cell-3>\", line 2\nNameError: name 'bfs' is not defined"
out_text, spans = context.cell_output([("out", "found 3 walls\n" + rows + "\n" + "z" * 900), ("error", err),
                                       ("state", "[state after step #7] game ls20\n" + "0" * 3000),
                                       ("memory", "[memory saved: plan]"), ("graph", "[code graph: -bfs]"),
                                       ("status", "[game: level 0/7]")])
ok("cell_output_spans", [k for k, *_ in spans] == ["out", "error", "state", "memory", "graph", "status"]
   and all(out_text[a:b] for _, a, b in spans) and out_text.endswith("[game: level 0/7]"))
cell_msg = {"role": "tool", "tool_call_id": "c9", "content": out_text, "_spans": spans,
            "_act_log": "act: 1 of 1 actions done\n#7 A1(up): obj 1 moved up"}
short = context.short_cell_output(cell_msg, 400)["content"]
ok("drain_masks_cell_output", short.startswith("found 3 walls\n[a printed board of 20 rows") and err in short
   and "[state after step #7: removed to save space]" in short and "0" * 100 not in short
   and "[code graph" not in short and "[memory saved: plan]" in short and "#7 A1(up)" in short
   and short.rstrip().endswith("#7 A1(up): obj 1 moved up") and len(short) < 1200, short[:600])
long_err = "Traceback\n" + "x" * 5000
kept = context.short_cell_output({"role": "tool", "content": long_err, "_spans": [["error", 0, len(long_err)]]}, 400)
ok("drain_keeps_error_end", kept["content"].endswith("x" * 100) and len(kept["content"]) <= context.ERROR_CHARS + 60
   and "_spans" not in kept)
gmsg = context.graph_message("code graph: 1 functions\nstep(s, m)")
dd2 = context.drain([{"role": "user", "content": "task"}, gmsg, reply(1, "t1"), cell_msg, {"role": "user", "content": "go"},
                     reply(2, "t2"), tool(2)], output_chars=400)
ok("drain_drops_graph_message_and_masks", not any(m.get("_kind") == context.GRAPH_KIND for m in dd2)
   and "[a printed board of 20 rows" in compaction._text(dd2[2]))
# the tail: the picture of the board now goes last and is never part of the history
tail_msg = {"role": "user", "_kind": context.TAIL_KIND,
            "content": [{"type": "text", "text": "[picture of the board now, after step #2]"},
                        {"type": "image_url", "image_url": {"url": "data:t"}}]}
text_hist = [{"role": "user", "content": "task"}, reply(1, "t1"), tool(1)]
w_tail = context.build("SYS", text_hist, append_only=True, vision=True, old_images=False, tail=tail_msg)
ok("build_tail_last", w_tail[:-1] == context.build("SYS", text_hist, append_only=True, vision=True, old_images=False)
   and w_tail[-1]["content"][1]["image_url"]["url"] == "data:t"
   and context.build("SYS", text_hist, append_only=True, vision=False, tail=tail_msg)[-1]["role"] == "tool")


# --- 3. a real session on ls20 with a scripted model ----------------------------------------------------------
class ScriptedLLM:
    """ipython and act turns in turn, each with ~600 tokens of thinking; the plan changes on every second act. The
    reported prompt size is chars/4 of the request plus 500 per picture, so the triggers fire as on the server."""

    def __init__(self):
        self.requests, self.sizes, self.turn, self.cold = [], [], 0, 0

    def chat(self, messages, tools=None, max_tokens=None, timeout_s=None, template_kwargs=None, thinking=None):
        size = sum(len(json.dumps({k: v for k, v in m.items() if k != "content"})) // 4 + (
            sum(len(p.get("text", "")) // 4 + (500 if p.get("type") == "image_url" else 0) for p in m["content"])
            if isinstance(m["content"], list) else len(str(m["content"])) // 4) for m in messages)
        if tools is None:   # no other model call is expected (the curator is off, there is no summary)
            raise AssertionError("a model call without tools")
        self.requests.append(messages)
        self.sizes.append(size)
        self.turn += 1
        n = self.turn
        say = ""
        if n % 2:
            call = {"name": "ipython", "arguments": json.dumps({"code": f"x = {n}"})}
        else:
            plan = f'PLAN = "plan {n}: keep going up"\n' if n % 4 == 0 else ""   # the memory is a kernel variable
            call = {"name": "ipython", "arguments": json.dumps({"code": plan + "r = await play(['1'])"})}
        tc = [{"id": f"call{n}", "type": "function", "function": call}]
        return Reply(say, f"thinking {n} " + "z" * 2400, tc, "tool_calls", size, 700)

    def mark_cold(self):
        self.cold += 1


def play(append_only, turns=36):
    from harness.game.arc_host import ArcHost
    from harness.game.games import build_games, make_arcade
    from harness.agent import prompts

    game = build_games(make_arcade("offline", environments_dir=str(Path(_ROOT) / "eval" / "real_games")), "offline",
                       only=["ls20"])[0]
    game.start()
    run = Path(tempfile.mkdtemp())
    # chain_reasoning_tokens is used by the old builder only; 16000 keeps its older thinking stable in this short game
    # (a drain now keeps up to 3/5 of the trigger, so the chain is longer than the old 4000 held)
    # every thinking kept (keep_reasoning on): the thinking sent back and the drains are what this suite tests
    c = build_config({"agent": {"append_only": append_only, "vision": True, "limits": {"max_turns": turns},
                                "chain_reasoning_tokens": 16000, "keep_reasoning": True,
                                "memory": {"curator": False, "in_context": True},   # the memory messages
                                "compaction": {"reserve_tokens": 4096,
                                               "trigger_tokens": 16000, "cap_tokens": 24000}}})
    fit_context(c["agent"], 40000)
    llm = ScriptedLLM()
    host = ArcHost(game, 400, should_stop=lambda: False, tokens_spent=lambda: 0)
    s = A.AgentSession(cfg=c["agent"], llm=llm, name="ls20", session_dir=run / "ls20",
                       task=prompts.task_message(game_id=game.game_id, max_actions=400, minutes=60), arc=host,
                       deadline=time.time() + 3600, stop_event=threading.Event(), memory_root=run / "memory")
    s.run()
    game.finish()
    events = [json.loads(x) for x in (run / "ls20" / "transcript.jsonl").read_text(encoding="utf-8").splitlines()]
    return s, llm, events


def broken(requests):
    """Indices k where request k does not start with the whole of request k-1 followed by its reply."""
    return [k for k in range(1, len(requests)) if requests[k][:len(requests[k - 1])] != requests[k - 1]
            or len(requests[k]) <= len(requests[k - 1]) or requests[k][len(requests[k - 1])]["role"] != "assistant"]


def is_picture(m):
    return m["role"] == "user" and "[picture of the board now" in json.dumps(m["content"])


def pictures(msgs):
    return sum(1 for m in msgs if isinstance(m["content"], list) for p in m["content"] if p.get("type") == "image_url")


try:
    s, llm, events = play(True)
    st = s.stats
    # a drain that cannot reach its target trims in the same step: one rewrite between two requests
    seq = [e.get("event") for e in events if e.get("event") in ("drain", "context_trim", "context_reset")]
    drain_trims = sum(1 for x, y in zip(seq, seq[1:]) if (x, y) == ("drain", "context_trim"))
    rewrites = st["drains"] + st["context_resets"] + st["context_trims"] - drain_trims
    bad = broken(llm.requests)
    ok("e2e_ran", len(llm.requests) >= 30 and st["act_calls"] >= 10 and not s.end_reason.startswith("crash"),
       f"{len(llm.requests)} requests, {s.end_reason}")
    ok("e2e_drained", st["drains"] >= 1, str(st))
    ok("e2e_prefix_breaks_only_at_rewrites", len(bad) == rewrites, f"broken {bad}, rewrites {rewrites}: {st}")
    first_rewrite = (bad or [len(llm.requests)])[0]
    early = llm.requests[first_rewrite - 1]
    ok("e2e_every_thinking_sent", all(m.get("reasoning_content") for m in early if m["role"] == "assistant")
       and sum(1 for m in early if m["role"] == "assistant") >= 4)
    # The state after moves goes in the cell output, not in a new user message: the turn goes on. The picture of the
    # board is its own user message after each new state, so a request after a move ends with it, and the pictures
    # already sent stay where they are until a rewrite.
    ok("e2e_state_in_cell_output", sum(1 for m in early if m["role"] == "tool"
       and prompts.TURN_NOTE in str(m["content"])) >= 3
       and sum(1 for m in early if m["role"] == "user" and not is_picture(m)) <= 3, str([m["role"] for m in early]))
    after_move = [r for r in llm.requests[1:] if is_picture(r[-1])]
    ok("e2e_picture_after_moves", s.picture_last and len(after_move) >= st["act_calls"] // 2
       and all(pictures(r) >= 1 for r in llm.requests) and pictures(early) >= 3,
       f"{len(after_move)} requests end with a picture, {st['act_calls']} act calls, {pictures(early)} pictures")
    # A picture is drawn once per state: images_sent counts the pictures logged
    sent = sum(1 for e in events if e.get("event") == "tail") + \
        sum(pictures([e]) for e in events if e.get("event") == "message" and "content" in e)
    ok("e2e_images_sent_counts_sent", st["images_sent"] == sent and sent <= st["act_actions"] + 2,
       f"images_sent {st['images_sent']}, logged {sent}, moves {st['act_actions']}")
    text = lambda msgs: "\n".join(m["content"] if isinstance(m["content"], str) else
                                  " ".join(p.get("text", "") for p in m["content"]) for m in msgs)
    ok("e2e_memory_full_first", "[memory] Your memory" in text(llm.requests[0][:2]))
    ok("e2e_memory_once", text(early).count("[memory] Your memory") == 1, str(st))
    after = llm.requests[bad[0]] if bad else []
    ok("e2e_after_drain", bad and pictures(after) == 1 and text(after[1:]).count("[memory] Your memory") == 1
       and "[memory update]" not in text(after[1:]), f"{pictures(after)} pictures")   # the system prompt names both
    # as the Duck harness drains: old turns go whole, the task stays first, the kept replies keep their thinking
    ok("e2e_drain_drops_whole_turns", bad and text(after[1:2]).startswith(text(llm.requests[0][1:2])[:200])
       and any(m["role"] == "assistant" and m.get("reasoning_content") for m in after)
       and len(after) < len(llm.requests[bad[0] - 1]))
    ok("e2e_cold_marked_per_rewrite", llm.cold == rewrites + drain_trims, f"{llm.cold} vs {rewrites}")
    ok("e2e_drain_logged", any(e.get("event") == "drain" for e in events))
    ok("e2e_under_cap", max(llm.sizes) <= 24000 and st["drains"] >= 2, f"max {max(llm.sizes)}, {st['drains']} drains")

    s0, llm0, _ = play(False)
    bad0 = broken(llm0.requests)
    # No board message after a move any more, so the old builder breaks only at rewrites too
    rewrites0 = s0.stats["drains"] + s0.stats["context_resets"] + s0.stats["context_trims"]
    ok("old_builder_no_break_per_move", len(bad0) <= rewrites0,
       f"{len(bad0)} breaks, {rewrites0} rewrites, {s0.stats['act_calls']} acts")
    print(f"     prefix breaks: append-only {len(bad)} of {len(llm.requests)} requests ({st['drains']} drains, "
          f"{st['context_trims']} trims, {st['context_resets']} resets); old builder {len(bad0)} of "
          f"{len(llm0.requests)}")
except ImportError as e:   # no arc-agi SDK or games in this checkout
    WARNS.append(f"e2e skipped: {e!r}")

# --- 5. the scheduler hands over a cold game -------------------------------------------------------------------
class Echo:
    def chat(self, *a, **k):
        return SimpleNamespace(completion_tokens=10)


log = Path(tempfile.mkdtemp()) / "sched.jsonl"
sch = PriorityScheduler(slots=1, quantum_calls=100, log_path=log)
g = ScheduledLLM(Echo(), sch, "g1", levels_won=lambda: 0, n_levels=7, should_stop=lambda: False)
g.mark_cold()
g.chat()                       # cold before any call: nothing to hand over
g.chat()
g.mark_cold()
g.chat()                       # cold after two calls: hands over (and wins the slot back, it is alone)
rows = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines()]
releases = [r["reason"] for r in rows if r["event"] == "release"]
ok("cold_hands_over_once", releases == ["cold"], str(releases))
ok("cold_resets_quantum", g._calls_in_quantum == 1 and g._cold is False)
g.close()

# --- 6. sysmon against a fake /proc ----------------------------------------------------------------------------
fake = Path(tempfile.mkdtemp())
(fake / "meminfo").write_text("MemTotal:       184549376 kB\nMemAvailable:    10485760 kB\nCached:  1048576 kB\n"
                              "Shmem: 0 kB\nSwapFree: 0 kB\n")
for pid, ppid, name, rss in ((100, 1, "python", 2097152), (101, 100, "sglang::sched", 104857600),
                             (102, 101, "sglang::detok", 1048576), (200, 1, "rlm.repl", 524288)):
    d = fake / str(pid)
    d.mkdir()
    (d / "stat").write_text(f"{pid} ({name}) S {ppid} {pid} {pid} 0")
    (d / "status").write_text(f"Name:\t{name}\nVmRSS:\t{rss} kB\n")
    (d / "comm").write_text(name + "\n")
    (d / "oom_score_adj").write_text("0")
real_proc = sysmon.PROC
sysmon.PROC = fake
try:
    ok("sysmon_tree", sorted(sysmon.tree(100)) == [100, 101, 102])
    ok("sysmon_protect_tree", sysmon.protect_tree(100) == 3
       and (fake / "102" / "oom_score_adj").read_text() == "-1000" and (fake / "200" / "oom_score_adj").read_text() == "0")
    snap = sysmon.snapshot(100)
    ok("sysmon_snapshot", snap["total_gb"] == 176.0 and snap["available_gb"] == 10.0 and snap["server_rss_gb"] == 103.0
       and snap["top"][0]["name"] == "sglang::sched", str(snap))
    ok("sysmon_missing_pid", sysmon.set_oom_score_adj(999, 500) is False)
    rows = []
    stub = SimpleNamespace(process=SimpleNamespace(pid=100, poll=lambda: None),
                           event=lambda name, echo=True, **d: rows.append((name, echo, d)))
    LlmServer.memory_tick(stub)
    ok("server_ram_row", rows and rows[0][0] == "ram" and rows[0][1] is False and rows[0][2]["marked"] == 3
       and rows[0][2]["server_rss_gb"] == 103.0, str(rows)[:300])
finally:
    sysmon.PROC = real_proc
ok("sysmon_noop_without_proc", sysmon.PROC.is_dir() or (sysmon.snapshot(1) == {} and not sysmon.set_oom_score_adj(1, 0)))

print("\n================ SUMMARY ================")
print(f"CHECKS: {CHECKS[0]}")
print(f"HARD FAILS: {len(FAILS)} -> {FAILS}")
print(f"WARNINGS  : {len(WARNS)} -> {WARNS}")
sys.exit(1 if FAILS else 0)

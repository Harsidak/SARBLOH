# --- path shim: these suites live in tests/ but import from the repo root ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_PRIME = _os.environ.get("PRIME_DIR", "Sarbloh-Experimentation")   # these changes are in the experimentation copy only
for _p in (_ROOT, _os.path.join(_ROOT, _PRIME)):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end shim ---
"""The SFT trace (Sarbloh-Experimentation) and the ``tracing`` switch, with a scripted model on ls20:
- every request the model received is rebuilt exactly from the transcript (``trace.sft_wire``), through drains,
  compactions and a level-up reset, with the append-only context and with the old builder;
- the level after a reset starts with the reset message and the full board, not the task;
- each row's replies follow exactly the messages they answered, the picture each one saw (the last message of its
  request) right before it; the pictures of the earlier replies of a row stay in place; a failed cell is kept as
  context, out of the loss;
- every picture is a file the row names; the report counts turns, drains and pictures;
- tracing off writes no transcript, no pictures and no trace, and sends the same requests.

Run: uv run python tests/unit/test_experimentation_trace.py
"""
import json
import re
import sys
import tempfile
import threading
import time
from pathlib import Path

from harness import trace
from harness.agent import agent as A, context
from harness.config import build_config
from harness.llm.client import Reply
from harness.run import fit_context

FAILS, WARNS, CHECKS = [], [], [0]


def ok(name, cond, detail=""):
    CHECKS[0] += 1
    print(("PASS " if cond else "FAIL ") + name + (f"  {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


RESET_TURN, ERROR_TURN = 20, 5


class ScriptedLLM:
    """ipython and act turns in turn, ~600 tokens of thinking each; turn ERROR_TURN runs a failing cell and turn
    RESET_TURN is treated as a level-up (the context is reset after it). Prompt size = chars/4 + 500 per picture."""

    def __init__(self):
        self.requests, self.efforts, self.turn, self.session = [], [], 0, None

    def chat(self, messages, tools=None, max_tokens=None, timeout_s=None, template_kwargs=None, thinking=None):
        size = sum(len(json.dumps({k: v for k, v in m.items() if k != "content"})) // 4 + (
            sum(len(p.get("text", "")) // 4 + (500 if p.get("type") == "image_url" else 0) for p in m["content"])
            if isinstance(m["content"], list) else len(str(m["content"])) // 4) for m in messages)
        last = messages[-1]["content"]
        handover = "Write a handover note to yourself" in (last if isinstance(last, str) else " ".join(p.get("text", "") for p in last))
        if tools is None or handover:   # a compaction summary (the curator is off)
            return Reply("SUMMARY of the play so far", "", [], "stop", size, 20)
        self.requests.append(messages)
        self.efforts.append(template_kwargs)
        self.turn += 1
        n = self.turn
        if n == RESET_TURN:
            self.session._level_up_text = "LEVEL UP (scripted): a new level begins"
        if n % 2:
            call = {"name": "ipython", "arguments": json.dumps({"code": "1/0" if n == ERROR_TURN else f"x = {n}"})}
        else:
            call = {"name": "ipython", "arguments": json.dumps({"code": f"PLAN = 'plan {n}'\nr = await play(['1'])"})}
        tc = [{"id": f"call{n}", "type": "function", "function": call}]
        return Reply("",f"thinking {n} " + "z" * 2400, tc, "tool_calls", size, 700)


def play(append_only=True, tracing=True, turns=36):
    from harness.agent import prompts
    from harness.game.arc_host import ArcHost
    from harness.game.games import build_games, make_arcade

    game = build_games(make_arcade("offline", environments_dir=str(Path(_ROOT) / "eval" / "real_games")), "offline",
                       only=["ls20"])[0]
    game.start()
    run = Path(tempfile.mkdtemp())
    # every thinking kept (keep_reasoning on): its ~600 tokens a turn fill the context to the drains under test
    c = build_config({"agent": {"append_only": append_only, "vision": True, "limits": {"max_turns": turns},
                                "keep_reasoning": True,
                                "memory": {"curator": False, "in_context": True}, "drain_max_fraction": 0.7,
                                "level_up_reset": True,
                                "compaction": {"reserve_tokens": 4096, "keep_recent_tokens": 3000,
                                               "trigger_tokens": 16000, "floor_gap_tokens": 1000}}})
    fit_context(c["agent"], 40000)
    llm = ScriptedLLM()
    host = ArcHost(game, 400, should_stop=lambda: False, tokens_spent=lambda: 0)
    s = A.AgentSession(cfg=c["agent"], llm=llm, name="ls20", session_dir=run / "games" / "ls20",
                       task=prompts.task_message(game_id=game.game_id, max_actions=400, minutes=60), arc=host,
                       deadline=time.time() + 3600, stop_event=threading.Event(), memory_root=run / "memory",
                       tracing=tracing)
    llm.session = s
    s.run()
    game.finish()
    path = run / "games" / "ls20" / "transcript.jsonl"
    events = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
    return s, llm, events, run, game


TAIL = re.compile(r"(?:\n\n)?\[picture of the board now[^\]]*\]")


def no_pictures(msgs):
    """The messages without the tail pictures (a message that was only a picture goes; a joined one keeps its text)."""
    out = []
    for m in msgs:
        if isinstance(m.get("content"), list):
            rest = TAIL.sub("", " ".join(p.get("text", "") for p in m["content"] if p.get("type") == "text"))
            if not rest:
                continue
            m = {**m, "content": rest}
        out.append(m)
    return out


def canon(request):
    """A request's digest with the minutes left blanked (the clock moves between two runs)."""
    text = json.dumps([{**m, "content": [{"type": "image_url"} if p.get("type") == "image_url" else p
                                         for p in m["content"]]} if isinstance(m["content"], list) else m
                       for m in request], sort_keys=True)
    return re.sub(r"\d+ min", "N min", text)


try:
    s, llm, events, run, game = play()
    st = s.stats
    run_info = {"game_id": "ls20", "levels_completed": 1, "number_of_levels": 7, "actions_per_level": [18],
                "baseline_actions": [21]}
    rows, counts = trace.sft_wire(events, run_info, {}, "games/ls20")
    replies = [e for e in events if e.get("event") == "message" and e.get("role") == "assistant"]
    ok("ran_with_rewrites", len(llm.requests) >= 30 and st["drains"] >= 1 and st["context_resets"] >= 1
       and not s.end_reason.startswith("crash"), f"{len(llm.requests)} requests, {st}, {s.end_reason}")
    ok("logged_digest_is_of_what_was_sent", [r.get("wire_sha") for r in replies] ==
       [context.digest(q) for q in llm.requests])
    ok("every_request_rebuilt", counts["turns"] == len(llm.requests) and counts["verified"] == counts["turns"],
       str(counts))
    ok("effort_recorded", all((r.get("effort") or None) == ((e or {}).get("reasoning_effort"))
                              for r, e in zip(replies, llm.efforts)))
    ok("snapshots_at_rewrites", sum(1 for e in events if e.get("event") == "context") ==
       st["drains"] + st["context_resets"] + st["context_trims"], str(st))

    # rows: each reply follows exactly the request it answered
    row_turns = [t for r in rows for t in r["turns"]]
    # a row whose turns are all flagged has nothing to train on, so the trace drops it (rows_without_target)
    bad_turns = {t for t, f in trace._turn_flags(events).items() if f & set(trace.BAD_FLAGS)}
    missing = set(range(1, len(llm.requests) + 1)) - set(row_turns)
    ok("rows_cover_every_turn", len(row_turns) == len(set(row_turns)) and missing <= bad_turns
       and bool(missing) == bool(counts.get("rows_without_target")), f"{row_turns}, missing {sorted(missing)}")
    exact = True
    for r in rows:
        msgs = [{"role": "system", "content": r["system"]}] + r["messages"]
        for i, m in enumerate(msgs):
            if m["role"] == "assistant" and m.get("turn") in r["turns"]:
                sent = llm.requests[m["turn"] - 1]
                # A target keeps its code (the model learns to write it); the request held the code graph in its
                # place, so both sides are compared with the code taken out (every_request_rebuilt checks the graph).
                back = [(context.without_code({**{k: v for k, v in x.items() if k not in ("train", "turn", "flags")},
                          **({"reasoning": x["reasoning_content"]} if x.get("reasoning_content") else {})})
                         if x["role"] == "assistant" else x) for x in msgs[:i]]
                held = [context.without_code(x) if x["role"] == "assistant" else x for x in sent]
                exact &= canon(no_pictures(back)) == canon(no_pictures(held)) and canon(back[-1:]) == canon(held[-1:])
    ok("row_prefix_equals_request", exact)
    reset_rows = [r for r in rows if r["segment_start"] == "level_up"]
    first_user = reset_rows[0]["messages"][0] if reset_rows else {}
    text = first_user.get("content") if isinstance(first_user.get("content"), str) else \
        " ".join(p.get("text", "") for p in first_user.get("content") or [])
    ok("level_start_is_the_reset_not_the_task", reset_rows and text.startswith("LEVEL UP (scripted)")
       and "[memory] Your memory" in text and RESET_TURN + 1 == reset_rows[0]["turns"][0], text[:200])
    # with hide_code (the default) every reply is stored as a note, so each turn is its own row and the failed
    # turn's row has nothing to train on and is dropped; either way it is never trained
    bad = [m for r in rows for m in r["messages"] if m.get("turn") == ERROR_TURN]
    ok("failed_cell_out_of_the_loss", "cell_error" in trace._turn_flags(events)[ERROR_TURN]
       and all(m["train"] is False for m in bad), str(bad)[:300])
    ok("earlier_rows_are_context_only", all(m.get("train") is False for r in rows for m in r["messages"]
                                            if m["role"] == "assistant" and m.get("turn") not in r["turns"]))
    ok("thinking_is_a_target", all(m.get("reasoning_content", "").startswith("thinking") for r in rows
                                   for m in r["messages"] if m.get("turn") in r["turns"]))
    urls = [p["image_url"]["url"] for r in rows for m in r["messages"] if isinstance(m.get("content"), list)
            for p in m["content"] if p.get("type") == "image_url"]
    ok("pictures_are_files", urls and all((run / u).is_file() for u in urls), str(urls[:3]))
    ok("rows_tagged", all(r["level"] == 0 and r["level_cleared"] and r["rhae"] == 1.15 and r["wire_verified"]
                          for r in rows))   # (21/18)^2 capped at 1.15

    # the whole build on a results.json
    (run / "results.json").write_text(json.dumps({"summary": {}, "config": {}, "sessions": {}, "runs": [
        {**run_info, "game_id": "ls20", "history": []}]}), encoding="utf-8")
    md = trace.build_run(run).read_text(encoding="utf-8")
    ok("build_run_reports_wire", "rebuilt exactly (100.0%)" in md, md[-600:])
    report = (run / "games" / "ls20" / "report.md").read_text(encoding="utf-8")
    tr = json.loads((run / "games" / "ls20" / "trace.json").read_text(encoding="utf-8"))["context"]
    ok("report_turns_and_context", tr["requests"] == len(llm.requests) and tr["turns"] == st["outer_turns"]
       and sum(tr["turn_ends"].values()) == tr["turns"] and len(tr["drains"]) == st["drains"]
       and tr["stored_pictures"] >= 1 and "turns ended by" in report and "Code graph:" in report, str(tr)[:400])

    # the old builder
    s0, llm0, events0, _, _ = play(append_only=False)
    rows0, counts0 = trace.sft_wire(events0, run_info, {}, "games/ls20")
    ok("old_builder_rebuilt", counts0["turns"] == len(llm0.requests) and counts0["verified"] == counts0["turns"],
       str(counts0))

    # tracing off: no transcript, the same requests (the pictures are still saved under frames/)
    s1, llm1, events1, run1, _ = play(tracing=False)
    ok("off_writes_nothing", not events1 and not list(run1.rglob("transcript.jsonl")))
    ok("off_same_requests", [canon(q) for q in llm1.requests] == [canon(q) for q in llm.requests],
       f"{len(llm1.requests)} vs {len(llm.requests)}")
except ImportError as e:   # no arc-agi SDK or games in this checkout
    WARNS.append(f"e2e skipped: {e!r}")

# --- the switch in the config ---------------------------------------------------------------------------------
ok("config_default_on", build_config({})["tracing"] is True)

print("\n================ SUMMARY ================")
print(f"CHECKS: {CHECKS[0]}")
print(f"HARD FAILS: {len(FAILS)} -> {FAILS}")
print(f"WARNINGS  : {len(WARNS)} -> {WARNS}")
sys.exit(1 if FAILS else 0)

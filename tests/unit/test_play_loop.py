# --- path shim: these suites live in tests/ but import from the repo root ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_PRIME = _os.environ.get("PRIME_DIR", "Sarbloh-Experimentation")   # these changes are in the experimentation copy only
for _p in (_ROOT, _os.path.join(_ROOT, _PRIME)):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end shim ---
"""``play()`` as the only way to move, and the notes around it (Sarbloh-Experimentation), in a real session on ls20
with a scripted model:
- the system prompt and the tool list teach ``play()`` and never ``act()``;
- a surprise in ``play()`` reaches the host, and the next turn gets the short cap (surprise_yield_tokens): a reply that
  runs code past it without moving ends the turn, with no note; four code-only requests stay in one turn;
- a level that has eaten stuck_note_tokens of replies without being won gets the "stuck" note at the end of a turn,
  once per share;
- ``act()`` in a cell raises and spends nothing; a path over act_max_actions is refused whole, nothing spent;
- after a level-up the next message names the functions and classes already in the kernel.

Run: uv run python tests/unit/test_play_loop.py
"""
import json
import tempfile
import threading
import time
from pathlib import Path

from harness.agent import agent as A, guardrails, prompts
from harness.config import build_config
from harness.llm.client import Reply
from harness.run import fit_context

FAILS, WARNS, CHECKS = [], [], [0]


def ok(name, cond, detail=""):
    CHECKS[0] += 1
    print(("PASS " if cond else "FAIL ") + name + (f"  {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


def code_call(n, code):
    return [{"id": f"call{n}", "type": "function", "function": {"name": "ipython",
                                                                 "arguments": json.dumps({"code": code})}}]


SCRIPT = {   # turn -> (code, completion tokens)
    1: ('r = await play(["1"], predict=lambda b, m: {(0, 0): "Z"})', 300),   # surprise: next turn's budget is 1000
    2: ("a = 1", 1200),                                                         # past 1000, no move: turn ends
    3: ("b = 1", 300), 4: ("c = 1", 300), 5: ("d = 1", 300), 6: ("e = 1", 300),  # 4 calls, no move: one turn
    7: ('r = await play(["1"])', 400),                                          # 3100 on the level: stuck
    8: ('r = await act(["1"])', 100),                                           # act() is gone
    9: ('r = await play(["1"] * 101)', 100),                                    # over the cap: refused
}


class ScriptLLM:
    def __init__(self):
        self.requests, self.turn = [], 0

    def chat(self, messages, tools=None, max_tokens=None, timeout_s=None, template_kwargs=None, thinking=None):
        self.requests.append((messages, tools))
        self.turn += 1
        code, tokens = SCRIPT.get(self.turn, ("z = 0", 100))
        return Reply("", f"thinking {self.turn}", code_call(self.turn, code), "tool_calls", 5000, tokens)

    def yield_turn(self):
        pass


def text(request):
    out = []
    for m in request:
        c = m.get("content") or ""
        out.append(c if isinstance(c, str) else " ".join(p.get("text", "") for p in c if isinstance(p, dict)))
    return "\n".join(out)


def since_last_assistant(request):
    i = max(k for k, m in enumerate(request) if m["role"] == "assistant")
    return text(request[i + 1:])


# --- prompt -------------------------------------------------------------------------------------------------------
cfg = build_config({})["agent"]
system = A.system_prompt(cfg)
ok("prompt_teaches_play", "await play(path" in system and "only call that changes the game" in system)
ok("prompt_has_no_act", "act(" not in system.replace("_act(", "") and "`act`" not in system, system[:200])
ok("prompt_look_and_move", "observe()" in system and "await play(" in system)
ok("tool_teaches_play", "await act" not in system and "await act" not in json.dumps(A.game_tools()))
ok("defaults", cfg["surprise_yield_tokens"] == 0 and cfg["turn_yield_tokens"] == 16384 and cfg["stuck_note_tokens"] == 0,
   str({k: cfg[k] for k in ("surprise_yield_tokens", "stuck_note_tokens")}))

try:
    from harness.game.arc_host import ArcHost
    from harness.game.games import build_games, make_arcade

    def session(run, llm, extra):
        game = build_games(make_arcade("offline", environments_dir=str(Path(_ROOT) / "eval" / "real_games")),
                           "offline", only=["ls20"])[0]
        game.start()
        c = build_config({"agent": {"vision": False, "memory": {"curator": False}, **extra}})
        fit_context(c["agent"], 131072)
        host = ArcHost(game, 400, should_stop=lambda: False, tokens_spent=lambda: 0)
        s = A.AgentSession(cfg=c["agent"], llm=llm, name="ls20", session_dir=run / "games" / "ls20",
                           task=prompts.task_message(game_id=game.game_id, max_actions=400, minutes=45), arc=host,
                           deadline=time.time() + 3600, stop_event=threading.Event(), memory_root=run / "memory")
        return s, game

    # --- the scripted session -------------------------------------------------------------------------------------
    run = Path(tempfile.mkdtemp())
    llm = ScriptLLM()
    s, game = session(run, llm, {"limits": {"max_turns": 10}, "turn_yield_tokens": 4092, "surprise_yield_tokens": 1000, "stuck_note_tokens": 3000})
    s.run()
    game.finish()
    events = [json.loads(x) for x in (run / "games" / "ls20" / "transcript.jsonl").read_text(encoding="utf-8")
              .splitlines()]
    R = [m for m, _ in llm.requests]
    st = s.stats
    ok("e2e_ran", len(R) == 10 and not s.end_reason.startswith("crash"), f"{len(R)} {s.end_reason}")
    ok("system_prompt_sent", "await play(path" in R[0][0]["content"])
    out1 = since_last_assistant(R[1])
    ok("surprise_printed", "One rule in your model was wrong" in out1, out1[:400])
    ok("surprise_reached_host", st["surprises"] == 1 and st["surprise_turns"] == 1
       and any(e.get("event") == "surprise" for e in events), str(st["surprises"]))
    ends = [(e["reason"], e["requests"]) for e in events if e.get("event") == "turn_end"]
    ok("short_cap_ends_turn", ends[:3] == [("moved", 1), ("token_cap", 1), ("moved", 5)], str(ends))
    ok("no_move_now_notes", not any(prompts.FIX_NOW in text(r) or prompts.ACT_NOW in text(r) for r in R)
       and not any(e.get("event") == "turn_yield" for e in events))
    ok("no_idle_note", "made no move" not in text(R[-1]) and "idle_notes" not in st)
    stuck = since_last_assistant(R[7])
    ok("stuck_note", "on this level for a long time" in stuck and st["stuck_notes"] == 1, stuck[-400:])
    ok("stuck_note_once_per_share", st["stuck_notes"] == 1)
    moves_before = [e for e in events if e.get("event") == "act"]
    ok("plays_logged", len(moves_before) == 2 and all(e["via"] == "play" and e.get("play_id") for e in moves_before)
       and moves_before[0]["predicted"] is True and moves_before[1]["predicted"] is False, str(moves_before[:2]))
    out8 = since_last_assistant(R[8])
    ok("act_is_gone", "NameError" in out8, out8[-300:])
    out9 = since_last_assistant(R[9])
    ok("path_over_cap_refused", "takes at most 100" in out9 and st["act_actions"] == 2, out9[-300:])

    # --- the level-start note, from the kernel's names ------------------------------------------------------------
    run2 = Path(tempfile.mkdtemp())
    s2, game2 = session(run2, ScriptLLM(), {})
    s2.kernel.start()
    s2.kernel.execute("def step(s, m):\n    return s\nclass Model:\n    pass\nseen = set()")
    guardrails.level_start_note(s2)
    note = s2.messages[-1]["content"] if s2.messages else ""
    ok("reuse_code_note", "still in the kernel: step, Model" in note and "seen" not in note, note)
    s2.kernel.close()
    game2.finish()
except ImportError as e:   # no arc-agi SDK or games in this checkout
    WARNS.append(f"e2e skipped: {e!r}")

print("\n================ SUMMARY ================")
print(f"CHECKS: {CHECKS[0]}")
print(f"HARD FAILS: {len(FAILS)} -> {FAILS}")
print(f"WARNINGS  : {len(WARNS)} -> {WARNS}")
raise SystemExit(1 if FAILS else 0)

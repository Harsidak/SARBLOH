# --- path shim: these suites live in tests/ but import from the repo root ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_PRIME = _os.environ.get("PRIME_DIR", "Sarbloh-Experimentation")   # these changes are in the experimentation copy only
for _p in (_ROOT, _os.path.join(_ROOT, _PRIME)):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end shim ---
"""The reset rule, the kernel's file guard and the prompt lines that came with them (Sarbloh-Experimentation):
- config reset_move 0: "reset" in a path is refused whole and nothing is spent; after a GAME_OVER the host restarts
  the level itself with one RESET, prints its line and the GAME_OVER note, and the game plays on; no prompt line tells
  the model that a lost level restarts by itself (it drained its move bar to restart, run 10-26); reset_move above 0
  (the default, 3) keeps "reset" as a move (its limit is tested in test_revisit_leash.py);
- the system prompt says moves count against the score and to test a goal with one move before a big model; the start
  message has no "nothing is known" / probe line; the stuck note no longer points at recall();
- config file_guard on: the kernel cannot open or list the game files (os.walk yields nothing) or its own harness code
  (only imports may read it), cannot start another
  program, and can still use its own folder, the standard library and numpy.

Run: uv run python tests/unit/test_reset_and_file_guard.py
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

FAILS, WARNS, CHECKS = [], [], [0]


def ok(name, cond, detail=""):
    CHECKS[0] += 1
    print(("PASS " if cond else "FAIL ") + name + (f"  {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


# --- prompts --------------------------------------------------------------------------------------------------------
cfg = build_config({})["agent"]
ok("defaults", cfg["reset_move"] == 3 and cfg["revisit_stop"] is False and cfg["wall_edge_band"] == 3
   and cfg["file_guard"] is True)
system = A.system_prompt({**cfg, "reset_move": 0})
ok("no_reset_move_taught", '"reset"' not in system and prompts.RESET_CONTROL in system
   and prompts.RESET_LINE in system, system[system.find("Controls"):][:400])
ok("moves_cost_score", "Every move you send counts against your score" in system)
ok("test_goal_first", "test that goal with one move" in system)
on = A.system_prompt(cfg)
ok("reset_move_on_taught", prompts.RESET_CONTROL_MOVE in on and '"7" undo, "reset", or a click' in on)
start = prompts.task_message(game_id="g", max_actions=400, minutes=45, win_levels=7)
ok("start_message", start.startswith("It has 7 levels. You have at most 400 moves") and "nothing is known" not in start
   and "probe" not in start and "level 1" not in start.lower().replace("this is level 1.", ""), start)
ok("stuck_note_no_recall", "recall" not in prompts.STUCK and "on this level for a long time" in prompts.STUCK)
skill = (Path(_ROOT) / _PRIME / "harness" / "agent" / "skills" / "composition.md").read_text(encoding="utf-8")
ok("composition_no_experiment_room", "room to experiment" not in skill)
ok("no_restart_hint", "by itself" not in system and "again" not in prompts.event_message("game_over")
   and "by itself" not in prompts.ACT_NO_RESET, system[system.find("Controls"):][:400])
ok("game_over_notes", prompts.event_message("game_over").startswith("GAME_OVER: this level was lost.")
   and 'play(["reset"])' in prompts.event_message("game_over_reset"))

# --- read_moves -----------------------------------------------------------------------------------------------------
fake = SimpleNamespace(cfg={"act_max_actions": 100, "reset_move": 0}, _resets_used={},
                       arc=SimpleNamespace(game=SimpleNamespace(state=SimpleNamespace(available_actions=[0, 1, 2], levels_completed=0))))
try:
    guardrails.read_moves(fake, {"actions": ["1", "reset"]})
    ok("reset_refused", False, "not refused")
except ValueError as e:
    ok("reset_refused", "There is no reset move" in str(e), str(e))
ok("plain_moves_read", guardrails.read_moves(fake, {"actions": ["1", "2"]}) == [(1, None, None), (2, None, None)])
fake.cfg["reset_move"] = 3
ok("reset_read_when_on", guardrails.read_moves(fake, {"actions": ["reset"]}) == [(0, None, None)])

# --- auto reset after GAME_OVER, in a real session on bp35 (moves 3, 4, 7 in a loop lose it after 44 moves) ----------
SCRIPT = {1: 'r = await play(["3", "4", "7"] * 20)\nprint("RESULT", r.stopped, r.state, r.done)',
          2: 'r = await play(["reset"])'}


class ScriptLLM:
    def __init__(self):
        self.requests, self.turn = [], 0

    def chat(self, messages, tools=None, max_tokens=None, timeout_s=None, template_kwargs=None, thinking=None):
        self.requests.append(messages)
        self.turn += 1
        code = SCRIPT.get(self.turn, "z = 0")
        call = [{"id": f"call{self.turn}", "type": "function",
                 "function": {"name": "ipython", "arguments": json.dumps({"code": code})}}]
        return Reply("", f"thinking {self.turn}", call, "tool_calls", 5000, 100)

    def yield_turn(self):
        pass


def text(request):
    out = []
    for m in request:
        c = m.get("content") or ""
        out.append(c if isinstance(c, str) else " ".join(p.get("text", "") for p in c if isinstance(p, dict)))
    return "\n".join(out)


def after(request, k):
    """The messages after the k-th assistant message of a request."""
    idx = [i for i, m in enumerate(request) if m["role"] == "assistant"]
    return text(request[idx[k] + 1:idx[k + 1]] if k + 1 < len(idx) else request[idx[k] + 1:])


try:
    from harness.game.arc_host import ArcHost
    from harness.game.games import build_games, make_arcade
    from harness.run import fit_context

    run = Path(tempfile.mkdtemp())
    game = build_games(make_arcade("offline", environments_dir=str(Path(_ROOT) / "eval" / "real_games")),
                       "offline", only=["bp35"])[0]
    game.start()
    c = build_config({"agent": {"vision": False, "memory": {"curator": False}, "limits": {"max_turns": 3},
                                "reset_move": 0, "revisit_stop": False}})
    fit_context(c["agent"], 131072)
    host = ArcHost(game, 400, should_stop=lambda: False, tokens_spent=lambda: 0)
    llm = ScriptLLM()
    s = A.AgentSession(cfg=c["agent"], llm=llm, name="bp35", session_dir=run / "games" / "bp35",
                       task=prompts.task_message(game_id=game.game_id, max_actions=400, minutes=45), arc=host,
                       deadline=time.time() + 3600, stop_event=threading.Event(), memory_root=run / "memory")
    s.run()
    R = llm.requests
    out1 = after(R[1], 0) if len(R) > 1 else ""
    ok("e2e_ran", len(R) == 3 and not s.end_reason.startswith("crash"), f"{len(R)} {s.end_reason}")
    ok("game_over_then_auto_reset", s.stats["auto_resets"] == 1 and "RESET (automatic after GAME_OVER)" in out1
       and "GAME_OVER: this level was lost." in out1, out1[-800:])
    ok("play_reports_game_over", "RESULT GAME_OVER GAME_OVER 44" in out1, out1[:600])
    ok("level_playing_again", game.state.engine_state.name not in ("GAME_OVER", "WIN") and not host.finished,
       game.state.engine_state.name)
    spent = game.action_count
    ok("reset_counted_once", spent == 45, str(spent))   # 44 moves (undo included by the engine count) + the reset
    out2 = after(R[2], 1) if len(R) > 2 else ""
    ok("model_reset_refused", "There is no reset move" in out2 and game.action_count == spent, out2[-400:])
    game.finish()
except ImportError as e:   # no arc-agi SDK or games in this checkout
    WARNS.append(f"e2e skipped: {e!r}")

# --- the kernel's file guard ----------------------------------------------------------------------------------------
from harness.runtime.kernel import Kernel

secret = Path(tempfile.mkdtemp(prefix="game_files_"))
(secret / "game.py").write_text("SECRET = 1\n", encoding="utf-8")
outside = Path(_ROOT) / "README.md"   # the repo root holds the games folder, so it is not an allowed root
games_dir = Path(_ROOT) / "eval" / "real_games"
a_game_file = next(games_dir.rglob("*.py"), None)


def kernel(guard):
    k = Kernel(Path(tempfile.mkdtemp()) / "work", lambda req: {},
               env={"SARBLOH_FILE_GUARD": "1" if guard else "0",
                    "SARBLOH_GAME_FILES": _os.pathsep.join([str(secret), str(games_dir)])})
    k.start()
    return k


k = kernel(True)
try:
    r = k.execute(f"open({str(secret / 'game.py')!r}).read()")
    ok("game_file_refused", r.status == "error" and "PermissionError" in (r.error or ""), r.render(600))
    r = k.execute(f"print('WALK', list(__import__('os').walk({str(secret)!r})))")
    ok("walk_finds_nothing", "WALK []" in r.stdout, r.render(600))
    r = k.execute(f"open({str(outside)!r}).read()")
    ok("outside_file_refused", r.status == "error" and "PermissionError" in (r.error or ""), r.render(600))
    if a_game_file is not None:
        r = k.execute(f"open({str(a_game_file)!r}).read()")
        ok("real_game_source_refused", r.status == "error" and "PermissionError" in (r.error or ""), r.render(600))
    r = k.execute("import subprocess, sys\nsubprocess.run([sys.executable, '-c', 'print(1)'])")
    ok("program_refused", r.status == "error" and "PermissionError" in (r.error or ""), r.render(600))
    r = k.execute("open('mine.txt', 'w').write('x')\nimport decimal, fractions, statistics\n"
                  "print('OWN', open('mine.txt').read(), np is not None and int(np.arange(3).sum()), "
                  "statistics.mean([1, 3]))")
    ok("own_folder_and_imports", "OWN x 3 2" in r.stdout, r.render(600))
    r = k.execute("print('OBS', type(observe).__name__)")
    ok("helpers_still_work", "OBS function" in r.stdout, r.render(600))
    # the agent's own code folder: only the import system may read it
    acting_src = Path(_ROOT) / "Sarbloh-Experimentation" / "harness" / "runtime" / "skills" / "acting.py"
    r = k.execute(f"open({str(acting_src)!r}).read()")
    ok("harness_source_refused", r.status == "error" and "PermissionError" in (r.error or ""), r.render(600))
    r = k.execute(f"print('LS', __import__('os').listdir({str(acting_src.parent)!r}))")
    ok("harness_folder_not_listed", r.status == "error" and "PermissionError" in (r.error or ""), r.render(600))
    r = k.execute("import inspect\ninspect.getsource(play)")
    ok("harness_getsource_refused", r.status == "error", r.render(600))
    r = k.execute("import harness.memory.timeline as _t\nprint('IMPORTED', _t.render_row({'kind': 'step', 'i': 1, "
                  "'level': 0, 'action': 'A1', 'change': 'no change'}))")
    ok("harness_import_still_works", "IMPORTED #1 L1 A1: no change" in r.stdout, r.render(600))
    # the base skills inside the code folder: recall() lists and reads them, nothing may write them
    skills_dir = Path(_ROOT) / _PRIME / "harness" / "agent" / "skills"
    r = k.execute(f"import os\nprint('SK', 'code_bfs.md' in os.listdir({str(skills_dir)!r}), "
                  f"len(open({str(skills_dir / 'code_bfs.md')!r}).read()) > 100)")
    ok("base_skills_readable", "SK True True" in r.stdout, r.render(600))
    r = k.execute(f"open({str(skills_dir / 'code_bfs.md')!r}, 'a').write('x')")
    ok("base_skills_not_writable", r.status == "error" and "PermissionError" in (r.error or ""), r.render(600))
    r = k.execute(f"open({str(skills_dir.parent / 'agent.py')!r}).read()")
    ok("next_to_skills_refused", r.status == "error" and "PermissionError" in (r.error or ""), r.render(600))
finally:
    k.close()
k = kernel(False)
try:
    r = k.execute(f"print('READ', open({str(secret / 'game.py')!r}).read().strip())")
    ok("guard_off_reads", "READ SECRET = 1" in r.stdout, r.render(600))
finally:
    k.close()

print("\n================ SUMMARY ================")
print(f"CHECKS: {CHECKS[0]}")
print(f"HARD FAILS: {len(FAILS)} -> {FAILS}")
print(f"WARNINGS  : {len(WARNS)} -> {WARNS}")
raise SystemExit(1 if FAILS else 0)

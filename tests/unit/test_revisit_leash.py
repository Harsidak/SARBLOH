# --- path shim: these suites live in tests/ but import from the repo root ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_PRIME = _os.environ.get("PRIME_DIR", "Sarbloh-Experimentation")
for _p in (_ROOT, _os.path.join(_ROOT, _PRIME)):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end shim ---
"""The revisit leash and the limited reset (run 10-26: the model burned its move bar to restart a level, and swept
boards it had already seen):
- config revisit_stop on (off by default since 10-27): the rest of a play() path is not sent after a move that brings back a board
  already seen since the level started or was last reset, the 3-cell edge band left out; the path stops with
  "board seen before", nothing is refused, and a reset or a level up starts a new try;
- config reset_move 3 (the default): "reset" may be sent 3 times per level; the 4th is refused whole and nothing is
  spent; the host's own reset after a GAME_OVER is not counted;
- the system prompt says what stops a path and that "reset" is a move, with no new wording beyond that.

Run: uv run python tests/unit/test_revisit_leash.py
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
ok("defaults", cfg["reset_move"] == 3 and cfg["revisit_stop"] is False and cfg["wall_edge_band"] == 3)
system = A.system_prompt({**cfg, "revisit_stop": True})
ok("prompt_path_stop", prompts.REVISIT_PATH_STOP in system and prompts.WALL_PATH_STOP not in system
   and '"board seen before", "wall hit twice"' in system, system[system.find("The whole path"):][:400])
ok("prompt_reset_is_a_move", prompts.RESET_CONTROL_MOVE in system and prompts.RESET_LINE_MOVE in system)
ok("prompt_no_limit_wording", "3 resets" not in system and "by itself" not in system)
off = A.system_prompt(cfg)
ok("prompt_leash_off", prompts.WALL_PATH_STOP_EDGE.format(band=3) in off and "board seen before" not in off)
ok("prompt_whole_frame_wall", prompts.WALL_PATH_STOP in A.system_prompt({**cfg, "wall_edge_band": 0}))


# --- the wall stop: the same move twice in a row with nothing changed outside the edge band -------------------------
def wall(band, frames, moves):
    s = SimpleNamespace(cfg={"wall_edge_band": band}, stats={"wall_hits": 0, "revisits": 0},
                        arc=SimpleNamespace(finished=False))
    cell = {}
    return [guardrails.batch_stop(s, cell, m, a, b, {}) for m, (a, b) in zip(moves, zip(frames, frames[1:]))]


b0 = [[0] * 64 for _ in range(64)]
b1 = [r[:] for r in b0]
b1[63][10] = 5                     # only the bar at the bottom edge moved
b2 = [r[:] for r in b1]
b2[63][11] = 5
b3 = [r[:] for r in b2]
b3[30][30] = 5                     # the playing area changed
ok("wall_edge_band_stops", wall(3, [b0, b1, b2], ["1", "1"]) == [None, "wall hit twice"])
ok("wall_whole_frame_misses_bar", wall(0, [b0, b1, b2], ["1", "1"]) == [None, None])
ok("wall_needs_same_move", wall(3, [b0, b1, b2], ["1", "2"]) == [None, None])
ok("wall_board_change_resets", wall(3, [b0, b1, b3, b3], ["1", "1", "1"]) == [None, None, None])

# --- the reset limit in read_moves ----------------------------------------------------------------------------------
state = SimpleNamespace(available_actions=[0, 1, 2], levels_completed=0)
fake = SimpleNamespace(cfg={"act_max_actions": 100, "reset_move": 3}, _resets_used={0: 2},
                       arc=SimpleNamespace(game=SimpleNamespace(state=state)))
ok("third_reset_read", guardrails.read_moves(fake, {"actions": ["reset"]}) == [(0, None, None)])
fake._resets_used = {0: 3}
try:
    guardrails.read_moves(fake, {"actions": ["1", "reset"]})
    ok("fourth_reset_refused", False, "not refused")
except ValueError as e:
    ok("fourth_reset_refused", "The 3 resets of this level are used" in str(e) and "no move was spent" in str(e), str(e))
state.levels_completed = 1
ok("next_level_has_its_own", guardrails.read_moves(fake, {"actions": ["reset"]}) == [(0, None, None)])
fake.cfg["reset_move"] = 0
try:
    guardrails.read_moves(fake, {"actions": ["reset"]})
    ok("reset_move_0_refused", False, "not refused")
except ValueError as e:
    ok("reset_move_0_refused", "There is no reset move" in str(e), str(e))

# --- the board key: the edge band is left out -----------------------------------------------------------------------
board = [[0] * 64 for _ in range(64)]
bar = [row[:] for row in board]
bar[63][10] = 5          # a move bar at the bottom edge
inner = [row[:] for row in board]
inner[30][30] = 5
ok("edge_band_ignored", guardrails._board_key(board) == guardrails._board_key(bar)
   and guardrails._board_key(board) != guardrails._board_key(inner))


# --- end to end on real games ---------------------------------------------------------------------------------------
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


def tool_text(requests):
    return "\n".join(str(m.get("content")) for m in requests[-1] if m["role"] == "tool") if requests else ""


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


try:
    from harness.game.arc_host import ArcHost
    from harness.game.games import build_games, make_arcade
    from harness.run import fit_context

    # ls20 level 1, the 35 moves that won it in run 20261023_115157: move 12 repeats move 11 (a wall, only the bar
    # at the edge changed), so the leash stops the path there and the other 23 moves are not sent.
    LS20_L1 = list("11113332341144411122333212121444111")
    # The default (leash off, the wall stop with the edge band): the whole winning path is sent in one play().
    s, game, llm = session("ls20", {1: f'r = await play({LS20_L1!r})\nprint("RESULT", r.done, r.stopped)'})
    out0 = tool_text(llm.requests[:2])
    ok("default_sends_winning_path", "RESULT 35 level up" in out0 and s.stats["revisits"] == 0
       and s.stats["wall_hits"] == 0, out0[:300])
    game.finish()

    s, game, llm = session("ls20", {
        1: f'r = await play({LS20_L1!r})\nprint("RESULT", r.done, r.stopped)',
        2: ('for i in range(4):\n    await play(["1"], quiet=True)\n    try:\n        await play(["reset"], quiet=True)\n'
            '        print("RESET OK")\n    except RuntimeError as e:\n        print("REFUSED", e)'),
    }, revisit_stop=True)
    out1 = tool_text(llm.requests[:2])
    ok("leash_stops_path", "RESULT 12 board seen before" in out1 and "stopped: board seen before" in out1, out1[:600])
    # moves that brought a board back: #12 here, and the first "1" of the next cell (the same wall again)
    ok("revisits_counted", s.stats["revisits"] == 2, str(s.stats["revisits"]))
    out2 = tool_text(llm.requests[:3])
    ok("three_resets_then_refused", out2.count("RESET OK") == 3 and out2.count("REFUSED") == 1
       and "The 3 resets of this level are used" in out2, out2[:800])
    ok("refused_reset_spends_nothing", game.action_count == 12 + 4 + 3, str(game.action_count))
    ok("reset_starts_a_new_try", s.stats["model_resets"] == 3 and len(s._life_boards) <= 2, str(len(s._life_boards)))
    game.finish()

    # bp35: moves 3, 4, 7 in a loop lose the level after 44 moves (test_reset_and_file_guard.py). With the leash off
    # and resets allowed, the host still restarts the lost level itself, and that reset is not one of the 3.
    s, game, llm = session("bp35", {1: 'r = await play(["3", "4", "7"] * 20)\nprint("RESULT", r.stopped, r.done)'},
                           revisit_stop=False)
    out = tool_text(llm.requests[:2])
    ok("auto_reset_with_reset_move_on", s.stats["auto_resets"] == 1 and "RESULT GAME_OVER 44" in out
       and "GAME_OVER: this level was lost." in out and s._resets_used == {}, out[-600:])
    game.finish()

    # bp35 with the leash on: the loop brings a board back, so the path stops before the level is lost.
    s, game, llm = session("bp35", {1: 'r = await play(["3", "4", "7"] * 20)\nprint("RESULT", r.stopped, r.done)'},
                           revisit_stop=True)
    out = tool_text(llm.requests[:2])
    ok("leash_stops_loop_before_game_over", "RESULT board seen before" in out and s.stats["auto_resets"] == 0
       and game.action_count < 44, out[:600])
    game.finish()
except ImportError as e:   # no arc-agi SDK or games in this checkout
    WARNS.append(f"e2e skipped: {e!r}")

print("\n================ SUMMARY ================")
print(f"CHECKS: {CHECKS[0]}")
print(f"HARD FAILS: {len(FAILS)} -> {FAILS}")
print(f"WARNINGS  : {len(WARNS)} -> {WARNS}")
raise SystemExit(1 if FAILS else 0)

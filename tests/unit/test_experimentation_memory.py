# --- path shim: these suites live in tests/ but import from the repo root ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_PRIME = _os.environ.get("PRIME_DIR", "Sarbloh-Experimentation")
for _p in (_ROOT, _os.path.join(_ROOT, _PRIME)):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end shim ---
"""The agent's memory kept per level: GOAL, PLAN, HYPOTHESES and FINDINGS are dicts keyed by level ("L1", "L2", ...);
goals and hypotheses are plain text with no status (a dict is read as its keys, so no write is refused for a status);
a goal saying the level cannot be won is not saved; only the host marks a goal "won", at a level-up, shown as [won] in
the block; a wrong shape is not saved and is named in ``rejected``; a level-up erases nothing and promotes only that
level's findings; a new game erases everything; the kernel values and the blocks show every level.
"""
import json
import sys
import tempfile
import threading
import time
from pathlib import Path

from harness.agent import agent as A, prompts
from harness.config import build_config
from harness.llm.client import Reply
from harness.memory.lifecycle import GameMemory
from harness.memory.working import level_key
from harness.run import fit_context

FAILS, CHECKS = [], [0]


def ok(name, cond, detail=""):
    CHECKS[0] += 1
    print(("PASS " if cond else "FAIL ") + name + (f"  {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


mcfg = build_config({})["agent"]["memory"]
m = GameMemory(Path(tempfile.mkdtemp()), "g1", mcfg)
w = m.working

ok("level_key", level_key(0) == "L1" and level_key(3) == "L4")

# --- writes ------------------------------------------------------------------------------------------------------
out = m.sync({"GOAL": {"L1": ["push the block onto the flag"]}, "PLAN": {"L1": "push right"},
              "HYPOTHESES": {"L1": ["1 moves obj 4 up", "5 rotates"]},
              "FINDINGS": {"L1": ["the bar loses one block per move"]}}, turn=1, level=0)
ok("write_all_four", not out["errors"] and out["plan"] == ["L1"] and len(out["goal"]) == 1
   and [h["id"] for h in out["hypotheses"]] == ["h1", "h2"] and len(out["findings"]) == 1, str(out))
ok("goal_event_shape", out["goal"][0] == {"level": 0, "goal": "push the block onto the flag", "status": "kept",
                                          "prev_status": None})
kv = w.kernel_values()
ok("kernel_values_by_level", kv["GOAL"] == {"L1": ["push the block onto the flag"]} and kv["PLAN"] == {"L1": "push right"}
   and kv["HYPOTHESES"] == {"L1": ["1 moves obj 4 up", "5 rotates"]}
   and kv["FINDINGS"] == {"L1": ["the bar loses one block per move"]}, str(kv))

same = m.sync(kv, turn=2, level=0)
ok("unchanged_writes_nothing", not any(same[k] for k in ("plan", "goal", "hypotheses", "findings", "removed")), str(same))

drop = m.sync({"GOAL": {"L1": ["match the colours"]}}, turn=3, level=0)
ok("ruled_out_goal_is_deleted", [g["goal"] for g in drop["goal"]] == ["match the colours"]
   and drop["removed"] == ["push the block onto the flag"], str(drop))

# The statuses the 10-26 run wrote (and had refused 1043 times) now save: a dict is read as its keys.
old = m.sync({"HYPOTHESES": {"L1": {"1 moves obj 4 up": "untested", "5 rotates": "kept (untested)", "x": "verified"}}},
             turn=4, level=0)
ok("status_values_ignored", not old["errors"] and not old["rejected"]
   and w.kernel_values()["HYPOTHESES"] == {"L1": ["1 moves obj 4 up", "5 rotates", "x"]}
   and [h["id"] for h in old["hypotheses"]] == ["h3"], str(old))

loose = m.sync({"FINDINGS": {"L1": "one fact"}, "HYPOTHESES": {"L1": "one rule"}}, turn=5, level=0)
ok("loose_shapes_accepted", not loose["errors"] and w.kernel_values()["FINDINGS"] == {"L1": ["one fact"]}
   and w.kernel_values()["HYPOTHESES"] == {"L1": ["one rule"]}, str(loose))

# --- refused writes ------------------------------------------------------------------------------------------------
before = w.kernel_values()
for name, values, needle in [
        ("no_win_goal", {"GOAL": {"L1": ["Level 1 is unsolvable as generated"]}}, "every level can be won"),
        ("no_win_goal_2", {"GOAL": {"L1": {"the level cannot be won from here": "kept"}}}, "every level can be won"),
        ("bad_key", {"PLAN": {"level one": "x"}}, "is not a level"),
        ("old_string_shape", {"GOAL": "push the block"}, "must be a dict of level"),
        ("old_list_shape", {"FINDINGS": ["a"]}, "must be a dict of level"),
        ("findings_dict", {"FINDINGS": {"L1": {"a": "kept"}}}, "must be a list of text")]:
    r = m.sync(values, turn=6, level=0)
    ok(f"refused_{name}", r["rejected"] == list(values) and any(needle in e for e in r["errors"])
       and w.kernel_values() == before, str(r["errors"]))
mixed = m.sync({"GOAL": {"L1": ["it is impossible to win"]}, "PLAN": {"L1": "still saved"}}, turn=7, level=0)
ok("one_refused_variable_does_not_block_others", mixed["rejected"] == ["GOAL"] and w.plan_for(0) == "still saved")
ok("positive_goal_with_word_win_ok", not m.sync({"GOAL": {"L1": ["match the colours", "reach the exit to win"]}},
                                                turn=8, level=0)["errors"])

# --- level-up: nothing erased ------------------------------------------------------------------------------------
promo = m.on_level_up(level_won=1, step=10, objects=[])
kv = w.kernel_values()
ok("level_up_marks_goals_won", kv["GOAL"]["L1"] == ["match the colours", "reach the exit to win"]
   and promo["won_goals"] == ["match the colours", "reach the exit to win"]
   and [g["status"] for g in w.goals] == ["won", "won"], str(w.goals))
ok("level_up_erases_nothing", kv["PLAN"] == {"L1": "still saved"} and kv["FINDINGS"] == {"L1": ["one fact"]}
   and kv["HYPOTHESES"] == {"L1": ["one rule"]} and promo["kept"]["findings"] == 1, str(kv))
ok("level_up_confirmation", promo["confirmation"] == "confirmed: won level 1 with goal: match the colours; reach the exit to win")
ok("won_goal_stays_won", not m.sync(kv, turn=11, level=1)["errors"] and [g["status"] for g in w.goals] == ["won", "won"])

m.sync({**kv, "GOAL": {**kv["GOAL"], "L2": ["match the colours"]}, "FINDINGS": {**kv["FINDINGS"], "L2": ["new fact"]}},
       turn=12, level=1)
promo2 = m.on_level_up(level_won=2, step=20, objects=[])
ok("promotes_only_won_level_findings", promo2["kept"]["findings"] == 1
   and [p["text"] for p in promo2["promoted"] if p["type"] == "rule"] == ["new fact"], str(promo2["promoted"]))
blocks = m.blocks()
ok("blocks_show_every_level", "L1 [won] match the colours" in blocks["goal"] and "L2 [won] match the colours" in blocks["goal"]
   and "L1 h4 one rule" in blocks["hypotheses"] and "[kept]" not in blocks["hypotheses"]
   and "L1 - one fact" in blocks["findings"] and "L2 - new fact" in blocks["findings"] and "L1: still saved" in blocks["plan"],
   str(blocks))

# --- persistence and new game ------------------------------------------------------------------------------------
again = GameMemory(m.root, "g1", mcfg)
ok("saved_and_reloaded", again.working.kernel_values() == w.kernel_values())
m.on_new_game()
ok("new_game_erases", w.kernel_values() == {"GOAL": {}, "PLAN": {}, "HYPOTHESES": {}, "FINDINGS": {}, "SKILLS": {}}
   and w.goal_log == [] and w.next_h == 1)

# --- prompts -------------------------------------------------------------------------------------------------------
system = prompts.game_system(act_max=100, board_text=False, memory_shown=False)
ok("prompt_says_solvable", "All the games are solvable." in system)
ok("prompt_memory_by_level", 'GOAL["L2"] = ["push the block onto the flag"]' in system
   and "Nothing is emptied when you win a level" in system and "are emptied" not in system)
ok("prompt_has_no_status", '"kept"' not in system and '"refuted"' not in system and '"won"' not in system, system)
lvl = prompts.event_message("level_up", levels_done=1, win_levels=7, level=2)
ok("level_up_message", "are all kept" in lvl and "empty" not in lvl and 'under "L2"' in lvl and '"won"' not in lvl, lvl)

# --- a real level-up on ls20 through play(): the host marks the goal "won", nothing is emptied ---------------------
# The moves that won ls20 level 1 in run 20261023_115157 (steps.jsonl, actions 0-34).
LS20_L1 = list("11113332341144411122333212121444111")


def code_call(n, code):
    return [{"id": f"call{n}", "type": "function", "function": {"name": "ipython", "arguments": json.dumps({"code": code})}}]


SCRIPT = {1: code_call(1, 'GOAL["L1"] = ["reach the exit"]\nPLAN["L1"] = "walk"\nFINDINGS["L1"] = ["f1"]\n'
                          'def walk(path):\n    """the moves of a path"""\n    return list(path)\n'
                          f'r = await play(walk({LS20_L1!r}))'),
          2: code_call(2, "print(repr(GOAL)); print(repr(PLAN)); print(repr(FINDINGS))")}


class ScriptLLM:
    def __init__(self):
        self.requests, self.turn = [], 0

    def chat(self, messages, tools=None, max_tokens=None, timeout_s=None, template_kwargs=None, thinking=None):
        self.requests.append(messages)
        self.turn += 1
        if self.turn in SCRIPT:
            return Reply("", "thinking", SCRIPT[self.turn], "tool_calls", 5000, 300)
        return Reply("done", "thinking", [], "stop", 5000, 300)

    def yield_turn(self):
        pass


try:
    from harness.game.arc_host import ArcHost
    from harness.game.games import build_games, make_arcade

    game = build_games(make_arcade("offline", environments_dir=str(Path(_ROOT) / "eval" / "real_games")), "offline",
                       only=["ls20"])[0]
    game.start()
    run = Path(tempfile.mkdtemp())
    # the hidden-code setting under test (keep_code off), with every thinking kept; the revisit leash off, since this
    # 35-move path passes boards twice (test_revisit_leash.py)
    c = build_config({"agent": {"limits": {"max_turns": 3}, "memory": {"curator": False},
                                "keep_code": False, "keep_reasoning": True, "revisit_stop": False}})
    fit_context(c["agent"], 131072)
    llm = ScriptLLM()
    s = A.AgentSession(cfg=c["agent"], llm=llm, name="ls20", session_dir=run / "games" / "ls20",
                       task=prompts.task_message(game_id=game.game_id, max_actions=400, minutes=45),
                       arc=ArcHost(game, 400, should_stop=lambda: False, tokens_spent=lambda: 0),
                       deadline=time.time() + 3600, stop_event=threading.Event(), memory_root=run / "memory")
    s.run()
    game.finish()
    ok("e2e_level_won", game.state.levels_completed >= 1, str(game.state.levels_completed))
    printed = next((m["content"] for r in llm.requests for m in r
                    if m["role"] == "tool" and "{'L1':" in str(m.get("content"))), "")
    ok("e2e_kernel_goal_kept_and_nothing_emptied", "{'L1': ['reach the exit']}" in printed
       and "{'L1': 'walk'}" in printed and "{'L1': ['f1']}" in printed, printed[:400])
    kv = s.memory.working.kernel_values()
    ok("e2e_host_memory", kv["GOAL"] == {"L1": ["reach the exit"]} and kv["PLAN"] == {"L1": "walk"}
       and [g["status"] for g in s.memory.working.goals] == ["won"], str(kv))
    # the context: a stored reply keeps its thinking but not its code; no game status line in the outputs
    # in place of the code: the code graph of what the cell defined (cell 1 defines walk, cell 2 nothing)
    sent = [m for r in llm.requests[1:] for m in r if m["role"] == "assistant" and m.get("tool_calls")]
    codes = {tc["id"]: json.loads(tc["function"]["arguments"])["code"] for m in sent for tc in m["tool_calls"]}
    ok("e2e_code_not_sent_back", set(codes) == {"call1", "call2"} and "await play" not in "".join(codes.values())
       and all(m.get("reasoning_content") or m.get("reasoning") for m in sent), str(codes))
    ok("e2e_graph_in_place_of_code", codes.get("call1") == "# code not kept in the messages; the code graph of what it "
       "defined:\n# walk(path)  # the moves of a path"
       and codes.get("call2") == "# code not kept in the messages; it defined no functions or classes", str(codes))
    ok("e2e_graph_decided_once", all(json.loads(tc["function"]["arguments"])["code"] == codes[tc["id"]]
                                     for m in sent for tc in m["tool_calls"]))
    events = [json.loads(x) for x in (run / "games" / "ls20" / "transcript.jsonl").read_text(encoding="utf-8").splitlines()]
    logged = [e for e in events if e.get("event") == "message" and e.get("role") == "assistant" and e.get("code_full")]
    ok("e2e_transcript_keeps_code", logged and "await play(" in logged[0]["code_full"]["tool_calls"][0]["function"]["arguments"])
    outputs = [str(m.get("content")) for r in llm.requests for m in r if m["role"] == "tool"]
    ok("e2e_no_status_line", outputs and not any("min left]" in o or "[game:" in o for o in outputs), outputs[-1][-200:])
    ok("e2e_prompt_print_memory", "`print(GOAL)`" in llm.requests[0][0]["content"]
       and "in its place you see the code graph" in llm.requests[0][0]["content"])
except Exception as exc:  # noqa: BLE001
    import traceback
    traceback.print_exc()
    ok("e2e_ran", False, repr(exc))

print("================ SUMMARY ================")
print(f"CHECKS: {CHECKS[0]}")
print(f"HARD FAILS: {len(FAILS)} -> {FAILS}")
sys.exit(1 if FAILS else 0)

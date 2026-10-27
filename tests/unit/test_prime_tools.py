# --- path shim: these suites live in tests/ but import from the repo root ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_PRIME = _os.environ.get("PRIME_DIR", "Sarbloh")   # PRIME_DIR=Sarbloh-Experimentation tests the copy
for _p in (_ROOT, _os.path.join(_ROOT, _PRIME), _os.path.join(_ROOT, _PRIME, "harness", "runtime", "skills")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end shim ---
"""E008 tools end to end without a model: a scripted fake LLM plays the real ls20 game through a real REPL kernel with
the three tools (ipython, act, recall). Checks: the first state arrives with the task; the REPL reads `scene` and
cannot reach `arc`; a bad action id, an over-cap batch or a missing plan fail without spending; a 3-action batch gives
3 change lines, then exactly one observation message with one image; one act per reply; recall by step and by words;
"reset" inside act; a batch stops at a level-up, which promotes verified items into the lessons graph, clears the
level's memory and runs the curator; every request carries the memory block second and at most one image; the trace
and the SFT export read the E008 events. Also unit checks of the action parser and the toolset.
On a copy with E116 perception (Sarbloh-Experimentation) the REPL reads `observe()` instead of `scene`, and observe()
sends the full state (briefing, objects, board) once per state: not at the start (already shown), once after an act
even when called twice.
On Sarbloh-Experimentation (E118) there is no toolset setting (`game_tools`), `plan` is optional (an act without it
keeps the old plan), and the curator's level-up pass runs inside `act` before the level's memory is cleared. There the
memory is joined to the newest user message and built once per user message, so every request starts with the one
before it (the server's prefix cache): all of it within a tool-call chain, and up to the previous user message after
a new one.
When the copy has no `act` tool (moves are made from Python with `await act(...)` in ipython), every scripted act is
sent as an ipython call that awaits act() with the same arguments, two acts in one reply both run, and a cell that won
a level cannot move again before the new level is read.
(E006's version of this suite tested the removed dedicated tools; it is in git history, commit f117ca4.)
"""
import importlib.util
import inspect
import json
import queue
import sys
import tempfile
import time
from pathlib import Path

from harness.agent import context
from harness.agent.agent import AgentSession
try:
    from harness.agent.tools import toolset
except ImportError:   # E118 copy: one toolset, no setting
    from harness.agent.tools import game_tools
    toolset = None
from harness.config import build_config
from harness.game.arc_host import ArcHost

FAILS, WARNS, CHECKS = [], [], [0]
E116 = importlib.util.find_spec("harness.agent.perception") is not None   # observe() replaces `scene`
LEVEL_RESET = hasattr(AgentSession, "_reset_if_level_up")   # the experimentation copy resets the context at a level-up
MEMORY_LAST = context.build("S", [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
                                  {"role": "user", "content": "c"}],
                            {"role": "user", "content": "[memory] m"})[-1]["content"].startswith("[memory]")
# The experimentation copy's default context is append-only: the memory is a stored message, every picture is kept,
# and each request starts with the whole previous one until a rewrite (here: the level-up reset).
APPEND_ONLY = bool(build_config({})["agent"].get("append_only"))


def ok(name, cond, detail=""):
    CHECKS[0] += 1
    print(("PASS " if cond else "FAIL ") + name + (f"  {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


# --- 1. units ------------------------------------------------------------------------------------------------
P = AgentSession._parse_action
ok("parse_plain", P("1") == (1, None, None) and P(3) == (3, None, None) and P("ACTION4") == (4, None, None))
ok("parse_click_row_col", P("6 12 40") == (6, 12, 40) and P("A6(12,40)") == (6, 12, 40)
   and P({"action": 6, "r": 1, "c": 2}) == (6, 1, 2))
ok("parse_reset", P("reset") == (0, None, None) and P("RESET") == (0, None, None) and P(0) == (0, None, None))
for bad in ("6 12", "1 2", "up", "9"):
    try:
        P(bad)
        ok(f"parse_rejects_{bad!r}", False)
    except ValueError:
        ok(f"parse_rejects_{bad!r}", True)


def tools(mode):
    if toolset is None:
        return game_tools(act_max=5)
    extra = {"depth": 0, "max_depth": 0} if "depth" in inspect.signature(toolset).parameters else {}
    return toolset(mode, act_max=5, **extra)


ACT_IN_CODE = "act" not in [t["function"]["name"] for t in tools("e008")]   # moves from Python: `await act(...)`
if ACT_IN_CODE:
    ok("toolset_e008", [t["function"]["name"] for t in tools("e008")] == ["ipython", "recall"]
       and "await act(" in tools("e008")[0]["function"]["description"])
    PLAN_REQUIRED = False
else:
    ok("toolset_e008", [t["function"]["name"] for t in tools("e008")] == ["ipython", "act", "recall"])
    if toolset is not None:
        ok("toolset_ipython", [t["function"]["name"] for t in tools("ipython")] == ["ipython"])
    act_schema = tools("e008")[1]["function"]
    PLAN_REQUIRED = "plan" in act_schema["parameters"]["required"]
    ok("act_schema_required", act_schema["parameters"]["required"] ==
       (["actions", "plan"] if PLAN_REQUIRED else ["actions"])
       and "Up to 5 actions" in act_schema["parameters"]["properties"]["actions"]["description"]
       and "expect" not in act_schema["parameters"]["properties"])


def as_sent(calls):
    """The scripted calls as the harness takes them: an act becomes ipython code awaiting act() when there is no act
    tool. ``_after`` is extra code for that cell (dropped when act is its own tool)."""
    out = []
    for name, args in calls:
        args = dict(args)
        after = args.pop("_after", "")
        if name == "act" and ACT_IN_CODE:
            moves = args.pop("actions")
            out.append(("ipython", {"code": f"r = await act({moves!r}, **{args!r})" + after}))
        else:
            out.append((name, args))
    return out


# --- 2. scripted end-to-end run ---------------------------------------------------------------------------
class Usage:
    def to_json(self):
        return {}


class Reply:
    def __init__(self, calls=None, content=""):
        self.content, self.reasoning = content, "scripted"
        self.tool_calls = [{"id": f"c{i}_{time.time_ns()}", "type": "function",
                            "function": {"name": n, "arguments": json.dumps(a)}} for i, (n, a) in enumerate(calls or [])]
        self.finish_reason, self.prompt_tokens, self.completion_tokens = "stop", 1000, 10


FORCE_LEVEL_UP = {"at": None}   # transition index that the patched host reports as a level-up
if E116:
    LOOK = ("obs = observe()\nprint(obs.step, len(obs.objects) > 0, obs.legend)\nprint(obs.ascii(60, 10, 63, 20))\n"
            "print([o['bbox'] for o in obs.objects if o['letter'] == 'b'][0])")
    HISTORY = "print(observe().history[-3:])"
else:
    LOOK = ("print(scene.step, len(scene.objects) > 0, scene.legend)\nprint(scene.ascii(60, 10, 63, 20))\n"
            "print(scene.find(letter='b')[0]['bbox'])")
    HISTORY = "print(scene.history(3))"
ROOT_SCRIPT = [
    [("ipython", {"code": LOOK})],
    *([[("ipython", {"code": "observe()"})]] if E116 else []),       # the full state is already the newest message
    [("ipython", {"code": "obs = await arc.observe()"})],                                       # refused: no arc
    [("act", {"actions": ["1", "6 3 4"], "plan": "try a click"})],                                # refused: not legal
    [("act", {"actions": ["1"] * 6, "plan": "too many"})],                                       # refused: cap
    [("act", {"actions": ["1"], "plan": ""}) if PLAN_REQUIRED                                    # refused: no plan
     else ("act", {"actions": ["1"], "hypotheses": [{"id": "h9", "status": "verified"}]})],    # refused: unknown id
    [("act", {"actions": ["1", "1", "3"], "plan": "probe the controls",
              "hypotheses": [{"text": "1 moves obj 10 up", "status": "proposed"},
                             {"text": "the bottom bar counts steps", "status": "proposed"}],
              "goal": "reach the exit"})],
    *([[("ipython", {"code": "o = observe()\nprint(o.change)"}),     # sends the full state once ...
        ("ipython", {"code": "o2 = observe()\no2.letters"})]] if E116 else []),   # ... not twice; old name hinted
    [("act", {"actions": ["4"], **({"plan": "a"} if PLAN_REQUIRED else {})}),                 # no plan: kept
     ("act", {"actions": ["4"], "plan": "b"})],                                                  # 2nd refused
    [("recall", {"query": "#0"})],
    [("recall", {"query": "bar", "scope": "hypotheses"})],
    [("ipython", {"code": HISTORY})],
    [("act", {"actions": ["reset"], "plan": "test reset",
              "hypotheses": [{"id": "h1", "status": "verified", "evidence": [0, 1]}],
              "findings": ["the bottom bar shrinks on every action"]})],
    [("act", {"actions": ["1", "1", "1"], "plan": "force a level up on the 2nd step",           # stops at level-up
              "_after": "\ntry:\n    await act(['1'])\nexcept RuntimeError as e:\n    print('REFUSED:', e)"})],
    [],                                                                                         # continuation
]
CURATOR_REPLY = json.dumps({"open_questions": ["What does the bottom bar count?"],
                            "lessons": [{"type": "rule", "text": "Action 1 moves the blue block up",
                                         "shapes": [], "edge": "moves"}],
                            "skills": [{"name": "counter_bars", "content": "A bar on the edge that shrinks each "
                                                                          "action is a step budget."}],
                            "load_skills": ["counter_bars", "planning"]})


class FakeLLM:
    model = "fake"

    def __init__(self):
        self.usage = Usage()
        self.root = queue.Queue()
        for s in ROOT_SCRIPT:
            self.root.put(s)
        self.requests = []
        self.curator_calls = 0
        self.curator_inputs = []

    def chat(self, messages, tools=None, max_tokens=None, timeout_s=None, thinking=None, template_kwargs=None):
        if "You review the memory of an agent" in messages[0]["content"]:
            self.curator_calls += 1
            self.curator_inputs.append(messages)
            return Reply(content=CURATOR_REPLY)
        self.requests.append(messages)
        try:
            calls = self.root.get_nowait()
        except queue.Empty:
            calls = []
        if calls and calls[0][0] == "act" and "force a level up" in calls[0][1].get("plan", ""):
            FORCE_LEVEL_UP["at"] = "next+1"
        return Reply(as_sent(calls), content="" if calls else "done")


_real_handle = ArcHost.handle


def _patched(self, req, *args, **kwargs):
    if FORCE_LEVEL_UP["at"] == "next+1":
        FORCE_LEVEL_UP["at"] = len(self.transitions) + 1
    res = _real_handle(self, req, *args, **kwargs)
    if FORCE_LEVEL_UP["at"] is not None and len(self.transitions) - 1 == FORCE_LEVEL_UP["at"]:
        res = {**res, "level_up": True}
        FORCE_LEVEL_UP["at"] = None
    return res


def images_in(msgs):
    return sum(1 for m in msgs if isinstance(m.get("content"), list)
               for p in m["content"] if p.get("type") == "image_url")


try:
    ArcHost.handle = _patched
    from harness import run as RUN
    from harness.game.games import build_games, make_arcade

    out = Path(tempfile.mkdtemp(prefix="e008_tools_"))
    cfg = build_config({"experiment": "E008_tools_unit", "games": ["ls20"], "concurrency": 1, "game_wall_s": 300,
                        "max_actions_per_game": 40, "notebook_budget_s": 360, "teardown_reserve_s": 0,
                        "agent": {"tool_mode": "native", "toolset": "e008", "vision": True,
                                  "act_max_actions": 5,
                                  "limits": {"max_turns": len(ROOT_SCRIPT) + 1, "max_output_tokens": 10**6}}})
    import os
    os.environ["ONLY_RESET_LEVELS"] = "true"
    arcade = make_arcade("offline", environments_dir=str(Path(_ROOT) / "eval" / "real_games"),
                         recordings_dir=str(out / "recordings"))
    games = build_games(arcade, "offline", only=["ls20"], record=True)
    llm = FakeLLM()
    RUN.run_games(games, cfg, llm, out, time.time() + 300)
    gdir = next((out / "games").iterdir())
    ev = [json.loads(line) for line in (gdir / "transcript.jsonl").read_text(encoding="utf-8").splitlines()]
    msgs = [e for e in ev if e.get("event") == "message"]
    acts = [e for e in ev if e.get("event") == "act"]
    errs = [e for e in ev if e.get("event") == "tool_error"]
    tool_msgs = [e for e in msgs if e.get("role") == "tool"]
    obs_msgs = [e for e in msgs if e.get("_kind") == "observation"]

    first_obs = 2 if APPEND_ONLY else 1   # append-only: the full memory is stored between the task and the state
    ok("first_state_with_task", msgs[0]["role"] == "user" and "Play the ARC-AGI-3 game" in msgs[0]["content"]
       and (not APPEND_ONLY or msgs[1].get("_kind") == "pinned")
       and msgs[first_obs].get("_kind") == "observation" and "board:" in msgs[first_obs]["content"][0]["text"])
    ok("transcript_has_no_png_bytes", all("base64" not in json.dumps(m) for m in msgs))
    ok("repl_reads_scene", any("W=" in m["content"] or "B=" in m["content"] for m in tool_msgs[:1])
       and "[" in tool_msgs[0]["content"], tool_msgs[0]["content"][:300])
    ok("repl_has_no_arc", any("there is no `arc` in this harness" in m["content"] for m in tool_msgs))
    ok("bad_id_refused_no_spend", any(e["tool"] == "act" and "not legal now" in e["error"] for e in errs))
    ok("cap_refused", any(e["tool"] == "act" and "at most 5" in e["error"] for e in errs))
    if PLAN_REQUIRED:
        ok("plan_required", any(e["tool"] == "act" and "plan is required" in e["error"] for e in errs))
    else:
        ok("bad_hypothesis_refused", any(e["tool"] == "act" and "h9" in e["error"] for e in errs), str(errs)[:600])
        ok("act_without_plan_runs", any(a.get("plan") is None and a.get("done") == 1 for a in acts), str(acts)[:600])
    first = acts[0] if acts else {}
    ok("act_three_steps", first.get("done") == 3 and len(first.get("steps") or []) == 3, str(first)[:300])
    t_first = next((m for m in tool_msgs if m["content"].startswith("act: 3 of 3")), None)
    ok("three_change_lines", t_first is not None and sum(1 for ln in t_first["content"].splitlines()
                                                           if ln.startswith("#")) == 3)
    idx = msgs.index(t_first) if t_first else -1
    after = [m for m in msgs[idx + 1:] if m.get("_kind") not in ("pinned", "memory_update")] if idx >= 0 else []
    nxt = after[0] if after else {}
    ok("one_observation_after_act", nxt.get("_kind") == "observation"
       and sum(1 for p in nxt["content"] if p.get("type") == "image_url") == 1
       and nxt["content"][0]["text"].startswith("[state after step #2]"), str(nxt)[:400])
    ok("hypotheses_logged", sum(1 for e in ev if e.get("event") == "hypothesis") >= 3)
    ok("goal_logged", any(e.get("event") == "goal" and e["goal"] == "reach the exit" for e in ev))
    if ACT_IN_CODE:
        ok("two_acts_in_one_reply_run", any(a.get("plan") == "b" for a in acts))
        ok("no_act_after_level_up_in_cell", any("not run, nothing spent: the level changed" in m["content"]
                                                for m in tool_msgs), str([m["content"][-300:] for m in tool_msgs][-3:]))
    else:
        ok("one_act_per_reply", any("second act in one reply" in str(e.get("error")) for e in errs)
           and not any(a.get("plan") == "b" for a in acts))
    rec = [e for e in ev if e.get("event") == "recall"]
    ok("recall_step_and_words", len(rec) == 2 and rec[0]["hits"] >= 1 and rec[1]["hits"] >= 1, str(rec))
    ok("recall_text", any(m["content"].startswith("recall '#0' in all") and "#0 L0 A1(up)" in m["content"]
                          for m in tool_msgs))
    ok("scene_history_in_repl", any("'change'" in m["content"] for m in tool_msgs))
    reset_act = next((a for a in acts if a.get("plan") == "test reset"), {})
    ok("reset_inside_act", reset_act.get("done") == 1 and reset_act["steps"][0]["action"] == "RESET")
    lu = next((e for e in ev if e.get("event") == "level_up"), None)
    ok("batch_stops_at_level_up", acts[-1].get("stop") == "level up" and acts[-1].get("done") == 2, str(acts[-1])[:300])
    ok("level_up_promotes_verified", lu is not None and any(p["type"] == "rule" for p in lu["promoted"])
       and any("bottom bar shrinks" in p["text"] for p in lu["promoted"]), str(lu)[:400])
    ok("level_up_erases", lu is not None and lu["erased"]["hypotheses"] == 2 and lu["erased"]["findings"] == 1)
    cur = [e for e in ev if e.get("event") == "curator"]
    ok("curator_ran_at_level_up", llm.curator_calls >= 1 and cur and cur[0]["reason"] == "level_up"
       and cur[0]["loaded"] == ["counter_bars", "planning"], str(cur)[:300])
    if not PLAN_REQUIRED:   # E118: the level-up pass reads the level's memory before it is cleared
        lu_inputs = [m[1]["content"] for m in llm.curator_inputs if "<trigger>level_up</trigger>" in m[1]["content"]]
        ok("curator_sees_level_memory_before_clear", lu_inputs and "bottom bar shrinks" in lu_inputs[0]
           and "Write 2 to 5 skills" in llm.curator_inputs[0][0]["content"], str(lu_inputs)[:600])
        lu_msg = next((m["content"] for m in tool_msgs if "LEVEL UP:" in m["content"]), "")
        # the forced level-up leaves the engine on level 1, so the note is the early-level one
        ok("level_up_message_names_skills_and_reasoning", "New skills from the level you just won: counter_bars"
           in lu_msg and "learn the controls by trying them" in lu_msg, lu_msg[-800:])
        ok("task_says_level_1_learn_by_trying", "This is level 1. Early levels count least" in msgs[0]["content"])
    after_lu = llm.requests[-1] if llm.requests else []

    def text0(m):
        return m["content"] if isinstance(m["content"], str) else m["content"][0]["text"]

    def last_user(r):
        return max(i for i, m in enumerate(r) if m["role"] == "user")

    if APPEND_ONLY:
        # Each request starts with the whole previous one, except at the level-up reset; the memory is in full
        # once per context (first message after the task or the reset) and as [memory update] after new states.
        broken, resets = [], 0
        for n, (a, b) in enumerate(zip(llm.requests, llm.requests[1:])):
            if b[:len(a)] != a:
                if LEVEL_RESET and "LEVEL UP:" in text0(b[1]) and "LEVEL UP:" not in text0(a[1]):
                    resets += 1
                else:
                    broken.append(n)
        ok("each_request_starts_with_the_last", not broken and resets == 1 and len(llm.requests) >= 8,
           f"broken at {broken}, resets {resets}")
        ok("memory_full_once_per_context", all(sum("[memory] Your memory" in text0(m) for m in r if m["role"] == "user")
                                               == 1 for r in llm.requests))
        ok("memory_updates_sent", any("[memory update]" in text0(m) for r in llm.requests for m in r))
        pinned = next((text0(m) for m in after_lu if m["role"] == "user" and "[memory] Your memory" in text0(m)), "")
    elif MEMORY_LAST:
        # The memory is joined to the front of the newest user message, built once per user message, so each
        # request starts with the one before it: all of it within a tool-call chain, and everything before the
        # previous newest user message once a new one arrives (that message loses the memory and its image, and
        # the chain after it loses its thinking).
        pinned = text0(after_lu[last_user(after_lu)]) if after_lu else ""
        ok("memory_once_in_newest_user_message", all(
            [i for i, m in enumerate(r) if m["role"] == "user" and "\n## Game\n" in text0(m)] == [last_user(r)]
            and "[memory] Your memory" in text0(r[last_user(r)]) for r in llm.requests))
        same, moved, broken, resets = 0, 0, [], 0
        for n, (a, b) in enumerate(zip(llm.requests, llm.requests[1:])):
            if LEVEL_RESET and "LEVEL UP:" in text0(b[1]) and "LEVEL UP:" not in text0(a[1]):
                resets += 1   # a level-up starts a clean context: the level-up message and the new full state
                continue
            k = last_user(a)
            if len(b) > k and b[k] == a[k]:
                same += 1
                if b[:len(a)] != a:
                    broken.append(n)
            else:
                moved += 1
                if b[:k] != a[:k]:
                    broken.append(n)
        ok("each_request_starts_with_the_last", not broken and same >= 3 and moved >= 3,
           f"broken at {broken}, same chain {same}, new message {moved}")
        if LEVEL_RESET:
            ok("level_up_resets_context_once", resets == 1 and len(after_lu) and after_lu[0]["role"] == "system",
               f"resets {resets}")
    else:
        pinned = text0(after_lu[1]) if len(after_lu) > 1 else ""
        ok("pinned_second_every_request", all(text0(r[1]).startswith("[memory]") for r in llm.requests))
    ok("pinned_after_level_up", "confirmed: won level" in pinned and "What does the bottom bar count?" in pinned
       and "### counter_bars" in pinned and "h1 [" not in pinned, pinned[:1500])
    if APPEND_ONLY:
        ok("every_picture_kept", any(images_in(r) >= 3 for r in llm.requests),
           str([images_in(r) for r in llm.requests]))
    else:
        ok("at_most_one_image_per_request", all(images_in(r) <= 1 for r in llm.requests)
           and any(images_in(r) == 1 for r in llm.requests))
    lessons = json.loads((out / "memory" / "lessons.json").read_text(encoding="utf-8"))
    ok("lessons_file_shared", any("bottom bar shrinks" in n["text"] for n in lessons["nodes"].values())
       and any(n["source"] == "curator" for n in lessons["nodes"].values()))
    sess = json.loads((gdir / "session.json").read_text(encoding="utf-8"))
    ok("stats_e008", sess["act_calls"] >= 4 and sess["images_sent"] >= 4 and sess["recalls"] == 2
       and sess["act_arg_errors"] >= 3 and sess["promotions"] >= 2, json.dumps({k: sess.get(k) for k in (
           "act_calls", "images_sent", "recalls", "act_arg_errors", "promotions")}))
    rep = (gdir / "report.md").read_text(encoding="utf-8")
    ok("report_has_sections", all(h in rep for h in ("## 1. Actions", "## 2. Understanding", "## 4. Waste",
                                                     "## 5. Run context"))
       and ("- E008:" in rep or "- memory:" in rep))
    tr = json.loads((gdir / "trace.json").read_text(encoding="utf-8"))
    ok("trace_plan_as_cause", tr["actions"] and tr["actions"][0]["expect"] == "probe the controls"
       and tr["actions"][0]["tool"] == "act")
    ok("trace_hypothesis_beliefs", any(b["kinds"] == ["proposed", "verified"] for b in tr["beliefs"]),
       str([b["kinds"] for b in tr["beliefs"]]))
    ok("trace_e008_context", tr["context"]["promotions"] >= 2 and tr["context"]["recalls"] == 2
       and tr["context"]["repeated_share"] is not None and tr["context"]["hypotheses"] >= 2)
    if E116:
        notes = [m["content"] for m in tool_msgs if "[observe: " in m["content"]]
        ok("e116_observe_at_start_not_resent", len(notes) >= 2 and "is already in the newest board message" in notes[1]
           and not any(m.get("_full") and m.get("_step") is None for m in obs_msgs[1:]), str(notes)[:600])
        i_obs = next((k for k, m in enumerate(msgs) if m.get("role") == "tool" and "A3(left)" in m["content"]
                      and "[observe: the full state at step #2 follows" in m["content"]), None)
        ok("e116_observe_sees_last_change", i_obs is not None, str([m["content"][:200] for m in tool_msgs])[:1500])
        second = msgs[i_obs + 1] if i_obs is not None else {}
        pushed = msgs[i_obs + 2] if i_obs is not None and i_obs + 2 < len(msgs) else {}
        text = pushed["content"][0]["text"] if isinstance(pushed.get("content"), list) else ""
        ok("e116_observe_old_name_hint", "obs.board" in str(second.get("content"))
           and "follows in the next message" in str(second.get("content")), str(second)[:600])
        ok("e116_full_state_pushed_once", pushed.get("_kind") == "observation" and pushed.get("_full")
           and text.startswith("[state after step #2]") and "last step: #2 A3(left)" in text
           and "BRIEFING (" in text and "\nboard:\n" in text
           and sum(1 for m in obs_msgs if m.get("_full") and m.get("_step") == 2) == 1, text[:600])
        ok("e116_no_display_marker", not any("[display:" in m["content"] for m in tool_msgs))
        ok("e116_stats", sess.get("observe_calls") == 5 and sess.get("observe_pushes") == 2
           and sess.get("perception_errors") == 0 and sess.get("perception_s_max", 9) < 5,
           json.dumps({k: sess.get(k) for k in ("observe_calls", "observe_pushes", "perception_errors",
                                                "perception_s_max", "perception_s_total")}))
    sft = [json.loads(line) for line in (out / "sft_levels.jsonl").read_text(encoding="utf-8").splitlines()]
    ok("sft_rows", sft and sft[0]["tools"] and sft[0]["system"] and sft[0]["messages"])
    ok("sft_roles_alternate_users", all(not (a["role"] == b["role"] == "user")
                                        for r in sft for a, b in zip(r["messages"], r["messages"][1:])))
    print(f"(run dir: {out})")
except Exception as e:  # noqa: BLE001
    import traceback
    traceback.print_exc()
    ok("end_to_end_ran", False, f"{type(e).__name__}: {e}")
finally:
    ArcHost.handle = _real_handle

print("\n================ SUMMARY ================")
print(f"CHECKS: {CHECKS[0]}")
print(f"HARD FAILS: {len(FAILS)} -> {FAILS}")
print(f"WARNINGS  : {len(WARNS)} -> {WARNS}")
sys.exit(1 if FAILS else 0)

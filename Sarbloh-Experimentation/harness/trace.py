"""Trace of a run: for every game, what the agent thought, ran, believed and spent, and a training export.

    uv run python Sarbloh-Experimentation/harness/trace.py <run_dir>     # run.py calls build_run

Sources in <run_dir>: ``results.json`` and, per game, ``games/<game_id>/transcript.jsonl`` (every message of the
session with the model's reasoning, and one event per cell, move, memory write, recall, code-graph change, drain, turn
end and curator pass), plus the ARC SDK recording ``recordings/<scorecard>/<game_id>-<guid>.jsonl`` (one line per
environment step). Without a recording the actions come from ``run.history``.

How the loop shows up in a transcript (``harness/agent/agent.py``): a *request* is one model call (the ``turn`` field
of a message is its number); a *turn* is the run of requests between two ``turn_end`` events, which give why it ended
(``moved``, ``request_cap``, ``token_cap``, ``answered``, ``stop``) and how many requests it took. There is one tool,
ipython: moves are ``play()`` calls inside cells (``act`` and ``step`` events), and ``recall()`` and ``graph()`` run in
the kernel (``recall`` and ``code_graph`` events).

Writes, per game (next to the transcript):
- ``steps.jsonl`` / ``steps.md``: one row per action / the request-by-request map (thought, code, output, events),
  with the turn boundaries.
- ``report.md`` / ``trace.json``: five sections, as markdown and as data.
  1. Actions and causes: per action the request, the turn, the call, the action, the plan written with it, what
     changed, and whether the (state, action) pair was already tried.
  2. Understanding over time: every hypothesis, finding and goal the agent wrote, when it first appeared and whether
     the agent removed it later; the goal statements (the "correct" column is left for a human to mark).
  3. Living code: per ipython cell the functions defined, the earlier helpers reused, printed boards and errors;
     living_code_ratio = cells that call a helper defined in an earlier cell / cells after the first helper. From the
     code graph: its change lines, the functions removed, the calls to missing names it flagged, and
     ``calls_to_removed``: cells that called a function the graph had reported removed (and not defined again).
  4. Waste per level: actions vs the human baseline, resets, the largest single spend, repeated (state, action)
     pairs, and actions spent after the last new belief (stagnation).
  5. Run context: config hash, git SHA, model, turns and why they ended, requests per turn, drains, trims, resets,
     curator passes and their time, recalls, surprises, notes, pictures, tokens, wall clock, end reason.
Writes, per run: ``trace.md`` (one row per game) and ``sft_levels.jsonl``, the LoRA SFT export, tagged per row with the
level, ``level_cleared`` and ``rhae``, so "successful trajectories" is ``[r for r in rows if r["level_cleared"]]``.

The SFT export is what the model was sent, not the message log (``sft_wire``). The agent rewrites its context while it
plays (drains, trims, resets), so the transcript has a snapshot of the stored context after every rewrite; the stored
context at any request is the last snapshot plus the messages appended after it, and each request is rebuilt from it
with the agent's own builder (``harness.agent.context.build``) and checked against the digest the agent logged
(``wire_sha``). Requests whose history extends the previous one plus its reply form one row, so every assistant message
in a row follows the messages it was answering. With the picture at the tail (``tail`` events: the newest picture is
the last message of a request and is never stored), each reply in a row follows the picture it saw, and the pictures
seen by the earlier replies of the row stay where they were. Per reply: ``turn``, ``flags`` (cut_off, tool_error,
cell_error, repeat, game_over, no_tool_call) and ``train``: false for a flagged reply (all flags but no_tool_call) and
for replies of earlier rows that are only context here. Pictures are files (``games/<id>/frames/``) named in the image
parts. Row fields beside ``system``, ``tools`` and ``messages``: ``segment_start`` (start, drain, trim, overflow,
level_up), ``template_kwargs`` (the reasoning effort), ``wire_verified``. A transcript without wire fields (tracing
off) has no SFT rows.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

if __package__ in (None, ""):   # run as a script: the package root is the parent of harness/
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness.agent import context as agent_context   # the agent's own request builder

TURN_END_REASONS = ("moved", "request_cap", "token_cap", "answered", "stop")


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            pass
    return rows


def _cut(text: Any, n: int) -> str:
    text = str(text or "").strip()
    return text if len(text) <= n else f"{text[:n // 2]}\n[... {len(text) - n} chars cut ...]\n{text[-n // 2:]}"


def _one(text: Any, n: int = 160) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[:n - 3] + "..."


def _md(text: Any, n: int = 160) -> str:
    return _one(text, n).replace("|", "\\|")


_ACTION_ID = {"RESET": 0, **{f"ACTION{i}": i for i in range(1, 8)}}


def action_rows(recording: Path | None, run: dict[str, Any] | None) -> list[dict[str, Any]]:
    """One row per agent action: the SDK recording if there is one, else ``run.history``."""
    rows: list[dict[str, Any]] = []
    if recording is not None and recording.exists():
        for n, ev in enumerate(_jsonl(recording)):
            d = ev.get("data") or {}
            ai = d.get("action_input") or {}
            ref = ai.get("reasoning") if isinstance(ai.get("reasoning"), dict) else {}
            if n == 0 and not ref:
                continue  # the SDK records its own reset at Arcade.make(); not an agent action
            xy = ai.get("data") or {}
            rows.append({"i": len(rows), "action": ai.get("id"), "x": xy.get("x"), "y": xy.get("y"),
                         "levels": d.get("levels_completed"), "state": d.get("state"),
                         "full_reset": bool(d.get("full_reset")), "turn": ref.get("turn"), "call": ref.get("call"),
                         "k": ref.get("k"), "after_cell": bool(ref.get("after_cell")), "ts": ev.get("timestamp")})
        return rows
    hist = (run or {}).get("history") or []
    for i, h in enumerate(hist):
        ref = h.get("ref") or {}
        after = hist[i + 1]["lvl"] if i + 1 < len(hist) else (run or {}).get("levels_completed")
        rows.append({"i": i, "action": h.get("a"), "x": (h.get("d") or {}).get("x"), "y": (h.get("d") or {}).get("y"),
                     "levels": after, "state": None, "full_reset": False, "turn": ref.get("turn"),
                     "call": ref.get("call"), "k": ref.get("k"), "after_cell": bool(ref.get("after_cell"))})
    return rows


def _name(r: dict[str, Any]) -> str:
    return str(r["action"]).replace("ACTION", "A") + (f"({r['x']},{r['y']})" if r.get("x") is not None else "")


def render_actions(rows: list[dict[str, Any]], levels_before: int | None) -> str:
    """Runs of the same action, broken at every event: ``#12-#14 A1x3, #15 A6(3,4) -> LEVEL UP 1``."""
    parts: list[str] = []
    run_start, prev = None, levels_before
    for j, r in enumerate(rows):
        marks = []
        if prev is not None and r.get("levels") is not None and r["levels"] > prev:
            marks.append(f"LEVEL UP {r['levels']}")
        if r.get("state") in ("GAME_OVER", "WIN"):
            marks.append(r["state"])
        if r.get("full_reset"):
            marks.append("FULL RESET (score wiped)")
        prev = r["levels"] if r.get("levels") is not None else prev
        if run_start is None:
            run_start = j
        nxt = rows[j + 1] if j + 1 < len(rows) else None
        if marks or nxt is None or _name(nxt) != _name(r):
            a, n = rows[run_start]["i"], j - run_start + 1
            parts.append((f"#{a}" if n == 1 else f"#{a}-#{r['i']}") + f" {_name(r)}" + (f"x{n}" if n > 1 else "")
                         + (" -> " + ", ".join(marks) if marks else ""))
            run_start = None
    return ", ".join(parts)


def _tool_calls(ev: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
    """(call id, tool name, parsed args) for an assistant message event."""
    out = []
    for tc in ev.get("tool_calls") or []:
        fn = tc.get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except (ValueError, TypeError):
            args = {"unparsed": fn.get("arguments")}
        out.append((str(tc.get("id")), fn.get("name") or "ipython", args if isinstance(args, dict) else {}))
    return out


def outer_turns(events: list[dict[str, Any]]) -> dict[Any, int]:
    """Request number -> the turn it belongs to (1-based): turn k is the requests after the (k-1)th ``turn_end``."""
    out, turn = {}, 1
    for ev in events:
        if ev.get("event") == "message" and ev.get("role") == "assistant" and ev.get("turn") is not None:
            out[ev["turn"]] = turn
        elif ev.get("event") == "turn_end":
            turn += 1
    return out


# --- per-game analysis -----------------------------------------------------------------------------------
_FENCED = re.compile(r"```(?:python|py|ipython|repl)?[ \t]*\n(.*?)```", re.DOTALL)   # = guardrails._FENCED


def _printed_board(output: str) -> bool:
    """A cell that printed a board into the context (what a drain masks)."""
    return "[a printed board of" in agent_context.mask_boards(output or "")


def _code_names(code: str) -> tuple[set[str], set[str], set[str], str | None]:
    """(defined functions/classes, top-level assigned names, loaded names, syntax error) of a cell."""
    try:
        tree = ast.parse(code or "")
    except SyntaxError as exc:
        return set(), set(), set(), f"SyntaxError: {exc.msg}"
    defs = {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    assigned: set[str] = set()
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(
            node, (ast.AnnAssign, ast.AugAssign)) else []
        for t in targets:
            assigned |= {n.id for n in ast.walk(t) if isinstance(n, ast.Name)}
    loads = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    return defs, assigned, loads, None


def _delta_names(line: str) -> tuple[set[str], set[str]]:
    """(added or changed, removed) names in one ``[code graph: ...]`` line."""
    body = line.strip()[len("[code graph:"):].rstrip("]") if line.strip().startswith("[code graph:") else line
    added, removed = set(), set()
    for part in body.split(" | "):
        part = part.strip()
        m = re.match(r"^([+~-])(?:class )?([A-Za-z_]\w*)", part)
        if m:
            (removed if m.group(1) == "-" else added).add(m.group(2))
    return added, removed


def code_graph_report(events: list[dict[str, Any]]) -> dict[str, Any]:
    """What the code graph saw: its change lines, the names it reported removed, its missing-call warnings, and the
    cells that later called a removed function without defining it again (the agent trusting an old line)."""
    deltas = [e for e in events if e.get("event") == "code_graph"]
    removed: set[str] = set()
    removed_ever: set[str] = set()
    stale: list[dict[str, Any]] = []
    for ev in events:
        if ev.get("event") == "code_graph":
            added, gone = _delta_names(ev.get("delta") or "")
            removed = (removed | gone) - added
            removed_ever |= gone
        elif ev.get("event") == "message" and ev.get("role") == "assistant":
            for cid, name, args in _tool_calls(ev):
                if name != "ipython":
                    continue
                defs, assigned, loads, _ = _code_names(str(args.get("code") or ""))
                hit = sorted((loads & removed) - defs - assigned)
                if hit:
                    stale.append({"turn": ev.get("turn"), "call": cid, "names": hit})
    return {"deltas": len(deltas), "removed": sorted(removed_ever),
            "missing_warnings": sum((e.get("delta") or "").count("! ") for e in deltas),
            "nodes_last": deltas[-1].get("nodes") if deltas else None,
            "calls_to_removed": len(stale), "calls_to_removed_at": stale[:20]}


def analyze_game(events: list[dict[str, Any]], rows: list[dict[str, Any]], run: dict[str, Any],
                 session: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    t0 = events[0]["t"] if events else 0.0
    turn_of_req = outer_turns(events)
    turn_of_call: dict[str, int] = {}
    tool_of_call: dict[str, str] = {}
    args_of_call: dict[str, dict[str, Any]] = {}
    output_of_call: dict[str, str] = {}
    fenced: str | None = None
    for ev in events:
        if ev.get("event") == "message" and ev.get("role") == "assistant":
            calls = _tool_calls(ev)
            for cid, name, args in calls:
                turn_of_call[cid], tool_of_call[cid], args_of_call[cid] = ev.get("turn"), name, args
            blocks = [] if calls else [b for b in _FENCED.findall(ev.get("content") or "") if b.strip()]
            fenced = f"fenced-t{ev.get('turn')}" if blocks else None   # the agent's call id for fenced cells
            if fenced:
                turn_of_call[fenced], tool_of_call[fenced] = ev.get("turn"), "ipython"
                args_of_call[fenced] = {"code": "\n\n".join(blocks)}
        elif ev.get("event") == "message" and ev.get("role") == "tool":
            output_of_call[str(ev.get("tool_call_id"))] = str(ev.get("content") or "")
        elif ev.get("event") == "message" and ev.get("role") == "user" and fenced \
                and str(ev.get("content") or "").startswith("[ipython output]"):
            output_of_call[fenced], fenced = str(ev.get("content") or ""), None
    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in events:
        by_kind[str(e.get("event"))].append(e)
    steps, acts, hyps = by_kind["step"], by_kind["act"], by_kind["hypothesis"]

    # 1. actions and their causes ------------------------------------------------------------------------
    act_by_call = {str(a.get("call")): a for a in acts}
    actions: list[dict[str, Any]] = []
    if steps:
        for s in steps:
            act = act_by_call.get(str(s.get("call")))
            actions.append({
                "i": s.get("i"), "request": s.get("turn"), "turn": turn_of_req.get(s.get("turn")),
                "call": s.get("call"), "action": s.get("action"), "x": s.get("x"), "y": s.get("y"),
                "level": s.get("level"), "plan": act.get("plan") if act else None,
                "happened": s.get("change"), "state": s.get("state"), "level_up": s.get("level_up"),
                "repeat_of": s.get("repeat_of"), "t": round(s.get("t", t0) - t0, 1)})
    else:  # no step events (a run without tracing of the host): the recording has the action, the call and the level
        prev_level = 0
        for r in rows:
            actions.append({"i": r["i"], "request": r.get("turn"), "turn": turn_of_req.get(r.get("turn")),
                            "call": r.get("call"), "action": _ACTION_ID.get(str(r["action"]), r["action"]),
                            "x": r.get("x"), "y": r.get("y"), "level": prev_level, "plan": None, "happened": None,
                            "state": r.get("state"), "level_up": None, "repeat_of": None, "t": None})
            prev_level = r["levels"] if r.get("levels") is not None else prev_level

    # 2. understanding over time -------------------------------------------------------------------------
    beliefs: dict[str, dict[str, Any]] = {}
    for h in hyps:   # the agent's hypotheses
        bid = f"{h.get('level')}:{h.get('id')}"
        b = beliefs.setdefault(bid, {"id": bid, "kind": "hypothesis", "first_turn": h.get("turn"),
                                     "first_action": h.get("action_count"), "first_level": h.get("level"),
                                     "versions": 0})
        b["versions"] += 1
        b["content"] = h.get("text")
    for e in by_kind["finding"]:
        bid = f"finding:{e.get('t')}"
        beliefs[bid] = {"id": bid, "kind": "finding", "first_turn": e.get("turn"), "first_action": e.get("action_count"),
                        "first_level": e.get("level"), "versions": 1, "content": e.get("text")}
    removed = {t for e in by_kind["memory_removed"] for t in e.get("texts") or []}
    for b in beliefs.values():
        b["removed_by_agent"] = b.get("content") in removed
    goals = [{"turn": g.get("turn"), "action_count": g.get("action_count"), "level": g.get("level"),
              "goal": g.get("goal"), "correct": None} for g in by_kind["goal"]]   # marked by hand after the run

    # 3. living code -------------------------------------------------------------------------------------
    cells = []
    defined_at: dict[str, int] = {}
    cell_status = {str(e.get("call")): e for e in by_kind["cell"]}
    for cid, name in tool_of_call.items():
        if name != "ipython":
            continue
        code = str(args_of_call.get(cid, {}).get("code") or "")
        defs, assigned, loads, syntax = _code_names(code)
        n = len(cells)
        reused = sorted(nm for nm in loads if nm in defined_at and defined_at[nm] < n)
        for d in defs:
            defined_at.setdefault(d, n)
        ce = cell_status.get(cid) or {}
        cells.append({"n": n, "request": turn_of_call.get(cid), "call": cid, "defines": sorted(defs),
                      "reuses": reused, "variables": sorted(assigned - defs)[:20],
                      "printed_board": _printed_board(output_of_call.get(cid, "")),
                      "status": ce.get("status") or ("syntax" if syntax else None),
                      "error": syntax or ("error" if ce.get("status") not in (None, "ok") else None),
                      "actions": ce.get("acts"), "output_chars": len(output_of_call.get(cid, "")),
                      "code_lines": code.count("\n") + 1 if code else 0})
    first_def = min(defined_at.values()) if defined_at else None
    after = [c for c in cells if first_def is not None and c["n"] > first_def]
    reused_names = {nm for c in cells for nm in c["reuses"]}
    living = {"cells": len(cells), "helpers_defined": len(defined_at), "helpers_reused": len(reused_names),
              "living_code_ratio": round(sum(1 for c in after if c["reuses"]) / len(after), 3) if after else 0.0,
              "printed_boards": sum(1 for c in cells if c["printed_board"]),
              "cell_errors": sum(1 for c in cells if c["error"]), "graph": code_graph_report(events)}

    # 4. waste per level -----------------------------------------------------------------------------------
    per_level = run.get("actions_per_level") or []
    base = run.get("baseline_actions") or []
    cleared = run.get("levels_completed") or 0
    belief_marks: dict[int, list[int]] = defaultdict(list)   # level -> action counts of new beliefs
    for h in hyps:
        if h.get("prev_status") != h.get("status"):
            belief_marks[h.get("level") or 0].append(h.get("action_count") or 0)
    for e in by_kind["finding"] + by_kind["goal"]:
        belief_marks[e.get("level") or 0].append(e.get("action_count") or 0)
    waste = []
    for lvl in range(len(per_level) or (max((a["level"] or 0 for a in actions), default=0) + 1)):
        mine = [a for a in actions if a.get("level") == lvl]
        n_act = per_level[lvl] if lvl < len(per_level) else len(mine)
        if not n_act and lvl > cleared:
            continue
        idx = [a["i"] for a in mine]
        last_belief = max(belief_marks.get(lvl, []), default=None)
        stagnation = sum(1 for a in mine if last_belief is not None and a["i"] >= last_belief) if steps else None
        calls = Counter(str(a["call"]) for a in mine if a.get("call"))
        waste.append({
            "level": lvl, "actions": n_act, "baseline": base[lvl] if lvl < len(base) else None,
            "cleared": lvl < cleared,
            "rhae": round(min((base[lvl] / n_act) ** 2, 1.15), 4) if lvl < cleared and lvl < len(base) and n_act else 0.0,
            "resets": [{"i": a["i"], "plan": a["plan"]} for a in mine if a["action"] in (0, "RESET", "reset")],
            "largest_spend": max(calls.values(), default=0),
            "repeated_pairs": sum(1 for a in mine if a.get("repeat_of") is not None) if steps else None,
            "actions_after_last_new_belief": stagnation if last_belief is not None else (len(mine) if steps else None),
            "first_i": min(idx, default=None), "last_i": max(idx, default=None)})

    # 5. run context ---------------------------------------------------------------------------------------
    end = next((e for e in reversed(events) if e.get("event") == "end"), {})
    stats = end.get("stats") or {}
    ends = by_kind["turn_end"]
    curations = by_kind["curator"]
    recalls = by_kind["recall"]
    drains = [{"t": round(e.get("t", t0) - t0, 1), "reason": e.get("reason"), "tokens_before": e.get("tokens_before"),
               "tokens_after_est": e.get("tokens_after_est"), "action_count": e.get("action_count")}
              for e in by_kind["drain"]]
    ctx = {**context, "end_reason": session.get("end_reason") or end.get("reason"),
           "wall_s": session.get("wall_s"), "tokens_total": session.get("tokens_total"),
           "output_tokens": stats.get("output_tokens"), "prompt_tokens_last": stats.get("prompt_tokens_last"),
           "requests": stats.get("turns"), "turns": len(ends),
           "turn_ends": dict(Counter(e.get("reason") for e in ends)),
           "requests_per_turn": round(sum(e.get("requests") or 0 for e in ends) / len(ends), 2) if ends else None,
           "requests_per_turn_max": max((e.get("requests") or 0 for e in ends), default=None),
           "tool_counts": stats.get("tool_counts") or dict(Counter(tool_of_call.values())),
           "tool_errors": stats.get("tool_errors"), "cell_errors": stats.get("cell_errors"),
           "play_calls": len(acts),
           "moves_per_play": round(sum(a.get("done", 0) for a in acts) / len(acts), 2) if acts else None,
           "drains": drains, "trims": len(by_kind["context_trim"]), "resets": len(by_kind["context_reset"]),
           "overflows": len(by_kind["context_overflow"]),
           "act_arg_errors": stats.get("act_arg_errors"), "recalls": len(recalls),
           "recall_scopes": dict(Counter(str(r.get("scope")) for r in recalls)),
           "skills_recalled": sorted({s for r in recalls for s in r.get("loaded") or []}),
           "hypotheses": len({(h.get("level"), h.get("id")) for h in hyps}),
           "levels_without_hypothesis": sorted({a["level"] for a in actions if a.get("level") is not None}
                                               - {h.get("level") for h in hyps}) if hyps or acts else None,
           "promotions": sum(len(e.get("promoted") or []) for e in by_kind["level_up"]),
           "curator_runs": dict(Counter(c.get("reason") or "level_up" for c in curations if not c.get("error"))),
           "curator_errors": len([c for c in curations if c.get("error")]),
           "curator_s": round(sum(c.get("duration_s") or 0 for c in curations), 1),
           "skills_written": sorted({s for c in curations for s in c.get("skills") or []}),
           "skills_loaded_by_curator": sorted({s for c in curations for s in c.get("loaded") or []}),
           "surprises": len(by_kind["surprise"]), "stuck_notes": len(by_kind["stuck_note"]),
           "cut_off_notes": len(by_kind["cut_off_note"]), "tail_pictures": len(by_kind["tail"]),
           "images_sent": stats.get("images_sent"), "observation_chars_max": stats.get("observation_chars_max"),
           "repeated_share": round(sum(1 for r in actions if r.get("repeat_of") is not None) / len(actions), 3)
           if actions and steps else None}
    return {"game_id": run.get("game_id"), "levels_completed": cleared, "number_of_levels": run.get("number_of_levels"),
            "score": run.get("final_score"), "actions": actions,
            "repeats": sum(1 for r in actions if r.get("repeat_of") is not None) if steps else None,
            "beliefs": list(beliefs.values()), "goals": goals, "cells": cells, "living_code": living,
            "waste": waste, "context": ctx}


def render_report(a: dict[str, Any]) -> str:
    c = a["context"]
    out = [f"# {a['game_id']}: trace report", "",
           f"level {a['levels_completed']}/{a['number_of_levels']} | score {a['score']} | {len(a['actions'])} actions "
           f"| end {c.get('end_reason')}", ""]
    # 1
    out += ["## 1. Actions and their causes", "",
            "`plan` = PLAN when the move was sent; `repeat` = same state and action as an earlier step.", "",
            "| # | request | turn | call | action | plan | what happened | repeat |",
            "|---|---|---|---|---|---|---|---|"]
    for r in a["actions"]:
        act = "RESET" if r["action"] == 0 else f"{r['action']}" + (f"({r['x']},{r['y']})" if r.get("x") is not None else "")
        out.append(f"| {r['i']} | {r.get('request')} | {r.get('turn')} | {str(r.get('call') or '')[-8:]} | {act} | "
                   f"{_md(r.get('plan'), 90)} | {_md(r.get('happened'), 110)}{' LEVEL UP' if r.get('level_up') else ''} | "
                   f"{'' if r.get('repeat_of') is None else '#' + str(r['repeat_of'])} |")
    planned = sum(1 for r in a["actions"] if r.get("plan"))
    out += ["", f"Actions with a plan: {planned}/{len(a['actions'])}. Repeated pairs: "
                f"{a['repeats'] if a['repeats'] is not None else 'n/a (no step events)'}.", ""]
    # 2
    out += ["## 2. Understanding over time", "", "### Hypotheses and findings", "",
            "| id | kind | first at action | level | versions | removed by agent | latest text |",
            "|---|---|---|---|---|---|---|"]
    for b in a["beliefs"]:
        out.append(f"| {b['id']} | {b['kind']} | {b['first_action']} (request {b['first_turn']}) | {b['first_level']} | "
                   f"{b['versions']} | {'yes' if b.get('removed_by_agent') else ''} | {_md(b.get('content'), 120)} |")
    if not a["beliefs"]:
        out.append("| (no memory writes) | | | | | | |")
    out += ["", "### Goal statements (mark `correct` by hand after the run)", "",
            "| request | at action | level | goal | correct? |", "|---|---|---|---|---|"]
    for g in a["goals"]:
        out.append(f"| {g['turn']} | {g['action_count']} | {g['level']} | {_md(g['goal'], 140)} | |")
    if not a["goals"]:
        out.append("| (no goal stated) | | | | |")
    # 3
    lv = a["living_code"]
    gr = lv["graph"]
    out += ["", "## 3. Living code", "",
            f"{lv['cells']} cells, {lv['helpers_defined']} helpers defined, {lv['helpers_reused']} reused later, "
            f"living-code ratio {lv['living_code_ratio']}, {lv['printed_boards']} cells printed a board, "
            f"{lv['cell_errors']} cell errors.",
            f"Code graph: {gr['deltas']} change lines, {gr['nodes_last']} functions and classes at the last change, "
            f"removed {', '.join(gr['removed']) or 'none'}, {gr['missing_warnings']} missing-call warnings, "
            f"{gr['calls_to_removed']} cells called a function after the graph reported it removed"
            + (": " + "; ".join(f"request {s['turn']} {', '.join(s['names'])}" for s in gr["calls_to_removed_at"])
               if gr["calls_to_removed_at"] else "") + ".", "",
            "| cell | request | defines | reuses | variables | printed board | error | moves |",
            "|---|---|---|---|---|---|---|---|"]
    for cl in a["cells"]:
        out.append(f"| {cl['n']} | {cl['request']} | {', '.join(cl['defines'])} | {', '.join(cl['reuses'])} | "
                   f"{_md(', '.join(cl['variables']), 60)} | {'yes' if cl['printed_board'] else ''} | "
                   f"{cl['error'] or ''} | {cl['actions'] or ''} |")
    # 4
    out += ["", "## 4. Waste per level", "",
            "| level | actions | human | cleared | rhae | resets | largest spend | repeated pairs | "
            "after last new belief |", "|---|---|---|---|---|---|---|---|---|"]
    for w in a["waste"]:
        out.append(f"| {w['level']} | {w['actions']} | {w['baseline']} | {'yes' if w['cleared'] else ''} | {w['rhae']} | "
                   f"{len(w['resets'])} | {w['largest_spend']} | "
                   f"{'n/a' if w['repeated_pairs'] is None else w['repeated_pairs']} | "
                   f"{w['actions_after_last_new_belief'] if w['actions_after_last_new_belief'] is not None else ''} |")
    # 5
    out += ["", "## 5. Run context", "",
            f"- experiment `{c.get('experiment')}`, config hash `{c.get('config_hash')}`, git `{c.get('git_sha')}`, "
            f"model `{c.get('model')}`",
            f"- end reason {c.get('end_reason')}, wall {c.get('wall_s')} s, output tokens {c.get('output_tokens')}, "
            f"all tokens {c.get('tokens_total')}, last prompt {c.get('prompt_tokens_last')}",
            f"- {c.get('requests')} requests in {c.get('turns')} turns ({c.get('requests_per_turn')} per turn, at most "
            f"{c.get('requests_per_turn_max')}); turns ended by {json.dumps(c.get('turn_ends'))}",
            f"- tool calls {json.dumps(c.get('tool_counts'))}, tool errors {c.get('tool_errors')}, cell errors "
            f"{c.get('cell_errors')}, play() calls {c.get('play_calls')} ({c.get('moves_per_play')} moves each), "
            f"play() argument errors {c.get('act_arg_errors')}, surprises {c.get('surprises')}",
            f"- context: {len(c.get('drains') or [])} drains, {c.get('trims')} trims, {c.get('resets')} resets, "
            f"{c.get('overflows')} overflows; {c.get('tail_pictures')} pictures at the tail, {c.get('images_sent')} "
            f"pictures drawn, largest observation {c.get('observation_chars_max')} chars",
            f"- memory: {c.get('hypotheses')} hypotheses, levels without one {c.get('levels_without_hypothesis')}, "
            f"promotions {c.get('promotions')}, recalls {c.get('recalls')} {json.dumps(c.get('recall_scopes'))}, "
            f"skills recalled {c.get('skills_recalled')}",
            f"- curator: passes {json.dumps(c.get('curator_runs'))} (errors {c.get('curator_errors')}), "
            f"{c.get('curator_s')} s in all, skills written {c.get('skills_written')}, loaded "
            f"{c.get('skills_loaded_by_curator')}",
            f"- notes: stuck {c.get('stuck_notes')}, cut off {c.get('cut_off_notes')}; repeated (state, action) share "
            f"{c.get('repeated_share')}"]
    for d in c.get("drains") or []:
        out.append(f"  - drain +{d['t']}s ({d['reason']}) at action {d['action_count']}: {d['tokens_before']} -> "
                   f"{d['tokens_after_est']} tokens")
    return "\n".join(out) + "\n"


_MAP_EVENTS = ("hypothesis", "finding", "goal", "memory_removed", "memory_error", "recall", "code_graph", "level_up",
               "curator", "memory_init", "tool_error", "surprise", "stuck_note", "cut_off_note")
_MAP_MARKS = ("drain", "context_trim", "context_reset", "context_overflow", "crash", "llm_error", "end")


def game_map(game_dir: Path, rows: list[dict[str, Any]], run: dict[str, Any], session: dict[str, Any]) -> str:
    by_call: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if not r["after_cell"] and r.get("call"):
            by_call[r["call"]].append(r)
    shown: set[int] = set()
    game_id = run.get("game_id", game_dir.name)
    out = [f"# {game_id}: step map", "",
           f"level {run.get('levels_completed')}/{run.get('number_of_levels')} | {len(rows)} actions | "
           f"score {run.get('final_score')} | end {session.get('end_reason')} | requests {session.get('turns')}",
           f"actions per level {run.get('actions_per_level')} vs baseline {run.get('baseline_actions')}", ""]
    events = _jsonl(game_dir / "transcript.jsonl") if (game_dir / "transcript.jsonl").exists() else []
    t0 = events[0]["t"] if events else 0.0
    turn_of_req = outer_turns(events)
    levels, turn = 0, 1
    for ev in events:
        kind, t = ev.get("event"), f"+{ev.get('t', t0) - t0:.0f}s"
        if kind == "message" and ev.get("role") == "assistant":
            usage = ev.get("usage") or ["?", "?"]
            out += ["", f"## Request {ev.get('turn', '?')} of turn {turn_of_req.get(ev.get('turn'), '?')} ({t}, prompt "
                        f"{usage[0]} / out {usage[1]} tokens{', CUT OFF' if ev.get('finish') == 'length' else ''})"]
            if ev.get("reasoning"):
                out += [f"**Thought** ({len(ev['reasoning'])} chars):", "", "> " + _cut(ev["reasoning"], 1600)
                        .replace("\n", "\n> "), ""]
            if (ev.get("content") or "").strip():
                out += ["**Said:** " + _cut(ev["content"], 1200), ""]
            for cid, name, args in _tool_calls(ev):
                if name == "ipython":
                    out += [f"**Code** (`{cid}`):", "````python", _cut(args.get("code") or args, 4000), "````"]
                else:
                    out += [f"**Tool `{name}`** (`{cid}`): `{_one(json.dumps(args, ensure_ascii=False), 600)}`"]
        elif kind == "message" and ev.get("role") == "tool":
            out += ["**Output:**", "````", _cut(ev.get("content"), 1500), "````"]
        elif kind == "message" and ev.get("role") == "user":
            first = str(ev.get("content") or "").strip().splitlines()[:1] if isinstance(ev.get("content"), str) else \
                ["(picture)"]
            out += ["", f"> **{ev.get('_kind') or 'host'} -> agent** ({t}): {_cut(first[0] if first else '', 300)}"]
        elif kind == "cell":
            acts = by_call.get(str(ev.get("call")), [])
            shown.update(r["i"] for r in acts)
            status = "" if ev.get("status") == "ok" else f" (cell {ev.get('status')})"
            if acts:
                out += [f"**Actions** ({len(acts)}){status}: " + render_actions(acts, levels), ""]
                levels = acts[-1].get("levels") if acts[-1].get("levels") is not None else levels
            elif status:
                out += [f"**No actions**{status}", ""]
        elif kind == "turn_end":
            out += ["", f"--- turn {turn} ended: {ev.get('reason')} after {ev.get('requests')} requests, "
                        f"{ev.get('tokens')} tokens since the last move ---"]
            turn += 1
        elif kind in _MAP_EVENTS:
            detail = {k: v for k, v in ev.items() if k not in ("t", "event", "steps")}
            out += [f"- **[{kind}]** {t} " + _cut(json.dumps(detail, default=str, ensure_ascii=False), 500)]
        elif kind in _MAP_MARKS:
            detail = {k: v for k, v in ev.items() if k not in ("t", "event", "stats", "traceback")}
            out += ["", f"- **[{kind}]** {t} " + _cut(json.dumps(detail, default=str), 600)]
            if kind == "crash":
                out += ["````", _cut(ev.get("traceback"), 3000), "````"]
    rest = [r for r in rows if r["i"] not in shown]
    if rest:
        out += ["", "## Actions not matched to a call (no transcript)", "", render_actions(rest, None)]
    return "\n".join(out) + "\n"


# --- SFT export --------------------------------------------------------------------------------------------
_EVENT_KEYS = {"t", "event", "reasoning", "usage", "finish", "turn", "wire_sha", "effort", "reasoning_unsent"}
BAD_FLAGS = ("cut_off", "tool_error", "cell_error", "repeat", "game_over")


def _stored(m: dict[str, Any], prefix: str) -> dict[str, Any]:
    """A logged message as the agent stored it, with its picture's file (``_image_file``) in the image part."""
    m = {k: v for k, v in m.items() if k not in _EVENT_KEYS}
    if m.get("_image_file") and isinstance(m.get("content"), list):
        url = f"{prefix}/{m['_image_file']}"
        m["content"] = [{"type": "image_url", "image_url": {"url": url}} if p.get("type") == "image_url" else p
                        for p in m["content"]]
    return m


def _turn_flags(events: list[dict[str, Any]]) -> dict[Any, set[str]]:
    """Per request, what went wrong in it: cut off, a refused tool call, a failed cell, a repeated (state, action)
    pair, a GAME_OVER, or no tool call at all."""
    flags: dict[Any, set[str]] = defaultdict(set)
    for ev in events:
        kind, turn = ev.get("event"), ev.get("turn")
        if kind == "message" and ev.get("role") == "assistant":
            if ev.get("finish") == "length":
                flags[turn].add("cut_off")
            if not ev.get("tool_calls") and not _FENCED.search(ev.get("content") or ""):
                flags[turn].add("no_tool_call")
        elif kind == "tool_error":
            flags[turn].add("tool_error")
        elif kind == "cell" and ev.get("status") not in (None, "ok"):
            flags[turn].add("cell_error")
        elif kind == "step" and turn is not None:
            if ev.get("repeat_of") is not None:
                flags[turn].add("repeat")
            if ev.get("state") == "GAME_OVER":
                flags[turn].add("game_over")
    return flags


def _as_sent(m: dict[str, Any]) -> dict[str, Any]:
    """A stored reply as the next append-only request carries it."""
    return agent_context.merge_users(agent_context.all_reasoning([m]))[0]


def sft_wire(events: list[dict[str, Any]], run: dict[str, Any], context: dict[str, Any],
             prefix: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rows of exact requests (see the module docstring) and their counts. ``prefix`` is the game directory relative to
    the run directory (where its frames are)."""
    system_ev = next((e for e in events if e.get("event") == "system_prompt"), {})
    system = system_ev.get("content") or ""
    vision = bool(system_ev.get("vision"))
    append_only = system_ev.get("append_only", True)
    old_images = system_ev.get("old_images", True)
    budget = int(system_ev.get("reasoning_budget") or agent_context.REASONING_BUDGET_TOKENS)
    flags = _turn_flags(events)
    base = run.get("baseline_actions") or []
    per_level = run.get("actions_per_level") or []
    cleared = run.get("levels_completed") or 0
    n_levels = run.get("number_of_levels") or 10 ** 6
    rows: list[dict[str, Any]] = []
    counts = Counter()

    def emit(g: dict[str, Any]) -> None:
        lvl = g["level"]
        n = per_level[lvl] if lvl < len(per_level) else None
        ok = lvl < cleared
        counts["turns"] += len(g["turns"])
        if lvl >= n_levels or (not ok and not n):
            counts["turns_dropped"] += len(g["turns"])
            return   # after the last level, or a level with no action spent: nothing to learn from
        msgs = [({k: v for k, v in m.items() if k != "reasoning"} if m["role"] == "assistant" else m)
                for m in g["history"]] + [g["turns"][-1]["target"]]
        msgs = [({**m, "train": False} if m["role"] == "assistant" else m) for m in msgs]
        for t in g["turns"]:
            bad = sorted(t["flags"] & set(BAD_FLAGS))
            msgs[t["pos"]] = {**t["target"], "turn": t["turn"], "flags": sorted(t["flags"]), "train": not bad}
            counts["trainable" if not bad else "flagged"] += 1
        for t in reversed(g["turns"]):   # each reply follows the picture it saw (newest first: positions hold)
            if t["last"] is None:
                continue
            if t["merged"]:   # the request joined the picture to its last user message
                msgs[t["pos"] - 1] = t["last"]
            else:
                msgs.insert(t["pos"], t["last"])
        if not any(m.get("train") for m in msgs if m["role"] == "assistant"):
            counts["rows_without_target"] += 1
            return
        sys_msg = msgs[0] if msgs and msgs[0]["role"] == "system" else None
        rows.append({
            "game_id": run.get("game_id"), "level": lvl, "level_cleared": ok,
            "rhae": round(min((base[lvl] / n) ** 2, 1.15), 4) if ok and n and lvl < len(base) else 0.0,
            "actions": n, "baseline": base[lvl] if lvl < len(base) else None,
            "segment_start": g["start"], "turns": [t["turn"] for t in g["turns"]],
            "template_kwargs": {"reasoning_effort": g["effort"]} if g["effort"] else {},
            "wire_verified": all(t["verified"] for t in g["turns"]),
            "system": sys_msg["content"] if sys_msg else system, "tools": system_ev.get("tools"),
            "messages": msgs[1:] if sys_msg else msgs, "source": "wire",
            **{k: context.get(k) for k in ("experiment", "config_hash", "git_sha", "model")}})

    state: list[dict[str, Any]] = []
    pinned, tail, start, level, group = None, None, "start", 0, None
    for ev in events:
        kind = ev.get("event")
        if kind == "context":
            state, start = [_stored(m, prefix) for m in ev.get("messages") or []], ev.get("reason") or "rewrite"
        elif kind == "pinned":
            pinned = _stored(ev, prefix)
        elif kind == "tail":   # the picture sent last in the next requests
            tail = _stored(ev, prefix)
        elif kind == "step":
            level = max(level, ev.get("level_after") or 0)
        elif kind == "message":
            m = _stored(ev, prefix)
            if ev.get("role") == "assistant":
                if ev.get("reasoning") and not ev.get("reasoning_unsent"):
                    m["_reasoning"] = ev["reasoning"]
                kw = dict(pinned=None if append_only else pinned, vision=vision, reasoning_budget=budget,
                          append_only=append_only, old_images=old_images)
                request = agent_context.build(system, state, tail=tail, **kw)
                history = agent_context.build(system, state, **kw) if tail is not None else request
                verified = agent_context.digest(request) == ev.get("wire_sha")
                counts["verified"] += verified
                target = {k: v for k, v in _as_sent(m).items() if k != "reasoning"}
                if ev.get("reasoning"):
                    target["reasoning_content"] = ev["reasoning"]   # what it generated, sent back or not
                pictured = tail is not None and request != history
                turn = {"turn": ev.get("turn"), "pos": len(history), "target": target, "verified": verified,
                        "flags": flags.get(ev.get("turn"), set()), "last": request[-1] if pictured else None,
                        "merged": pictured and len(request) == len(history)}
                g = group
                if g is not None and g["effort"] == ev.get("effort") and g["level"] == level \
                        and len(history) > len(g["history"]) and history[:len(g["history"])] == g["history"] \
                        and history[len(g["history"])] == g["reply"]:
                    g["turns"].append(turn)
                    g["history"], g["reply"] = history, _as_sent(m)
                else:
                    if g is not None:
                        emit(g)
                    group = {"turns": [turn], "history": history, "reply": _as_sent(m), "effort": ev.get("effort"),
                             "level": level, "start": start}
                    start = "continued"   # a row that breaks without a rewrite (the old builder, a new effort)
            state.append(m)
    if group is not None:
        emit(group)
    counts["rows"] = len(rows)
    counts["rows_cleared"] = sum(1 for r in rows if r["level_cleared"])
    return rows, dict(counts)


def build_run(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    runs = {r["game_id"]: r for r in results.get("runs", [])}
    sessions = results.get("sessions", {})
    summ = results.get("summary") or {}
    cfg = results.get("config") or {}
    context = {"experiment": summ.get("experiment") or cfg.get("experiment"), "config_hash": summ.get("config_hash"),
               "git_sha": summ.get("git_sha"), "model": summ.get("model") or (cfg.get("llm") or {}).get("model")}
    table = ["# Trace index", "",
             f"experiment `{context['experiment']}` | config `{context['config_hash']}` | git `{context['git_sha']}` | "
             f"model `{context['model']}`", "",
             "| game | levels | actions | requests | turns | play calls | resets | GAME_OVERs | repeats | beliefs | "
             "living code | calls to removed | drains | curator s | end | report | map |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    sft: list[dict[str, Any]] = []
    sft_counts: Counter = Counter()
    unexported = 0
    for game_id, run in sorted(runs.items()):
        game_dir = run_dir / "games" / game_id
        session = sessions.get(game_id, {})
        rec_name = Path(session["recording"]).name if session.get("recording") else None
        hits = sorted((run_dir / "recordings").rglob(rec_name)) if rec_name else []
        recording = hits[0] if hits else None
        rows = action_rows(recording, run)
        game_dir.mkdir(parents=True, exist_ok=True)
        with (game_dir / "steps.jsonl").open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        (game_dir / "steps.md").write_text(game_map(game_dir, rows, run, session), encoding="utf-8")
        events = _jsonl(game_dir / "transcript.jsonl") if (game_dir / "transcript.jsonl").exists() else []
        a = analyze_game(events, rows, run, session, context)
        (game_dir / "trace.json").write_text(json.dumps(a, indent=1, default=str), encoding="utf-8")
        (game_dir / "report.md").write_text(render_report(a), encoding="utf-8")
        if any(e.get("wire_sha") for e in events):
            got, counts = sft_wire(events, run, context, f"games/{game_id}")
            sft += got
            sft_counts.update(counts)
        else:
            unexported += 1
        acts, c = a["actions"], a["context"]
        table.append(
            f"| {game_id} | {run.get('levels_completed')}/{run.get('number_of_levels')} | {run.get('actions')} | "
            f"{c.get('requests')} | {c.get('turns')} | {c.get('play_calls')} | "
            f"{sum(1 for r in acts if r['action'] in (0, 'RESET', 'reset'))} | "
            f"{sum(1 for r in acts if r.get('state') == 'GAME_OVER')} | "
            f"{sum(1 for r in acts if r.get('repeat_of') is not None)} | {len(a['beliefs'])} | "
            f"{a['living_code']['living_code_ratio']} | {a['living_code']['graph']['calls_to_removed']} | "
            f"{len(c.get('drains') or [])} | {c.get('curator_s')} | {session.get('end_reason')} | "
            f"[report](games/{game_id}/report.md) | [map](games/{game_id}/steps.md) |")
    with (run_dir / "sft_levels.jsonl").open("w", encoding="utf-8") as fh:
        for r in sft:
            fh.write(json.dumps(r, default=str, ensure_ascii=False) + "\n")
    c = sft_counts
    frames = list((run_dir / "games").glob("*/frames/*.png"))
    table += ["", (f"SFT export: `sft_levels.jsonl`, {len(sft)} rows, {sum(1 for r in sft if r['level_cleared'])} on "
                   f"cleared levels.")]
    if c:
        table.append(f"From the wire: {c['turns']} replies, {c['verified']} rebuilt exactly "
                     f"({100 * c['verified'] / max(1, c['turns']):.1f}%), {c['trainable']} trained, {c['flagged']} "
                     f"flagged out (train false), {c['turns_dropped']} after the last level or on levels with no "
                     f"action, {c['rows_without_target']} rows dropped with nothing to train; {len(frames)} pictures "
                     f"({sum(f.stat().st_size for f in frames) / 1e6:.1f} MB).")
    if unexported:
        table.append(f"{unexported} games without wire fields (tracing off): not in the SFT export.")
    out = run_dir / "trace.md"
    out.write_text("\n".join(table) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    print(build_run(Path(sys.argv[1] if len(sys.argv) > 1 else ".")))

"""Trace of a Prime run: for every game, what the agent thought, ran, believed and spent, and a training export.

    uv run python Sarbloh/harness/trace.py <run_dir>     # e.g. runs/kaggle_.../prime_run; run.py calls build_run

Sources in <run_dir>: ``results.json`` and, per game, ``games/<game_id>/transcript.jsonl`` (every message of the
root session with the model's reasoning, every cell, and one structured event per step, act, memory write,
recall and curator pass; older transcripts also have reset, plan, delegation, message and world-model events), plus the ARC SDK recording
``recordings/<scorecard>/<game_id>-<guid>.jsonl`` (one line per environment step). Without a recording the actions
come from ``run.history``.

Writes, per game (next to the transcript):
- ``steps.jsonl`` / ``steps.md``: one row per action / the turn-by-turn map (thought, code, tool calls, actions).
- ``report.md`` / ``trace.json``: five sections, as markdown and as data.
  1. Actions and causes: per action the turn, call, tool, action, expect, what changed, whether the prediction held,
     and whether the (state, action) pair was already tried.
  2. Understanding over time: every memory write and refinement with its evidence, when each belief first appeared,
     whether it was later refuted, and the plan's goal statements (the "goal first stated correctly" column is left
     for a human to mark).
  3. Living code: per ipython cell the functions defined, the earlier helpers reused, full-grid prints and errors.
     living_code_ratio = cells that call a helper defined in an earlier cell / cells after the first helper.
  4. Waste per level: actions vs the human baseline, resets with reasons, the largest single spend, repeated
     (state, action) pairs, and actions spent after the last new belief (stagnation).
  5. Run context: config hash, git SHA, model, compactions, tokens, wall clock, end reason.
Writes, per run: ``trace.md`` (one row per game) and ``sft_levels.jsonl``: one row per (game, level) with the
level's message sequence in chat format, tagged ``level_cleared`` and ``rhae``, so "successful trajectories for LoRA"
is ``[r for r in rows if r["level_cleared"]]``. A level's messages are those produced while that level was being
played; rows for level > 0 start mid-conversation (``starts_mid_conversation``) and carry the task message first.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


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


# --- per-game analysis -----------------------------------------------------------------------------------
_FENCED = re.compile(r"```(?:python|py|ipython|repl)?[ \t]*\n(.*?)```", re.DOTALL)   # = agent._FENCED
_GRID_LINE = re.compile(r"^\s*\d{1,2} [0-9a-f]{16,}\s*$")


def _full_grid_print(code: str, output: str) -> bool:
    """A cell that dumped a (near) whole grid into the context: 30+ lines of hex rows in its output, or an
    ``arc.show(x)`` call without a window."""
    if sum(1 for ln in (output or "").splitlines() if _GRID_LINE.match(ln)) >= 30:
        return True
    return bool(re.search(r"arc\.show\(\s*[^,()]+(\([^()]*\))?\s*\)", code or ""))


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


def analyze_game(events: list[dict[str, Any]], rows: list[dict[str, Any]], run: dict[str, Any],
                 session: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    t0 = events[0]["t"] if events else 0.0
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
    steps = [e for e in events if e.get("event") == "step"]
    acts = [e for e in events if e.get("event") == "act"]
    resets = [e for e in events if e.get("event") == "reset"]
    remembers = [e for e in events if e.get("event") == "remember"]
    plans = [e for e in events if e.get("event") == "plan"]
    wm_events = [e for e in events if e.get("event") == "wm"]
    hyps = [e for e in events if e.get("event") == "hypothesis"]
    goal_events = [e for e in events if e.get("event") == "goal"]
    level_ups = [e for e in events if e.get("event") == "level_up"]
    curations = [e for e in events if e.get("event") == "curator"]
    recalls = [e for e in events if e.get("event") == "recall"]

    # 1. actions and their causes ------------------------------------------------------------------------
    act_by_call = {str(a.get("call")): a for a in acts}
    next_verdict: dict[str, Any] = {}
    for a, b in zip(acts, acts[1:]):
        next_verdict[str(a.get("call"))] = b.get("was_right")
    reset_by_i = {r.get("i"): r for r in resets}
    actions: list[dict[str, Any]] = []
    if steps:
        wm_of: dict[int, Any] = {}
        for a in acts:
            for s in a.get("steps") or []:
                wm_of[s.get("i")] = (s.get("wm") or {}).get("match")
        for s in steps:
            call = str(s.get("call"))
            act = act_by_call.get(call)
            tool = "act" if act is not None else "reset_level" if s.get("source") == "tool" else "ipython"
            actions.append({
                "i": s.get("i"), "turn": s.get("turn"), "call": s.get("call"), "tool": tool,
                "action": s.get("action"), "x": s.get("x"), "y": s.get("y"), "level": s.get("level"),
                "expect": (act.get("expect") or act.get("plan")) if act else
                (reset_by_i.get(s.get("i")) or {}).get("reason"),
                "happened": s.get("change"), "changed_px": s.get("changed"), "state": s.get("state"),
                "level_up": s.get("level_up"), "wm_match": wm_of.get(s.get("i")),
                "self_verdict": next_verdict.get(call) if act else None, "repeat_of": s.get("repeat_of"),
                "t": round(s.get("t", t0) - t0, 1)})
    else:  # old transcripts without step events: the recording has the action, the call and the level
        prev_level = 0
        for r in rows:
            actions.append({"i": r["i"], "turn": r.get("turn"), "call": r.get("call"), "tool": "ipython",
                            "action": _ACTION_ID.get(str(r["action"]), r["action"]), "x": r.get("x"), "y": r.get("y"),
                            "level": prev_level, "expect": None, "happened": None, "changed_px": None,
                            "state": r.get("state"), "level_up": None, "wm_match": None, "self_verdict": None,
                            "repeat_of": None, "t": None})
            prev_level = r["levels"] if r.get("levels") is not None else prev_level

    # 2. understanding over time -------------------------------------------------------------------------
    beliefs: dict[str, dict[str, Any]] = {}
    for m in remembers:
        b = beliefs.setdefault(m["id"], {"id": m["id"], "title": m.get("title"), "first_turn": m.get("turn"),
                                         "first_action": m.get("action_count"), "first_level": m.get("level"),
                                         "kinds": [], "versions": 0, "refuted_at": None, "evidence": []})
        b["kinds"].append(m.get("kind"))
        b["versions"] = m.get("version")
        b["evidence"] = sorted(set(b["evidence"]) | set(m.get("evidence") or []))
        b["content"] = m.get("content")
        if m.get("kind") == "refuted" and b["refuted_at"] is None and len(b["kinds"]) > 1:
            b["refuted_at"] = m.get("action_count")
    for cid, name in tool_of_call.items():
        code = str(args_of_call.get(cid, {}).get("code") or "") if name == "ipython" else ""
        for m in re.finditer(r"rlm\.harness\.(create|update)_memory\((.{0,300})", code, re.S):
            title = re.search(r"title\s*=\s*(['\"])(.*?)\1", m.group(2))
            bid = f"repl:{title.group(2)[:60] if title else cid}"
            b = beliefs.setdefault(bid, {"id": bid, "title": title.group(2) if title else None,
                                         "first_turn": turn_of_call.get(cid), "first_action": None, "first_level": None,
                                         "kinds": [], "versions": 0, "refuted_at": None, "evidence": [],
                                         "content": "(written in ipython: no evidence field)"})
            b["kinds"].append(f"repl-{m.group(1)}")
            b["versions"] += 1
    for e in events:
        if e.get("event") == "auto_refine" and e.get("applied"):
            bid = f"refine:{e.get('t')}"
            beliefs[bid] = {"id": bid, "title": None, "first_turn": e.get("turn"), "first_action": e.get("action_count"),
                            "first_level": None, "kinds": ["refine"], "versions": 1, "refuted_at": None,
                            "evidence": [], "content": _one(json.dumps(e.get("applied"), default=str), 300)}
    for h in hyps:   # the agent's hypotheses, with the statuses it set
        bid = f"{h.get('level')}:{h.get('id')}"
        b = beliefs.setdefault(bid, {"id": bid, "title": None, "first_turn": h.get("turn"),
                                     "first_action": h.get("action_count"), "first_level": h.get("level"),
                                     "kinds": [], "versions": 0, "refuted_at": None, "evidence": []})
        if not b["kinds"] or b["kinds"][-1] != h.get("status"):
            b["kinds"].append(h.get("status"))
        b["versions"] += 1
        b["evidence"] = sorted(set(b["evidence"]) | set(h.get("evidence") or []))
        b["content"] = h.get("text")
        if h.get("status") == "refuted" and b["refuted_at"] is None:
            b["refuted_at"] = h.get("action_count")
    for e in events:
        if e.get("event") == "finding":
            bid = f"finding:{e.get('t')}"
            beliefs[bid] = {"id": bid, "title": None, "first_turn": e.get("turn"), "first_action": e.get("action_count"),
                            "first_level": e.get("level"), "kinds": ["finding"], "versions": 1, "refuted_at": None,
                            "evidence": e.get("evidence") or [], "content": e.get("text")}
    goals = []
    for g in goal_events:
        goals.append({"version": f"goal (turn {g.get('turn')})", "turn": g.get("turn"),
                      "action_count": g.get("action_count"), "level": g.get("level"), "phase": "act",
                      "goal": g.get("goal"), "correct": None})
    for p in plans:
        if not goals or goals[-1]["goal"] != p.get("goal"):
            goals.append({"version": p.get("version"), "turn": p.get("turn"), "action_count": p.get("action_count"),
                          "level": p.get("level"), "phase": p.get("phase"), "goal": p.get("goal"),
                          "correct": None})  # marked by hand after the run
    for m in remembers:
        if m.get("kind") == "goal":
            goals.append({"version": f"memory {m['id']} v{m.get('version')}", "turn": m.get("turn"),
                          "action_count": m.get("action_count"), "level": m.get("level"), "phase": "memory",
                          "goal": m.get("content"), "correct": None})
    goals.sort(key=lambda g: (g["action_count"] or 0, g["turn"] or 0))

    # 3. living code -------------------------------------------------------------------------------------
    cells = []
    defined_at: dict[str, int] = {}
    cell_status = {str(e.get("call")): e for e in events if e.get("event") == "cell"}
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
        cells.append({"n": n, "turn": turn_of_call.get(cid), "call": cid, "defines": sorted(defs),
                      "reuses": reused, "variables": sorted(assigned - defs)[:20],
                      "full_grid_print": _full_grid_print(code, output_of_call.get(cid, "")),
                      "status": ce.get("status") or ("syntax" if syntax else None),
                      "error": syntax or ("error" if ce.get("status") not in (None, "ok") else None),
                      "actions": ce.get("actions"), "output_chars": len(output_of_call.get(cid, "")),
                      "code_lines": code.count("\n") + 1 if code else 0})
    first_def = min(defined_at.values()) if defined_at else None
    after = [c for c in cells if first_def is not None and c["n"] > first_def]
    reused_names = {nm for c in cells for nm in c["reuses"]}
    living = {"cells": len(cells), "helpers_defined": len(defined_at), "helpers_reused": len(reused_names),
              "living_code_ratio": round(sum(1 for c in after if c["reuses"]) / len(after), 3) if after else 0.0,
              "full_grid_prints": sum(1 for c in cells if c["full_grid_print"]),
              "cell_errors": sum(1 for c in cells if c["error"])}

    # 4. waste per level -----------------------------------------------------------------------------------
    per_level = run.get("actions_per_level") or []
    base = run.get("baseline_actions") or []
    cleared = run.get("levels_completed") or 0
    spend_by_call = Counter(str(a["call"]) for a in actions if a.get("call"))
    belief_marks: dict[int, list[int]] = defaultdict(list)   # level -> action counts of new beliefs
    for m in remembers:
        if m.get("created") or (m.get("prev_kind") and m.get("prev_kind") != m.get("kind")):
            belief_marks[m.get("level") or 0].append(m.get("action_count") or 0)
    for p in plans:
        if p.get("goal_changed") or p.get("version") == 1:
            belief_marks[p.get("level") or 0].append(p.get("action_count") or 0)
    for h in hyps:
        if h.get("prev_status") != h.get("status"):
            belief_marks[h.get("level") or 0].append(h.get("action_count") or 0)
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
            "resets": [{"i": a["i"], "reason": a["expect"]} for a in mine if a["action"] == 0],
            "largest_spend": max(calls.values(), default=0),
            "repeated_pairs": sum(1 for a in mine if a.get("repeat_of") is not None) if steps else None,
            "actions_after_last_new_belief": stagnation if last_belief is not None else (len(mine) if steps else None),
            "first_i": min(idx, default=None), "last_i": max(idx, default=None)})

    # 5. run context ---------------------------------------------------------------------------------------
    comps = [{"turn": None, "t": round(e.get("t", t0) - t0, 1), "reason": e.get("reason"),
              "tokens_before": e.get("tokens_before"), "summarized_messages": e.get("summarized_messages"),
              "kept_messages": e.get("kept_messages"), "summary_chars": e.get("summary_chars"),
              "action_count": e.get("action_count")} for e in events if e.get("event") == "compaction"]
    end = next((e for e in reversed(events) if e.get("event") == "end"), {})
    stats = end.get("stats") or {}
    tool_counts = stats.get("tool_counts") or dict(Counter(tool_of_call.values()))
    ctx = {**context, "end_reason": session.get("end_reason") or end.get("reason"),
           "wall_s": session.get("wall_s"), "tokens_total": session.get("tokens_total"),
           "output_tokens": stats.get("output_tokens"), "prompt_tokens_last": stats.get("prompt_tokens_last"),
           "turns": stats.get("turns"), "compactions": comps, "tool_counts": tool_counts,
           "tool_errors": stats.get("tool_errors"), "act_calls": len(acts),
           "act_batch_mean": round(sum(a.get("done", 0) for a in acts) / len(acts), 2) if acts else None,
           "wm_events": [{k: v for k, v in e.items() if k not in ("event",)} for e in wm_events],
           "wm_checks": stats.get("wm_checks"), "wm_mispredictions": stats.get("wm_mispredictions"),
           "plan_versions": len(plans),
           # memory and curator
           "act_arg_errors": stats.get("act_arg_errors"), "recalls": len(recalls),
           "recall_scopes": dict(Counter(str(r.get("scope")) for r in recalls)),
           "hypotheses": len({(h.get("level"), h.get("id")) for h in hyps}),
           "hypothesis_statuses": dict(Counter(h.get("status") for h in hyps)),
           "levels_without_hypothesis": sorted({a["level"] for a in actions if a.get("level") is not None}
                                               - {h.get("level") for h in hyps}) if hyps or acts else None,
           "promotions": sum(len(e.get("promoted") or []) for e in level_ups),
           "curator_runs": len([c for c in curations if not c.get("error")]),
           "curator_errors": len([c for c in curations if c.get("error")]),
           "skills_written": sorted({s for c in curations for s in c.get("skills") or []}),
           "images_sent": stats.get("images_sent"), "observation_chars_max": stats.get("observation_chars_max"),
           "pinned_chars_max": stats.get("pinned_chars_max"),
           "repeated_share": round(sum(1 for r in actions if r.get("repeat_of") is not None) / len(actions), 3)
           if actions and steps else None}
    return {"game_id": run.get("game_id"), "levels_completed": cleared, "number_of_levels": run.get("number_of_levels"),
            "score": run.get("final_score"), "actions": actions,
            "repeats": sum(1 for r in actions if r.get("repeat_of") is not None) if steps else None, "beliefs": list(beliefs.values()), "goals": goals,
            "memory_events": remembers, "plans": plans, "cells": cells, "living_code": living, "waste": waste,
            "context": ctx}


def render_report(a: dict[str, Any]) -> str:
    c = a["context"]
    out = [f"# {a['game_id']}: trace report", "",
           f"level {a['levels_completed']}/{a['number_of_levels']} | score {a['score']} | {len(a['actions'])} actions "
           f"| end {c.get('end_reason')}", ""]
    # 1
    out += ["## 1. Actions and their causes", "",
            "`expect / plan` = the plan written with the act (a prediction in old transcripts); `wm` / `self` = "
            "old transcripts only; `repeat` = "
            "same state and action as an earlier step.", "",
            "| # | turn | call | tool | action | expect / plan | what happened | wm | self | repeat |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for r in a["actions"]:
        act = "RESET" if r["action"] == 0 else f"A{r['action']}" + (f"({r['x']},{r['y']})" if r.get("x") is not None else "")
        wm = {True: "ok", False: "WRONG", None: ""}.get(r.get("wm_match"), "")
        out.append(f"| {r['i']} | {r.get('turn')} | {str(r.get('call') or '')[-8:]} | {r['tool']} | {act} | {_md(r.get('expect'), 90)} | "
                   f"{_md(r.get('happened'), 110)}{' LEVEL UP' if r.get('level_up') else ''} | {wm} | "
                   f"{r.get('self_verdict') or ''} | {'' if r.get('repeat_of') is None else '#' + str(r['repeat_of'])} |")
    n_act = sum(1 for r in a["actions"] if r["tool"] == "act")
    stated = sum(1 for r in a["actions"] if r.get("expect"))
    out += ["", f"Actions with a stated cause: {stated}/{len(a['actions'])}. Through act: {n_act}. "
                f"World-model checked: {sum(1 for r in a['actions'] if r.get('wm_match') is not None)} "
                f"(wrong: {sum(1 for r in a['actions'] if r.get('wm_match') is False)}). "
                f"Repeated pairs: {a['repeats'] if a['repeats'] is not None else 'n/a (no step events)'}.", ""]
    # 2
    out += ["## 2. Understanding over time", "", "### Beliefs (memory writes)", "",
            "| id | kinds over time | first at action | level | versions | refuted at | evidence | latest content |",
            "|---|---|---|---|---|---|---|---|"]
    for b in a["beliefs"]:
        out.append(f"| {b['id']} | {' -> '.join(b['kinds'])} | {b['first_action']} (turn {b['first_turn']}) | "
                   f"{b['first_level']} | {b['versions']} | {b['refuted_at'] if b['refuted_at'] is not None else ''} | "
                   f"{', '.join('#' + str(e) for e in b['evidence'][:8])} | {_md(b.get('content'), 120)} |")
    if not a["beliefs"]:
        out.append("| (no memory writes) | | | | | | | |")
    out += ["", "### Goal statements (mark `correct` by hand after the run)", "",
            "| source | at action | level | phase | goal | correct? |", "|---|---|---|---|---|---|"]
    for g in a["goals"]:
        out.append(f"| plan v{g['version']} | {g['action_count']} | {g['level']} | {g['phase']} | {_md(g['goal'], 140)} | |"
                   if isinstance(g["version"], int) else
                   f"| {g['version']} | {g['action_count']} | {g['level']} | memory | {_md(g['goal'], 140)} | |")
    if not a["goals"]:
        out.append("| (no goal stated) | | | | | |")
    # 3
    lv = a["living_code"]
    out += ["", "## 3. Living code", "",
            f"{lv['cells']} cells, {lv['helpers_defined']} helpers defined, {lv['helpers_reused']} reused later, "
            f"living-code ratio {lv['living_code_ratio']}, {lv['full_grid_prints']} full-grid prints, "
            f"{lv['cell_errors']} cell errors.", "",
            "| cell | turn | defines | reuses | variables | full grid | error | actions |", "|---|---|---|---|---|---|---|---|"]
    for cl in a["cells"]:
        out.append(f"| {cl['n']} | {cl['turn']} | {', '.join(cl['defines'])} | {', '.join(cl['reuses'])} | "
                   f"{_md(', '.join(cl['variables']), 60)} | {'yes' if cl['full_grid_print'] else ''} | "
                   f"{cl['error'] or ''} | {cl['actions'] or ''} |")
    # 4
    out += ["", "## 4. Waste per level", "",
            "| level | actions | human | cleared | rhae | resets (reason) | largest spend | repeated pairs | "
            "after last new belief |", "|---|---|---|---|---|---|---|---|---|"]
    for w in a["waste"]:
        rs = "; ".join(f"#{r['i']}: {_md(r['reason'], 60)}" for r in w["resets"][:5]) + \
             (f" (+{len(w['resets']) - 5})" if len(w["resets"]) > 5 else "")
        out.append(f"| {w['level']} | {w['actions']} | {w['baseline']} | {'yes' if w['cleared'] else ''} | {w['rhae']} | "
                   f"{len(w['resets'])}{': ' + rs if rs else ''} | {w['largest_spend']} | "
                   f"{'n/a' if w['repeated_pairs'] is None else w['repeated_pairs']} | "
                   f"{w['actions_after_last_new_belief'] if w['actions_after_last_new_belief'] is not None else ''} |")
    # 5
    out += ["", "## 5. Run context", "",
            f"- experiment `{c.get('experiment')}`, config hash `{c.get('config_hash')}`, git `{c.get('git_sha')}`, "
            f"model `{c.get('model')}`",
            f"- end reason {c.get('end_reason')}, wall {c.get('wall_s')} s, turns {c.get('turns')}, output tokens "
            f"{c.get('output_tokens')}, all tokens {c.get('tokens_total')}, last prompt "
            f"{c.get('prompt_tokens_last')}",
            f"- tool calls {json.dumps(c.get('tool_counts'))}, tool errors {c.get('tool_errors')}, act calls "
            f"{c.get('act_calls')} (mean batch {c.get('act_batch_mean')}), plan versions {c.get('plan_versions')}",
            f"- world model: {c.get('wm_checks')} act steps checked, {c.get('wm_mispredictions')} wrong; events: "
            + (", ".join(f"{e.get('kind')} {e.get('name', '')} v{e.get('version', '')}"
                         + (f" {e.get('passed')}/{e.get('tested')}" if e.get('kind') == 'check' else "")
                         for e in c.get("wm_events") or []) or "none"),
            f"- memory: act argument errors {c.get('act_arg_errors')}, recalls {c.get('recalls')} "
            f"{json.dumps(c.get('recall_scopes'))}, hypotheses {c.get('hypotheses')} "
            f"{json.dumps(c.get('hypothesis_statuses'))}, levels without a hypothesis "
            f"{c.get('levels_without_hypothesis')}, promotions {c.get('promotions')}, curator runs "
            f"{c.get('curator_runs')} (errors {c.get('curator_errors')}), skills written {c.get('skills_written')}, "
            f"images {c.get('images_sent')}, largest observation {c.get('observation_chars_max')} chars, largest "
            f"memory block {c.get('pinned_chars_max')} chars, repeated (state, action) share {c.get('repeated_share')}",
            f"- compactions: {len(c.get('compactions') or [])}"]
    for cp in c.get("compactions") or []:
        out.append(f"  - +{cp['t']}s ({cp['reason']}) at action {cp['action_count']}: {cp['summarized_messages']} "
                   f"messages summarized into {cp['summary_chars']} chars, {cp['kept_messages']} kept, "
                   f"{cp['tokens_before']} tokens before")
    return "\n".join(out) + "\n"


def game_map(game_dir: Path, rows: list[dict[str, Any]], run: dict[str, Any], session: dict[str, Any]) -> str:
    by_call: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if not r["after_cell"] and r.get("call"):
            by_call[r["call"]].append(r)
    shown: set[int] = set()
    game_id = run.get("game_id", game_dir.name)
    out = [f"# {game_id}: step map", "",
           f"level {run.get('levels_completed')}/{run.get('number_of_levels')} | {len(rows)} actions | "
           f"score {run.get('final_score')} | end {session.get('end_reason')} | turns {session.get('turns')} | "
           f"compactions {session.get('compactions')}",
           f"actions per level {run.get('actions_per_level')} vs baseline {run.get('baseline_actions')}", ""]
    events = _jsonl(game_dir / "transcript.jsonl") if (game_dir / "transcript.jsonl").exists() else []
    t0 = events[0]["t"] if events else 0.0
    levels = 0
    for ev in events:
        kind, t = ev.get("event"), f"+{ev.get('t', t0) - t0:.0f}s"
        if kind == "message" and ev.get("role") == "assistant":
            usage = ev.get("usage") or ["?", "?"]
            out += ["", f"## Turn {ev.get('turn', '?')} ({t}, prompt {usage[0]} / out {usage[1]} tokens"
                        f"{', CUT OFF' if ev.get('finish') == 'length' else ''})"]
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
            first = str(ev.get("content") or "").strip().splitlines()[:1]
            label = ev.get("_kind") or "host"
            out += ["", f"> **{label} -> agent** ({t}): {_cut(first[0] if first else '', 300)}"]
        elif kind in ("cell", "act", "reset"):
            acts = by_call.get(str(ev.get("call")), [])
            shown.update(r["i"] for r in acts)
            status = "" if kind != "cell" or ev.get("status") == "ok" else f" (cell {ev.get('status')})"
            if acts:
                out += [f"**Actions** ({len(acts)}){status}: " + render_actions(acts, levels), ""]
                levels = acts[-1].get("levels") if acts[-1].get("levels") is not None else levels
            elif status:
                out += [f"**No actions**{status}", ""]
        elif kind in ("plan", "remember", "wm", "delegate", "message_sent", "tool_error", "hypothesis", "finding",
                      "goal", "recall", "level_up", "curator", "memory_init"):
            detail = {k: v for k, v in ev.items() if k not in ("t", "event", "prev_content", "steps")}
            out += [f"- **[{kind}]** {t} " + _cut(json.dumps(detail, default=str, ensure_ascii=False), 500)]
        elif kind in ("compaction", "compaction_error", "compaction_skipped", "auto_refine", "spawn", "crash",
                      "llm_error", "reflection_due", "reflection_done", "reflection_skipped", "end"):
            detail = {k: v for k, v in ev.items() if k not in ("t", "event", "stats", "traceback")}
            out += ["", f"- **[{kind}]** {t} " + _cut(json.dumps(detail, default=str), 600)]
            if kind == "crash":
                out += ["````", _cut(ev.get("traceback"), 3000), "````"]
    rest = [r for r in rows if r["i"] not in shown]
    if rest:
        out += ["", "## Actions not matched to a call (background tasks or no transcript)", "",
                render_actions(rest, None)]
    return "\n".join(out) + "\n"


# --- SFT export --------------------------------------------------------------------------------------------
def _chat_message(ev: dict[str, Any]) -> dict[str, Any]:
    m = {"role": ev["role"], "content": ev.get("content") or ""}
    if ev.get("tool_calls"):
        m["tool_calls"] = ev["tool_calls"]
    if ev.get("tool_call_id"):
        m["tool_call_id"] = ev["tool_call_id"]
    if ev.get("reasoning"):
        m["reasoning_content"] = ev["reasoning"]
    if ev.get("_kind"):
        m["kind"] = ev["_kind"]
    return m


def _merge_users(msgs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Consecutive user messages joined, as the agent sends them (agent._wire): chat templates need alternation."""
    out: list[dict[str, Any]] = []
    for m in msgs:
        if out and m["role"] == "user" and out[-1]["role"] == "user":
            a, b = out[-1]["content"], m["content"]
            if isinstance(a, str) and isinstance(b, str):
                out[-1] = {**out[-1], "content": f"{a}\n\n{b}"}
            else:   # observations are lists of parts (text + image marker)
                out[-1] = {**out[-1], "content": _as_parts(a) + _as_parts(b)}
        else:
            out.append(m)
    return out


def _as_parts(content: Any) -> list[dict[str, Any]]:
    return content if isinstance(content, list) else [{"type": "text", "text": str(content)}]


def sft_levels(events: list[dict[str, Any]], run: dict[str, Any], context: dict[str, Any],
               rows: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """One row per level played: the messages produced while that level was current, tagged for filtering.

    The level switches after the tool round that cleared it (at the next message that is not a tool result), so a
    row never splits an assistant tool call from its result. Level-ups come from ``step`` events or, for older
    transcripts, from the recording rows (the call that raised ``levels``)."""
    system = next((e for e in events if e.get("event") == "system_prompt"), {})
    task = next((e for e in events if e.get("event") == "message" and e.get("role") == "user" and not e.get("_kind")),
                None)
    per_level = run.get("actions_per_level") or []
    base = run.get("baseline_actions") or []
    cleared = run.get("levels_completed") or 0
    n_levels = run.get("number_of_levels") or 10 ** 6
    has_steps = any(e.get("event") == "step" for e in events)
    level_of_call: dict[str, int] = {}     # fallback: call id -> levels completed after it
    if not has_steps:
        prev = 0
        for r in rows or []:
            if r.get("levels") is not None and r["levels"] > prev and r.get("call"):
                level_of_call[str(r["call"])] = r["levels"]
            prev = r["levels"] if r.get("levels") is not None else prev
    level, pending, last_turn = 0, None, None
    segs: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ev in events:
        if ev.get("event") == "message":
            is_result = ev.get("role") == "tool" or (
                ev.get("role") == "user" and str(ev.get("content") or "").startswith("[ipython output]"))
            if pending is not None and not is_result:
                level, pending = pending, None
            segs[level].append(_chat_message(ev))
            if ev.get("role") == "assistant":
                last_turn = ev.get("turn")
            elif not has_steps and is_result:
                cid = str(ev.get("tool_call_id") or f"fenced-t{last_turn}")
                if level_of_call.get(cid, 0) > level:
                    pending = level_of_call[cid]
        elif ev.get("event") == "step" and (ev.get("level_after") or 0) > max(level, pending or 0):
            pending = ev["level_after"]
    out = []
    for lvl in sorted(segs):
        n = per_level[lvl] if lvl < len(per_level) else None
        ok = lvl < cleared
        if lvl >= n_levels or (not ok and not n):
            continue   # after the last level, or a level with no action spent: nothing to learn from
        msgs = segs[lvl]
        if lvl > 0 and task is not None:
            msgs = [_chat_message(task), *msgs]
        out.append({
            "game_id": run.get("game_id"), "level": lvl, "level_cleared": ok,
            "rhae": round(min((base[lvl] / n) ** 2, 1.15), 4) if ok and n and lvl < len(base) else 0.0,
            "actions": n, "baseline": base[lvl] if lvl < len(base) else None, "starts_mid_conversation": lvl > 0,
            "system": system.get("content"), "tools": system.get("tools"), "messages": _merge_users(msgs),
            "level_source": "steps" if has_steps else "recording",
            **{k: context.get(k) for k in ("experiment", "config_hash", "git_sha", "model")}})
    return out


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
             "| game | levels | actions | turns | tools used | act calls | resets | GAME_OVERs | repeats | "
             "beliefs | living code | wm checks | promotions | compactions | end | report | map |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    sft: list[dict[str, Any]] = []
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
        sft += sft_levels(events, run, context, rows)
        acts = a["actions"]
        tools = a["context"].get("tool_counts") or {}
        table.append(
            f"| {game_id} | {run.get('levels_completed')}/{run.get('number_of_levels')} | {run.get('actions')} | "
            f"{session.get('turns')} | {', '.join(f'{k} {v}' for k, v in sorted(tools.items()))} | "
            f"{a['context'].get('act_calls')} | {sum(1 for r in acts if r['action'] == 0)} | "
            f"{sum(1 for r in acts if r.get('state') == 'GAME_OVER')} | "
            f"{sum(1 for r in acts if r.get('repeat_of') is not None)} | {len(a['beliefs'])} | "
            f"{a['living_code']['living_code_ratio']} | {a['context'].get('wm_checks')} | "
            f"{a['context'].get('promotions')} | {session.get('compactions')} | "
            f"{session.get('end_reason')} | [report](games/{game_id}/report.md) | [map](games/{game_id}/steps.md) |")
    with (run_dir / "sft_levels.jsonl").open("w", encoding="utf-8") as fh:
        for r in sft:
            fh.write(json.dumps(r, default=str, ensure_ascii=False) + "\n")
    table += ["", f"SFT export: `sft_levels.jsonl`, {len(sft)} level rows, "
                  f"{sum(1 for r in sft if r['level_cleared'])} cleared."]
    out = run_dir / "trace.md"
    out.write_text("\n".join(table) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    print(build_run(Path(sys.argv[1] if len(sys.argv) > 1 else ".")))

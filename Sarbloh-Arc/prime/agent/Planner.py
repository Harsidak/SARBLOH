"""Planner: the agent's explicit, versioned plan, written through the native ``plan`` tool.

Design (E006). The plan is the agent's working theory in four fields, stated in words, not code:
- ``phase``: explore (what do I control, what changes?) -> model (write and check the rules) -> execute (reach the goal).
- ``goal``: the current guess of how a level is won. "unknown" is a valid answer early on.
- ``hypotheses``: open claims, each with the test that would settle it ("A1 moves the 5x5 block up -> act [1]").
- ``steps``: the next few steps, first one first.

What the host does with it:
- Keeps every version (the trace reads them: when the goal was first stated, how often it changed).
- Shows a one-line plan in the status line after every tool result, and the full plan in the compaction head, so the
  plan survives compaction.
- Warns, never blocks: ``execute`` with no certified world model, or a plan that has gone stale (many actions, a level
  up or a GAME_OVER since the last update). The warning goes into the next ``act`` result.
Search (the "optimal decisions" part) is not here: it runs in the REPL, in the agent's own world model
(``wm.plan``), and its result is what the agent writes into ``steps``.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any

PHASES = ("explore", "model", "execute")

PLAN_TOOL = {
    "type": "function",
    "function": {
        "name": "plan",
        "description": "Write or update your plan. Free. Call it at the start, after each surprise, after each level "
                       "up, and before a long execution. The plan stays visible after compaction.",
        "parameters": {
            "type": "object",
            "properties": {
                "phase": {"type": "string", "enum": list(PHASES),
                          "description": "explore: find what you control and what changes. model: write and check the "
                                         "rules in the REPL. execute: reach the goal with a checked plan."},
                "goal": {"type": "string",
                         "description": "Your current guess of how the level is won, or 'unknown'."},
                "hypotheses": {"type": "array", "items": {"type": "string"},
                               "description": "Open claims, each with its test: 'claim -> test'. At most 5."},
                "steps": {"type": "array", "items": {"type": "string"},
                          "description": "The next steps, first one first. At most 8."},
            },
            "required": ["phase", "goal", "steps"],
        },
    },
}


@dataclass
class PlanVersion:
    version: int
    phase: str
    goal: str
    hypotheses: list[str]
    steps: list[str]
    turn: int
    action_count: int
    level: int
    t: float = field(default_factory=lambda: round(time.time(), 2))


class Planner:
    def __init__(self, stale_actions: int = 15) -> None:
        self.versions: list[PlanVersion] = []
        self.stale_actions = stale_actions
        self._warned_at: int | None = None   # action count of the last stale warning
        self._warned: set = set()             # triggers already warned for the current plan version

    @property
    def current(self) -> PlanVersion | None:
        return self.versions[-1] if self.versions else None

    def update(self, args: dict[str, Any], *, turn: int, action_count: int, level: int,
               certified: bool) -> tuple[str, dict[str, Any]]:
        """Apply a ``plan`` call. Returns (text for the model, event for the transcript)."""
        phase = str(args.get("phase") or "").strip().lower()
        if phase not in PHASES:
            raise ValueError(f"phase must be one of {list(PHASES)}")
        goal = str(args.get("goal") or "").strip()
        if not goal:
            raise ValueError("goal is required (write 'unknown' if you do not know it yet)")
        hyps = [h[:300] for h in _as_list(args.get("hypotheses"))][:5]
        steps = [s[:300] for s in _as_list(args.get("steps"))][:8]
        if not steps:
            raise ValueError("steps is required: at least the next step")
        prev = self.current
        v = PlanVersion(len(self.versions) + 1, phase, goal[:500], hyps, steps, turn, action_count, level)
        self.versions.append(v)
        self._warned_at = None
        self._warned = set()
        notes = []
        if phase == "execute" and not certified:
            notes.append("note: phase is execute but no world model has passed `await wm.check()`; if a step "
                         "surprises you, go back to model")
        if prev is not None and prev.goal != v.goal:
            notes.append(f"goal changed (was: {prev.goal[:120]})")
        text = f"plan v{v.version} saved ({phase}). Next: {steps[0][:160]}" + ("".join("\n" + n for n in notes))
        event = {"event": "plan", **asdict(v), "goal_changed": prev is not None and prev.goal != v.goal,
                 "phase_changed": prev is not None and prev.phase != v.phase, "certified": certified}
        return text, event

    def status(self) -> str:
        """One line for the status suffix after every tool result."""
        v = self.current
        if v is None:
            return "no plan yet (use the plan tool)"
        return f"plan v{v.version} {v.phase}: next {v.steps[0][:80]}"

    def render(self) -> str:
        """The whole plan, for the compaction head."""
        v = self.current
        if v is None:
            return ""
        lines = [f"[current plan v{v.version}, phase {v.phase}, written at action {v.action_count}]",
                 f"Goal: {v.goal}"]
        if v.hypotheses:
            lines += ["Open hypotheses:"] + [f"- {h}" for h in v.hypotheses]
        lines += ["Next steps:"] + [f"{i}. {s}" for i, s in enumerate(v.steps, 1)]
        return "\n".join(lines)

    def stale_note(self, *, action_count: int, level: int, game_over: bool) -> str | None:
        """A re-plan nudge for the next act result, at most once per plan version and trigger."""
        v = self.current
        if v is None:
            if action_count >= 5 and self._warned_at is None:
                self._warned_at = action_count
                return "[planner] you have acted without a plan. Call plan: phase, goal guess, hypotheses, next steps."
            return None
        reason, key = None, None
        if level > v.level:
            reason, key = f"level up (plan v{v.version} was written for level {v.level})", ("level", level)
        elif game_over:
            reason, key = "GAME_OVER", ("game_over", action_count)
        elif action_count - v.action_count >= self.stale_actions:
            reason = f"{action_count - v.action_count} actions since plan v{v.version}"
            key = ("stale", (action_count - v.action_count) // self.stale_actions)
        if reason is None or key in self._warned:
            return None
        self._warned.add(key)
        self._warned_at = action_count
        return f"[planner] {reason}. Update the plan: what did you learn, what is the goal now, what is next?"

    def to_json(self) -> list[dict[str, Any]]:
        return [asdict(v) for v in self.versions]


def _as_list(value: Any) -> list[str]:
    """Weak models send a string where the schema asks for an array: never iterate it letter by letter."""
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            try:
                value = json.loads(text)
            except ValueError:
                value = [text]
        else:
            value = text.replace(";", "\n").split("\n")
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"expected a list of strings, got {type(value).__name__}")
    return [str(v).strip() for v in value if str(v).strip()]

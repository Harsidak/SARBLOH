"""``act()``: make game moves from Python. Pre-imported by the kernel.

The REPL is the agent's only game tool: it models the game in code, searches the model and sends the moves it finds,
in the same cell. Each call goes to the host (``AgentSession._host``, type ``arc.act``), which runs the moves exactly
like the old act tool did: the same parsing, legality check, memory writes, change lines and stops (level up,
GAME_OVER, game end). The change lines are printed; the host sends the new state after the reply.

    r = await act(["4", "4", "1"], plan="push the block right, then up")
    r.lines        # one change line per move: "#12 A4(right): obj 4 R 3x3 moved right 3 -> ..."
    r.done         # moves made; r.stopped is why it stopped early ("level up", "GAME_OVER", ...) or None
    r.level_up, r.state
    obs = observe()   # the state after the moves

A move is "1".."5", "7", "reset", "6 r c" (a click at row r, column c), an int 1..7, or a tuple (6, r, c).
"""

from __future__ import annotations

from typing import Any

from rlm import repl as _repl


class ActResult:
    """What one ``act()`` call did."""

    def __init__(self, data: dict[str, Any]) -> None:
        self.text: str = data.get("text") or ""
        self.lines: list[str] = [ln for ln in self.text.splitlines() if ln.startswith("#")]
        self.done: int = int(data.get("done") or 0)
        self.stopped: str | None = data.get("stopped")
        self.level_up: bool = bool(data.get("level_up"))
        self.state: str = data.get("state") or ""

    def __repr__(self) -> str:
        return f"ActResult(done={self.done}, stopped={self.stopped!r}, state={self.state!r})"


def _move(a: Any) -> str:
    if isinstance(a, str):
        return a
    if isinstance(a, (tuple, list)):
        return " ".join(str(int(x)) for x in a)
    return str(int(a))   # int, numpy int


async def act(actions: Any, plan: str | None = None, hypotheses: list | None = None, findings: list | None = None,
              goal: str | None = None, quiet: bool = False) -> ActResult:
    """Make ``actions`` in order (each costs one move) and write your memory in the same call. Prints the change
    lines unless ``quiet``. Raises RuntimeError when the call is refused (nothing is spent then)."""
    if isinstance(actions, (str, int)) or (isinstance(actions, tuple) and actions and not
                                           isinstance(actions[0], (tuple, list, str))):
        actions = [actions]
    args: dict[str, Any] = {"actions": [_move(a) for a in actions]}
    for key, val in (("plan", plan), ("hypotheses", hypotheses), ("findings", findings), ("goal", goal)):
        if val is not None:
            args[key] = val
    reply = await _repl.host_request({"type": "arc.act", "args": args})
    if reply.get("status") != "ok":
        raise RuntimeError(str(reply.get("error") or "act failed"))
    res = ActResult(reply.get("result") or {})
    if not quiet:
        print(res.text)
    return res

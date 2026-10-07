"""Tool schemas the host offers the model.

Two tools: ``ipython`` to think, compute and act (``observe()`` reads the state for free, ``await act([...])`` makes
moves and writes the agent's plan, hypotheses, findings and goal), and ``recall`` to search memory and skills. Code and
moves share one tool so the agent can model the game, search its model and send the path in one call. The handlers live
in ``harness.agent.agent.AgentSession`` (``act()`` is a host request answered by ``_tool_act``).
"""

from __future__ import annotations


def _fn(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": properties, "required": required}}}


IPYTHON = _fn(
    "ipython",
    "Your one tool for the game: run Python to read the board, model the game, search for a path and make the moves. "
    "Variables, functions and imports stay for the whole game, across levels, so reuse your code and change it when "
    "the game shows something new. `obs = observe()` gives the current state, read-only and free. "
    "`await act([\"4\", \"4\", \"1\"], plan=...)` makes moves, each costing one move, and prints one change line "
    "per move. The full reference is in your instructions under \"`ipython` in detail\" and \"`act()` in detail\".",
    {"code": {"type": "string", "description": "Python code. Top-level `await` works."}},
    ["code"])

RECALL = _fn(
    "recall",
    "Search your memory. It is free. Use it to recall steps (\"#12\", \"12-20\", \"level 1\", or words), "
    "hypotheses and findings of earlier levels, your goal history, lessons from other levels and games, and skills.",
    {"query": {"type": "string", "description": "A step, a range, \"level N\", or words; \"\" for the latest."},
     "scope": {"type": "string", "enum": ["all", "timeline", "hypotheses", "findings", "goal", "lessons", "skills"],
               "description": "Where to search. Default all."}},
    ["query"])


def game_tools(**_: object) -> list[dict]:
    """The tool list for a session (the same for every game, so the server caches it)."""
    return [IPYTHON, RECALL]

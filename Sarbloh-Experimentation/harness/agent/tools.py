"""Tool schemas the host offers the model.

Two tools: ``ipython`` to think, compute and act (``observe()`` reads the state for free, ``await act([...])`` makes
moves), and ``recall`` to search memory and skills. The agent writes its memory as text before the code. Code and
moves share one tool so the agent can model the game, search its model and send the path in one call. The handlers live
in ``harness.agent.agent.AgentSession`` (``act()`` is a host request answered by ``_tool_act``).
"""

from __future__ import annotations


def _fn(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": properties, "required": required}}}


IPYTHON = _fn(
    "ipython",
    "Run Python: read the board with `observe()`, model the game and make moves with `await act([...])`. "
    "See \"The `ipython` tool\" in your instructions.",
    {"code": {"type": "string", "description": "Python code. Top-level `await` works."}},
    ["code"])

RECALL = _fn(
    "recall",
    "Search your memory and skills. See \"`recall` in detail\" in your instructions.",
    {"query": {"type": "string", "description": "A step, a range, \"level N\", or words; \"\" for the latest."},
     "scope": {"type": "string", "enum": ["all", "timeline", "plan", "hypotheses", "findings", "goal", "lessons", "skills"],
               "description": "Where to search. Default all."}},
    ["query"])


def game_tools(**_: object) -> list[dict]:
    """The tool list for a session (the same for every game, so the server caches it)."""
    return [IPYTHON, RECALL]

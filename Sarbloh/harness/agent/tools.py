"""Tool schemas the host offers the model.

One tool: ``ipython``, to think, compute and act. In the kernel ``observe()`` reads the state for free, ``await
play([...])`` makes moves, ``recall()`` searches the timeline and the skills (found skills go into ``SKILLS``) and
``graph()`` prints the map of the agent's own functions. Code, memory and moves share one tool so the agent can model
the game, search its model and send the path in one call. The handlers live in ``harness.agent.agent.AgentSession``
(each move of ``play()`` is a host request answered by ``_tool_act``).
"""

from __future__ import annotations


def _fn(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": properties, "required": required}}}


IPYTHON = _fn(
    "ipython",
    "your coding tool",
    {"code": {"type": "string", "description": "Implement your code in Python."}},
    ["code"])


def game_tools(**_: object) -> list[dict]:
    """The tool list for a session (the same for every game, so the server caches it)."""
    return [IPYTHON]

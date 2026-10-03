"""Tool schemas the host offers the model.

Three tools: ``ipython`` to think and compute (it cannot act and cannot read the raw game; it reads the state with
``observe()``), ``act`` to move (1 to N actions, plus the agent's plan, hypotheses, findings and goal), and ``recall``
to search memory and skills. The observation is pushed after every act; there is nothing to fetch. The handlers live
in ``harness.agent.agent.AgentSession``.
"""

from __future__ import annotations


def _fn(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": properties, "required": required}}}


IPYTHON = _fn(
    "ipython this is your primary tool to reason and you can use code to reason and come to take an optimal action",
    "Run Python in a persistent REPL, to perform resoning you can write code snippets to verify your hypotheses or plans.",
    " Variables, functions and imports persist for the",
    "whole game, across levels: write code and also edit the existing one when the game shows something new, and "
    "reuse them. `obs = observe()` gives the current state, read-only, and the full state (briefing, objects, board "
    "and picture) is shown to you in the next message, once per state. For your code: `obs.board[r][c]` (the colour "
    "letter at row r, column c), `obs.objects` (list of dicts: id, letter, name, size, bbox [r0,c0,r1,c1], hash, "
    "corners, parent, children, adjacent, hud, cells), `obs.change` (the newest change line), `obs.briefing`, "
    "`obs.history` (the last 20 steps), `print(obs.ascii(r0, c0, r1, c1))`. An `obs` does not update: call observe() "
    "again after an act. It cannot spend actions: make the moves your code finds with act.",
    {"code": {"type": "string", "description": "Python code. Top-level `await` works."}},
    ["code"])

ACT = _fn(
    "act",
    "Spend 1 to {max} game actions, in order; each costs 1 action against the score. Write your memory with the "
    "same call: `plan` often, and `hypotheses`, `findings` and `goal` when they change. The result gives one change "
    "line per action, then the new state (the changed region, its objects and a picture) arrives as the next message. "
    "The batch stops early at a level-up or a GAME_OVER.",
    {"actions": {"type": "array", "items": {"type": "string"},
                 "description": "Up to {max} actions: \"1\" up, \"2\" down, \"3\" left, \"4\" right, \"5\" space "
                                "(the game's special action), \"6 r c\" click the cell at row r, column c, \"7\" undo, "
                                "\"reset\" restart the level. Only the legal ones work; the meaning of each is yours "
                                "to verify. Example: [\"1\", \"1\", \"4\"] or [\"6 12 40\"]."},
     "plan": {"type": "string",
              "description": "Optional. Your next steps and to win the game (by clearing all levels in minimum steps)"},
     "hypotheses": {"type": "array", "description": "New hypotheses, or status changes of old ones (by id). You "
                                                    "decide the status: proposed, verified (it predicted steps it did "
                                                    "not come from) or refuted (a step contradicted it).",
                    "items": {"type": "object", "properties": {
                        "id": {"type": "string", "description": "e.g. \"h3\"; leave out for a new hypothesis"},
                        "text": {"type": "string", "description": "The rule, concrete: objects, directions, counts."},
                        "status": {"type": "string", "enum": ["proposed", "verified", "refuted"]},
                        "evidence": {"type": "array", "items": {"type": "integer"},
                                     "description": "Step numbers (#i in the change lines) that support or refute it."}},
                        "required": ["status"]}},
     "findings": {"type": "array", "items": {"type": "string"},
                  "description": "Facts you have established in this level, one short sentence each."},
     "goal": {"type": "string", "description": "What wins the level, as you now believe it. Kept across levels."}},
    ["actions"])

RECALL = _fn(
    "recall",
    "Search your memory. it is completely free to use by you. Use it to recall steps (\"#12\", \"12-20\", \"level 1\", or words),hypotheses and findings of "
    "earlier levels, your goal history, lessons from other levels and games, and skills.",
    {"query": {"type": "string", "description": "A step, a range, \"level N\", or words; \"\" for the latest."},
     "scope": {"type": "string", "enum": ["all", "timeline", "hypotheses", "findings", "goal", "lessons", "skills"],
               "description": "Where to search. Default all."}},
    ["query"])


def game_tools(*, act_max: int) -> list[dict]:
    """The tool list for a session, with ``act_max`` filled into the act schema."""
    fn = ACT["function"]
    props = dict(fn["parameters"]["properties"])
    props["actions"] = {**props["actions"], "description": props["actions"]["description"].format(max=act_max)}
    act = {"type": "function", "function": {**fn, "description": fn["description"].format(max=act_max),
                                            "parameters": {**fn["parameters"], "properties": props}}}
    return [IPYTHON, act, RECALL]

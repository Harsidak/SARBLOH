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
    "ipython",
    "Your main tool for reasoning: run Python to check your ideas and plans before you act. Variables, functions and "
    "imports stay for the whole game, across levels, so reuse your code and change it when the game shows something "
    "new. `obs = observe()` gives the current state, read-only; call it again after every act. It is free, and it "
    "cannot make moves: make the moves your code finds with `act`. The full reference is in your instructions under "
    "\"`ipython` in detail\".",
    {"code": {"type": "string", "description": "Python code. Top-level `await` works."}},
    ["code"])

ACT = _fn(
    "act",
    "Spend 1 to {max} game actions, in order; each costs 1 action against the score. In the same call you can "
    "update your memory (`plan`, `hypotheses`, `findings`, `goal`), but only what changed: what you leave out is "
    "kept. The result gives one change line per action, then the new state (the changed region, its objects and a "
    "picture when pictures are on) arrives as the next message. The batch stops early at a level-up or a GAME_OVER.",
    {"actions": {"type": "array", "items": {"type": "string"},
                 "description": "Up to {max} actions: \"1\" up, \"2\" down, \"3\" left, \"4\" right, \"5\" space "
                                "(the game's special action), \"6 r c\" click the cell at row r, column c, \"7\" often undo, "
                                "\"reset\" restart the level. Only the legal ones work; the meaning of each is yours "
                                "to verify. Example: [\"1\", \"1\", \"4\"] or [\"6 12 40\"]."},
     "plan": {"type": "string",
              "description": "Optional, only when your next steps change. Your next steps and why, in one or "
                             "two sentences. It replaces the old plan; leave it out to keep the old one."},
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
                  "description": "New facts you have established in this level, one short sentence each. They "
                                 "are added to the old ones, so send only new ones."},
     "goal": {"type": "string", "description": "What wins the level, as you now believe it, only when it "
                                               "changes. Kept across levels."}},
    ["actions"])

RECALL = _fn(
    "recall",
    "Search your memory. It is free. Use it to recall steps (\"#12\", \"12-20\", \"level 1\", or words), "
    "hypotheses and findings of earlier levels, your goal history, lessons from other levels and games, and skills.",
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

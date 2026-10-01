"""Tool schemas the host offers the model.

Upstream Prime Agent exposes one tool, the persistent ``ipython`` REPL; that is ``toolset: "ipython"`` (E003-E005, the
control arm). ``toolset: "e008"`` gives three: ``ipython`` to think and compute (it cannot act and cannot read the
raw game; it reads the read-only ``scene``), ``act`` to move (1 to N actions, plus the agent's plan, hypotheses,
findings and goal), and ``recall`` to search memory and skills. The observation is pushed after every act; there is
nothing to fetch. The handlers live in ``prime.agent.agent.AgentSession``.

E006's ``dedicated`` toolset (plan, reset_level, remember, delegate, message, and ``expect`` / ``was_right`` on act)
was removed on 2026-10-01 for E008; it is in git history (commit f117ca4).
"""

from __future__ import annotations


def _fn(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": properties, "required": required}}}


IPYTHON_TOOL = _fn(
    "ipython",
    "Execute Python code in a persistent Python REPL. Top-level `await` is supported. Variables, imports, and loaded "
    "data persist across calls. Run shell commands with `bash('cmd')` / `await bash('cmd')`.",
    {"code": {"type": "string", "description": "Python code to execute in the persistent Python REPL."}},
    ["code"])

IPYTHON_E008 = _fn(
    "ipython",
    "Run Python in a persistent REPL, to think and compute. Free. `scene` is the current state, read-only: "
    "`scene.objects` (list of dicts: id, letter, colour, size, bbox [r0,c0,r1,c1], hash, parent, children, adjacent, "
    "hud, cells), `scene.letters` (rows as strings of colour letters), `print(scene.ascii(r0, c0, r1, c1))`, "
    "`scene.find(letter=..., hash=..., id=...)`, `scene.history(n)` (the last n steps as change lines and objects). "
    "Variables and functions persist. It cannot spend actions: use act.",
    {"code": {"type": "string", "description": "Python code. Top-level `await` works."}},
    ["code"])

ACT_E008 = _fn(
    "act",
    "Spend 1 to {max} game actions, in order; each costs 1 action against the score. Write your memory with the "
    "same call: `plan` always; `hypotheses`, `findings` and `goal` when they change. The result gives one change line "
    "per action, then the new state (objects, the changed region and an image) arrives as the next message. The "
    "batch stops early at a level-up or a GAME_OVER.",
    {"actions": {"type": "array", "items": {"type": "string"},
                 "description": "Up to {max} actions: \"1\" up, \"2\" down, \"3\" left, \"4\" right, \"5\" space "
                                "(the game's special action), \"6 r c\" click the cell at row r, column c, \"7\" undo, "
                                "\"reset\" restart the level. Only the legal ones work; the meaning of each is yours "
                                "to verify. Example: [\"1\", \"1\", \"4\"] or [\"6 12 40\"]."},
     "plan": {"type": "string",
              "description": "Your next steps and why, in one or two sentences. Replaces the previous plan."},
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
    ["actions", "plan"])

# E021 (``memory.goal_versioning``): ``goal`` also takes a list of goals, to keep rivals and refute them.
GOAL_E021 = {
    "anyOf": [
        {"type": "string"},
        {"type": "array", "items": {"type": "object", "properties": {
            "id": {"type": "string", "description": "e.g. \"g2\"; leave out for a new goal"},
            "text": {"type": "string", "description": "What wins the level, concrete: objects, places, counts."},
            "status": {"type": "string", "enum": ["active", "candidate", "refuted"]},
            "evidence": {"type": "array", "items": {"type": "integer"},
                         "description": "Step numbers (#i) that support or refute it."},
            "why": {"type": "string", "description": "Why you add, switch or refute it, in a few words."}}}}],
    "description": "What wins the level. A string is a new version of your active goal. A list keeps up to 3 goals: "
                   "one active (your plan pursues it), the rest candidate; refute one with evidence and why. Kept "
                   "across levels."}

RECALL_E008 = _fn(
    "recall",
    "Search your memory. Free. Finds steps (\"#12\", \"12-20\", \"level 1\", or words), hypotheses and findings of "
    "earlier levels, your goal history, lessons from other levels and games, and skills.",
    {"query": {"type": "string", "description": "A step, a range, \"level N\", or words; \"\" for the latest."},
     "scope": {"type": "string", "enum": ["all", "timeline", "hypotheses", "findings", "goal", "lessons", "skills"],
               "description": "Where to search. Default all."}},
    ["query"])

E008_TOOLS = ("ipython", "act", "recall")


def toolset(mode: str, *, depth: int, max_depth: int, act_max: int, goal_versioning: bool = False) -> list[dict]:
    """The tool list for a session. ``mode`` is the config's ``agent.toolset``: "e008" or "ipython"."""
    if mode == "e008" and depth == 0:
        fn = ACT_E008["function"]
        props = dict(fn["parameters"]["properties"])
        props["actions"] = {**props["actions"], "description": props["actions"]["description"].format(max=act_max)}
        if goal_versioning:
            props["goal"] = GOAL_E021
        act = {"type": "function", "function": {**fn, "description": fn["description"].format(max=act_max),
                                                "parameters": {**fn["parameters"], "properties": props}}}
        return [IPYTHON_E008, act, RECALL_E008]
    if mode not in ("e008", "ipython"):
        raise ValueError(f"unknown toolset {mode!r}: 'e008' or 'ipython'")
    return [IPYTHON_TOOL]

"""Tool schemas the host offers the model.

Upstream Prime Agent exposes one tool, the persistent ``ipython`` REPL; that is ``toolset: "ipython"`` (E003-E005).
``toolset: "dedicated"`` (E006) takes actions, resets, memory and delegation out of the REPL and gives each a native
tool, so that every action carries a prediction and every reset a reason. Observation, transitions, diffs, analysis and
the world model stay in ``ipython`` and stay free. The handlers live in ``prime.agent.agent.AgentSession``.
"""

from __future__ import annotations

from prime.agent.Planner import PLAN_TOOL


def _fn(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": properties, "required": required}}}


IPYTHON_TOOL = _fn(
    "ipython",
    "Execute Python code in a persistent Python REPL. Top-level `await` is supported. Variables, imports, and loaded "
    "data persist across calls. Run shell commands with `bash('cmd')` / `await bash('cmd')`.",
    {"code": {"type": "string", "description": "Python code to execute in the persistent Python REPL."}},
    ["code"])

# Dedicated mode: the REPL is for looking and thinking, not acting.
IPYTHON_TOOL_DEDICATED = _fn(
    "ipython",
    "Run Python in the persistent REPL. Free. Use it to look and think: `await arc.observe()`, `arc.show(...)`, "
    "`await arc.transitions()`, `arc.diff(a, b)`, `wm.objects(grid)`, and to write and check your world model "
    "(`wm.register`, `await wm.check()`, `wm.plan`). Variables and functions persist. It cannot spend actions: use act.",
    {"code": {"type": "string", "description": "Python code. Top-level `await` works."}},
    ["code"])

ACT_TOOL = _fn(
    "act",
    "Spend 1 to {max} game actions, in order. Each costs 1 action against the score. Say first what you expect. The "
    "result gives, per step, what changed (objects moved, colours changed) and the new game status. The batch stops "
    "early at a level up, a GAME_OVER, or when your registered world model predicts a step wrong.",
    {"actions": {"type": "array", "items": {"type": "string"},
                 "description": "Action ids from the legal list, e.g. [\"1\", \"1\", \"4\"]. Action 6 is a click: "
                                "\"6 x y\" with x = column, y = row, e.g. \"6 12 40\"."},
     "expect": {"type": "string",
                "description": "One sentence: what you predict these actions will do, and why."},
     "was_right": {"type": "string", "enum": ["yes", "no", "partly", "n/a"],
                   "description": "Was the expect of your previous act right? n/a for the first act."}},
    ["actions", "expect"])

RESET_TOOL = _fn(
    "reset_level",
    "Restart the current level. Costs 1 action, and the actions already spent in the level still count. The level's "
    "state is lost; your REPL variables, memories and plan are kept. Needed after GAME_OVER.",
    {"reason": {"type": "string", "description": "Why a restart is better than continuing from here."}},
    ["reason"])

REMEMBER_TOOL = _fn(
    "remember",
    "Save or update a memory. Free. Memories survive compaction and are shown at every compaction. Same title = "
    "update (use it to correct or refute an old belief).",
    {"kind": {"type": "string", "enum": ["fact", "hypothesis", "refuted", "goal", "procedure"],
              "description": "fact: verified by transitions (evidence required). hypothesis: not yet verified. "
                             "refuted: shown false (evidence required). goal: how a level is won. procedure: a "
                             "reusable way to do something (e.g. a REPL helper's name and use)."},
     "title": {"type": "string", "description": "Short, stable name, e.g. 'ACTION1 moves player up'."},
     "content": {"type": "string", "description": "The memory itself: precise, with numbers and coordinates."},
     "evidence": {"type": "array", "items": {"type": "integer"},
                  "description": "Transition indices (from act results or arc.transitions()) that show it."},
     "all_games": {"type": "boolean", "description": "Also useful in other games (a general lesson). Default false."}},
    ["kind", "title", "content"])

RECALL_TOOL = _fn(
    "recall",
    "Search your memories (this game, and general lessons from other games). Free. An empty query lists all of this "
    "game's memories.",
    {"query": {"type": "string", "description": "Words to search for, or \"\" for everything."}},
    ["query"])

DELEGATE_TOOL = _fn(
    "delegate",
    "Start a child agent on a self-contained analysis task. Free. It has its own REPL with read-only game access "
    "(`arc.observe`, `arc.transitions`, `wm`) and cannot act. It works in parallel; its answer arrives later as a "
    "message. Use it for heavy analysis (e.g. 'fit a movement rule to transitions 0-40') while you continue.",
    {"name": {"type": "string", "description": "Unique short name: letters, digits, '_', '-'."},
     "task": {"type": "string", "description": "The full task: what to analyse, what to return. The child does not "
                                               "see your conversation."}},
    ["name", "task"])

MESSAGE_TOOL = _fn(
    "message",
    "Send a message to your parent (to=\"parent\") or to one of your children (to=its name). Free.",
    {"to": {"type": "string", "description": "\"parent\" or a child's name."},
     "text": {"type": "string", "description": "The message."}},
    ["to", "text"])

ROOT_TOOLS = ("ipython", "plan", "act", "reset_level", "remember", "recall", "delegate", "message")
CHILD_TOOLS = ("ipython", "recall", "message")


def toolset(mode: str, *, depth: int, max_depth: int, act_max: int) -> list[dict]:
    """The tool list for a session. ``mode`` is the config's ``agent.toolset``."""
    if mode != "dedicated":
        return [IPYTHON_TOOL]
    act = {**ACT_TOOL, "function": {**ACT_TOOL["function"],
                                    "description": ACT_TOOL["function"]["description"].format(max=act_max)}}
    table = {"ipython": IPYTHON_TOOL_DEDICATED, "plan": PLAN_TOOL, "act": act, "reset_level": RESET_TOOL,
             "remember": REMEMBER_TOOL, "recall": RECALL_TOOL, "delegate": DELEGATE_TOOL, "message": MESSAGE_TOOL}
    names = CHILD_TOOLS if depth > 0 else ROOT_TOOLS
    if depth >= max_depth:
        names = tuple(n for n in names if n != "delegate")
    return [table[n] for n in names]

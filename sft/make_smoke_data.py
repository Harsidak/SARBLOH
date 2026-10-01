"""Writes data/smoke3.jsonl: three short, hand-written Prime-style tool trajectories for the pipeline smoke test.

They are not training data. They exercise every input shape the real exports have: a plain {"messages", "tools"} row
with a system message, a Prime ``sft_levels.jsonl`` row (``system`` beside ``messages``, JSON-string tool arguments,
an image part), reasoning_content, several tool calls, and a second user turn (which makes Qwen3.5's template
non-prefix-stable and forces per-turn masking).
"""

import json
from pathlib import Path


def fn(name, desc, props, req):
    return {"type": "function", "function": {"name": name, "description": desc,
                                             "parameters": {"type": "object", "properties": props, "required": req}}}


TOOLS = [
    fn("ipython", "Run Python in the persistent REPL. Free. Use it to look and think: `await arc.observe()`, "
       "`arc.diff(a, b)`, `wm.objects(grid)`, `wm.register`, `await wm.check()`. It cannot spend actions: use act.",
       {"code": {"type": "string", "description": "Python code. Top-level `await` works."}}, ["code"]),
    fn("act", "Spend 1 to 8 game actions, in order. Each costs 1 action against the score. Say first what you expect.",
       {"actions": {"type": "array", "items": {"type": "string"}, "description": "Action ids, e.g. [\"1\", \"4\"]."},
        "expect": {"type": "string", "description": "One sentence: what these actions will do, and why."},
        "was_right": {"type": "string", "enum": ["yes", "no", "partly", "n/a"]}}, ["actions", "expect"]),
    fn("remember", "Save or update a memory. Free.",
       {"kind": {"type": "string", "enum": ["fact", "hypothesis", "refuted", "goal", "procedure"]},
        "title": {"type": "string"}, "content": {"type": "string"},
        "evidence": {"type": "array", "items": {"type": "integer"}}}, ["kind", "title", "content"]),
]
SYSTEM = ("You play an ARC-AGI-3 game. Actions cost score; thinking and Python are free. Write your world model as "
          "code, check it against every recorded transition with `await wm.check()`, and act only on a checked model.")


def call(cid, name, args, as_string=False):
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args) if as_string else args}}


ROWS = [
    {   # 1. plain shape, system message inside messages, dict arguments
        "id": "smoke-ls20-push",
        "tools": TOOLS,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Game ls20, level 1. Legal actions: 1 2 3 4. Grid 16x16. A blue 2x2 block "
                                        "(colour 1) at x=3,y=5; a yellow cell (colour 4) at x=9,y=5."},
            {"role": "assistant", "reasoning_content": "First look at the objects before spending anything.",
             "content": "", "tool_calls": [call("c1", "ipython", {
                 "code": "g = await arc.observe()\nobjs = wm.objects(g)\nprint([(o.colour, o.bbox) for o in objs])"})]},
            {"role": "tool", "tool_call_id": "c1",
             "content": "[(1, (3, 5, 4, 6)), (4, (9, 5, 9, 5)), (5, (0, 0, 15, 0))]"},
            {"role": "assistant", "reasoning_content": "Action 4 is probably right. Test one step and predict it.",
             "content": "", "tool_calls": [call("c2", "act", {
                 "actions": ["4"], "expect": "The blue block moves one cell right, to x=4.", "was_right": "n/a"})]},
            {"role": "tool", "tool_call_id": "c2", "content": "step 0: object colour 1 moved (+1, 0). status PLAYING"},
            {"role": "assistant", "content": "", "tool_calls": [call("c3", "remember", {
                "kind": "fact", "title": "ACTION4 moves the block right",
                "content": "ACTION4 moves the blue 2x2 block +1 in x.", "evidence": [0]})]},
            {"role": "tool", "tool_call_id": "c3", "content": "saved"},
            {"role": "assistant", "reasoning_content": "The block is at x=4; the target is at x=9: five more steps.",
             "content": "", "tool_calls": [call("c4", "act", {
                 "actions": ["4", "4", "4", "4", "4"],
                 "expect": "The block reaches the yellow cell and the level ends.", "was_right": "yes"})]},
            {"role": "tool", "tool_call_id": "c4", "content": "steps 1-5: colour 1 moved (+1, 0) each. LEVEL UP"},
        ],
    },
    {   # 2. Prime sft_levels.jsonl row: system beside messages, JSON-string arguments, an image part
        "game_id": "ft09", "level": 0, "level_cleared": True, "rhae": 0.81, "actions": 4, "baseline": 4,
        "starts_mid_conversation": False, "system": SYSTEM, "tools": TOOLS,
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "Game ft09, level 1. Legal actions: 6 (click x y). The board:"},
                {"type": "image_url", "image_url": {"url": "[png of step 0]"}},
                {"type": "text", "text": "Four grey tiles in a row at y=8, x=2,6,10,14. Clicking flips a tile."}]},
            {"role": "assistant", "reasoning_content": "Register a flip model and check it before clicking.",
             "content": "", "tool_calls": [call("p1", "ipython", {
                 "code": "def step(s, a):\n    x, y = a\n    s = s.copy()\n    s[y][x] = 3 if s[y][x] == 5 else 5\n"
                         "    return s\nwm.register(step)\nprint(await wm.check())"}, as_string=True)]},
            {"role": "tool", "tool_call_id": "p1", "content": "wm.check: 0 transitions, nothing to refute yet"},
            {"role": "assistant", "content": "", "tool_calls": [call("p2", "act", {
                "actions": ["6 2 8", "6 6 8", "6 10 8", "6 14 8"],
                "expect": "Each click flips one grey tile to green; all four green wins.", "was_right": "n/a"},
                as_string=True)]},
            {"role": "tool", "tool_call_id": "p2", "content": "4 steps, the model predicted all 4. LEVEL UP"},
        ],
    },
    {   # 3. two user turns (the second is a harness nudge): the template drops turn 1's <think>, so per-turn masking
        "id": "smoke-vc33-nudge",
        "tools": TOOLS,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Game vc33, level 2. Legal actions: 1 2 3 4 5. Your memories: ACTION5 "
                                        "swaps the two red cells (fact, evidence [3, 7])."},
            {"role": "assistant", "reasoning_content": "Check what changed since level 1.", "content": "",
             "tool_calls": [call("v1", "ipython", {"code": "print(arc.diff(0, -1))"})]},
            {"role": "tool", "tool_call_id": "v1", "content": "level 2: 3 red cells instead of 2; one wall moved"},
            {"role": "user", "content": "Reminder: 12 actions used on this level; the human baseline is 9."},
            {"role": "assistant", "reasoning_content": "Three red cells: the swap rule may not carry over. Record "
                                                       "it as a hypothesis, not a fact.",
             "content": "The old swap rule is unverified on this level.",
             "tool_calls": [call("v2", "remember", {
                 "kind": "hypothesis", "title": "ACTION5 with three red cells",
                 "content": "With 3 red cells ACTION5 may rotate them rather than swap two."})]},
            {"role": "tool", "tool_call_id": "v2", "content": "saved"},
        ],
    },
]

if __name__ == "__main__":
    out = Path(__file__).parent / "data" / "smoke3.jsonl"
    out.parent.mkdir(exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ROWS), encoding="utf-8")
    print(f"wrote {out} ({len(ROWS)} rows)")

"""Prompts for SARBLOH, written as small snippets and assembled per situation.

How the game loop uses prompts (read from ``harness/agent/agent.py`` and ``harness/run.py``):

    run.py      -> start message (game start)            : task_message()
    agent.py    -> system prompt, fixed for the game    : game_system() / base_prompt() + arc_section()
    agent.py    -> after every act: change lines + new state (built in code, not here)
    agent.py    -> the model ends a turn with no tool    : continuation()
    agent.py    -> level up, GAME_OVER, WIN, refused act, reply cut off, compaction : event_message(kind, ...)

The system prompt never changes during a game (cheap prefix caching). Everything that depends on the moment (a new
level, a lost level, a refused act, a long silence, low budget) is sent as a short message built by ``event_message``.
So "which prompt goes in when" is the table ``EVENT_SNIPPETS`` plus the if/else in the builders.

Tools are described twice on purpose: ``TOOLS_BRIEFING`` says what each tool is for and what it costs;
``TOOLS_DETAILED`` (= ``IPYTHON_TOOL_DETAILED`` + ``ACT_DETAILED`` + ``RECALL_DETAILED``) gives every argument, limit,
refusal and return format, taken from ``tools.py``, ``agent.py`` (``_tool_act``, ``_tool_recall``),
``runtime/skills/observation.py`` and ``agent/perception.py`` (E116). If those change, change these.

Style (owner rule, 2026-10-02): simple English in full sentences that flow, written as instructions to a human. Exact
tool and function names, no stories. The score formula is not shown to the agent on purpose (owner decision
2026-10-02). ROLE_OBJECTIVE_COMMUNICATION, GAME_INTUITION and IPYTHON_BRIEFING are the owner's wording; edit them only
when the owner asks. LONG_RUNNING_WORK is upstream text.

Placeholders are filled with ``str.format``: a literal brace in a template is doubled.
"""

from __future__ import annotations

from typing import Any

# =====================================================================================================================
# 1. The owner's snippets (shared by both toolsets)
# =====================================================================================================================

ROLE_OBJECTIVE_COMMUNICATION = """Role:
You are an agent playing a multi-level grid puzzle game. Your goal is to solve the entire game by clearing every level in as few moves as possible.

Objective:
The game board is a 64x64 grid made of colors. {legend}
You will play the game in a continuous cycle: look at the board, use Python to check your ideas, take an action, and check the results. Keep in mind that game rules and layouts can change between levels.

Communication:
Write your thinking and your notes in short, plain sentences."""

# The colour legend in the form each toolset shows the board: letters (``observe()``) or numbers (``arc``, ``obs.grid``).
LEGEND_LETTERS = ("Each color is written as one letter: W=white, w=light grey, g=grey, G=dark grey, c=charcoal, "
                  "B=black, M=magenta, P=pink, R=red, b=blue, S=sky blue, Y=yellow, O=orange, r=dark red, N=green, "
                  "p=purple.")
LEGEND_NUMBERS = ("Each color is a number from 0 to 15, and `arc.show` prints it as one hex digit (a=10 to f=15): "
                  "0=white, 1=light grey, 2=grey, 3=dark grey, 4=charcoal, 5=black, 6=magenta, 7=pink, 8=red, 9=blue, "
                  "10=sky blue, 11=yellow, 12=orange, 13=dark red, 14=green, 15=purple.")

GAME_INTUITION = """Game Intuition:
- Visualizing the Board: Treat the board as a picture with objects, obstacles, and targets.
- Objects: Puzzle pieces are usually groups of blocks (like 2x2 or 3x3 shapes) or single blocks.
- No Player Character: Do not assume you control a specific "player." The puzzle might be controlled by a cursor, or by changing the whole board at once.
- Backgrounds: Backgrounds are usually large, stable areas. Do not assume the background is a specific color; figure it out by looking at what takes up the most space and doesn't move.
- Timers and HUDs (Important): If you see a strip of blocks on the edge of the screen that changes on every move, it is likely a timer or a "moves-remaining" bar. **Do not treat these as puzzle pieces.** Do not click on them unless you are absolutely sure they are part of the puzzle.
- Clicks and Goals: A click needs an exact row and column. The row counts down from the top, and the column counts across from the left. Describe your goals with objects (for example, "move the red block onto the green one"), not with coordinates, because positions change from level to level.
- Game Progress: If the number of finished levels goes up, or the whole board changes at once, you probably finished a level. Stop and look at the new board carefully before you use an old plan. `WIN` means you have beaten the entire game."""

IPYTHON_BRIEFING = """iPython tool:
- Write short, focused Python code instead of massive, complicated scripts. Never print the entire game board. Print only small, useful summaries (like object lists, coordinate changes, or counts).
- Unlimited Uses: You can use the Python tool as many times as you need to investigate the board before making an actual game move. Do not rush."""

LONG_RUNNING_WORK = "\n".join([
    "For slow or independently completing work, use a nonblocking control loop: start the work, record its handle or "
    "output location, then end your turn.",
    "Do not keep the turn open by polling with `time.sleep()` or shell `sleep`, and do not replace polling with a long "
    "blocking `await`. Await only the short operation needed to start work or inspect a result that is already "
    "available; otherwise end the turn.",
])

# =====================================================================================================================
# 2. System-prompt snippets for the game agent (tools: ipython, act, recall)
# =====================================================================================================================

GAME_FACTS = """Game:
You are playing the ARC-AGI-3 game `{game_id}`. It has {win_levels} levels. The game is turn based, so nothing moves until you make a move. Nobody will tell you the controls, the rules or the goal. You find them out by looking and by trying moves. Early levels usually teach one idea each, and later levels mix them. You win the game when every level is done.
You work alone. There is no human to answer questions, so never ask one. Thinking and using Python are free. Only game moves count, so make each move for a reason."""

READING_THE_STATE = """What you see:
1. After every `act`, the result has one line per move. For example:
   `#12 A1(up): obj 4 R 3x3 moved up 5 -> r10-12 c20-22; obj 17 Y shrank 40->38 cells (hud)`.
   `#12` is the step number, and you use step numbers as evidence. A line that ends with `[repeat: same state and action as #k]` means you already made this exact move from this exact board.
2. The new state follows in the next message. It has a status line, the last step, the objects in the area that changed and that area drawn as letters{picture_after}.
3. At the start of each level, after old messages are shortened, and whenever you call `observe()` in `ipython`, you get the full state instead: a briefing, the full list of objects and the whole board as letters{picture_start}.

The briefing is worked out by code from the board, and it has two parts:
- MEASURED facts are computed, so you can trust them. They say where the object most likely to be you can go and in how many steps, what encloses an empty space and what would fit in it, which lines tie objects together, which objects line up in a row or a column, which objects have the same shape (turned, mirrored, resized or in other colors), which objects are identical copies, and which objects have lopsided colors.
- GUESSES are only guesses. Each one gives a role, such as YOU, GOAL, MOVES LEFT or HANDLE, and a quick test. On new games they are wrong about half the time, so check a guess with one move before you build a plan on it.
{picture_help}
How to read it:
- Positions are written as rows and columns. `r10-12 c20-22` means rows 10 to 12 and columns 20 to 22. Row 0 is the top, and column 0 is the left.
- An object is one connected area of one color. Its line gives its id, its color letter and its size (`R 3x3` is a full rectangle, `R 7px` is any other shape), where it is, and a `#hash`. The same shape always has the same hash, in any place and on any level. Then come its relations: `in 2` means it is inside object 2, `has 5,6` means it contains objects 5 and 6, and `adj 3` means it touches object 3. `hud` means it sits on the edge of the screen, where it is often a counter, a bar or a number of lives.
- Object ids stay the same for the whole level, and the same ids are used in the briefing{picture_ids}. A thing made of several colors is named after the id of its biggest part. The most common color is the background, and it is not listed as an object."""

PICTURE_HELP = """
The picture is a map of the board. Open floor is light, walls and other large areas are dark and hatched, and the HUD at the edge of the screen is grey-blue and is drawn again, larger, on the right. Objects keep their real colors. Each object has a box and a tag with its id and its guessed role, for example `10: YOU?`. A dashed line joins objects with the same shape (pink when one is turned or resized, blue when only the colors differ), a yellow ring marks an object held between two others, and thin lines show the floor's tile grid. The letters are exact, so use the picture to see the layout at a glance and the letters for exact positions.
"""

TOOLS_BRIEFING = """Tools:
You have three tools, and only `act` spends moves.
- `ipython` (free) runs Python, so you can look at the board and compute things. It reads the current state with `observe()`. It cannot make moves.
- `act` makes 1 to {act_max} moves, and in the same call you write your memory (plan, hypotheses, findings and goal). Each move costs one move. Use one `act` per reply.
- `recall` (free) searches your memory: earlier steps, hypotheses, findings, goals, lessons and skills.
The cycle is: read the new board, check your idea in `ipython`, then `act`. Use `recall` when you need something that is no longer in the conversation."""

IPYTHON_TOOL_DETAILED = """`ipython` in detail:
- Its argument is `code`, the Python to run. Top-level `await` works, and variables, functions and imports stay between calls.
- `obs = observe()` gives you the current state. It is free, and the full state ({full_state}) also comes to you in the next message, once per state. There is no number grid: colors are letters, and positions are (row, column).
  - `obs.board` is the board, one string of letters per row. `obs.board[r][c]` is the color at row r and column c.
  - `obs.objects` is a list of dicts, one per object, with the keys `id`, `letter`, `name`, `size`, `bbox` [r0, c0, r1, c1], `hash`, `corners`, `parent`, `children`, `adjacent`, `hud` and `cells`. `cells` is a list of (row, column), or None when the object has more than 256 cells. To find objects, filter the list, for example `[o for o in obs.objects if o["letter"] == "R"]`.
  - `print(obs.ascii(r0, c0, r1, c1))` prints a window of the board with row and column labels.
  - `obs.change` is the change line of the newest step, and `obs.history` lists the last 20 steps. Each step is a dict with `i` (the step number), `level`, `action`, `change` and `state`.
  - `obs.briefing` has every line of the briefing, `obs.step` is the newest step number, `obs.status` is the status line, `obs.background` is the background letter, and `obs.legend` maps each letter to its color name.
- An `obs` does not change after you get it. Call `observe()` again after every `act`, and call it inside your helper functions instead of keeping an old board in a variable."""

ACT_DETAILED = """`act` in detail:
Arguments:
- `actions` (required) is a list of 1 to {act_max} moves, made in order.
  - "1" is up, "2" down, "3" left, "4" right and "5" space (the game's special action). "7" is often undo, but check that in this game before you rely on it.
  - "6 r c" clicks the cell at row r and column c, for example "6 12 40". The row comes first, and each click is its own item in the list.
  - "reset" restarts the current level.
  For example, ["1", "1", "4"] or ["6 12 40"]. These names are only labels. What a move really does in this game is a guess until you have seen it happen.
- `plan` (required), `hypotheses`, `findings` and `goal` are your memory. They are explained below under "Your memory".
Limits:
- Use one `act` per reply. A second `act` in the same reply is refused and spends nothing.
- Only the legal moves in the status line work.
- The whole call is refused before any move, with nothing spent and nothing saved, if there are more than {act_max} moves, if a move cannot be read or is not legal, if `plan` is missing, or if a hypothesis is wrong (an unknown id, a bad status, or no text). Fix it and send the call again.
- The moves stop early at a level up, at GAME_OVER, or when the game ends.
- If the game refuses a move, the moves before it still count, and your memory writes are kept.
What comes back:
- The first line is `act: 3 of 3 actions done`, plus `; stopped: <reason>` when it stopped early.
- Then comes one change line per move, for example `#12 A1(up): <what changed>`. A click is written `A6(r12,c40)`. `[repeat: same state and action as #7]` means you made this exact move from this exact board before, and `[GAME_OVER]` marks the move that lost the level.
- After a level up, a GAME_OVER or a WIN there is a short note that tells you what to do next.
- The new board follows in the next message."""

RECALL_DETAILED = """`recall` in detail:
Arguments:
- `query` (required) can be "#12" (one step), "12-20" (a range of steps), "level 1" (one level), or words such as "red block". An empty query "" gives the latest steps.
- `scope` (optional, default "all") can be "timeline" (your steps), "hypotheses", "findings", "goal" (your goal and its history), "lessons" (from earlier levels and games), "skills", or "all".
What comes back is the matches, grouped by scope. Long results are cut, so ask for something narrow."""

TOOLS_DETAILED = "\n\n".join([IPYTHON_TOOL_DETAILED, ACT_DETAILED, RECALL_DETAILED])

MEMORY = """Your memory:
At the top of every turn there is a [memory] block. It is rebuilt from what you write with `act`, and it stays when old messages are removed. Old messages are not kept, so anything you want to remember must go into your memory. In every `act` you write:
- `plan` (always): your next steps and the reason, in one or two sentences. It replaces your old plan.
- `hypotheses` (when they change): a list of rules you think are true. Each one has a `status`, which is "proposed", "verified" (it correctly predicted moves you made after you wrote it) or "refuted" (a move showed it is wrong). A new hypothesis also needs `text`, the rule written concretely (objects, directions, counts), and it gets an id such as "h3". To change an old one, give its `id`, for example {{"id": "h2", "status": "refuted", "evidence": [14]}}. Always give step numbers in `evidence`.
- `findings` (when you learn one): facts about this level, one short sentence each.
- `goal` (when it changes): what you think wins the level. While you are not sure, keep 2 or 3 different goals and test them against each other.
When you finish a level, your verified hypotheses and findings are saved as lessons for later levels, and your plan, hypotheses and findings are cleared. Your goal is kept. So mark what you have verified before the level ends."""

LEVEL_METHOD = """How to play a level:
1. Look (free). Read the briefing, the board and the object list. Name the objects: what you might control, the walls, the targets, and any counter on the edge. The GUESSES in the briefing are a good place to start, but each one needs its test.
2. Guess (free). Write 2 to 4 hypotheses and a goal guess in your first `act`.
3. Test (cheap). Test one idea per `act` with 1 or 2 moves. Pick the move whose result tells your guesses apart, then read the change lines and update the statuses.
4. Solve. When the rules you need are verified, write a search in `ipython` (for example a breadth-first search over the moves you verified) to find the shortest path. Then send the path in `act` calls of up to {act_max} moves."""

RULES = """Rules:
- Do not read, count or copy cells in your head. Ask `ipython` instead: filter `observe().objects`, print a small `obs.ascii(...)` window, or compute distances and find objects with the same hash. One short Python call is faster and more exact than a long thought.
- Keep your thinking short, then call a tool. A reply that is too long is cut off and lost.
- "reset" costs a move and throws away your progress in the level. Use it after GAME_OVER, or when the level is clearly stuck. Do not use it to experiment.
- Every level can be solved. If your search finds no way to the goal, one of your rules is wrong, so test the rule you are least sure about."""

EXAMPLE = """Example from another game:
The board shows `4 R 3x3 r10-12 c20-22`, `9 N 3x3 r10-12 c41-43` and `17 G 1x30 r63 c0-29 hud`.
You call `act` with actions ["4"], plan "test if 4 moves object 4 right", hypotheses [{{"text": "4 moves object 4 right", "status": "proposed"}}, {{"text": "object 17 counts the moves left", "status": "proposed"}}] and goal "move object 4 onto object 9".
The result is `#0 A4(right): obj 4 R 3x3 moved right 3 -> r10-12 c23-25; obj 17 G shrank 30->29 cells (hud)`.
Object 4 moves 3 columns per move and still has 18 columns to go, so it needs 6 more moves. You check the path in `ipython`, mark both hypotheses verified with evidence [0], and send the 6 right moves in `act` calls of up to {act_max} moves. The level is done."""

ENVIRONMENT = """Working directory: {cwd}
Python packages: numpy. `observe` (the current state, read-only) is already imported."""

# =====================================================================================================================
# 3. System-prompt snippets for the plain ipython toolset (the agent makes moves from Python with `arc`)
# =====================================================================================================================

BASE_AGENT_IDENTITY = "\n".join([
    "You are a specialized coding agent that uses code to solve problems.",
    "You solve problems by breaking them down into sub-tasks, writing and executing code, observing results, and "
    "iterating one step at a time.",
    "When you are done, stop calling tools and state your final answer.",
])

IPYTHON_DETAILED = """The `ipython` tool is a Python session that stays alive between calls. Variables, functions and imports you create stay available in every later call, and top-level `await` works.
Use it to look at the game, keep notes in variables, and write small helper functions you can reuse. Run shell commands with `bash("cmd")`, not with `subprocess` or `os.system`.

The game is the module `arc`, which is already imported. These calls are free:
- `obs = await arc.observe()` gives the current board. `obs.grid` is a 64x64 numpy array, read as `obs.grid[row, column]`. `obs.state` is "NOT_FINISHED", "WIN" or "GAME_OVER". `obs.levels_completed` tells you how many levels are done, and `obs.available_actions` lists the legal moves.
- `ts = await arc.transitions()` gives every move so far, oldest first, each with the board before and after. Use it instead of making a move to find out something you have already seen.
- `print(arc.show(grid, x0, y0, x1, y1))` prints a small window of a board, and `arc.diff(a, b)` lists the cells that changed between two boards.
Print short summaries, not whole boards.

Long-term notes: `rlm.harness.create_memory(title=..., content=...)` and `rlm.harness.update_memory(id, title, content)` save notes that stay when old messages are removed, and `rlm.harness.overview()` lists them."""

ROOT_ACT_LINES = """- `obs = await arc.step(a)` makes move `a`, which must be an id from `obs.available_actions`. Move 6 is a click and needs a cell: `await arc.step(6, x=column, y=row)`. Each call costs one move, and `obs.level_up` is True when the move finished a level.
- `obs = await arc.reset()` restarts the current level. It costs one move, the progress in the level is lost, and the moves you already made still count. You need it after GAME_OVER, and it is refused when the level is already at its start.
- When you know the rules but not the steps, write a search in Python (for example a breadth-first search) to find the shortest path before you make the moves.
"""

CAP_LINE = ("- At most {cap} `arc.step` or `arc.reset` calls fit in one `ipython` call. After that you get an error, so "
            "read the results and continue in a new `ipython` call.\n")

# =====================================================================================================================
# 4. Situation messages (user turns or notes sent by the harness at a given moment)
# =====================================================================================================================

TASK_GAME = """Play the ARC-AGI-3 game `{game_id}` and win it. You have at most {max_actions} moves and about {minutes} minutes for the whole game.
The first board is shown below. This is level 1, so nothing is known yet. Do this first:
1. Read the briefing, the board and the object list. Use `ipython` (`obs = observe()`, then `obs.objects`) to check what you think you see.
2. In one `act` call, write your first plan, 2 to 4 hypotheses and a goal guess, and make 1 or 2 test moves."""

TASK_IPYTHON = ("Play the ARC-AGI-3 game `{game_id}` and win it. You have at most {max_actions} moves and about {minutes} "
                "minutes. There is no human, so do not ask questions. Start with "
                "`obs = await arc.observe(); print(obs); print(arc.show(obs.grid))`.")

CONTINUE_GAME = """You ended your turn without calling a tool. Nobody else will reply, so keep playing until the game is won or you run out of moves or time.
Where you are now: {status}.
Next, read your [memory] block and the newest board. If you were waiting for an answer, make the most likely guess yourself and test it. Then call `ipython` to check an idea, or `act` to make your next move."""

CONTINUE_IPYTHON = """You ended your turn without calling a tool. Nobody else will reply, so keep playing until the game is won or you run out of moves or time.
Where you are now: {status}.
Next, look at `await arc.transitions()` to see what your last moves did, make the most likely guess yourself, and continue in `ipython`."""

LEVEL_UP = """LEVEL UP: level {levels_done} of {win_levels} is done. Well played. A new level starts now, and it is shown in full below.
Your memory changed: your verified hypotheses and findings are now lessons, your plan, hypotheses and findings are empty, and your goal is kept and marked as won.
On the new level:
1. Look at the new board before you move. Find what is the same as before (the same colors, the same `#hash`) and what is new.
2. Keep the rules you verified, and do not test them again.
3. New objects usually bring a new rule, so test them first with 1 or 2 moves.
4. Check that your goal still makes sense on this board before you follow it."""

GAME_OVER = """GAME_OVER: this level is lost. Before you do anything else:
1. Find the step that caused it in the change lines. Use `recall` with the step numbers if you need to.
2. Write it into your memory in your next `act`. Refute the hypothesis that led there, or add a finding such as "touching the red block ends the level".
3. Send "reset" in that same `act` to restart the level. Your memory is kept."""

WIN = "WIN: every level is done, and the game is complete."

CUT_OFF = "Your last reply was too long, so it was cut off and lost. Think less this time and call a tool soon."

COMPACTED = """Older messages were shortened to save space. Your [memory] block is complete, and the newest board is shown below. Use `recall` if you need an earlier step."""

ACT_TWICE = ("Your second `act` in the same reply was not run, and no move was spent. Read the new board that came "
             "after your first `act`, then act in your next reply.")

ACT_TOO_MANY = ("You sent {sent} moves, but one `act` takes at most {act_max}. Nothing was run and no move was spent. "
                "Send the first {act_max}, read the result, then send the rest.")

ACT_ILLEGAL = ("{bad} is not legal now, so nothing was run and no move was spent. The legal moves are {legal} (and "
               "\"reset\"). Check the status line before you act.")

ACT_BAD_ARGS = ("{problem}. Nothing was run and no move was spent. A correct call looks like actions [\"1\", \"4\"] or "
                "[\"6 12 40\"] with a plan, for example plan \"test if 1 moves object 4 up\". The full rules are in your "
                "instructions under \"`act` in detail\".")

LOW_BUDGET = ("Note: only {left} moves are left for this game. Do not spend them on tests you can do in `ipython`. "
              "Send only moves that your verified rules say will help.")

LOW_TIME = ("Note: only about {minutes} minutes are left. Stop long tests, and use what you have verified to finish the "
            "level you are on.")

# =====================================================================================================================
# 5. Assembly tables: which snippets go in for which situation
# =====================================================================================================================

SNIPPETS: dict[str, str] = {
    "ROLE_OBJECTIVE_COMMUNICATION": ROLE_OBJECTIVE_COMMUNICATION,
    "GAME_INTUITION": GAME_INTUITION,
    "IPYTHON_BRIEFING": IPYTHON_BRIEFING,
    "LONG_RUNNING_WORK": LONG_RUNNING_WORK,
    "GAME_FACTS": GAME_FACTS,
    "READING_THE_STATE": READING_THE_STATE,
    "TOOLS_BRIEFING": TOOLS_BRIEFING,
    "TOOLS_DETAILED": TOOLS_DETAILED,
    "MEMORY": MEMORY,
    "LEVEL_METHOD": LEVEL_METHOD,
    "RULES": RULES,
    "EXAMPLE": EXAMPLE,
    "ENVIRONMENT": ENVIRONMENT,
    "BASE_AGENT_IDENTITY": BASE_AGENT_IDENTITY,
    "IPYTHON_DETAILED": IPYTHON_DETAILED,
}

# The system prompt per toolset. Order matters: who you are, the game, how to see, how to act, how to remember, how
# to play, the rules, one example.
SYSTEM_PLAN: dict[str, list[str]] = {
    "game": ["ROLE_OBJECTIVE_COMMUNICATION", "GAME_FACTS", "GAME_INTUITION", "READING_THE_STATE", "TOOLS_BRIEFING",
             "IPYTHON_BRIEFING", "TOOLS_DETAILED", "MEMORY", "LEVEL_METHOD", "RULES", "EXAMPLE", "ENVIRONMENT"],
    "ipython": ["BASE_AGENT_IDENTITY", "LONG_RUNNING_WORK", "IPYTHON_DETAILED"],
    "ipython_arc": ["ROLE_OBJECTIVE_COMMUNICATION", "GAME_INTUITION", "IPYTHON_BRIEFING"],
}

# The message per situation, for each toolset.
EVENT_SNIPPETS: dict[str, dict[str, str]] = {
    "game": {"start": TASK_GAME, "continue": CONTINUE_GAME, "level_up": LEVEL_UP, "game_over": GAME_OVER,
             "win": WIN, "cut_off": CUT_OFF, "compacted": COMPACTED, "act_twice": ACT_TWICE,
             "act_too_many": ACT_TOO_MANY, "act_illegal": ACT_ILLEGAL, "act_bad_args": ACT_BAD_ARGS,
             "low_budget": LOW_BUDGET, "low_time": LOW_TIME},
    "ipython": {"start": TASK_IPYTHON, "continue": CONTINUE_IPYTHON, "win": WIN, "cut_off": CUT_OFF,
                "low_budget": LOW_BUDGET, "low_time": LOW_TIME},
}

# The config still calls the game toolset by its old name; both names select the same prompts.
TOOLSET_NAMES = {"e008": "game"}

LOW_BUDGET_MOVES = 20     # add the low-budget note when this many moves or fewer are left
LOW_TIME_MINUTES = 5      # add the low-time note when this many minutes or fewer are left


def _fill(text: str, ctx: dict[str, Any]) -> str:
    try:
        return text.format(**ctx)
    except KeyError as exc:
        raise KeyError(f"prompt needs {exc} (given: {sorted(ctx)})") from None


def assemble_system(toolset: str, **ctx: Any) -> str:
    """The system prompt for ``toolset``, built from ``SYSTEM_PLAN``. Snippets with placeholders are filled from
    ``ctx``."""
    toolset = TOOLSET_NAMES.get(toolset, toolset)
    if toolset not in SYSTEM_PLAN:
        raise ValueError(f"unknown toolset {toolset!r}; known: {sorted(SYSTEM_PLAN)}")
    return "\n\n".join(_fill(SNIPPETS[k], ctx) for k in SYSTEM_PLAN[toolset])


def event_message(kind: str, toolset: str = "game", *, moves_left: int | None = None,
                  minutes_left: int | None = None, **ctx: Any) -> str:
    """The message for a situation. ``kind`` is one of the keys of ``EVENT_SNIPPETS[toolset]``.
    When ``moves_left`` or ``minutes_left`` is low, a short note is added to continue, level_up, game_over and
    compacted messages."""
    table = EVENT_SNIPPETS.get(TOOLSET_NAMES.get(toolset, toolset))
    if table is None:
        raise ValueError(f"unknown toolset {toolset!r}")
    if kind not in table:
        raise ValueError(f"no {kind!r} message for toolset {toolset!r}; known: {sorted(table)}")
    parts = [_fill(table[kind], ctx)]
    if kind in ("continue", "level_up", "game_over", "compacted"):
        if moves_left is not None and moves_left <= LOW_BUDGET_MOVES:
            parts.append(_fill(table["low_budget"], {"left": moves_left}))
        if minutes_left is not None and minutes_left <= LOW_TIME_MINUTES:
            parts.append(_fill(table["low_time"], {"minutes": minutes_left}))
    return "\n\n".join(parts)


# =====================================================================================================================
# 6. Interface used by agent.py and run.py
# =====================================================================================================================

def game_system(*, game_id: str, win_levels: int, act_max: int, cwd: str, vision: bool = True, **_: Any) -> str:
    """System prompt of the game agent (tools ipython, act, recall). ``vision`` says whether pictures are sent."""
    pictures = {"picture_after": ", and a picture of the whole board (only the newest picture is kept)",
                "picture_start": ", and the picture", "full_state": "briefing, objects, board and picture",
                "picture_ids": " and on the picture", "picture_help": PICTURE_HELP} if vision else \
        {"picture_after": "", "picture_start": "", "full_state": "briefing, objects and board", "picture_ids": "",
         "picture_help": ""}
    return assemble_system("game", legend=LEGEND_LETTERS, game_id=game_id, win_levels=win_levels, act_max=act_max,
                           cwd=cwd, **pictures)


def base_prompt(*, cwd: str, transcript: str, **_: Any) -> str:
    """System prompt of the plain ipython toolset."""
    env = (f"Working directory: {cwd}\nConversation log: {transcript}\nPython packages: numpy. `arc` is already "
           "imported. Inspect a module with `help(arc)`.")
    return assemble_system("ipython") + "\n\n" + env


def arc_section(*, game_id: str, win_levels: int, cell_cap: int | None, output_chars: int, **_: Any) -> str:
    """The game part appended to ``base_prompt`` for the plain ipython toolset."""
    facts = (f"You are playing the ARC-AGI-3 game `{game_id}`. It has {win_levels} levels. Tool output longer than "
             f"{output_chars} characters is cut in the middle, so print short summaries.")
    moves = "Making moves (each one costs a move):\n" + ROOT_ACT_LINES + (CAP_LINE.format(cap=cell_cap) if cell_cap else "")
    return "\n\n".join([assemble_system("ipython_arc", legend=LEGEND_NUMBERS), facts, moves.strip()])


def task_message(*, game_id: str, max_actions: int, minutes: int, toolset: str = "game") -> str:
    return event_message("start", toolset, game_id=game_id, max_actions=max_actions, minutes=minutes)


def continuation(*, status: str, toolset: str = "game", moves_left: int | None = None,
                 minutes_left: int | None = None) -> str:
    return event_message("continue", toolset, status=status, moves_left=moves_left, minutes_left=minutes_left)


# Forced reflection (ipython toolset only, off by default).
REFLECT = """[reflection checkpoint: {reason}] ({status})
`arc.step` and `arc.reset` are blocked until you save what you learned. In one `ipython` call:
1. Look at what your recent moves did with `await arc.transitions()`. Do not make a move.
2. Save each fact you have checked with `rlm.harness.create_memory(title=..., content=...)`, or fix a wrong one with `rlm.harness.update_memory(id, title, content)`. Save the ideas that turned out wrong too.
3. If you repeated a procedure, save the helper with `rlm.harness.create_skill(...)`.
Then print `rlm.harness.overview()` and write your next plan in one sentence."""

REFLECT_AGAIN = ("Nothing was saved yet, so `arc.step` is still blocked. Call `rlm.harness.create_memory(...)` or "
                 "`rlm.harness.update_memory(...)` now, in an `ipython` call.")

"""Prompts for SARBLOH, written as small snippets and assembled per situation.

How the game loop uses prompts (read from ``harness/agent/agent.py`` and ``harness/run.py``):

    run.py      -> start message (game start)            : task_message()
    agent.py    -> system prompt, fixed for the game    : game_system()
    agent.py    -> after every move: change lines + new state (built in code, not here)
    agent.py    -> the model ends a turn with no tool    : continuation()
    agent.py    -> level up, GAME_OVER, WIN, refused move, turn budget spent, stuck level, compaction :
                   event_message(kind, ...)

The system prompt is the same text for every game in a run: nothing game-specific (name, level count, folder) is in
it, so the server keeps one cached copy of it (and of the tool list after it) for all games. The game's name and level
count go in the start message, and the status line repeats the level count with every state. It never changes during
a game either (cheap prefix caching). Everything that depends on the moment (a new level, a lost level, a refused move,
a spent thinking budget, low budget) is sent as a short message built by ``event_message``. So "which prompt goes in
when" is the table ``EVENT_SNIPPETS`` plus the if/else in the builders. The messages that start a level (game start,
level up) also get ``LEVEL_NOTE``. After a compaction the message is only "continue".

The system prompt is the API contract only. ``TOOLS_DETAILED`` (one snippet for the one tool, ipython, and the kernel
functions ``observe()``, ``play()``, ``recall()`` and ``graph()``) gives every argument, limit, refusal and return
format, taken from ``tools.py``, ``agent.py`` (``_tool_act``), ``runtime/skills/observation.py``, ``acting.py``,
``recalling.py``, ``codegraph.py`` and ``agent/perception.py``. If those change, change these. The agent's memory is
five variables in its kernel (GOAL, PLAN, HYPOTHESES and FINDINGS, dicts keyed by level, and SKILLS; the host saves them,
``AgentSession._read_memory``), shown in a [memory] block only with config ``memory.in_context``.

Style (owner rule, 2026-10-02, tightened 2026-10-14): short, simple sentences written as instructions to a human. Say
what to do, not how carefully to do it: no "do not rush", "take your time" or "investigate first", because every such
line makes the model think longer before it moves. Exact tool and function names, no stories. The action meanings are
the standard ARC-AGI-3 ones and are stated outright. The score formula is not shown to the agent on purpose (owner
decision 2026-10-02); it is told only that every move counts against its score (owner decision 2026-10-21).

Placeholders are filled with ``str.format``: a literal brace in a template is doubled.
"""

from __future__ import annotations

from typing import Any

# =====================================================================================================================
# 1. Role and the Python tool
# =====================================================================================================================

ROLE_OBJECTIVE_COMMUNICATION = """Role: You are a coding agent playing a multi-level game.

Objective:
Read the board with code and look at the picture. You need to implement your code in python and create your world model to take optimal actions and execute the actions. 

Working:
Here is the thing, your goal is to win the game, to win the game you must take optimal actions(you send the moves with `play()`), to take optimal 
actions you must understand the game from board(Read the board with `observe()`) and image provided and build or update the world model in python 
simulating the game rules, making sure that it plays same moves in the model and the real game to find where they disagree, then taking actions to win game.
Every move you send counts against your score, while thinking and code cost nothing, so send only the moves you need.
Before you build a big model or a search for a goal, test that goal with one move. If the game does not react the way the goal says it should, the goal is wrong: change it before you write more code.
"""

# The colour legend, in the letters the board is written with (``observe()``).
LEGEND_LETTERS = ("W=white, w=light grey, g=grey, G=dark grey, c=charcoal, "
                  "B=black, M=magenta, P=pink, R=red, b=blue, S=sky blue, Y=yellow, O=orange, r=dark red, N=green, "
                  "p=purple.")

# =====================================================================================================================
# 2. System-prompt snippets (tools: ipython with play(), recall)
# =====================================================================================================================

GAME_FACTS = """Game:
You are playing a game having board of 64x64 grid of colors. {legend}
you have to understand the game related rules or goal while playing the game.
Each level keeps what the earlier levels showed and can add something new.
All the games are solvable.
You would be able to win when you understand the game and taking optimal actions to achieve the goal from your understanding.

Controls:
- "1" is up, "2" is down, "3" is left and "4" is right.
- "5" is the special action: it can select, use, rotate, attach or detach something, or carry out a step.
- "6 r c" clicks the cell at row r and column c.
- "7" is undo. It takes back your last move and is not counted as a move.
{reset_control}
you have to take the legal actions particular the game you are playing(in some games actions 7 might not work, or some games are click only). 

What to watch for:
- Some games have no player. What you control may be a cursor, a selected object, a region or the whole board.
- A long bar or a row of small blocks at the edge or side of the screen is usually a move counter or lives. It changes with every move. Do not click through it and do not count its change as progress. Check that the playing area changed.
- When a bar shrinks with every move, count it into your plan: running out loses the level.
- Describe the goal by objects (put the block on the target, match the colors)"""

READING_THE_STATE = """What you see:
1. Every `play()` prints one line per move in your `ipython` output, for example:
   `#12 A1(up): 18 cells changed at r10-15 c20-22; the HUD changed (2 cells)`.
   `#12` is the step number. The line only says how many cells changed and where. `no change` means the move did nothing, and `no change on the board (only the HUD changed, ...)` means only the counter or bar at the edge changed. To see what moved, look at the board or compare two `observe()` calls in your code. `[repeat: same state and action as #k]` means you already made this exact move from this exact board.
2. After a call that made moves, the end of the same output has the new state: a status line, the last step{short_objects}{board_after}.
3. At the start of each level{observe_full}, you get the full state: the status line{briefing}{object_list}{board_start}{picture_start}.{observe_code}
{board_note} {picture_help}
How to read it:
- Positions are rows and columns. `r10-12 c20-22` means rows 10 to 12 and columns 20 to 22. Row 0 is the top, and column 0 is the left.
- An object is one connected area of one color. Its line gives its id, its color letter, its size (`R 3x3` is a full rectangle, `R 7px` is any other shape), where it is, and a `#hash`. The same shape always has the same hash, on any level. `in 2` means it is inside object 2, `has 5,6` means it contains objects 5 and 6, `adj 3` means it touches object 3, and `hud` means it sits on the edge of the screen, where it is often a counter, a bar or lives.
- Object ids stay the same for the whole level{picture_ids}. The most common color is the background and it is not listed."""

PICTURE_HELP = """{picture_where}
The picture is a map of the board. Open floor is light, walls are dark and hatched, and the HUD at the edge is drawn again, larger, on the right. 
Each object has a box and a tag with its id and its guessed role, for example `10: YOU?`. Use the picture to see the layout and understanding, and your code for exact cells.
"""

TOOLS_DETAILED = """
How to use tools:
- Never read, count or copy cells in your head. Filter `observe().objects`, print a small `obs.ascii(...)` window, or compute it.
- You can call ipython tool as many times in one turn, and you see each output right away. Call `observe()` and read the state in the output,
 then write your code and `play()` in the next call. Read what changed and keep going in the same turn.
- Your code stays. Variables, functions and imports live for the whole game, across levels. Write functions you can call again,
 such as `step(state, move)` and a search over it, and fix them when the game shows you something new.

ipython tool:
The tool is a live Python kernel tied to the game and you can implement your code here(python only). 

Apart from implementing python here are certain things which you can use:
observe():
obs = observe() is a frozen copy of the state at the moment it is called. It does not update after moves; a new call to `observe()`, or `r.obs` from `play()`, gives the newer state.
- obs.board is the board as a list of 64 strings, one per row, one color letter per cell. `obs.board[r][c]` is the color at row r, column c. It is plain Python data, so numpy can turn it into an array: `np.array([list(row) for row in obs.board])`.
- obs.objects is a list of dicts, one per object (a connected area of one color). Each has `id`, `letter`, `name`, `size`, `bbox` [r0, c0, r1, c1], `hash` (same shape, same hash), `corners`, `parent`, `children`, `adjacent`, `hud` (True for counters and bars at the edge) and `cells` (a list of (row, column), or None for objects over 256 cells). A list comprehension picks out what matters, for example `[o for o in obs.objects if o["letter"] == "R" and not o["hud"]]`.
- obs.ascii(r0, c0, r1, c1) returns a labelled text window of the rows r0..r1 and columns c0..c1. A window around the objects of interest is a few hundred characters; the whole 64x64 board is about 4,000.
- obs.change is the change line of the newest move. `obs.history` is the last 20 steps, newest first, each a dict with `i`, `level`, `action`, `change` and `state`.
- obs.before is the board before the newest move, in the same form as `obs.board` (None before the first move).
- obs.status holds the level, the moves used and the legal moves; {obs_briefing}`obs.step`, `obs.background` and `obs.legend` complete it.

moves():
`moves()` is every move of this level so far, oldest first, the tries before a GAME_OVER included. `moves(all_levels=True)` is every move of the game. Each is a dict with `i`, `level`, `move` (written as in a `play()` path), `before` and `after` (the boards before and after the move, in the same form as `obs.board`), `change` and `state`. It is free.

frame_diff():
`frame_diff(before, after)` compares two boards and returns what changed as objects: `changed` (the number of cells that differ) and the lists `moved`, `rotated`, `changed_color`, `appeared` and `disappeared`, at most 30 entries each. Each entry has the `letter`, `size` and its boxes [r0, c0, r1, c1]; `moved` also has `by` [rows, columns] and `rotated` has `turn` (degrees clockwise). `frame_diff()` with no arguments compares `obs.before` with the board now. It is free.

play():
`r = await play(path)` is the only call that changes the game. Each move in `path` counts as one action, except undo ("7"), which is not counted. A call without `await` builds a coroutine and makes no move.
- `path` is a list of 1 to {act_max} moves made in order. A move is "1" up, "2" down, "3" left, "4" right, "5" the special action, "7" undo,{reset_in_path} or a click "6 r c" (row first: "6 12 40" clicks row 12, column 40; the tuple (6, 12, 40) is the same). One move is a path of one: `await play(["5"])`.
- The whole path is sent unless one of these happens: the level is won, GAME_OVER, the game ends, {path_stop}. A timer or animation never stops a path.
- Before any move is spent, a path is refused with a `RuntimeError` when it has more than {act_max} moves, or when its first move is illegal or unreadable. Catching or reading that error costs nothing. An illegal or unreadable move later in the path stops the path there: the moves before it are made, the rest are not sent.
- After a level up, further `play()` calls in the same cell are refused, so the new level is seen before anything else is sent.
- `predict` checks the path against your model as it runs. It is a function `predict(board, move)`: it receives `obs.board` before each move and returns the board it expects after it, or a dict {{(r, c): letter}} with only the cells it cares about, or None for a move it does not check. Typically it wraps your step model: `await play(path, predict=lambda b, m: step(b, m))`.
- `state` lets the model work on its own state instead of the board: with `state=my_state_of`, the check compares `predict(my_state_of(obs), move)` with `my_state_of(observe())` after the move, using `==`.
- Cells of `hud` objects are left out of the check, so counters never count as a surprise.
- A wrong prediction does not stop the path. The first one is recorded as the surprise and printed after the path ends: the move, the cells that differ (predicted and actual) and a small window of the board around them.
- The call prints `play: 3 of 3 moves done` (with `; stopped: <reason>` when it stopped early) and one change line per move. A click is written `A6(r12,c40)` and `[GAME_OVER]` marks the move that lost the level. `quiet=True` prints nothing.
- It returns `r`: `r.done` (moves made), `r.stopped` ("level up", "GAME_OVER", {stop_names}, ... or None when the whole path ran), `r.surprise` (None or a dict with `move`, `action` and `cells`), `r.lines` (the change lines), `r.level_up`, `r.state` and `r.obs` (the state after the last move).
- {reset_line}"7" takes back the last move where the game allows it, and is not counted.

recall():
`recall(query, scope)` searches the record of this game and the skills, prints what it finds and changes nothing in the game. It is free.
- `query` picks what to find: "#12" is one step, "12-20" a range of steps, "level 1" a whole level, words such as "red block" or "bfs walls" search the text, and "" gives the latest steps and the list of every skill by name.
- `scope` picks where to look: "timeline" (your steps and what they changed), "skills" (methods kept from earlier levels and games, and the ones listed below), or "all" (the default, both).
- The skills it prints in full are also loaded into `SKILLS`, a dict of skill name to text. A word that is in a skill's name, such as "bfs", loads that skill. It holds at most 5: loading one more drops the oldest. `print(SKILLS["code_bfs"])` shows one again without a new search.
- Long results are cut at a fixed length, so a narrow query returns the whole answer and a broad one returns only its start.

The kernel itself:
- Everything defined in a cell stays for the whole game, across levels: variables, functions, classes and imports. A `step()` model, a search, a helper written on level 1 is still there on level 5.
- `dir()` lists the names defined so far, and `show_source("name")` prints the code of a function or class from an earlier cell.{graph_note}
- Several `ipython` calls fit in one turn, and each output comes back before the next call, so one cell can read and the next can act on what it read.
- Standard Python works: `collections`, `heapq`, `itertools`, `functools` and `numpy` are available to import.

Skills you can load with `recall()`:
- code_step_model, `step(state, move)`: a function that returns the next state the way the game would, used to plan without spending moves. Keep your model updated across 5 parts: (1) Objects (controlled avatar, walls, targets, keys, counters), (2) Actions (what each move does), (3) Goal (winning condition), (4) Unknowns (untested mechanics), and (5) Plan (next moves and reasoning). Every search below runs on it.
- code_bfs (breadth-first search with `collections.deque` and a `seen` set): tries all 1-move paths, then all 2-move paths, so the first path that reaches the goal is a shortest one. Used when you move something to a target around walls and every move costs the same, and to check whether a goal can be reached at all.
- code_flood_fill: spreads from one cell to every cell it can reach. Used to find what is reachable, which areas are closed off, and which cells form one room or one connected shape.
- code_weighted_search `heapq` (Dijkstra, or A* with a distance guess): takes the cheapest state first. Used when moves cost different amounts, such as slides or clicks, or when there are too many states for BFS.
- code_sequence_search (`itertools.product`): tries every list of moves up to a small length and keeps the best by a score. Used when a move changes many objects at once, or to find the best short combination of buttons or clicks.
- code_beam_search : keeps only the best few states at each depth, rated by a score. Used when there are far too many states and you have a good score.
- code_scoring function, `score(state)`: a number for how close a state is to the goal, such as distance to the target or cells that match. Used to compare candidates, in beam search and A*, and to choose between clicks.
- code_probe: one test move whose result tells two of your ideas apart. Used when you do not know what a move does or what wins. Once you know what you control, what each move does and the goal, searching the model is cheaper than probing.
- code_object_tracking: matching objects between two `observe()` calls by `hash`, then color, then nearest `bbox`. Used to find what you control and how far it moved.
- code_frame_diff: the objects that moved, turned, changed colour, appeared or disappeared between two boards. `frame_diff(before, after)` does exactly this, so use it instead of writing your own. Used after `play()` to compare the board with what `step()` predicted.
- code_act_loop: send the path one move at a time and compare each move with `step()`. `play(path, predict=...)` does exactly this, so use it instead of writing the loop. It does not stop at a wrong prediction: the whole path is sent and the first difference is reported at the end. If your `step()` works on your own state, pass `state=` too, because `predict` gets the board by default. Used for any path from your search, trusted or not: a correct model costs nothing extra.
"""


MEMORY = """Your memory:
{earlier_kept} stay in the messages. {memory_where}
Your memory is five variables in your `ipython` kernel. The first four are dicts with one entry per level: "L1" for level 1, "L2" for level 2, and so on. Change them in your code when they change, and they are saved:
- `GOAL`: for each level, a list of the goals you think win it, each written as what wins the level. For example `GOAL["L2"] = ["push the block onto the flag"]`.
- `PLAN`: for each level, your next steps and why, in one or two sentences. For example `PLAN["L2"] = "push the block right"`.
- `HYPOTHESES`: for each level, a list of the rules you think the game follows, each written concretely (objects, directions, counts). For example `HYPOTHESES.setdefault("L2", []).append("1 moves obj 4 up by 5 cells")`.
- `FINDINGS`: for each level, a list of facts you saw, one short text each. For example `FINDINGS.setdefault("L2", []).append("the bar loses one block per move")`.
- `SKILLS`: a dict of skill name to text, the skills you loaded with `recall()` and the ones picked for you when you win a level.
Only change GOAL, PLAN, HYPOTHESES and FINDINGS at the top level of a cell, as in the examples above. Do not rename them, do not copy them into other variables, and do not use them inside your functions or your search: they are notes for you, not inputs to your code. A plain assignment inside a function only makes a local copy and is not saved.
Nothing is emptied when you win a level: the entries of every earlier level stay. On a new level, read them and carry over the goals, plans and hypotheses that still fit.{thinking_note}
"""

# Config keep_reasoning off: the thinking of a reply is not kept in the messages.
THINKING_NOT_KEPT = ("\nYour thinking is not kept. Once a reply ends, only its code, its text and the tool outputs stay in "
                     "the messages, so the next reply cannot read what you thought. Whatever you work out and will need "
                     "later, such as a goal, the plan, a rule or a fact you saw, write it into GOAL, PLAN, HYPOTHESES or "
                     "FINDINGS in the same cell.")

MEMORY_HIDDEN = ("Your memory is not shown in the messages. Print GOAL, PLAN, HYPOTHESES or FINDINGS whenever you "
                 "want to read them, for example `print(GOAL)`.")
MEMORY_SHOWN = ("A [memory] block shows your goal, plan, hypotheses, findings and the names of your SKILLS, and a "
                "[memory update] shows the parts that changed.")

GRAPH_NOTE = ("\n- When a call adds, changes or removes a function or class, its output ends with one `[code graph: ...]` "
              "line: `+` a new one, `~` a changed one, `-` a removed one, and `!` one that calls a name that no longer "
              "exists. These lines are changes, so an older line can be out of date. `graph()` prints the map of your "
              "functions as they are now, each followed by what it calls, and `graph(\"name\")` shows one function "
              "with what calls it.")

PICTURE_LAST = "\nEvery time the board changes, a picture of the new board comes right after the output that changed it."

ENVIRONMENT = """Python packages: `collections`, `heapq`, `itertools`, `functools` and `numpy`. `observe` and `play` are already imported. The working directory of `ipython` is your own folder for this game."""

# =====================================================================================================================
# 3. Situation messages (user turns or notes sent by the harness at a given moment)
# =====================================================================================================================

TASK_GAME = """{levels}You have at most {max_actions} moves for the whole game.
The first state is below. In your first `ipython` call, read `observe()`, find what you might control and what might be the goal, and test that goal with one move. Use what changed to write your model."""

CONTINUE_GAME = "continue"

LEVEL_UP = """LEVEL UP: level {levels_done} of {win_levels} is done. The new level is shown below.
Your GOAL, PLAN, HYPOTHESES, FINDINGS and SKILLS are all kept.
Start from what you learned on the last level. Do not test known actions again. Your functions are still in the kernel.
If in case you encounter new objects or a changed layout update your code accordingly.
"""

NEW_SKILLS = "New skills from the level you just won: {names}."

LOADED_SKILLS = "These skills are now in SKILLS: {names}. `print(SKILLS[\"name\"])` shows one."

# The note for the level the agent is on, added to the messages that start a level.
LEVEL_NOTE = "This is level {level}. Write its notes under \"L{level}\" in your memory."

EARLY_LEVELS = 2   # config knobs with "early"/"later" values use "early" on levels 1 and 2

GAME_OVER = """GAME_OVER: this level was lost. Continue"""

# GAME_OVER when the host could not restart the level (no moves or time left).
GAME_OVER_RESET = """GAME_OVER: this level is lost. send `await play(["reset"])`. Your memory is kept."""

# The reset wording of the system prompt: config reset_move 0 (no reset move) or more.
RESET_CONTROL = "- There is no reset move."
RESET_CONTROL_MOVE = "- \"reset\" restarts the current level."
RESET_LINE = "There is no reset move. "
RESET_LINE_MOVE = "\"reset\" is a move too: it restarts the level and loses its progress. "

# What else stops a play() path: config revisit_stop on, or off (only the wall rule).
REVISIT_PATH_STOP = ("or a move brings back a board already seen since the level started or was last reset (a loop, "
                     "or a move that changed nothing; the 3 rows and columns at each edge are not compared)")
WALL_PATH_STOP = "or the same move twice in a row changes not one pixel (a wall)"
# The wall rule with config wall_edge_band above 0: the edge rows and columns are not compared.
WALL_PATH_STOP_EDGE = ("or the same move twice in a row changes nothing outside the {band} rows and columns at each edge "
                       "(a wall)")

WIN = "WIN: every level is done, and the game is complete."

# Added to the state after moves, every turn.
TURN_NOTE = ("Next: read what changed in code. "
             "If the last moves did not do what you expected, fix the model first.")

# The turn's thinking budget is spent with no move (config turn_yield_tokens). After a reply that ran code:
ACT_NOW = ("Your thinking budget for this turn is spent and you have not moved. Your tool results above are still "
           "valid, so do not start over. Send your moves now with `await play(path, predict=...)`.")
# After a reply that called no tool (or was cut off at the length limit):
CALL_TOOL_NOW = ("You spent this turn's thinking budget without calling a tool. you can use the tools provided")

# The turn after a surprise has a smaller thinking budget (config surprise_yield_tokens). When it is spent with no move:
FIX_NOW = ("A surprise needs a small fix, not a new model. Change the one rule in `step()` that was wrong and send the "
           "new path with `await play(path, predict=...)` now.")


# The level has eaten another config stuck_note_tokens of thinking without being won:
STUCK = ("You have been on this level for a long time. Time is running out")

# A level was won: the code already in the kernel.
REUSE_CODE = ("Your code from the last level is still in the kernel: {names}.")

COMPACTED = "continue"

ACT_TOO_MANY = ("You sent {sent} moves, but one `play()` takes at most {act_max}. Nothing was run and no move was "
                "spent")

ACT_ILLEGAL = ("{bad} is not legal now, so nothing was run and no move was spent. The legal moves are {legal}")

ACT_BAD_ARGS = (""" play() refused this path: {problem}. No move was made and no move was spent""")

ACT_NO_RESET = ("There is no reset move, so nothing was run and no move was spent.")

ACT_NO_RESETS_LEFT = ("The {allowed} resets of this level are used, so nothing was run and no move was spent.")

LOW_BUDGET = "Only {left} moves are left for this game."

LOW_TIME = "Only about {minutes} minutes are left."

# =====================================================================================================================
# 4. Assembly tables: which snippets go in for which situation
# =====================================================================================================================

SNIPPETS: dict[str, str] = {
    "ROLE_OBJECTIVE_COMMUNICATION": ROLE_OBJECTIVE_COMMUNICATION,
    "GAME_FACTS": GAME_FACTS,
    "READING_THE_STATE": READING_THE_STATE,
    "TOOLS_DETAILED": TOOLS_DETAILED,
    "MEMORY": MEMORY,
    "ENVIRONMENT": ENVIRONMENT,
}

# The system prompt carries only the contract the agent cannot discover: the game, how to read the state, the tools,
# the memory and the environment. Advice on how to play is in the skills (``harness/agent/skills/*.md``),
# which the agent reads with `recall`.
SYSTEM_PLAN: list[str] = ["ROLE_OBJECTIVE_COMMUNICATION", "GAME_FACTS", "READING_THE_STATE", "TOOLS_DETAILED", "MEMORY", "ENVIRONMENT"]

# The message per situation.
EVENT_SNIPPETS: dict[str, str] = {
    "start": TASK_GAME, "continue": CONTINUE_GAME, "level_up": LEVEL_UP, "game_over": GAME_OVER,
    "game_over_reset": GAME_OVER_RESET, "win": WIN,
    "act_now": ACT_NOW, "call_tool_now": CALL_TOOL_NOW, "compacted": COMPACTED, "act_too_many": ACT_TOO_MANY,
    "act_illegal": ACT_ILLEGAL, "act_bad_args": ACT_BAD_ARGS, "act_no_reset": ACT_NO_RESET,
    "act_no_resets_left": ACT_NO_RESETS_LEFT, "low_budget": LOW_BUDGET,
    "low_time": LOW_TIME,
    "fix_now": FIX_NOW, "stuck": STUCK, "reuse_code": REUSE_CODE,
}

LEVEL_START_KINDS = ("start", "level_up")   # these get the level note when the level is known
# these get the low-budget and low-time notes
NOTE_KINDS = ("continue", "level_up", "game_over", "game_over_reset", "compacted", "act_now", "call_tool_now", "fix_now")

LOW_BUDGET_MOVES = 20     # add the low-budget note when this many moves or fewer are left
LOW_TIME_MINUTES = 5      # add the low-time note when this many minutes or fewer are left


def _fill(text: str, ctx: dict[str, Any]) -> str:
    try:
        return text.format(**ctx)
    except KeyError as exc:
        raise KeyError(f"prompt needs {exc} (given: {sorted(ctx)})") from None


def assemble_system(**ctx: Any) -> str:
    """The system prompt, built from ``SYSTEM_PLAN``. Snippets with placeholders are filled from ``ctx``."""
    return "\n\n".join(_fill(SNIPPETS[k], ctx) for k in SYSTEM_PLAN)


def reasoning_note(level: int) -> str:
    """The note for ``level`` (1-based): model, search, send the path."""
    return _fill(LEVEL_NOTE, {"level": level})


def event_message(kind: str, *, moves_left: int | None = None, minutes_left: int | None = None,
                  level: int | None = None, new_skills: list[str] | None = None,
                  loaded_skills: list[str] | None = None, **ctx: Any) -> str:
    """The message for a situation. ``kind`` is one of the keys of ``EVENT_SNIPPETS``.
    ``level`` (the level the agent is now on, 1-based) adds the level note to the level start messages,
    ``new_skills`` names the skills written at a level up and ``loaded_skills`` the ones put into SKILLS. When
    ``moves_left`` or ``minutes_left`` is low, a short note is added to the messages in ``NOTE_KINDS``."""
    if kind not in EVENT_SNIPPETS:
        raise ValueError(f"no {kind!r} message; known: {sorted(EVENT_SNIPPETS)}")
    parts = [_fill(EVENT_SNIPPETS[kind], ctx)]
    if kind == "level_up" and new_skills:
        parts.append(_fill(NEW_SKILLS, {"names": ", ".join(new_skills)}))
    if kind == "level_up" and loaded_skills:
        parts.append(_fill(LOADED_SKILLS, {"names": ", ".join(loaded_skills)}))
    if kind in LEVEL_START_KINDS and level is not None:
        parts.append(reasoning_note(level))
    if kind in NOTE_KINDS:
        if moves_left is not None and moves_left <= LOW_BUDGET_MOVES:
            parts.append(_fill(LOW_BUDGET, {"left": moves_left}))
        if minutes_left is not None and minutes_left <= LOW_TIME_MINUTES:
            parts.append(_fill(LOW_TIME, {"minutes": minutes_left}))
    return "\n\n".join(parts)


# =====================================================================================================================
# 5. Interface used by agent.py and run.py
# =====================================================================================================================

def game_system(*, act_max: int, vision: bool = True, picture_last: bool = True, code_graph: bool = True,
                board_text: bool = False, memory_shown: bool = False, short_objects: bool = True,
                observe_view: bool = True, segmentation: bool = True, briefing: bool = True, code_kept: bool = False,
                reasoning_kept: bool = True, reset_move: bool = False, revisit_stop: bool = False,
                wall_edge_band: int = 0, **_: Any) -> str:
    """The system prompt (one tool, ipython, with observe(), play(), recall() and graph()), the same for every game.
    ``vision`` says whether pictures are sent and ``picture_last`` whether the picture of the board now is the last
    message of every request (config ``perception.old_images`` off); ``code_graph`` adds the note on the code-graph
    lines and ``graph()`` (config ``code_graph``); ``board_text`` whether the messages carry the board as letters
    (config ``perception.ascii``); ``memory_shown`` whether a [memory] block is in the messages (config
    ``memory.in_context``); ``short_objects`` and ``observe_view`` say whether the state after moves lists the changed
    objects and whether `observe()` also puts the full state in the messages (config ``perception``); ``reset_move``
    whether "reset" is a move the agent may send (config ``reset_move``); ``revisit_stop`` whether a path stops at a
    board seen before in this try (config ``revisit_stop``); ``wall_edge_band`` how many edge rows and columns the wall
    rule leaves out (config ``wall_edge_band``);
    ``code_kept`` and ``reasoning_kept`` whether stored replies keep their code and their thinking (config ``keep_code``,
    ``keep_reasoning``). Other arguments
    (``game_id``, ``win_levels``, ``cwd``) are accepted and not used: they would make the prompt differ between games."""
    help_text = _fill(PICTURE_HELP, {"picture_where": PICTURE_LAST if picture_last else ""})
    pictures = {"picture_after": ", and a picture of the whole board", "picture_start": ", and the picture",
                "picture_ids": " and on the picture", "picture_help": help_text} if vision else \
        {"picture_after": "", "picture_start": "", "picture_ids": "", "picture_help": ""}
    board = {"board_after": ", that area written as letters", "board_start": ", the whole board written as letters",
             "board_note": ""} if board_text else \
        {"board_after": "", "board_start": "",
         "board_note": "The board itself is not written in the messages. Read cells in code: `observe().board`, "
                       "`obs.objects` or a small `obs.ascii(...)` window.\n"}
    memory = MEMORY_SHOWN if memory_shown else MEMORY_HIDDEN
    views = {"short_objects": " and the objects in the area that changed" if short_objects else "",
             "briefing": ", a briefing" if briefing else "",
             "obs_briefing": "`obs.briefing`, " if briefing else "",
             "object_list": " and the list of objects" if segmentation else "",
             "observe_full": ", after old messages are shortened, and in the output of a call that runs `observe()`" if observe_view else
             " and after old messages are shortened",
             "observe_code": "" if observe_view else " `observe()` gives the same state in your code only."}
    resets = {"reset_control": RESET_CONTROL_MOVE, "reset_in_path": " \"reset\",", "reset_line": RESET_LINE_MOVE} \
        if reset_move else {"reset_control": RESET_CONTROL, "reset_in_path": "", "reset_line": RESET_LINE}
    stops = {"path_stop": REVISIT_PATH_STOP, "stop_names": '"board seen before", "wall hit twice"'} if revisit_stop \
        else {"path_stop": WALL_PATH_STOP_EDGE.format(band=wall_edge_band) if wall_edge_band > 0 else WALL_PATH_STOP,
              "stop_names": '"wall hit twice"'}
    thinking = "thinking, " if reasoning_kept else ""
    earlier = f"Your earlier {thinking}code and tool outputs" if code_kept else \
        (f"Your earlier {thinking}tool outputs (not your code: in its place you see the code graph of the functions and "
         "classes it defined)")
    return assemble_system(legend=LEGEND_LETTERS, act_max=act_max, graph_note=GRAPH_NOTE if code_graph else "",
                           memory_where=memory, **pictures, **board, **views, **resets, **stops, earlier_kept=earlier,
                           thinking_note="" if reasoning_kept else THINKING_NOT_KEPT)


def task_message(*, game_id: str, max_actions: int, minutes: int, win_levels: int | None = None) -> str:
    levels = f"It has {win_levels} levels. " if win_levels else ""
    return event_message("start", game_id=game_id, max_actions=max_actions, minutes=minutes, level=1, levels=levels)


def continuation(*, moves_left: int | None = None, minutes_left: int | None = None) -> str:
    return event_message("continue", moves_left=moves_left, minutes_left=minutes_left)

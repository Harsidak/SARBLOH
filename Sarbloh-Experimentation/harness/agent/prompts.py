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

The system prompt is the API contract only. ``TOOLS_DETAILED`` (= ``IPYTHON_TOOL_DETAILED``, which includes ``play()``, +
``RECALL_DETAILED``) gives every argument, limit, refusal and return format, taken from ``tools.py``, ``agent.py``
(``_tool_act``, ``_tool_recall``), ``runtime/skills/observation.py`` and ``agent/perception.py``. If those change,
change these. The agent writes its memory as four variables in its kernel (GOAL, PLAN, HYPOTHESES, FINDINGS; the host
saves them, ``AgentSession._read_memory``) and reads it with ``recall`` or in the [memory] block (config
``memory.in_context``).

Style (owner rule, 2026-10-02, tightened 2026-10-14): short, simple sentences written as instructions to a human. Say
what to do, not how carefully to do it: no "do not rush", "take your time" or "investigate first", because every such
line makes the model think longer before it moves. Exact tool and function names, no stories. The action meanings are
the standard ARC-AGI-3 ones and are stated outright. The score formula is not shown to the agent on purpose (owner
decision 2026-10-02).

Placeholders are filled with ``str.format``: a literal brace in a template is doubled.
"""

from __future__ import annotations

from typing import Any

# =====================================================================================================================
# 1. Role and the Python tool
# =====================================================================================================================

ROLE_OBJECTIVE_COMMUNICATION = """Role: You are a coding agent playing a multi-level game. Your goal is to win every level with as few moves as possible.

Objective:
Read the board with code and look at the picture. Write what you learn as a Python model of the game, `step(state, move)`. Then search that model for the shortest path to the goal and send it.

Working:
You have to work in observe-plan-act loop in order to win the games observe. you must observe and develop your understanding from the image provided to you
and you build/update your world model and implement the actions or sequence of actions in every call. Compares each new board with what
your model predicted, and update the first move the model got wrong. otherwise keep executing.
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
Earlier levels provide introduce game mechanics later levels combine or use more of these mechanics. 
You would be able to win when you understand the game and taking optimal actions to achieve the goal from your understanding.

Controls:
- "1" is up, "2" is down, "3" is left and "4" is right.
- "5" is the special action: it can select, use, rotate, attach or detach something, or carry out a step.
- "6 r c" clicks the cell at row r and column c.
- "7" is undo. It takes back your last move, it doesn't count as a move.
- "reset" restarts the current level.
you have to take the legal actions particular the game you are playing(in some games actions 7 might not work, or some games are click only). 

How to work:
Here is the thing, your goal is to win the game, to win the game you must take optimal actions(you send the moves with `play()`), to take optimal 
actions you must understand the game from board(Read the board with `observe()`) and image provided and build or update the world model in python 
simulating the game rules, making sure that it plays same moves in the model and the real game to find where they disagree, then taking actions to win game.

What to watch for:
- Some games have no player. What you control may be a cursor, a selected object, a region or the whole board.
- A long bar or a row of small blocks at the edge or side of the screen is usually a move counter or lives. It changes with every move. Do not click through it and do not count its change as progress. Check that the playing area changed.
- When a bar shrinks with every move, count it into your plan: running out loses the level.
- Describe the goal by objects (put the block on the target, match the colors)"""

READING_THE_STATE = """What you see:
1. Every `play()` prints one line per move in your `ipython` output, for example:
   `#12 A1(up): obj 4 R 3x3 moved up 5 -> r10-12 c20-22; obj 17 Y shrank 40->38 cells (hud)`.
   `#12` is the step number. `[repeat: same state and action as #k]` means you already made this exact move from this exact board.
2. After a call that made moves, the end of the same output has the new state: a status line, the last step{short_objects}{board_after}.
3. At the start of each level{observe_full}, you get the full state: a briefing{object_list}{board_start}{picture_start}.
{board_note} {picture_help}
How to read it:
- Positions are rows and columns. `r10-12 c20-22` means rows 10 to 12 and columns 20 to 22. Row 0 is the top, and column 0 is the left.
- An object is one connected area of one color. Its line gives its id, its color letter, its size (`R 3x3` is a full rectangle, `R 7px` is any other shape), where it is, and a `#hash`. The same shape always has the same hash, on any level. `in 2` means it is inside object 2, `has 5,6` means it contains objects 5 and 6, `adj 3` means it touches object 3, and `hud` means it sits on the edge of the screen, where it is often a counter, a bar or lives.
- Object ids stay the same for the whole level{picture_ids}. The most common color is the background and it is not listed."""

PICTURE_HELP = """
The picture is a map of the board. Open floor is light, walls are dark and hatched, and the HUD at the edge is drawn again, larger, on the right. 
Each object has a box and a tag with its id and its guessed role, for example `10: YOU?`. Use the picture to see the layout and understanding, and your code for exact cells.
"""

IPYTHON_TOOL_DETAILED = """The `ipython` tool:
This is the tool you use to read the board, write your understanding of the game as code, and make moves. You give it Python in its one argument, `code`. It runs the code and shows you what the code printed. Top-level `await` works.

How to use it:
- Never read, count or copy cells in your head. Filter `observe().objects`, print a small `obs.ascii(...)` window, or compute it.
- You can call `ipython` many times in one turn, and you see each output right away. Call `observe()` and read the state in the output, then write your code and `play()` in the next call. Read what changed and keep going in the same turn. You have a thinking budget of about 2,000 tokens between moves. When it runs out, run your code and move.
- Long code is fine, even hundreds of lines.
- Print short results: counts, positions, or a small window of the board. Don't print the whole board.
- Your code stays. Variables, functions and imports live for the whole game, across levels. Write functions you can call again, such as `step(state, move)` and a search over it, and fix them when the game shows you something new.{old_code}

What you can call inside it:
1. `obs = observe()` gives you the current state. It is free and costs no move. Colors are letters, and positions are (row, column). An `obs` does not change after you get it, so call `observe()` again after every `play()` (or use `r.obs`). It has:
   - `obs.board`: the board, one string of letters per row. `obs.board[r][c]` is the color at row r, column c.
   - `obs.objects`: a list of dicts with the keys `id`, `letter`, `name`, `size`, `bbox` [r0, c0, r1, c1], `hash`, `corners`, `parent`, `children`, `adjacent`, `hud` and `cells` (a list of (row, column), or None above 256 cells). Filter it, for example `[o for o in obs.objects if o["letter"] == "R"]`.
   - `obs.ascii(r0, c0, r1, c1)`: a window of the board with row and column labels. Print it.
   - `obs.change`: the newest change line. `obs.history`: the last 20 steps, as dicts with `i`, `level`, `action`, `change` and `state`.
   - `obs.briefing`, `obs.step`, `obs.status`, `obs.background` and `obs.legend`.
2. `r = await play(path, predict=None, state=None)` is the only way to make moves. Every move counts. It makes the moves of `path` one at a time and sends the whole path unless the level is won, the game ends, or the same move hits a wall twice in a row.
   - `path` is a list of 1 to {act_max} moves, made in order: "1" to "5", "7", "6 r c" (row first, for example "6 12 40") and "reset". A tuple (6, r, c) works too. For example ["1", "1", "4"] or ["6 12 40"]. One move alone is fine: `await play(["5"])`.
   - Do not forget `await`: without it nothing happens.
   - `predict(board, move)` gets `obs.board` before the move and returns the board you expect after it. It can also return a dict {{(r, c): letter}} with only the cells you care about, or None to not check that move. Without `predict`, no move is checked.
   - If your `step()` works on your own state, pass `state=my_state_of`: then `predict(my_state_of(obs), move)` is compared with `my_state_of(observe())` after the move.
   - Cells of `hud` objects are not compared, so counters do not count as a surprise.
   - A timer or an animation changing never stops the path. Only the same move twice in a row with not one pixel changed stops it, as a wall hit.
   - A move that did not do what you predicted does not stop the path: the first one is reported after the path ends.
   - "reset" costs a move and throws away the level's progress. Use it after GAME_OVER or when the level is stuck.
   - Only the legal moves in the status line work. The call is refused before any move, with nothing spent, if the path has more than {act_max} moves or if the first move is not legal or cannot be read. The refusal is a `RuntimeError` in your code. Fix it and call again.
   - It also stops at a level up, at GAME_OVER, or when the game ends. After a level up `play()` refuses for the rest of that call, so you see the new level first.
   - It prints `play: 3 of 3 moves done` (plus `; stopped: <reason>`) and one change line per move. A click is written `A6(r12,c40)`, and `[GAME_OVER]` marks the move that lost the level. On a surprise it also prints the cells that differ (predicted and actual) and a small window of the board around them: fix that rule in `step()`, then search again from the new board. `quiet=True` prints nothing.
   - It returns `r` with `r.done`, `r.stopped` ("level up", "GAME_OVER", "wall hit twice", ... or None when the whole path was made), `r.surprise`, `r.lines`, `r.level_up`, `r.state` and `r.obs`, the board after the last move.
3. `dir()` lists the names you have already defined.
4. `show_source("name")` prints the code of a function or class you defined in an earlier call. Read it before you change it.

Methods you can use:
- World model, `step(state, move)`: a function that returns the next state the way the game would, used to plan without spending moves. Keep your model updated across 5 parts: (1) Objects (controlled avatar, walls, targets, keys, counters), (2) Actions (what each move does), (3) Goal (winning condition), (4) Unknowns (untested mechanics), and (5) Plan (next moves and reasoning). Every search below runs on it.
- BFS (breadth-first search with `collections.deque` and a `seen` set): tries all 1-move paths, then all 2-move paths, so the first path that reaches the goal is a shortest one. Used when you move something to a target around walls and every move costs the same, and to check whether a goal can be reached at all.
- Flood fill: spreads from one cell to every cell it can reach. Used to find what is reachable, which areas are closed off, and which cells form one room or one connected shape.
- Shortest path with `heapq` (Dijkstra, or A* with a distance guess): takes the cheapest state first. Used when moves cost different amounts, such as slides or clicks, or when there are too many states for BFS.
- Short sequence search (`itertools.product`): tries every list of moves up to a small length and keeps the best by a score. Used when a move changes many objects at once, or to find the best short combination of buttons or clicks.
- Beam search: keeps only the best few states at each depth, rated by a score. Used when there are far too many states and you have a good score.
- Score function, `score(state)`: a number for how close a state is to the goal, such as distance to the target or cells that match. Used to compare candidates, in beam search and A*, and to choose between clicks.
- Probe: one test move whose result tells two of your ideas apart. Used when you do not know what a move does or what wins. Once you know what you control, what each move does and the goal, searching the model is cheaper than probing.
- Object tracking: matching objects between two `observe()` calls by `hash`, then color, then nearest `bbox`. Used to find what you control and how far it moved.
- Board diff: the list of objects that moved, appeared, disappeared or changed size between two `observe()` calls. Used after `play()` to compare the board with what `step()` predicted. If only `hud` objects changed, the move did nothing useful on the board.
- Act loop: send the path one move at a time, compare with `step()` after each, and stop at the first difference. `play(path, predict=...)` does exactly this, so use it instead of writing the loop. Used for any path from your search, trusted or not: a correct model costs nothing extra.

you can access these in detail by using the `recall` tool, scope "skills", and specifying its name: code_step_model, code_bfs, code_flood_fill, code_weighted_search, code_sequence_search, code_beam_search, code_scoring, code_probe, code_object_tracking, code_frame_diff, code_act_loop."""

RECALL_DETAILED = """`recall` in detail:
- `query` (required): "#12" (one step), "12-20" (a range), "level 1", or words such as "red block". "" gives the latest.
- `scope` (optional, default "all"): "timeline" (your steps), "plan", "hypotheses", "findings", "goal", "lessons" (from earlier levels and games), "skills", or "all".
- Scope "skills" or "lessons" gives advice from earlier levels and games.
Long results are cut, so ask for something narrow."""

TOOLS_DETAILED = "\n\n".join([IPYTHON_TOOL_DETAILED, RECALL_DETAILED])

MEMORY = """Your memory:
{earlier_kept} stay in the messages, and old messages are removed from time to time. {memory_where}
Your memory is four variables in your `ipython` kernel. Change them in your code when they change, and they are saved after every call and before every move:
- `GOAL`: text, what you think wins the level.
- `PLAN`: text, your next steps and why, in one or two sentences.
- `HYPOTHESES`: a list of the rules you think the game follows, each written concretely (objects, directions, counts). For example `HYPOTHESES.append("1 moves obj 4 up by 5 cells")`. Write any hypothesis you like and remove one when you stop believing it. You do not mark them true or false.
- `FINDINGS`: a list of facts about this level, one short text each.
The tool output says what was saved. At a level up, the hypotheses that the level proved and your findings become lessons and skills, `GOAL` is kept, and `PLAN`, `HYPOTHESES` and `FINDINGS` are emptied."""

MEMORY_HIDDEN = "Print the variables, or use `recall`, when you need your memory."
MEMORY_SHOWN = ("A [memory] block shows your plan, hypotheses, goal and findings, and a [memory update] shows the parts "
                "that changed. `recall` searches them too.")

OLD_CODE = (" All the code you write lives in your ipython kernel, and it is not shown back to you: once a call has "
            "run, its code is replaced by a note listing what it defined. Those names are still in the kernel, so call "
            "them directly. Start every function and class with a one-line docstring saying what it does and what it "
            "returns: the note keeps that line, so later you know what each one does without reading its code. "
            "`show_source(\"name\")` prints the full code of one when you need to change it.")

ENVIRONMENT = """Python packages: numpy. `observe` and `play` are already imported. The working directory of `ipython` is your own folder for this game."""

# =====================================================================================================================
# 3. Situation messages (user turns or notes sent by the harness at a given moment)
# =====================================================================================================================

TASK_GAME = """Play the ARC-AGI-3 game `{game_id}` and win it. {levels}You have at most {max_actions} moves and about {minutes} minutes for the whole game.
The first state is below. This is level 1, so nothing is known yet. In your first `ipython` call, read `observe()`, find what you might control and what might be the goal, and send probe moves that test it. Use what changed to write your first model."""

CONTINUE_GAME = "continue"

LEVEL_UP = """LEVEL UP: level {levels_done} of {win_levels} is done. The new level is shown below.
The hypotheses the level proved and your findings are now lessons, your goal is kept, and your plan, hypotheses and findings are empty.
Start from what you learned on the last level. Do not test known actions again. Your functions are still in the kernel, so do not write them again: run your `step()` and your search on the new board first.
Look for new objects or a changed layout. New objects usually bring a new rule this level needs: add that rule to `step()` and test it with a few moves. Check whether the goal is still the same.
Then search and send the whole path with `play(path, predict=...)`. Where it surprises you, fix only that rule."""

NEW_SKILLS = "New skills from the level you just won: {names}. Read them with `recall` and scope \"skills\"."

# The note for the level the agent is on, added to the messages that start a level.
LEVEL_NOTE = "This is level {level}."

EARLY_LEVELS = 2   # config knobs with "early"/"later" values use "early" on levels 1 and 2

GAME_OVER = """GAME_OVER: this level is lost. Find the step that caused it in the change lines, fix your model, refute the hypothesis that led there, and send `await play(["reset"])`. Your memory is kept."""

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
STUCK = ("This level has taken about {tokens}k tokens of thinking and is not won yet, so something you believe about it "
         "is probably wrong. Write down the goal and the rules you are assuming, pick the one you have tested least, "
         "and test it with one or two moves. If the level is in a bad state, send \"reset\".")

# A level was won: the code already in the kernel.
REUSE_CODE = ("Your code from the last level is still in the kernel: {names}. Check it on the new board before you write "
              "new code, and fix only what is different.")

COMPACTED = "continue"

ACT_TOO_MANY = ("You sent {sent} moves, but one `play()` takes at most {act_max}. Nothing was run and no move was "
                "spent")

ACT_ILLEGAL = ("{bad} is not legal now, so nothing was run and no move was spent. The legal moves are {legal}")

ACT_BAD_ARGS = ("{problem}. Nothing was run and no move was spent. A correct call looks like `await play([\"1\", \"4\"])` "
                "or `await play([\"6 12 40\"])`.")

LOW_BUDGET = "Only {left} moves are left for this game. Send the moves your model says reach the goal."

LOW_TIME = "Only about {minutes} minutes are left. Use the model you have to finish this level."

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
    "start": TASK_GAME, "continue": CONTINUE_GAME, "level_up": LEVEL_UP, "game_over": GAME_OVER, "win": WIN,
    "act_now": ACT_NOW, "call_tool_now": CALL_TOOL_NOW, "compacted": COMPACTED, "act_too_many": ACT_TOO_MANY,
    "act_illegal": ACT_ILLEGAL, "act_bad_args": ACT_BAD_ARGS, "low_budget": LOW_BUDGET, "low_time": LOW_TIME,
    "fix_now": FIX_NOW, "stuck": STUCK, "reuse_code": REUSE_CODE,
}

LEVEL_START_KINDS = ("start", "level_up")   # these get the level note when the level is known
# these get the low-budget and low-time notes
NOTE_KINDS = ("continue", "level_up", "game_over", "compacted", "act_now", "call_tool_now", "fix_now")

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
                  level: int | None = None, new_skills: list[str] | None = None, **ctx: Any) -> str:
    """The message for a situation. ``kind`` is one of the keys of ``EVENT_SNIPPETS``.
    ``level`` (the level the agent is now on, 1-based) adds the level note to the level start messages, and
    ``new_skills`` names the skills written at a level up. When ``moves_left`` or ``minutes_left`` is low, a short note
    is added to the messages in ``NOTE_KINDS``."""
    if kind not in EVENT_SNIPPETS:
        raise ValueError(f"no {kind!r} message; known: {sorted(EVENT_SNIPPETS)}")
    parts = [_fill(EVENT_SNIPPETS[kind], ctx)]
    if kind == "level_up" and new_skills:
        parts.append(_fill(NEW_SKILLS, {"names": ", ".join(new_skills)}))
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

def game_system(*, act_max: int, vision: bool = True, code_hidden: bool = False, board_text: bool = False,
                memory_shown: bool = False, short_objects: bool = True, observe_view: bool = True,
                segmentation: bool = True, thinking_dropped: bool = False, **_: Any) -> str:
    """The system prompt (tools ipython with play(), recall), the same for every game. ``vision`` says whether pictures are
    sent; ``board_text`` whether the messages carry the board as letters (config ``perception.ascii``);
    ``memory_shown`` whether a [memory] block is in the messages (config ``memory.in_context``);
    ``code_hidden`` adds the note that code is replaced once it has run (config ``hide_code``); ``short_objects`` and
    ``observe_view`` say whether the state after moves lists the changed objects and whether `observe()` also puts the
    full state in the messages (config ``perception``). Other arguments
    (``game_id``, ``win_levels``, ``cwd``) are accepted and not used: they would make the prompt differ between games."""
    pictures = {"picture_after": ", and a picture of the whole board", "picture_start": ", and the picture",
                "picture_ids": " and on the picture", "picture_help": PICTURE_HELP} if vision else \
        {"picture_after": "", "picture_start": "", "picture_ids": "", "picture_help": ""}
    board = {"board_after": ", that area written as letters", "board_start": ", the whole board written as letters",
             "board_note": ""} if board_text else \
        {"board_after": "", "board_start": "",
         "board_note": "The board itself is not written in the messages. Read cells in code: `observe().board`, "
                       "`obs.objects` or a small `obs.ascii(...)` window.\n"}
    memory = MEMORY_SHOWN if memory_shown else MEMORY_HIDDEN
    old_code = OLD_CODE if code_hidden else ""
    views = {"short_objects": " and the objects in the area that changed" if short_objects else "",
             "object_list": " and the list of objects" if segmentation else "",
             "observe_full": ", after old messages are shortened, and in the output of a call that runs `observe()`" if observe_view else
             " and after old messages are shortened. `observe()` gives the same state in your code only"}
    return assemble_system(legend=LEGEND_LETTERS, act_max=act_max, old_code=old_code,
                           memory_where=memory, **pictures, **board, **views,
                           earlier_kept="Your earlier code and tool outputs (and your thinking, when a reply wrote "
                                        "little or no code)" if thinking_dropped else
                                        "Your earlier thinking and tool outputs")


def task_message(*, game_id: str, max_actions: int, minutes: int, win_levels: int | None = None) -> str:
    levels = f"It has {win_levels} levels. " if win_levels else ""
    return event_message("start", game_id=game_id, max_actions=max_actions, minutes=minutes, level=1, levels=levels)


def continuation(*, moves_left: int | None = None, minutes_left: int | None = None) -> str:
    return event_message("continue", moves_left=moves_left, minutes_left=minutes_left)

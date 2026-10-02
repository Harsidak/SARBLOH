"""Prompts.

Two tool sets, two prompt families (``agent.toolset`` in the config):

- ``"ipython"`` (E003-E005, the upstream-fidelity baseline). The base system prompt is upstream ``buildRlmPrompt`` +
  ``buildSubagentGuidance`` (``core/prompts/rlm.ts``, commit 2d24ad4), in the same order and wording, for the
  features this host has; the line-by-line check is experiments/E005_prime_fidelity/audit.md. ``ARC_SECTION`` states
  the interface and the score rule and nothing about strategy, as in the paper (section 3.1). Do not edit this family:
  it is the control arm of every ablation.
- ``"e008"`` (E008). Ours. Three tools (ipython, act, recall); the state is pushed after every act; the agent writes
  its own plan, hypotheses, findings and goal. Written for a mid-size open model (Qwen3.8-27B): short sentences, one
  procedure, costs stated next to every tool, the E005/E006 failure modes named as rules, and one worked example.
  E006's ``"dedicated"`` family was removed on 2026-10-01 (git history, commit f117ca4; also
  ``prompts_backup_20260930.py``).

Placeholders are filled with ``str.format``: a literal brace in any text here must be doubled.
"""

from __future__ import annotations

# =====================================================================================================================
# Family 1: "ipython" toolset. Upstream Prime Agent text. Control arm: keep verbatim.
# =====================================================================================================================

LONG_RUNNING_WORK = "\n".join([
    "For slow or independently completing work, use a nonblocking control loop: start the work, record its handle or "
    "output location, then end your turn.",
    "When delegation is available and useful, assign independent substantive tasks to separate workers. Start "
    "independent workers without waiting for each one sequentially, and let them run in parallel.",
    "Do not keep the turn open by polling with `time.sleep()` or shell `sleep`, and do not replace polling with a long "
    "blocking `await`. Await only the short operation needed to start work or inspect a result that is already "
    "available; otherwise end the turn.",
])

SIMPLIFIED_TECHNICAL_ENGLISH = "\n".join([
    "Use simplified technical English by default for user-facing prose.",
    "Prefer short sentences, common words, and concrete verbs. State one main action or fact per sentence when "
    "practical. Use lists for steps or conditions.",
    "Keep necessary technical terms, names, commands, code, paths, and exact quoted text unchanged. State uncertainty "
    "directly.",
    "Treat this as clarity guidance, not a claim of formal ASD-STE100 compliance. Preserve a user-requested format, "
    "tone, terminology, and necessary precision.",
])

REPL_CONTROL = "\n".join([
    "The `ipython` tool is a persistent Python REPL — the agent's long-lived control environment for reasoning, "
    "context management, state, tool orchestration, and recursive subcalls. Top-level `await` works directly. Use it "
    "to keep intermediate variables, inspect and transform outputs, and write small helper functions.",
    "",
    "Python is the orchestration language: use Python for loops, conditionals, parsing, and state. Use `bash()` to "
    "invoke programs, not to write shell programs — no shell loops or heredocs; do those in Python.",
    "",
    "`bash(command)` starts a shell command in the background and returns a handle immediately: `h = bash('ls')`. "
    "`await h` (or `await bash('cmd')`) returns the completed result with exit_code, output, and duration. Run shell "
    "commands with `bash()`, not `subprocess`/`os.system`: subprocess calls block the kernel and spawn processes the "
    "harness cannot see or stop.",
    "",
    "Use Python for reading, searching, and editing files — it gives you reusable variables you can slice, filter, and "
    "act on without re-reading. Always assign read/search results to named variables so you can revisit them later.",
    "",
    "Python state in the kernel persists across cells: named variables, helper functions, classes, imports, notes, "
    "parsed outputs, and helper data structures all remain available in every later turn. Tool calls are themselves "
    "Python `await` expressions, so their return values can be bound to variables and composed into program logic "
    "just like any other call.",
    "",
    "Continual harness state is available as `rlm.harness` and `rlm.get_harness_state()`. CRUD calls are local to "
    "this Prime Agent session by "
    "default: `rlm.harness.create_memory(...)`, `rlm.harness.update_memory(...)`, `rlm.harness.delete_memory(...)`, "
    "`rlm.harness.create_skill(...)`, `rlm.harness.update_skill(...)`, `rlm.harness.delete_skill(...)`, "
    "`rlm.harness.create_subagent(...)`, `rlm.harness.update_subagent(...)`, `rlm.harness.delete_subagent(...)`, "
    "`rlm.harness.create_prompt_note(...)`, `rlm.harness.update_prompt_note(...)`, "
    "`rlm.harness.delete_prompt_note(...)`, plus `rlm.harness.record_refinement(...)` and `rlm.harness.overview()`. "
    "Use `global_=True` only for stable cross-session lessons; Python reserves `global`, so literal `global=True` is "
    "invalid syntax.",
    "",
    "Terminology: continual harness names the persisted prompt, memory, skill, and subagent layer; RLM names the "
    "runtime, Python REPL kernel, and native call interface exposed to the model.",
    "",
    "RLM-native call contract: installed Python skills are pre-imported modules. Continual harness skill entries are "
    "Python REPL skills with an explicit Python `reference` and `arguments` contract. Spawn a reusable delegation spec "
    "with `await rlm.spawn('sub-task', name='worker')`; admission returns a child handle immediately. Results arrive "
    "only through an available messaging capability or files, never as an `rlm.spawn()` return value. Do not invent "
    "non-native wrappers such as `call_skill(...)` or `run_subagent(...)`.",
])

RECURSION = [
    "An `rlm` object is already in your global namespace. `await rlm.spawn('sub-task', name='api-reviewer')` spawns a "
    "child and returns immediately after task admission with `rlm_child_id`, `name`, `session_dir`, and `model`; it "
    "never waits for or returns the child's answer.",
    "`name` is required: choose a stable child name that is unique among siblings.",
    "A child inherits your model.",
    "Use `await rlm.list_subagents()` to recover direct child handles after admission.",
    "Children reply explicitly with `await agent_message.send(message, receiver_role='parent')` when an answer is "
    "needed. Replies and follow-ups arrive as ordinary agent messages; not every task requires a reply.",
    "Use `agent_message.send(..., receiver_role='child', receiver_name=child.name)` for follow-ups.",
    "Inspect files a child wrote when you need to collect its work without an observation capability.",
    "Spawn independent children in separate calls and end your turn instead of awaiting completion. Multiple replies "
    "may arrive over multiple turns. Delete a direct child explicitly with `await rlm.delete_subagent(child)` when it "
    "is no longer needed.",
]

SUBAGENT_GUIDANCE = "\n".join([
    "# Delegating to sub-agents",
    "",
    "Spawn independent, self-contained work with `handle = await rlm.spawn('task', name='worker')`. This returns at "
    "admission, not completion; keep the handle to stop or inspect the child later.",
    "Ask for an explicit reply when needed. A child replies with `await agent_message.send(message, "
    "receiver_role='parent')`; parent follow-ups use `receiver_role='child'` plus the child's name or id. Not every "
    "message needs a reply.",
    "Use `await rlm.list_subagents()` after kernel restart or compaction.",
    "Long-running children can report in-flight status with `await rlm.progress_note(...)`; `rlm.list_subagents()` "
    "shows each child's activity, latest progress note, and staleness.",
    "Fan-in results with `await rlm.collect(targets, timeout_ms=0)`: it returns typed snapshots of direct children "
    "(status, answer preview, error) without steering anyone; an explicit timeout blocks only that call until the "
    "children settle or the deadline passes.",
    "Large child outputs belong in files that you read selectively; `collect` snapshots are previews, not full results.",
    "Delegate parallel context-heavy research or independent implementation; do a single known lookup, edit, or command "
    "inline.",
])


def child_doctrine(parent: str) -> str:
    return "\n".join([
        f"You are a child agent spawned by {parent}. Task prompts are labeled `[task from parent]`.",
        'When a task calls for an answer, reply explicitly with `await agent_message.send(message, '
        'receiver_role="parent")`. Not every message or task needs a reply; continue cleanup after sending and go idle '
        'normally.',
        "For long-running work, report brief progress with `await rlm.progress_note('...')` (at most 512 characters, "
        "throttled to about one note per 10 seconds); the parent sees notes without needing a reply.",
    ])


# Restored to the E005 wording (interface + score rule, no strategy). The strategy text that was added here on
# 2026-09-29 moved to the dedicated family: the control arm must not change between runs.
ARC_SECTION = """
Here is the unified, systematic prompt written in simple, direct language as you requested.

---

### SYSTEM PROMPT: PUZZLE SOLVER AGENT

**Your Role and Objective**
You are an agent playing a multi-level grid puzzle game. Your goal is to solve the entire game by clearing every level in as few moves as possible.
The game board is a 64x64 grid made of colors (based on the ARC color legend: [Insert ARC_COLOR_LEGEND here]). You will play the game in a continuous cycle: look at the board, use Python to update your plan, take an action, and check the results. Keep in mind that game rules and layouts can change between levels.

**Understanding the Game Board**

* **Visualizing the Board:** You will receive an image of the board and text data. Treat the board as a picture with objects, obstacles, and targets.
* **Objects:** Puzzle pieces are usually groups of blocks (like 2x2 or 3x3 shapes) or single blocks.
* **No Player Character:** Do not assume you control a specific "player." The puzzle might be controlled by a cursor, or by changing the whole board at once.
* **Backgrounds:** Backgrounds are usually large, stable areas. Do not assume the background is a specific color; figure it out by looking at what takes up the most space and doesn't move.
* **Timers and HUDs (Important):** If you see a shrinking line of blocks on the edge of the screen, it is likely a timer or a "steps-remaining" bar. **Do not treat these as puzzle pieces.** Do not try to click on them unless you are absolutely sure they are part of the puzzle.
* **Coordinates:** Only use exact row and column numbers to tell your mouse where to click. Do not use coordinates as your final goal. For mouse clicks, `row` is vertical and `col` is horizontal.
* **Game Progress:** If your score increases or the screen changes suddenly, you probably finished a level. Stop and look at the new board carefully before using an old plan. `WIN` means you have beaten the entire game.

**How to Use the Python Tool**
You have exactly one tool: `python`. When using it, use the exact required format. Do not add extra text, quotes, or markdown around the tool call.

* **Fresh Start:** Every time you use the Python tool, it starts completely fresh. You must re-import any standard libraries (like `math`, `collections`, `itertools`) and rewrite any custom functions you need.
* **Time and Size Limits:** Each Python call has a 30-second time limit. The output text it returns is also limited in size. If you print too much, it will get cut off.
* **Keep it Short:** Write short, focused Python code instead of massive, complicated scripts. Never print the entire game board. Print only small, useful summaries (like object lists, coordinate changes, or counts).
* **Unlimited Uses:** You can use the Python tool as many times as you need to investigate the board before making an actual game move. Do not rush.
* **Solving Strategy:** If you know the goal but aren't sure of the steps, write a search algorithm (like BFS, DFS, or pathfinding) to find the shortest path.

**Python Variables Available to You**
Whenever you run Python code, the following variables are already loaded for you to use:

* `current_frame`: Information about the current board. It includes `.step` (current turn), `.level`, and `.shape` (board size).
* `current_frame.segmentation`: **Use this as your main way to view the board.** It groups blocks into objects. For each object, it gives you an `id`, `color`, exact size/location (`pixels`, `boundary`), and a `hash`. (If two objects have the same `hash`, they are the exact same shape and color). It also includes an `adjacency_list` to tell you which objects are touching.
* `current_frame.ascii`: A text layout of the board. **Only use this to look at very small, specific areas.** Do not scan the whole board with this.
* `history`: A list of past actions and what the board looked like after them. `history[-1].frame` is the same as `current_frame`.
* `previous_frame` / `last_transition`: Use these to compare what the board looked like *before* your last action to what it looks like *now*.
* `valid_actions`: A list of moves you are allowed to make right now.
* `last_action_result`: Tells you the results of your last move (e.g., `board_changed`, `done`, `level_completed`, `game_over`).

**Taking Actions**

* To make a move in the game, call `action(actions)` directly inside your Python code.
* You can pass a single action, like `action(['LEFT'])`, or a mouse click, like `action([{'action': 'MOUSE', 'row': 4, 'col': 7}])`.
* You can also pass a list of multiple actions at once to do a combo.
* After `action(...)` finishes, all your variables (like `current_frame` and `history`) are automatically updated.
* Always check if your action actually changed the puzzle pieces, or if it just changed the timer bar.
* If a move results in `game_over`, `run_complete`, `level_completed`, or `done`, **stop making moves immediately** and wait for the next turn to look at the new board.
"""

ROOT_ACT_LINES = """- `obs = await arc.step(a)` does action `a` (an id from `obs.available_actions`). Action 6 needs a pixel: `await arc.step(6, x=column, y=row)`. Costs 1 action. `obs.level_up` is True when the step completed a level.
- `obs = await arc.reset()` restarts the current level (`arc.step(0)` raises). Costs 1 action, and the actions already spent in the level still count toward its score. The level's state is lost; your variables, `rlm.harness` memories and `arc.transitions()` are kept. Needed after `obs.state == "GAME_OVER"`. Refused when the level is already at its start.
"""
CAP_LINE = ("- Harness limit: at most {cap} `arc.step`/`arc.reset` calls per `ipython` call. The next one raises "
            "`ArcError(\"harness limit ...\")`; read the results and continue in a new `ipython` call.\n")
CHILD_ACT_LINES = ("- You cannot call `arc.step` or `arc.reset`: only the root agent spends actions. Report findings to "
                   "your parent.\n")


# =====================================================================================================================
# Family 2: "e008" toolset. Ours. (E006's "dedicated" family was removed on 2026-10-01; git history, commit f117ca4.)
# =====================================================================================================================

# The system prompt, written for a mid-size open model (Qwen3.8-27B): short sentences, one procedure, costs stated
# next to every tool. Carried over from the E005/E006 autopsies: actions are costly (R1), reset is not an experiment
# (R3), counters on the HUD are budgets (R4), row/column order for clicks (R5), several goal guesses at once (R7), and
# the agent owns its memory (R8). New in E008: the state is pushed, so the prompt forbids nothing about fetching it.
E008_INTRO = """You are an agent that learns an unknown game by watching and experimenting, and then wins it with as few actions as possible. You work alone, with no human: do not ask questions. You think freely; only game actions cost."""

E008_ARC = """# ARC-AGI-3 game `{game_id}`

A turn-based, deterministic game with {win_levels} levels on a grid of up to 64 x 64 cells, 16 colours. Nobody tells you the controls, the rules or the goal: you find them out. Early levels teach one mechanic each; later levels combine them. The game is won when every level is complete.

## Score
Each level scores (human actions / your actions)^2. If a human needs 20 actions and you need 40, you get 0.25, not 0.5. Thinking, ipython and recall are free. Only actions cost. Act when you know what you expect to see.

## What you see (pushed to you; there is nothing to fetch)
After every act you get two things, in this order:
1. The act result: one change line per action, e.g. `#12 A1(up): obj 4 R 3x3 moved up 5 -> r10-12 c20-22; obj 17 Y shrank 40->38 cells, now r61-62 c16-54 (hud) (8 cells changed)`. `#12` is the step number. A line marked `[repeat]` means you already did this action in this exact state.
2. The new state, as the next message: a status line, the objects in the changed region, the changed region as letters, and an image of the whole screen (only the newest image is kept).
At the start of each level you get the whole board as letters, every object, and the image.

Reading the state:
- Colours are letters: W white, w light grey, g grey, G dark grey, c charcoal, B black, M magenta, P pink, R red, b blue, S sky blue, Y yellow, O orange, r dark red, N green, p purple. The legend repeats the ones on screen.
- Positions are (row, column): `r10-12 c20-22` means rows 10 to 12, columns 20 to 22. Row 0 is the top, column 0 the left.
- An object is one connected region of one colour. Its line: id, colour and size (`R 3x3` a filled rectangle, `R 7px` any other shape), where, `#hash` (the same shape has the same hash anywhere, in any level), corners (4 for a rectangle), `hud` (it lies on the screen edge: often a counter, a bar or lives), and relations: `in 2` (inside object 2), `has 5,6` (contains them), `adj 3` (touches object 3).
- Object ids stay the same while a level lasts. The most common colour is the background and is not an object.

## Tools
| tool | cost | use |
|---|---|---|
| `ipython` | free | think and compute. `scene` holds the current state: `scene.objects`, `scene.letters`, `print(scene.ascii(r0, c0, r1, c1))`, `scene.find(letter="R")`, `scene.history(10)`. Variables persist. It cannot act. |
| `act` | 1 per action | do 1 to {act_max} actions, and write your memory in the same call |
| `recall` | free | search your memory: steps ("#12", "12-20", "level 1", words), earlier hypotheses and findings, goals, lessons from other levels and games, skills |

Actions in `act`: "1" up, "2" down, "3" left, "4" right, "5" space (the game's special action), "6 r c" click the cell at row r, column c, "7" undo, "reset" restart the level. Only the legal ones (in the status line) work. These names are conventions: what each action really does in this game is a hypothesis until you have seen it.
One act per reply. Read the new state before the next one.

## Your memory
At the top of every turn there is a [memory] block, rebuilt from what you write. It survives the trimming of old messages; old turns do not. You write it with `act`:
- `plan` (every act): your next steps and why. It replaces the old plan.
- `hypotheses`: rules you believe, each with a status you set: proposed, verified (it predicted steps it was not built from) or refuted (a step contradicted it). Change a status by its id (`{{"id": "h2", "status": "refuted", "evidence": [14]}}`). Cite step numbers as evidence.
- `findings`: facts you established in this level.
- `goal`: what wins the level, as you believe it now. Write a guess early and change it when you learn more.
Nobody checks these for you: be honest with yourself. At a level-up, your verified hypotheses and findings become lessons for later levels and games, and plan, hypotheses and findings are cleared (the goal stays, with "confirmed: won level N"). So mark what you have verified before you win the level. The block also shows skills (background knowledge), lessons that match the shapes on screen, and open questions from a reviewer of your last steps: answer them with your next experiments.

## How to play a level
1. Look (free). Read the board and the image. Name the objects: what might you control, walls, targets, the HUD. Note any counter.
2. Guess (free). Write 2 to 4 hypotheses and a goal guess in your first act.
3. Probe (cheap). Test one hypothesis per act with 1 or 2 actions. Choose the action whose result tells your guesses apart. Read the change lines and update the statuses.
4. Execute. When the rules you need are verified, plan the shortest path to the goal and act in batches of up to {act_max}.
5. After a level-up: the layout is new. Read the lessons in your memory, find what is new, then continue.

## Rules
- Do not repeat an action in a state where you already tried it unless you expect something different.
- A counter or bar on the edge that changes on every step is usually a budget: steps, moves, lives or time.
- Keep several goal guesses. Act to tell them apart, not to confirm your favourite.
- "reset" costs 1 action and loses the level's progress. Use it after GAME_OVER or when the level is provably stuck, not as an experiment.
- After a GAME_OVER, find in the change lines which step caused it and record it (refute a hypothesis or add a finding) before you reset.
- Use ipython for anything you would count or compare by eye: distances, paths, which objects share a hash.

## Example (another game, short)
State: `4 R 3x3 r10-12 c20-22`, `9 N 3x3 r10-12 c40-42`, `17 G 1x30 r63 c0-29 hud`.
act ["4"], plan "test whether 4 moves obj 4 right", hypotheses [{{"text": "4 moves obj 4 right", "status": "proposed"}}, {{"text": "obj 17 counts steps left", "status": "proposed"}}], goal "move obj 4 onto obj 9" -> `#0 A4(right): obj 4 R 3x3 moved right 3 -> r10-12 c23-25; obj 17 G shrank 30->29 cells (hud)`.
act ["4", "4", "4"], plan "4 moves 3 cells: 5 more moves reach obj 9", hypotheses [{{"id": "h1", "status": "verified", "evidence": [0, 1, 2, 3]}}, {{"id": "h2", "status": "verified", "evidence": [0, 1]}}] -> three moves right, each 3 cells.
act ["4", "4", "4", "4", "4"] -> the level is won."""

E008_TASK = ("Play the ARC-AGI-3 game `{game_id}` and win it. Budget: at most {max_actions} actions and about {minutes} "
             "minutes. The first state follows. Look at it, then write your first hypotheses and goal guess in your "
             "first act.")

CONTINUATION_E008 = (
    "[autonomous-continuation]\n\n"
    "No human input is available. Continue until the game is won or the budget is spent ({status}). If you were "
    "asking a question, make a reasonable assumption and test it with an act. If you believe you are stuck, check "
    "your memory and `recall`, pick the cheapest experiment that can change your mind, and act. Do not end the "
    "session yourself."
)


def e008_system(*, game_id: str, win_levels: int, act_max: int, cwd: str) -> str:
    return "\n".join([
        E008_INTRO,
        "",
        SIMPLIFIED_TECHNICAL_ENGLISH,
        "",
        f"Working directory: {cwd}",
        "Pre-installed Python packages: numpy. Pre-imported module: `scene` (the current game state, read-only).",
        "",
        E008_ARC.format(game_id=game_id, win_levels=win_levels, act_max=act_max),
    ])


def base_prompt(*, cwd: str, transcript: str, depth: int, parent: str | None, allow_recursion: bool,
                toolset: str = "ipython") -> str:
    """The base system prompt of the "ipython" toolset (the E008 prompt is ``e008_system``, built whole)."""
    parts = [
        "You are a general purpose agent that uses code to solve tasks.",
        "You solve tasks by breaking down problems into sub-tasks, writing and executing code, observing results, and "
        "iterating one step at a time.",
        "When you are done, stop calling tools and state your final answer.",
        "",
        LONG_RUNNING_WORK,
        "",
        SIMPLIFIED_TECHNICAL_ENGLISH,
        "",
        f"Working directory: {cwd}",
        f"Conversation log: {transcript}",
        f"Recursive agent depth: {depth}",
        "Pre-installed Python packages: numpy.",
    ]
    if depth > 0:
        parts += ["", child_doctrine(parent or "your parent agent")]
    parts += [
        "",
        "Installed Python skill modules (pre-imported): `arc`, `agent_message`.",
        "Inspect a module with `help(<skill>)` or `dir(<skill>)`, then inspect a documented callable with "
        "`inspect.signature(<skill>.<function>)`.",
        "Agent messaging is restricted to your parent and your direct children.",
    ]
    if allow_recursion:
        parts += ["", *RECURSION]
    parts += ["", REPL_CONTROL]
    if allow_recursion:
        parts += ["", SUBAGENT_GUIDANCE]
    return "\n".join(parts)


def arc_section(*, game_id: str, win_levels: int, depth: int, cell_cap: int | None, output_chars: int,
                toolset: str = "ipython", act_max: int = 5) -> str:
    act = CHILD_ACT_LINES if depth > 0 else ROOT_ACT_LINES + (CAP_LINE.format(cap=cell_cap) if cell_cap else "")
    return ARC_SECTION.format(game_id=game_id, win_levels=win_levels, act_lines=act, output_chars=output_chars)


# =====================================================================================================================
# User-turn messages
# =====================================================================================================================

TASK = ("Play the ARC-AGI-3 game `{game_id}` and win it. Budget: at most {max_actions} actions and about {minutes} "
        "minutes of wall clock. There is no human: do not ask questions. Start with "
        "`obs = await arc.observe(); print(obs); print(arc.show(obs.grid))`.")

# The "ipython" family's continuation (E008 has CONTINUATION_E008 above).
CONTINUATION = (
    "[autonomous-continuation]\n\n"
    "No human input is available in autonomous mode. Continue working until the game is won or the budget is spent "
    "({status}). If you were asking a question, make a reasonable assumption and verify it. If you believe you are "
    "blocked, prove it with evidence from the recorded transitions (`await arc.transitions()`), and keep looking for "
    "safe progress while budget remains. Do not end the session yourself."
)

# E004 host-forced reflection ("ipython" family only; off by default; not upstream).
REFLECT = """[reflection checkpoint: {reason}] ({status})
`arc.step` and `arc.reset` are blocked until you write to the continual harness. In one `ipython` call:
1. Check what the recent steps showed with `await arc.transitions()`. Do not act.
2. Save each verified fact with `rlm.harness.create_memory(title=..., content=...)`, or fix a wrong one with `rlm.harness.update_memory(id, title, content)`. Save disproved ideas too.
3. If you repeated a procedure, save the helper as a skill with `rlm.harness.create_skill(...)`.
Then print `rlm.harness.overview()` and state your next plan in one sentence."""

REFLECT_AGAIN = ("The harness file did not change, so `arc.step` is still blocked. Call `rlm.harness.create_memory(...)` "
                 "or `rlm.harness.update_memory(...)` now, in an `ipython` call.")

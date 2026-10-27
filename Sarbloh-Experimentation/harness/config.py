"""Every knob of a Prime Agent run. The notebook passes overrides; the merged config is printed and hashed."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

# Local trace collection for LoRA SFT: `python -m harness.run --local` plays the games with the DeepSeek API
# (harness/llm/deepseek.py) instead of a local server, and saves each run under runs/sft_deepseek/<time>/.
# Never used on Kaggle (no internet there): main() ignores it.
local_SFT_RUN = True

DEFAULT: dict[str, Any] = {
    "experiment": "sarbloh_experimentation",   # the ledger label; the notebook sets the experiment directory name
    "games": None,                     # None = every environment; else ids or id prefixes (offline only)
    "concurrency": 8,                  # games played at once (each is a root session with its own kernel)
    "notebook_budget_s": 3600.0,       # whole notebook wall clock, measured from its first cell
    "teardown_reserve_s": 300.0,       # games are stopped this long before the budget ends
    "game_wall_s": 1800.0,             # cap per game
    "max_actions_per_game": 2000,       # hard action budget per game (host-enforced)
    "stop_after_levels": None,         # end each game after this many levels (local tests); None = play them all
    # ARC SDK recording (Arcade.make(save_recording=True)): one JSONL line per env step, with frames and the agent's
    # step ref (turn, tool call, and on an act's first action its thought and arguments). Off on the competition rerun.
    "record": True,
    # The trace for analysis and LoRA SFT: each game's transcript.jsonl (with a snapshot of the context after every
    # rewrite, the reasoning effort and a digest of every request), the pictures sent (games/<id>/frames/), and after
    # the run harness/trace.py (reports + sft_levels.jsonl). Logging only: the agent's context and requests are the same
    # either way. Off: none of it is written. Always off on the competition rerun (its outputs are never seen).
    "tracing": True,
    # Priority scheduler (harness/scheduler.py). Off: `concurrency` games at a time, each for game_wall_s.
    # On: every game starts at once and only `slots` of them hold the GPU; a game keeps its slot for whole turns until
    # it has made `quantum_calls` requests, then at its next turn end the waiting game with the best (next-level weight
    # x hope) gets it. Each hand-over costs the game one request prefilled in full (a waiting game's prefix is evicted
    # from the server cache), so the cache hit rate is about 1 - 1/quantum_calls. Set game_wall_s to the budget.
    "scheduler": {"enabled": True, "slots": 10, "quantum_calls": 20, "token_scale": 40000.0,
                  "action_scale": 115.0, "human_actions": 25.0, "tail_fade_fraction": 0.2},
    # Before the games start, one 1-token request per reasoning effort with the shared system prompt and tools, so
    # every game's first request (and each one after a drain) finds them in the server's prefix cache.
    "prewarm_system": True,
    # --- agent: one tool, ipython (`observe()` reads, `await play([...])` moves, `recall()` searches the timeline and
    # the skills, `graph()` maps the agent's code; harness/agent/tools.py)
    "agent": {
        "act_max_actions": 100,        # moves per play() path: a whole searched path in one call (max_actions caps the game)
        "allow_fenced_code": True,     # run ```python blocks as ipython when no native tool call came back
        # Run a tool call the model wrote as text when the server did not parse it: every <tool_call> in the reply's
        # text, else the last one in its thinking (the JSON or the <function=...> form). Runs 10-23, 10-26 and 10-26b
        # had 70 replies with <tool_call> text and no tool call, each answered with a note instead of running the code.
        "recover_tool_calls": True,
        # The most one reply may write, thinking and code together: "early" on levels 1 and 2, "later" after, or one
        # int for every level. A reply cut at this cap gets a note to call the tool sooner. run.py caps it at
        # reserve_tokens.
        "max_tokens_per_turn": 12000,
        # One turn of the outer loop is up to this many model requests, sent back to back. While the model only runs
        # Python (looks, computes, reads its own output) the turn goes on: no note, no change of scheduler slot. The
        # turn ends at the first request whose cells moved (await play), or at a cap; then the slot is handed over.
        "max_inner_requests": 12,
        # The thinking cap of a turn: once the replies since the last move have written this many tokens without a
        # move, the turn ends and the slot goes to a waiting game. No note is sent and a reply is never cut. 0 = off.
        "turn_yield_tokens": 16384,
        # The thinking cap of the turn after play() stopped at a surprise: one rule was wrong, so a short fix (never a
        # cut, like turn_yield_tokens). 0 = the normal cap.
        "surprise_yield_tokens": 0,
        # Each time the current level has eaten this many more reply tokens without being won, a note says to question
        # the goal and the least tested rule. 0 = off.
        "stuck_note_tokens": 0,
        # How many times per level the agent may send "reset" in play(); a reset over it is refused (nothing spent).
        # 0: there is no reset move (the 9h run spent 81% of its tokens on levels with a voluntary reset). After a
        # GAME_OVER the host always restarts the level by itself, as the Duck harness does, and that reset is not counted.
        # Run 10-26 had none and the model burned its move bar to restart (ka59 240 moves, g50t 466).
        "reset_move": 3,
        # On: the rest of a play() path is not sent after a move that brings back a board already seen since the level
        # started or was last reset (a loop, or a move that changed nothing); the 3 rows and columns at each edge, where
        # move bars and counters sit, are not compared. Nothing is refused. Run 10-26, counted offline: 2331 of 9913
        # moves would not have been sent, 6 of them winning moves. Off since 10-27: in run 10-26b the model routed round it
        # with one-move play() loops (22% of its moves were sent after it fired), 282 turns reasoned about the "loop
        # detector", and ls20 got slower; the 35-move path that won ls20 level 1 on 10-23 is stopped 9 times.
        "revisit_stop": False,
        # The wall stop (the same move twice in a row and the board unchanged) leaves out this many rows and columns at
        # each edge, where move bars and counters sit; 0 compares the whole frame, so a wall bump that moves the bar never
        # counts. Counted offline on runs 10-23, 10-26 and 10-26b (50,555 moves): it fires 1,493 times instead of 328, and
        # once before a level up later in the same cell.
        "wall_edge_band": 3,
        # On: a play() path that would repeat, from the start of the level (or its last reset), the exact moves of a try
        # that ended in GAME_OVER is refused before the losing move is sent: the whole path when the repeat is in it, or
        # that one move when it comes in a later play(). The try must also start from the same board (edge band left
        # out), so a level whose first state changes between tries is never refused. Runs 10-23, 10-26 and 10-26b,
        # counted offline: 23 exact repeats, 1,137 moves.
        "replay_stop": True,
        # On: the agent's kernel may open and list files only in its own game folder, the memory folder recall()
        # reads, the temp folder and the Python libraries, and may not start other programs (runtime/skills/
        # fileguard.py). Off: the whole file system is readable, including the game files.
        "file_guard": True,
        # The Qwen3.8 chat template's reasoning effort: "xhigh" (its default, a "think carefully" line at the top of
        # the system prompt), "medium" (no line) or "low"; there is no "high". "early" is used on levels 1 and 2
        # (prompts.EARLY_LEVELS), "later" after. None sends nothing (the template's default).
        "reasoning_effort": {"early": "medium", "later": "medium"},
        "request_timeout_s": 900.0,
        "cell_timeout_s": 300.0,
        # The most one cell's output (and its error) may hold: about 1,000 tokens, the Duck harness's cap. A longer one keeps
        # its first two thirds and its last third (the notes at the end survive); 3-4% of the cells of runs 10-23, 10-26
        # and 10-26b were longer.
        "tool_output_chars": 4096,
        "context_window": 131072,      # served context; run.py sets it from the server profile / local server
        # No summary: the context is kept under cap_tokens by dropping old messages in code (no model call), as the
        # Duck harness does. Above trigger_tokens (or window - reserve, if lower) a drain drops the oldest whole turns
        # until the context is under drain_target_tokens; everything kept stays word for word (thinking, boards,
        # outputs), and the first message (the task, or the note of the last reset) is never dropped. If the newest
        # turn alone is over the target, the history is then trimmed back to the newest full state.
        # The counts include the system prompt, the tools and the code graph. trigger_tokens + reserve_tokens (one
        # 12k reply) must fit under cap_tokens.
        "compaction": {"reserve_tokens": 12288, "cap_tokens": 131072, "trigger_tokens": 118000,
                       "drain_target_tokens": 70000},
        # The prompt only grows between rewrites, so the server's prefix cache keeps it (harness/agent/context.py):
        # every thinking stays, the newest picture goes last in every request (never stored, see perception
        # old_images), and the memory is a stored message, in full after a rewrite and every memory_full_every
        # states, else only its changed sections. A drain drops whole old turns and the memory and code-graph
        # messages (a full memory and the whole code graph go back).
        "append_only": True,
        "memory_full_every": 10,
        # The thinking sent back in the current tool-call chain, newest reply first, up to this many tokens (chars/4);
        # older replies keep their tool calls and outputs but lose their thinking. Each turn still thinks freely.
        "chain_reasoning_tokens": 2000,
        # On: a stored reply keeps its code. Off: each ipython call's code becomes a one-line stub (the code graph maps
        # what the code defined, show_source(name) prints it). Decided once, when the reply is stored, so the prompt
        # stays a prefix of the next one. The transcript keeps the code as written. Off failed on 10-25 and 10-27: the
        # model wrote the stub itself as its code (97.7% and 98% of its calls).
        "keep_code": True,
        # Off: a stored reply keeps its code and text but not its thinking, so no old thinking is sent back; the model
        # keeps what it works out in GOAL, PLAN, HYPOTHESES and FINDINGS. Each reply still thinks freely, and the
        # transcript keeps the thinking (trace, SFT export). On: every thinking stays in the messages.
        "keep_reasoning": True,
        # The code graph (harness/runtime/skills/codegraph.py): after every ipython call the host reads the agent's
        # live functions and classes and what each calls, and ends the output with one line of what changed (added,
        # changed, removed, calls a missing name); after a drain, trim or reset the whole map goes back in once.
        # `graph()` prints it in the kernel either way. Off: no change lines and no map in the context.
        "code_graph": True,
        # On: every level-up resets the context to the level-up message and the new level's full state (the whole
        # prompt is prefilled again). Off: both are appended and the old level stays until a drain.
        "level_up_reset": False,
        "max_consecutive_llm_failures": 6,
        "limits": {"max_turns": 400, "max_output_tokens": 3_000_000},
        "vision": False,               # set by run.py: True when the served profile passed the image smoke test
        # harness/agent/perception.py: the full view (briefing, objects, whole board, picture) at a level start,
        # after a compaction and on `observe()`; after an act the short view (changed region) and the picture.
        "perception": {
            "image": True,             # the picture with every observation (only if "vision" is True)
            # Off: no picture is stored in the history; the picture of the board now is the last message of every
            # request, so a new one never changes the cached prefix (every picture is still saved under frames/).
            # On: every picture stays in the prompt where it was sent (prefix-stable, but the prompt grows each turn).
            "old_images": False,
            "cell": 10,                # picture pixels per grid cell: 64x64 -> 640x640 map, plus the HUD panel
            "ascii": False,            # letter board in the message (whole, or the changed region); off: code reads it
            "segmentation": False,     # object list with ids, hashes, containment, adjacency
            # briefing: MEASURED facts and GUESSES, in the full view. It also draws the picture: on, the labelled map
            # (terrain, panels, tile grid, HUD panel) the system prompt describes; off, the plain frame (one colour per
            # cell). Off from 10-19 to 10-25 in Sarbloh, until 10-27 here.
            "briefing": True,
            "briefing_lines": 20,      # most briefing lines shown (the rest: `observe().briefing` in ipython)
            "crop_margin": 3,          # cells around the changed region
            "max_objects": 35,         # object rows shown per observation (the rest: `observe().objects`)
            "observation_tokens": 600,  # cap on the short view's text (chars / 4)
            "short_objects": False,    # the short view lists the objects in the changed region; off: status, last step
            # observe() in ipython also pushes the full view (briefing, every object, picture) as a new message; off:
            # observe() is read in code only and the full view comes at a level start and after a drain
            "observe_view": False,
        },
        "memory": {
            # The memory block (GOAL, PLAN, HYPOTHESES, FINDINGS and the names of the SKILLS loaded) as messages in the
            # context. Off: the agent still has it in its kernel variables; nothing is pushed into the prompt.
            "in_context": False,
            "lessons": True,           # the cross-game lessons graph (read and written only by the curator)
            "curator": True,           # hidden curator: lessons, skills (2 to 5 at every level-up), skills to load
            # The curator always runs at every level-up (before its memory is cleared); above 0 also at the end of a
            # turn once this many requests have gone by since its last pass.
            "curator_every_turns": 0,
            "curator_max_tokens": 4000,
            # The stuck review, once per game: at the first turn end after this much run time (from the notebook
            # start), the agent's turn waits while the curator reads the whole level it is on (every move, the beliefs
            # held and dropped) with thinking on, and its review comes back to the agent as a message. By 4 h most
            # games are stuck (owner, 10-27: ~120 of 183 levels by then). 0 = off.
            "review_at_s": 4 * 3600,
            "review_max_tokens": 24000,   # the review's reply, thinking included
            "recall_tokens": 1000,     # the most one recall() prints (chars / 4)
            # memory block sections, in tokens (chars / 4); a section over its cap is trimmed oldest first
            "caps": {"goal": 150, "plan": 200, "skills": 200, "hypotheses": 400, "findings": 300},
        },
    },
    # --- model ---------------------------------------------------------------------------------------------
    # "gemma" | "qwen" | "qwen_flash_next" | "nemotron" | "deepseek_v4". Picks harness/llm/<name>.py: weights, server profiles, parsers, sampling and
    # thinking policy all come from there (build_config merges them under the overrides below).
    "model": "qwen",
    # Models tried in order when every profile of `model` fails to start (harness/run.py): each entry is
    # {"model": name, "profile_chain": [...] or None for that model's own chain}. Empty: no fallback.
    "model_fallback": [],
    "llm": {
        "base_url": "http://127.0.0.1:8000/v1",
        "request_timeout_s": 900.0,    # whole budget per call, retries and waiting for a restart included
        "retries": 4,
    },
    # --- DeepSeek API (only when local_SFT_RUN is on; key from $DEEPSEEK_API_KEY or the repo .env) -------------------
    "deepseek": {"base_url": "https://api.deepseek.com/v1", "model": "deepseek-reasoner", "send_reasoning": True},
    # --- model server on Kaggle ----------------------------------------------------------------------------
    "server": {
        "wheelhouse_dataset": "banwait13/sarbloh-vllm-wheelhouse",  # kaggle/wheels.ipynb, vLLM 0.19.0
        "port": 8000,
        "profile_chain": None,         # None = the model spec's chain
        # A start fails when the server process exits, or when it is not healthy by startup_timeout_s.
        "startup_timeout_s": 5400.0,   # Kaggle model loads reached 40-60+ min in the 10-22..10-25 cache incident
        "max_inflight": 32,            # requests admitted at once; ramps 2 -> 32 after a watchdog restart
        "bench": True,
        "watchdog": True,
        "watchdog_cfg": {},            # overrides of harness.llm.server.WATCHDOG
    },
}


def merge(base: dict[str, Any], over: dict[str, Any] | None) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def build_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """DEFAULT, then the chosen model's spec defaults (served name, sampling, thinking), then ``overrides``."""
    from harness.llm.spec import get_spec

    if "vllm" in (overrides or {}):  # renamed 2026-10-02; merge() would keep it as a dead key and drop its profile_chain
        raise ValueError('config key "vllm" was renamed to "server"')
    spec = get_spec((overrides or {}).get("model", DEFAULT["model"]))
    base = merge(DEFAULT, {"llm": {"model": spec.served_model_name, **spec.llm}})
    return merge(base, overrides)


def config_hash(cfg: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]

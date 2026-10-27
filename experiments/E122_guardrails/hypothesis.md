# E122 — Guardrails: what the TAAF (Duck) agent enforces, and which of them we lack

- **Date opened:** 2026-10-16
- **Axis varied:** budget / action policy (harness rules around the model, not the model)
- **Baseline:** `Sarbloh-Experimentation/harness` as of commit ed98ced
- **Split:** none yet (catalogue only); a port is measured on the experimentation games first
- **Status:** catalogue written 2026-10-16 at the owner's order. No code changed, nothing run.

## Source

`kaggle/datasets/Taaf agent/src/ARC3-Inference/inference/` (the Tufa Labs Duck bundle, gitignored, licence
UNCONFIRMED). Kaggle values come from the env vars set in `serving_setup.py` (~line 2936) and from
`SOURCE_IDENTITY.json` `run_contract`. Line numbers were read on 2026-10-16. The agent has one `python` tool; the
model's code runs in a sandboxed subprocess and calls `action(actions)` from inside it.

## The guardrails

### 1. Turn loop (`agent/tool_agent.py`)
| # | Guardrail | Kaggle value | What it does |
| --- | --- | --- | --- |
| 1 | Turn yield timer `LOCAL_ANALYZER_YIELD_SECONDS` (~l.1777) | 60 s | After 60 s with no move, the turn ends and control goes back to the solver, which starts a new turn. Checked before each request and after each tool call. Their "stop thinking, act" rule. |
| 2 | Tool-step cap `LOCAL_ANALYZER_TOOL_STEPS` (~l.1783) | 0 = off (12 off Kaggle) | Most requests per turn. The 60 s timer does this job on Kaggle. |
| 3 | Turn ends on the first move (~l.1950-1956) | always | Once a python call has executed a real move, the turn ends; any further tool calls in that reply are dropped. |
| 4 | Nudge on no tool call (~l.1900-1925) | always | A text-only reply gets "You have not acted yet. Investigate first…", restating the method (segmentation, small ascii region, search, then `action()`). |
| 5 | Tool call recovered from text (~l.1867-1870, 1907) | always | `<tool_call>` markup inside reasoning or text is parsed and run; if parsing fails, a retry message asks for exactly one `python` call, no notes. |
| 6 | Malformed JSON arguments (~l.1874-1882) | always | Bad arguments are recorded and the call runs with `{}` instead of crashing. |
| 7 | Failed turn rolled back (`preserve_history`, ~l.1960-2000) | always | On a request error, or several calls after a move, the history goes back to its pre-turn state; the result is marked retryable. |

### 2. Context (`agent/tool_agent.py`)
| # | Guardrail | Kaggle value | What it does |
| --- | --- | --- | --- |
| 8 | Context budget (`_trim_messages_for_context`) | 32,768 window − reply reserve − 512 margin | Before each request, drops the oldest history blocks until the request fits. Never summarises. |
| 9 | Persistent history `_PERSISTENT_HISTORY_ASSISTANT_TURNS` | last 30 assistant turns | Older turns are dropped between turns. |
| 10 | Overflow recovery (~l.1838-1852) | 512 tokens per retry | If the server says "too long", trims 512 more tokens and retries; if nothing was dropped, force-drops one block. |
| 11 | Tool output cap `LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS` (`_trim_tool_text`, ~l.1336) | 1024 tokens | stdout, errors and results are cut, with a "truncated N chars" note. Plus the tool description: use `.ascii` only for a small region. Their answer to our ascii issue. |

### 3. Python sandbox (`agent/python_tool_sandbox.py`)
| # | Guardrail | Kaggle value | What it does |
| --- | --- | --- | --- |
| 12 | Fresh subprocess per call, killed on timeout `LOCAL_ANALYZER_TOOL_TIMEOUT` | 30 s (clamped 1-30) | Wall-clock deadline; the whole process group is SIGKILLed ("Tool timed out after 30s"). Nothing persists between calls. |
| 13 | OS rlimits (`_set_limits`) | CPU = timeout + 1 s, 1 MB file size, 32 open files | Hard resource caps on the snippet. |
| 14 | Import allowlist `SAFE_MODULES` | bisect, collections, copy, fractions, functools, heapq, itertools, json, math, operator, random, re, statistics, string | Anything else: "Module 'x' is not allowed in the sandbox." No numpy, no os. |
| 15 | Builtins allowlist `SAFE_BUILTINS` | no open / eval / exec | `__import__` is replaced by `_safe_import`. |
| 16 | Syntax check | always | `compile()` before running; empty code is rejected. |
| — | No raw grid | always | Frames expose only `.ascii`, `.segmentation`, `.step`, `.level`. |

### 4. Actions (`action()` in `tool_agent.py`, `step_env` in `framework/solver.py`)
| # | Guardrail | Kaggle value | What it does |
| --- | --- | --- | --- |
| 17 | Action validation (`_normalize_python_actions`, `step_env`) | always | Name must be known and in `available_actions`; MOUSE needs `row`/`col` (legacy `x`/`y` rejected); ACTION6 coordinates clamped to 0-63; empty batches and `action`+`actions` together rejected. |
| 18 | Batch stops on events | always | Stops at the first invalid action, error, level completion, GAME_OVER or WIN; reports `stopped_early` + `stop_reason`. |
| 19 | Terminal-state latch | always | After level-up / GAME_OVER / run complete inside a snippet, later `action()` calls in that snippet are refused (`previous_<reason>`, "re-ground on the new scene before acting again"). |

### 5. Game loop and time (`framework/solver.py`)
| # | Guardrail | Kaggle value | What it does |
| --- | --- | --- | --- |
| 20 | `should_stop()` before every turn and move | per-game 7,920 s; `max_actions_per_game` null | Stops on game complete, stop event, per-game runtime limit or action cap. |
| 21 | Request timeout | min(900 s, game time left, time to soft end) | Soft end = 32,400 s overall budget − 600 s reserve. |
| — | Auto RESET on GAME_OVER | always | The harness resets before the next turn; the model never has to. |
| — | Retry backoff | 1 s | Retryable failures repeat the same analysis step. Crashes mark the game crashed and the run continues; cancel drain 120 s. |

### 6. Serving (`vllm_server_watchdog.py`)
- Health check every 15 s; restart after 4 failures; at most 2 restarts; a restart only after the old server is
  proven gone. Concurrency 28, prefix caching off.

## Ours vs theirs (as of ed98ced, `Sarbloh-Experimentation/harness/config.py`)
| Theirs | Ours | Gap |
| --- | --- | --- |
| 60 s turn yield | `turn_yield_tokens` 4092 (a note, reply never cut) | different unit; ours is softer |
| 1024-token tool output cap | `tool_output_chars` 5000 (~1250 tokens) | similar |
| Turn ends on first move | turn ends with a move | similar |
| 30 s cell, fresh process | `cell_timeout_s` 300, persistent kernel | ours 10x longer, state kept (by design) |
| Import / builtins allowlist | none | lacking (numpy is useful to us; not obviously worth porting) |
| Terminal-state latch (#19) | UNCONFIRMED whether `act()` stops a batch after level-up / GAME_OVER | check |
| Drop oldest history freely, no prefix cache | append-only + drain (prefix cache kept) | by design; not portable |
| Tool call recovered from reasoning (#5) | `allow_fenced_code` runs ```python blocks | partial; `<tool_call>` inside reasoning UNCONFIRMED |

## Hypothesis (candidate; UNCONFIRMED until the owner picks which guardrail to port)

Porting the guardrails that remove wasted actions or wasted wall clock — #19 (no moves on a scene the model has not
seen) and #11 (hard cap on board-sized output, see the "ascii issue") — raises levels per hour, because a move made
on the old scene after a level-up is a blind move, and a full-board print costs prefill on every later request.
Guardrails that only harden the sandbox (#13-#15) are not expected to change the score.

Note: §3 of CLAUDE.md forbids gates that hold back an action because a rule is unverified. #19 is not that: it stops a
batch that was planned for a scene that no longer exists. Any port must stay within that principle.

## Prediction

To be committed before a port is run: one guardrail, one number (levels cleared in 45 min on the experimentation
games vs ed98ced).

## Kill criterion

To be committed with the prediction. Default if none is set: kill a port that does not raise levels cleared on the
experimentation games in one 45-minute run at equal tok/s.

## Measurement

- First: from existing transcripts, count (a) actions sent after a level-up / GAME_OVER inside the same `act()` call,
  (b) ipython outputs over 1000 tokens. If both are near zero, there is nothing to port.
- Then: levels cleared and actions per level, 45-minute Kaggle run, experimentation games.

## Port of 2026-10-27 (owner: add the "worth adding" ones to Sarbloh-Experimentation)

Counted offline first (scratchpad `guard_counts.py`) on runs 10-23, 10-26 and 10-26b:
| Guardrail | Config | Count that motivates it |
| --- | --- | --- |
| Replay stop: a path that repeats, from the level start (or last reset) and the same first board, the exact moves of a try that ended in GAME_OVER is refused before any move is spent | `replay_stop` True | 23 exact repeats, 1,137 moves (16/632, 5/305, 2/200) |
| GAME_OVER note asks why (the [GAME_OVER] move, `moves()`, a bar running out); no restart wording (run 10-26 bar burning) | text | 457 GAME_OVERs (322, 70, 65) |
| No-tool reply gets "call ipython now" instead of "continue" (#4) | text | continuations 136 and 105 on 10-26/10-26b |
| `<tool_call>` written as text runs (#5): all in the reply text, else the last one in the thinking | `recover_tool_calls` True | 70 replies (42, 9, 19) |
| Tool output cap 8000 -> 4096 chars (#11), head 2/3 + tail 1/3 kept | `tool_output_chars` 4096 (config + 3 notebooks) | 3-4% of cells over 4096 |
| Click row/column outside 0-63 refused (#17) | always | not counted |

Test: `tests/unit/test_game_over_guardrails.py` (28 checks; sp80 level 1 lost by FIRE x5 on the local copy). Suite 29/37,
the same 8 stale failures as before. Not run on Kaggle. Kill criterion: the default above (levels cleared at equal tok/s).

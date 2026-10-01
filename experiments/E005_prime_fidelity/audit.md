# E005 — Port-fidelity audit: `Sarbloh-Prime/prime/` vs upstream Prime Agent (2026-09-28)

Upstream: `resources/repos/prime-agent`, commit 2d24ad4 (MIT). The harness is TypeScript (`packages/coding-agent`,
~151k lines); only the Python kernel `rlm/` is upstream code in our tree (byte-identical; `mcp.py`, `mcp_base.py`,
`skill.py` dropped). `prime/` is our Python port of the host. This table compares each host behaviour, line by line,
with the upstream file it claims to follow. "Before" is the E004 code (commit 35dcaec); "Now" is this change.

## L1 — context and compaction (`core/compaction/compaction.ts`, `utils.ts`, `agent-session.ts`)

| Behaviour | Upstream | Before (E004) | Now |
|---|---|---|---|
| Trigger | context = last usage (prompt+output) + chars/4 of later messages; compact when > window − `reserveTokens` (16384) | last prompt size ≥ fixed `compact_at_tokens` (96k Kaggle, 12k local) | upstream formula; estimate scaled by the measured token ratio (see "found in testing") |
| What survives | newest `keepRecentTokens` (20000) kept **verbatim**; cut only at a user/assistant message | **only the first user message (task) + a summary**; every recent observation dropped | ported (`compaction.find_cut_point`) |
| Summarizer | separate call, `SUMMARIZATION_SYSTEM_PROMPT` ("do NOT continue the conversation"), conversation serialized as text, fixed Goal/Progress/Next Steps/Critical Context format, kernel-persistence note, thinking off | our own prompt appended to the live chat (agent system prompt + tools context), free-form | ported verbatim |
| Later compactions | update prompt merges into `<previous-summary>` | old summary was just another message | ported |
| Split turn / recency anchor | a cut inside a turn gets a second "turn prefix" summary; newest kept assistant text anchors the summary | none | ported |
| After a threshold compaction | an `[autonomous-continuation]` message is queued (autonomous mode) | none: next call saw task + summary only | ported |
| Overflow | one compact-and-retry, then stop | unlimited retries | ported |
| Summary message | user message `[compaction-summary] … retained messages below are authoritative …` with the harness digest attached mechanically | `[compaction #n] Earlier messages were replaced …` | ported |

The "before" column explains the E004 finding "5/5 post-compaction windows start with a reset": after compaction the
model saw the task text and a summary, nothing it had just done, so it restarted like a new game. **Caveat (found
later, see "Found in testing" 2):** the 4B model also resets before any compaction (`for a in range(5): reset();
step(a)`), and `arc.step(0)` was a silent RESET, so reset counts from E004 and the v1 smoke do not isolate compaction.

## L3 — Continual Harness (`core/refinement/refinement.ts`, `messages.ts`, `agent-session.ts`)

| Behaviour | Upstream | Before (E004) | Now |
|---|---|---|---|
| Digest delivery | `[harness-digest]` user message on the first turn and on every compaction head; system prompt stays fixed | re-rendered into the **system prompt every turn** | ported |
| Digest format | `formatHarnessStateForPrompt`: fixed header, ≤6 entries/kind ranked by relevance to the last 4 messages, 180 chars each, last 5 refinements | prompt notes in full + `overview()` text, 6000-char cap | ported (query terms: Latin/digits only, no CJK bigrams) |
| Model told about refine edits | `[auto-refinement]` notice listing applied edits | **no** | ported |
| Auto-refine default | **on**: every 25 turns + after compaction, 20 min cooldown | off (local CLI flag only) | on |
| Refine system prompts | — | claimed "verbatim"; **was not**: 5 lines dropped from the planner prompt (terminology, 3 global-scope rules, metadata), subagent line cut, one line of ours added; review prompt missing its global-scope sentence | verbatim |
| Refine inputs | `serializeConversation` (numbered calls, tool results ≤2000 chars), merged global+local overview (40/kind, 240 chars), history format, scope text | raw `[role]` dump, local-only overview | ported |
| Auto-refine instructions | includes "Do not promote anything global unless explicitly requested." | missing | fixed |
| Timing | headless runs (`serializedRefine`): between turns, before threshold compaction | between turns | unchanged (already matched); also at turn end |

## L2 — REPL, subagents, messages (`core/prompts/rlm.ts`, `system-prompt.ts`, `messages.ts`, `agent-messages.ts`)

| Behaviour | Upstream | Before (E004) | Now |
|---|---|---|---|
| Base system prompt | `buildRlmPrompt` | condensed paraphrase | upstream wording and order; omissions listed in `prompts.py` docstring. Checked line by line against the TS source (test `base_prompt_lines_verbatim_*`): every line verbatim except 9 allowlisted ones (5 interpolated: depth, cwd, log path, packages, skill modules; 4 documented cuts/changes: `bash` handle methods, SKILL.md clause, child-parent name, parent+children-only messaging). The check found `rlm.get_harness_state()` cut without reason; restored |
| Delegation block | `buildSubagentGuidance` appended | missing | included |
| Message headers | `[task from parent]\n\n…`, `[agent-message from child:x]`, `[child-exited: no-reply child:x]`, `[child-failed child:x]` | `[message from child x] finished (…). Final answer: …` on every exit | upstream |
| Subagent keep-alive | `[autonomous-continuation: subagent-keep-alive]` text | ordinary continuation | ported |
| `ipython` tool schema | upstream description | ours | upstream (minus session-revive and project-env clauses) |
| Tool output cap | 65,536 chars per stream, head kept | 6000 chars, head+tail | **unchanged deviation** (small windows) |
| Kernel wire protocol | `repl-manager.ts` | reduced port | unchanged |

Not ported (unchanged): daemon/worker/TUI, goals, heartbeats/cron, `create_session`, MCP, model switching, crash
recovery / kernel snapshots, `bash()` completion follow-ups, agent-callable `refine.run()`, file-op lists in
summaries, auxiliary-model routing.

## ARC layer (ours, not upstream)

| Item | Before | Now |
|---|---|---|
| Where the interface lives | first user message, pinned through compaction (not upstream) | system prompt (upstream `appendSystemPrompt` mechanism); task message is 62 tokens |
| RESET | "restarts the current level"; `arc.step(0)` = RESET and 0 listed in `obs.available_actions` | cost 1 action, loses the level's progress, only after GAME_OVER or proven dead end; only `arc.reset()` resets (own host request); `arc.step(0)` raises; 0 not in `available_actions` |
| `arc.diff` | grids only (3 `TypeError`s in the v1 smoke from passing observations) | grids or observations |
| `arc.transitions()` | "(before grid, action, x, y, after grid, …)" — read as a tuple; 8 `t[0]`-style errors in E004 | dict with named keys + example; `t[0]` now raises an error naming the keys; `t.after` works |
| "cell" | meant grid square in the ARC text and code block in the cap message | grid square = "pixel"; code block = "ipython call" |
| Cap / reflection messages | `cell action cap reached …` (read as a game rule, filed under "Falsified Hypotheses") | "harness limit: … This is a harness rule, not a game rule." Both mechanisms off by default |
| Strategy doctrine | predict()/simulator/plan rules | removed: paper gives "only the environment interface and an autonomous prompt" |

Prompt size (chars/4): root system prompt ~2,340 tokens (base 1,744 + ARC 593), fixed per session; task 62; digest
~700 on the first turn and in compaction heads. Before: ~1,230 base + 693 task + digest (≤~1,500) in every turn's
system prompt.

## Found in testing (2026-09-28)

1. First local smoke (`runs/prime_local_E005_smoke_v0_estimate_bug`): the session ended after 4 turns with
`context_overflow`. Upstream's chars/4 estimate assumes English. ARC grids are digit text, and the Qwen tokenizer
gives every digit its own token (Gemma is assumed to do the same; UNCONFIRMED), so a printed 64×64 grid is ~4k real
tokens against a ~1k estimate. The cut-point walk never reached 4k "estimated" tokens, found nothing to summarize,
and the request overflowed. Fix: every estimate is scaled by the ratio measured on the last call (real prompt
tokens / estimate), clamped to [0.5, 4].

2. Second local smoke (`runs/prime_local_E005_smoke_v1_step0_reset_trap`, Qwen3.5-4B, ls20, 16k window, 150
actions): the harness ran clean: 9.4 min, 28 turns, ended on `action_budget`, 0 crashes, 0 LLM failures, 9
compactions (0 failed; scale 1.35-2.35; 2-19 messages kept verbatim), 2 auto-refine reviews (2 memories applied, both
`[auto-refinement]` notices delivered), 0 tuple-indexing errors, and the model read `t['action']` after compaction.
But **100 of 150 actions were RESET**. From turn 2 the model tested actions as `for action in range(5): await
arc.reset(); await arc.step(action)`, and `arc.step(0)` was itself a RESET, because the SDK id 0 is RESET and it
was listed in `obs.available_actions`. So each loop spent 6 resets for 4 real actions. The trap is in our ARC layer,
not upstream. Fixed (see the ARC layer table) and pinned by tests `step_0_refused`, `available_actions_hide_reset`,
`reset_request_spends_one_action`, `kernel_step_0_refused`. Compaction fired every ~3 turns only because the local
window is 16k and each printed 64×64 grid is ~4k real tokens; at 131k on Kaggle it is not expected at 300 actions.

3. Third local smoke with the RESET fix (`runs/prime_local_E005_smoke`, Qwen3.5-4B, ls20, 16k window, 8 min, 60
actions): 8.1 min, 28 turns, stopped on time, 0 crashes, 0 LLM failures (40 calls), 7 compactions (0 failed), and the
first acting call after each of them was a `step`, never a reset. 2 auto-refine reviews, 5 memories applied. **48
actions, 0 RESET.** The model still tried `arc.step(0)` once; the guard refused it and no action was spent. 44 of the
48 actions were ACTION3, in loops of 10, because the model decided the colour-11 bar on rows 61-62 was a wall that
ACTION3 clears (probably the step-counter HUD; UNCONFIRMED). That is model behaviour, not harness behaviour, and a 4B
model's play is not evidence for Gemma. The illegal-action error still listed `0` as legal; fixed, test
`illegal_action_error_hides_reset`.

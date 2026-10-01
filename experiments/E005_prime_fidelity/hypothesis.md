# E005 — Faithful Prime host + precise ARC interface (Gemma-4-31B, Kaggle, 1 h)

- **Date opened:** 2026-09-28
- **Axis varied:** harness fidelity (L1 compaction, L3 digest/refine delivery) and the ARC interface text. See
  `audit.md` for every change, row by row, against upstream commit 2d24ad4.
- **Baseline:** E003 v2 (Gemma-4-31B, Kaggle, same 6 games, 300 actions, 1 h): 0 compactions, 0 spawns, 0
  `rlm.harness` writes, 85% of actions in ≥10-action code blocks. E004 (local Qwen 4B): every post-compaction window
  began with a reset; 8 tuple-indexing errors on `arc.transitions()`; the cap message was filed as a game rule.
- **Split:** Kaggle OFFLINE public games (`ls20 vc33 ft09 sp80 tr87 lp85`). Public-set score is a smoke number, not
  evidence (CLAUDE.md §4.3). This run tests the harness, not the method.

## Mechanism
1. Compaction as upstream: keep the newest 20k tokens verbatim, separate structured summarizer, iterative update,
   `[autonomous-continuation]` after a threshold compaction. The agent keeps its recent observations.
2. Harness digest as a message (first turn + compaction head), `[auto-refinement]` notices, auto-refine on by
   default (25 turns / after compaction / 20 min cooldown), refine prompts verbatim.
3. ARC interface in the system prompt: RESET cost stated, `transitions()` keys named, "pixel" vs "ipython call",
   no strategy doctrine. Cell cap and reflection lock off.

## Prediction (committed before the Kaggle run)
1. 0 session crashes; `compaction_error` and `refine_errors` each ≤ 1 per game.
2. Every game that reaches 25 turns gets ≥ 1 auto-refine review; ≥ 3 of 6 games end with ≥ 1 applied harness entry.
3. 0 `KeyError: 0`-style tuple-indexing errors on transitions.
4. RESET share of all actions ≤ E003 v2's 3.4% (61/1800, counted from
   `runs/kaggle_sarbloh-prime_v2_20260928_011233/prime_run/results.json`; includes resets after GAME_OVER).
5. Compaction: only in games whose context passes ~114k tokens (131k window − 16k reserve). E003 reached ~40k at 300
   actions, so with 300 actions per game **no compaction is expected**; L1 is then untested on Gemma.
6. Score: not predicted.

## Kill criterion
Any session crash traced to `prime/compaction.py`, `prime/refine.py` or the new session code, or refine/compaction
errors in > 50% of attempts: stop, fix, rerun locally before any other experiment.

## Measurement
`prime_run/games/*/transcript.jsonl` events (`compaction`, `auto_refine`, `crash`), `session.json` counters,
`results.json` run history (RESET count, GAME_OVER positions), `harness/harness_state.json` per game.

## Amendment (2026-09-28, before the Kaggle run)
The local v1 smoke found that `arc.step(0)` was a silent RESET (see `audit.md`, "Found in testing" 2); it is fixed
in the code this run uses. E003 v2's 3.4% RESET share (prediction 4's baseline) may include such silent resets, so
prediction 4 is a weak comparison; also report RESETs that follow a GAME_OVER separately from the rest.

## Amendment 2 (2026-09-28, before the Kaggle run): 10 games, step trace, trigger 40k, reset guard
Owner order: run on 10 games and make every agent step traceable with the official tools.

- **Games (DEV split only, stratified):** keyboard `ls20 tr87 wa30`, click `lp85 tn36 su15`, keyboard+click
  `sp80 tu93 sc25 bp35`. `vc33` and `ft09` are dropped because they are HELD-OUT (`eval/splits/heldout_split.json`),
  and this run's traces are meant to be read step by step, which the held-out protocol forbids.
- **Budget:** 1 h notebook, 10 games at once, 500 actions per game, so the hour rather than the action cap
  should end most games (E003 v2 spent its 300-action cap in 14 min).
- **Code since the predictions above:** compaction also fires above 40k tokens (`trigger_tokens`, keeping the newest
  16k). `arc.reset()` is refused when no action has been taken since the level began or was last reset. RESET is
  gone from `available_actions`.
- **Tracing (instrument, not mechanism):** the ARC SDK's own recording is on (`Arcade.make(save_recording=True)`),
  giving one JSONL line per environment action, with frames. Every action carries a `reasoning` ref (session, turn,
  tool-call id, index within the cell); the first action of a cell also carries the model's thought, text and code.
  The SDK's per-game scorecard is saved, and `prime.trace` builds a per-game `steps.md` step map from the
  transcript plus the recording.

### Predictions (replace 4 and 5 above; 1 to 3 stand)
4'. RESET share ≤ 3.1% of actions: E003 v2 on the 4 DEV games it shares with this run (`ls20 sp80 tr87 lp85`) was
    37/1200.
5'. Compaction fires at least once in ≥ 5 of 10 games (E003 v2 peaked at ~40k context by 300 actions).
    `compaction_error` ≤ 1 per game.
6'. Trace completeness: 10/10 games have an SDK recording whose action count equals the harness history. Every
    non-initial recording line carries a turn + call ref. `steps.md` exists for 10/10 games.
7'. KV fit: no vLLM preemption at 10 concurrent. E003's log gave 118,912 KV tokens, "6.39x at 131k", so
    10 × (40k + 16k) fits. UNCONFIRMED until measured.

# E110 — Prompts that make thinking short: compute in code, batch the moves, carry levels over

- **Date opened:** 2026-10-01
- **Axis varied:** action policy / budget (prompt only)
- **Baseline:** E008 arm A, Kaggle run `banwait13/sarbloh-prime` v354374653 (2026-10-01, 17 public dev games, vLLM).
- **Split:** none. Public dev games; a smoke signal, not a held-out result.

## In simple words

The model is like a child doing sums in its head when it has a calculator on the desk. In today's run it read the
board letter by letter in its head ("position 0 is a dot, position 1 is a dot, ... position 17 is a dot") for
thousands of words, before every single move. That is slow: about 4 minutes per move. Sometimes it talked so long
that it was cut off and the whole turn was thrown away.

The new prompt tells it three things, in plain words:
1. **Use the calculator.** Counting, comparing and finding paths go in code (ipython), not in your head. Code is
   faster than thinking.
2. **Thinking is slow and the clock is running.** Every 1,000 words of thinking cost about half a minute. Think
   short, then act.
3. **When you know the way, take several steps at once**, and after a level-up, start from what you already
   learned instead of from zero.

## Evidence that motivates it (today's run)

- 2,680,625 completion tokens for 421 actions: **6,367 tokens per action**. Median reasoning 15,600 characters per
  turn.
- **37 of 392 turns (9.4%) hit the 16,384-token output limit** (`finish: length`): the whole turn is lost.
- The ls20 trace shows the model transcribing grid rows by hand ("Position 0: c14 = . … Position 17: c31 = .").
- 194 act calls for 421 actions: **2.2 actions per act** of the 5 allowed.
- Level-1 wins came after 10k-149k output tokens (median ~47k).

## Mechanism

Switch `agent.prompt_version` (default `"e008"` = unchanged control text). With `"e110"`, `prompts.e008_system`
edits the E008 system prompt in place (each replaced line is asserted present once, as E021 does):
- **Score section** gets a time paragraph: decode speed per game (~30 tokens/s when games share the GPU), what a
  long thought costs in minutes, that a reply cut at the output limit is lost, and a target of a short thought per
  turn.
- **Rules** get: never read, count or compare cells in your thinking (do it in ipython: `scene.find`, distances,
  a BFS over verified moves); when the controls and the goal are known, compute the path in ipython and send it as
  one act of up to N actions; an edge strip that changes on every step is a budget, never click its segments; the
  game is solvable, so a search that finds nothing means one rule is wrong: test the most doubtful one.
- **The level-up line of the act result** (agent.py) gets the carry-over instruction: start from the mechanics you
  verified, look for elements you have not seen, test those first, then check whether the goal still holds.
- **The length cut-off message** says what was lost and asks for code instead of thought.
The wording is ours. The ideas come from reading the Taaf agent prompts (`kaggle/datasets/Taaf agent/`,
`ARC3-Inference/inference/agent/prompts.py`) and the milestone-2 reference patch; their text is not copied (licence
UNCONFIRMED).

## Why it should work

The agent loses on the clock, not on the actions (it beats the human count on most levels it wins). Every token not
spent re-reading the board by hand is a token spent on the next move. Code answers counting questions in one short
tool call (a few hundred tokens) where hand-reading takes thousands.

## Prediction

Committed before the run, on the same 17 games as the baseline:
- completion tokens per action ≤ 4,000 (baseline 6,367);
- turns ending at the output limit ≤ 3% (baseline 9.4%);
- actions per act call ≥ 3.0 (baseline 2.2);
- levels completed ≥ 10 (baseline 10), RHAE on the won levels not lower.

## Kill criterion

Committed before the run. Binding.

> Kill if completion tokens per action do not drop by at least 25% (≥ 4,775), or if levels completed drop below 8
> on the same 17 games. If the run also changes the server or the scheduler (E109, E111), only the per-action and
> per-turn token numbers are attributed to E110; levels are reported as the combined effect.

## Measurement

- `summary.json` → `llm.completion_tokens`, `actions`, `act_calls`, `levels_completed`.
- New counter `length_cutoffs` (turns with `finish_reason == "length"`), from `session.json` per game.
- `tests/unit/test_prime_e110_prompts.py`: the switch off leaves the E008 text byte-identical.

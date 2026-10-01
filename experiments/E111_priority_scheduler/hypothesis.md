# E111 — Priority scheduler: GPU time goes to the games most likely to score next

- **Date opened:** 2026-10-01
- **Axis varied:** budget (backlog E65 "marginal-gain compute allocation", E64 "early abort")
- **Baseline:** E008 arm A, Kaggle run `banwait13/sarbloh-prime` v354374653 (2026-10-01, 17 public dev games).
- **Split:** none. Public dev games; a smoke signal, not a held-out result.

## In simple words

There is one teacher (the GPU) and 17 children (the games). Today the teacher sat with 6 children for exactly 90
minutes each, then the next 6, then the last 5. Some children were close to the answer when their 90 minutes ended
(sb26 had just finished its second level). Others had been stuck on the first page for an hour and still got the
full 90 minutes. The last group only started after three hours.

With the scheduler, every child sits in the room from the start. The teacher can help only a few at a time. Every
few minutes the teacher asks: "who is most likely to finish a page soon?" and helps that child next. A child who
just finished a page goes up the list. A child who has been stuck for a long time slowly goes down, so a child who
has not been helped yet gets a turn first. Nobody has a fixed 90 minutes any more.

## Evidence that motivates it (today's run)

- 3 waves (6, 6, 5 games) x 5,400 s = the whole 4.6 h; every game ended on `wall_clock`.
- Level-1 wins came after 10k-149k output tokens (median ~47k, 6 of 9 under 52k). The 8 games that never won level 1
  each burned ~150k tokens on it.
- sb26 won level 2 at the very end of its slot; it might have gone on.

## Mechanism

- `prime/scheduler.py` (new, ours): `PriorityScheduler(slots, quantum_calls, token_scale, log_path)`. A game must
  hold a slot to call the LLM. `ScheduledLLM` wraps the shared client per game: the first `chat` acquires a slot,
  the game keeps it for `quantum_calls` LLM calls (its tool runs in between take milliseconds), then gives it back
  and competes again. At a level-up it competes again at once. When a slot is free, the waiting game with the
  highest priority gets it (ties: whoever waited longest).
- Priority of a game on level L+1 (L levels won) of n:
  `gain x hope`, with `gain = (L + 1) / (n (n + 1) / 2)` (the RHAE weight of the next level in that game's score)
  and `hope = 0.5 ** ((t / T) ** 2)`, `t` = output tokens spent on the current level, `T` = `token_scale` = 80,000
  (calibrated on today's level-1 wins: hope 0.76 at 50k, 0.09 at 150k).
- `run.py`: with `scheduler.enabled`, every game starts at once (pool size = number of games) and the per-game wall
  is the whole budget; only `slots` games are on the GPU at a time. The waits and grants go to
  `prime_run/scheduler.jsonl`; `summary.json` gets per-game LLM-seconds and wait-seconds.
- The wait is outside the request timeout and ends at once on stop or on the game's deadline.
- **Added 2026-10-01, after the local smoke and before any Kaggle run:** a game that has never held a slot goes
  before every game that has. Without this rule, in the 1-slot local smoke tr87 (6 levels, gain 1/21) outranked ls20
  (7 levels, gain 1/28) until it had spent ~43k tokens, so ls20 never played in 8 minutes. On 17 games and 6 slots, the
  games with many levels would wait until the leaders had burned 50k-85k tokens each, which breaks the prediction
  "every game gets at least one slot in the first hour".

## Why it should work

RHAE rewards levels, weighted by level index, and only levels. Tokens spent on a game that cannot clear its level are
lost; tokens moved to a game that is one level deeper, or to a fresh game, buy expected score. The 12-hour budget is
fixed, so moving the same tokens to better games raises the total.

## Prediction

Committed before the run, same 17 games, same 5 h budget, vs today's 10 levels / 1.88 mean RHAE:
- more levels completed than the baseline (≥ 12), with the same server and prompts;
- the top quarter of games by levels gets ≥ 40% of the LLM-seconds (today: equal shares by construction);
- every game gets at least one slot in the first hour.

## Kill criterion

Committed before the run. Binding.

> Kill if levels completed are not higher than the comparison arm on the same games and budget, or if any game never
> gets a slot (starvation bug). If the run also changes the prompts (E110) or the server (E109), the verdict is
> INCONCLUSIVE for levels, and only the allocation numbers (LLM-seconds share, waits, starvation) are attributed to
> E111.

## Measurement

- `summary.json` → `levels_completed`, `mean_score`, `scheduler` (grants, waits, LLM-seconds per game).
- `scheduler.jsonl`: one line per grant / release with the priority at that moment.
- `tests/unit/test_prime_scheduler.py`: ordering, quantum hand-over, level-up re-compete, stop wakes waiters,
  no starvation of fresh games.

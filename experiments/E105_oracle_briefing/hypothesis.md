# E105 — Oracle briefing: understanding vs execution
- **Axis varied:** diagnosis
- **Baseline:** same model, screenshots + action space only

> Logged retrospectively on 2026-10-01 from the owner's report. Qualitative result; no held-out RHAE yet (Rule 2 open).

## Mechanism
Gemini plays an ls20 level twice: (a) only screenshots and the action space; (b) plus an owner-written briefing — action meanings, goal, mechanics, on-screen components.

## Why it should work
If the agent fails without the briefing but succeeds with it, execution is adequate and understanding (rule inference) is the bottleneck.

## Prediction
Briefed arm clears the level; un-briefed arm does not.

## Kill criterion
If the briefed arm also fails, execution is the bottleneck — fix the harness first.

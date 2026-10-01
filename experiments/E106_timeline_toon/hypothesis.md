# E106 — Timeline in TOON instead of JSON
- **Date opened:** 2026-10-01
- **Axis varied:** substrate
- **Baseline:** timeline sub-memory as JSON in the REPL (E08 verdict)

## Mechanism
Serialise the timeline sub-memory as TOON (Token-Oriented Object Notation) instead of JSON. Nothing else changes.

## Why it should work
TOON drops the repeated keys, quotes and braces JSON spends on every row of a uniform list, so the same timeline costs fewer tokens and more history fits before compaction.

## Prediction
Fewer tokens per timeline entry with no loss in how accurately the agent reads its own history.

## Kill criterion
Kill if the agent misreads or misquotes TOON timeline entries more often than JSON entries on the same games, regardless of token savings.

## Measurement
- Tokens per timeline entry (JSON vs TOON), same episodes
- Accuracy: agent's answers to questions about its own past actions, checked against the log
- Compaction events per game

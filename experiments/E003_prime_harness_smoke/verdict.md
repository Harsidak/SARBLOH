# E003 verdict — PLUMBING PASS, AGENT FAIL (2026-09-28)

**Kill criterion not hit:** vLLM started (gemma_fast, native tools), 0/6 sessions crashed. The serving stack is settled.
Predictions: 1 met, 2 met (>=20 actions, no crash), 3 missed (275 vs 300 tok/s), 4 met barely (1 level).

**But the run did not test Prime Agent.** Of the four state levels, only L0 (frozen Gemma) and the bare REPL of L2
were exercised:
- L1 compaction: 0 events. The 300-action cap ended every game at ~40k tokens, before the 96k trigger.
- L2 `rlm.spawn`: 0 children.
- L3 Continual Harness: 0 `rlm.harness` calls. The digest was empty all run, so nothing was learned or carried.

What ran was Gemma 4 + an `ipython` tool with `arc.step`. Everything Prime adds is opt-in, and Gemma-4-31B did
not opt in. Its failure mode was spending actions in Python loops (85% of actions in >=10-action cells), resets,
and misreading HUD elements as goals.

**Conclusion:** the paper's gains come from a model that chooses to write memories/skills and delegate. Handing
those tools to Gemma 31B via prompt is not enough. Score is a public-set smoke number, not evidence (§4.3).
Next steps are proposed in chat, not started.

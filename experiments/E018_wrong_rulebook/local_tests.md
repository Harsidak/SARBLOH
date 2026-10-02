# E018 local checks (2026-10-02)

**Plumbing only, no model.** No live run, so this is not a scored run and gets no ledger row.

- **Code:** `Sarbloh-Experimentation/prime/` on top of bf2d52e, uncommitted.
  - `memory/wrong_rules.py` (new): `WrongRulebook`.
  - `memory/lessons.py`: `SkillBook` use-case names, `BASE_USE_CASES`.
  - `memory/lifecycle.py`: `_route_wrong`, `_promote`, blocks, recall.
  - `agent/curator.py`: `system_prompt`, input, apply.
  - `agent/context.py`: block "wrong".
  - `agent/agent.py`: act note, stats, event `wrong_rule_reproposed`.
  - `agent/prompts.py`: one sentence.
  - `config.py`: `wrong_rulebook` False, `skill_names` "snake", cap `wrong` 200.
- **Lesson cap: no change.** It is 80 *tokens*, about 320 characters, not 80 characters (see hypothesis.md item 2).

## Tests
- `tests/unit/test_prime_wrong_rulebook.py` (new): **34 checks, 0 fails.** It covers:
  - the store: dedupe, this game first, cap, persistence;
  - a refuted hypothesis is in the file at once;
  - a re-proposal is kept, reported and counted once, and taking a rule back from refuted counts too;
  - a bad act writes nothing;
  - refuted rules from the level-up and the E022 review go to the rulebook and never become graph nodes;
  - the pinned block, recall scope lessons and the system prompt line;
  - the curator: its input has the rulebook, its prompt edits are on, its refuted lessons go to the rulebook;
  - use-case skill names: stored, an 11-word name dropped with an error, loaded and found in any case or punctuation,
    base skills read-only under their use-case names, snake mode unchanged;
  - switches off: no file, no block, E008 curator prompt.
- **Existing prime suites against the copy** (`PRIME_DIR=Sarbloh-Experimentation`, switches off):
  - memory 40, curator 15, context 17 and e110_prompts 19 checks: 0 fails.
  - goals, level_review and tools: every check passes until the scripted real-game section, which stops on
    `ModuleNotFoundError: sarbloh`. This failure is not caused by E018. `sarbloh/` (31 files) is deleted in the
    working tree but present in HEAD, and `prime/game/arc_host.py` (both copies) imports `sarbloh.harness.games`.
    Sarbloh-Arc fails the same way.
- **Not tested end to end:** the act note in `agent.py`, for the same reason. It compiles. The text it formats comes
  from `write_act`'s `wrong` list, which the tests check.
- **Ruff:** the new and edited files are clean apart from patterns that were already there (the RUF100 at
  `lifecycle.py` `clean`, the ISC004 in `prompts.py`).

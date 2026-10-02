# E018 local checks (2026-10-02)

**Plumbing only, no model.** No live run, so this is not a scored run and gets no ledger row.

- **Code:** `Sarbloh-Experimentation/harness/` on top of bf2d52e, uncommitted.
  - `memory/wrong_rules.py` (new): `WrongRulebook`, one per game (`<game>/wrong_rules.json`).
  - `memory/lessons.py`: `SkillBook` use-case names, `BASE_USE_CASES`.
  - `memory/lifecycle.py`: `_route_wrong` (refuted -> rulebook at once, re-proposals counted), `_promote`.
  - `agent/curator.py`: use-case naming edit of the system prompt; `refuted` lessons -> rulebook.
  - `agent/agent.py`: stats `wrong_rules`, `wrong_rule_reproposed`, event `wrong_rule_reproposed` (log only).
  - `config.py`: `wrong_rulebook` False, `skill_names` "snake".
- **Amendment (owner, same day, before any run):** per game instead of cross-game, and hidden from the agent. Removed:
  the pinned block "Wrong rulebook" (`context.py`, cap `wrong`), the `recall` section, the system-prompt sentence
  (`prompts.E018_WRONG_TEXT`), the act note on a re-proposal, and the curator's `<wrong_rulebook>` input and prompt
  edit.
- **Lesson cap: no change.** It is 80 *tokens*, about 320 characters, not 80 characters (see hypothesis.md item 2).

## Tests
- `tests/unit/test_prime_wrong_rulebook.py`: **35 checks, 0 fails** after the amendment. It covers:
  - the store: dedupe, levels and sources, cap, persistence;
  - a refuted hypothesis is in `<game>/wrong_rules.json` at once; another game gets its own rulebook;
  - a re-proposal is kept and counted once, and taking a rule back from refuted counts too;
  - a bad act writes nothing;
  - refuted rules from the level-up, the E022 review and the curator go to the rulebook, never to graph nodes;
  - hidden: no pinned block, no recall hits, no prompt line, no act note, not in the curator's input or prompt;
  - use-case skill names: stored, an 11-word name dropped with an error, loaded and found in any case or punctuation,
    base skills read-only under their use-case names, snake mode unchanged;
  - switches off: no file, E008 curator prompt.
- **Existing prime suites against the copy** (`PRIME_DIR=Sarbloh-Experimentation`, switches off):
  - memory 40, curator 15, context 17 and e110_prompts 19 checks: 0 fails.
  - goals, level_review and tools first stopped on `ModuleNotFoundError: sarbloh` (078d816 deleted `sarbloh/`, and
    `harness/game/arc_host.py` and `harness/run.py` imported `sarbloh.harness.games`). Fixed the same day: the module
    moved unchanged to `harness/game/games.py` in both trees. Rerun: fidelity 37, goals 75, level_review 40, tools 43
    checks, 0 fails.
- **Not tested end to end:** the `agent.py` stats and event (no model run). `harness.agent.agent` and `harness.run` import.

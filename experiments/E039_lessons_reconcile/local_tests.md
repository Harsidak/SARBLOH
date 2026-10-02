# E039 local checks (2026-10-02)

**Plumbing only, no model.** No live run, so this is not a scored run and gets no ledger row.

- **Code:** `Sarbloh-Experimentation/harness/` on top of bf2d52e and E018, uncommitted.
  - `agent/lessons_reconcile.py` (new): `RECONCILE_SYSTEM`, `reconcile_input`, `reconcile`.
  - `memory/lessons.py`: `LessonsGraph.remove`, `LessonsGraph.merge` (one id = a rewrite).
  - `memory/lifecycle.py`: `GameMemory.reconcile_lessons` (guards, applies, saves).
  - `agent/agent.py`: `_reconcile_pending` filled at each level-up, `_maybe_reconcile` between the E022 review and
    the curator; event `lessons_reconcile`; stats `reconciles`, `reconcile_errors`, `lessons_dropped|rewritten|merged`.
  - `config.py`: `lessons_reconcile` False, `lessons_reconcile_max_tokens` 2048, `lessons_reconcile_tokens` 2000.

## Tests
- `tests/unit/test_prime_lessons_reconcile.py` (new): **28 checks, 0 fails.** It covers:
  - graph operations: merge keeps games, levels, uses and moves edges; a rewrite to the same text keeps the node;
    mixed types and shapes refused; remove drops edges, keeps shapes;
  - the input: ids, this game first, where each lesson came from, the wrong rules with their levels, the cap;
  - a pass with a fake model: drop applied; drop of another game's lesson refused; unknown id refused; rewrite
    applied with the shape edge kept; empty rewrite and one-id merge skipped; saved to `lessons.json`;
  - merge across games; the 10-change limit; an unparseable reply raises;
  - skipped with the lessons graph off or empty; works without the rulebook; off by default.
- **Existing prime suites against the copy** (switches off): memory 40, curator 15, context 17, e110_prompts 19:
  0 fails. After the games module moved to `harness/game/games.py` (see E018 local_tests.md): fidelity 37, goals 75,
  level_review 40, tools 43 checks, 0 fails.
- **Not tested end to end:** `_maybe_reconcile` in `agent.py` (no model run). It imports and follows `_maybe_review`
  line for line.

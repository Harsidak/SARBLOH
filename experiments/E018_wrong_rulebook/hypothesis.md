# E018 — Wrong rulebook: refuted rules get their own per-game store, kept out of the lessons graph

- **Date opened:** 2026-10-02
- **Axis varied:** consolidation (backlog E18, "falsified-hypothesis log — refuted rules never re-proposed", as
  restated by the owner)
- **Baseline:** E008_perception_memory as it is (`agent.memory.wrong_rulebook: False`, `skill_names: "snake"`), same
  code, same game. Independent of E021 / E022; E039 (lessons reconcile) reads this store.
- **Split:** none yet. Local tests and plumbing only. Not a result.

## Mechanism

Owner's statements (2026-10-02):
1. "We don't need to verify it, the agent doesn't need to verify it; if the agent verifies it then add that in
   skills, otherwise add it to that new log, the wrong rulebook, a new memory arch (E18). Also increase that 80
   characters cap, it is too short. Also rather than saving a skill name by a word, save it by its use case under
   10 words."
2. Amendment, same day, before any run: "The wrong rulebook shouldn't be shared by all games. Don't show the wrong
   rulebook to the agent, just store it. The wrong rulebook should be logged constantly, but after one level is
   complete the lessons and the wrong rulebook should be compared, and then the lessons updated" (that comparison is
   E039).

The host still judges nothing: the status the agent writes decides where a rule goes. In E008 a refuted rule becomes
a `refuted` node in the lessons graph, mixed with the verified rules. E018 adds, behind two switches of `agent.memory`
in `Sarbloh-Experimentation/harness/`:

1. **`wrong_rulebook: true`.** A per-game store `harness/memory/wrong_rules.py` (`WrongRulebook`,
   `<memory root>/<game>/wrong_rules.json`):
   - **Writes, constantly.** A hypothesis goes in as soon as the agent marks it `refuted` in an act. It does not wait
     for the level-up, so a level the agent never wins still keeps its refuted rules. The `refuted` rules from the
     curator, the level-up and the E022 level review go here too. With the switch on, the lessons graph gets no
     `refuted` nodes. Verified rules follow the E008 path: lessons graph `rule` nodes, then the curator distils them
     into skills.
   - **Reads: none by the agent.** No pinned block, no `recall` hits, no prompt line, no act note. The curator does
     not read it either. Its one reader is E039, after each level won.
   - **Re-proposals, logged only.** A new or changed hypothesis that matches a wrong rule of this game is written as
     usual; the event `wrong_rule_reproposed` and the stat of the same name count it (the CLAUDE.md "waste event").
     The match is a word-set Jaccard of at least 0.75 after lower-casing and dropping punctuation (UNCONFIRMED
     threshold).
2. **Lesson length: no change.** The owner asked to raise the "80 character" cap on lesson text. That figure was my
   mistake: `clip(text, 80)` in `LessonsGraph.add_node` caps at 80 *tokens*, and `store.clip` multiplies by 4, so a
   lesson already keeps about 320 characters (two or three sentences). The only short character cap on this path was
   the 40-character snake_case skill name, which item 3 replaces.
3. **`skill_names: "use_case"`** (E008: `"snake"`, a one-word snake_case name).
   - **Naming.** A curated skill is named by its use case in at most 10 words, e.g. "when blocks must be pushed onto
     matching targets". Names are compared after lower-casing and dropping punctuation, so the curator can reuse a
     name to update a skill. A name over 10 words is dropped and logged.
   - **Base skills.** The three base skills get use-case names in this mode. Their files and texts are unchanged.

## Why it should work

Refuted rules mixed into the lessons graph take pinned-context room from verified ones, and a rule wrong in one game
shown to every game can stop the agent testing a rule that holds there. Keeping them per game and out of the context
keeps the lessons block verified-only; E039 then uses them to correct the lessons once per level. Use-case names let
the curator pick the skill for the situation from its name alone.

## Prediction

Committed before the run.

- **Local (tests, no model):**
  - With both switches off, behaviour and tool schemas are identical to E008: the existing prime suites pass against
    the copy.
  - With them on: an act that refutes a hypothesis writes it to `<game>/wrong_rules.json` at once; another game has
    its own file; a re-proposal is counted, not shown; the agent's context, recall and prompt never contain it; the
    curator's `refuted` lessons land in the rulebook, not the graph; a skill named by its use case is stored, found
    and loaded by that name.
- **Kaggle (not run; owner's call):** vs arm A on the 5 dev games, no fewer levels won, and a lessons block with no
  `[refuted]` lines. The re-proposal count is recorded in both arms for E039.

## Kill criterion

Committed before the run. Binding.

> Local: kill as a bug if the switch-off path changes any existing suite result, if a refuted hypothesis is lost
> (in neither the rulebook nor the graph), or if the rulebook reaches the agent's context. Kaggle (when it runs): kill
> if levels won drop by 2 or more over the 5 games. If not and RHAE is within noise, INCONCLUSIVE.

## Measurement

- **Tests:** `tests/unit/test_prime_wrong_rulebook.py` (store, routing, per-game files, hidden from the agent,
  curator, use-case skill names, switches off). The existing prime suites run against the copy
  (`PRIME_DIR=Sarbloh-Experimentation`).
- **Kaggle (later):** `kaggle_push.py experimentation` with `EXPERIMENT_OVERRIDES` from `config.yaml`. The results are
  `prime_run/results.json` stats (`wrong_rule_reproposed`, `wrong_rules`) and events, plus per-game RHAE via
  `eval/official_score.py`.

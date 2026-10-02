# E018 — Wrong rulebook: refuted rules get their own memory, are shown to the agent, and are flagged when re-proposed

- **Date opened:** 2026-10-02
- **Axis varied:** consolidation (backlog E18, "falsified-hypothesis log — refuted rules never re-proposed", as
  restated by the owner)
- **Baseline:** E008_perception_memory as it is (`agent.memory.wrong_rulebook: False`, `skill_names: "snake"`), same code, same game. Independent of E021 / E022: all switches can be on together.
- **Split:** none yet. Local tests and plumbing only. Not a result.

## Mechanism

Owner's statement (2026-10-02): "we don't need to verify it, the agent doesn't need to verify it; if the agent
verifies it then add that in skills, otherwise add it to that new log, the wrong rulebook, a new memory arch (E18).
Also increase that 80 characters cap, it is too short. Also rather than saving a skill name by a word, save it by its
use case under 10 words."

The host still judges nothing: the status the agent writes decides where a rule goes. In E008 a refuted rule becomes
a `refuted` node in the lessons graph, mixed with the verified rules, and nothing notices when the agent proposes it
again. E018 adds, behind two switches of `agent.memory` in `Sarbloh-Experimentation/prime/`:

1. **`wrong_rulebook: true`.** A new cross-game memory `prime/memory/wrong_rules.py` (`WrongRulebook`,
   `<memory root>/wrong_rules.json`):
   - **Writes.** A hypothesis goes in as soon as the agent marks it `refuted` in an act. It does not wait for the
     level-up, so a level the agent never wins still keeps its refuted rules. The `refuted` rules from the curator and
     the E022 level review go here too. With the switch on, the lessons graph gets no `refuted` nodes. Verified rules
     follow the E008 path: lessons graph `rule` nodes, then the curator distils them into skills.
   - **Reads.**
     - A new pinned block, "Wrong rulebook", lists this game's wrong rules first, then other games', under cap
       `caps.wrong` (200 tokens).
     - `recall` scope `lessons` (and `all`) also searches it.
     - The curator sees it in its input.
   - **Re-proposals.** A new or changed hypothesis that is not refuted and matches a wrong rule is still written: a
     rule wrong in one game can be right in another. The act result adds a note naming the wrong rule and where it
     was refuted, and the stat `wrong_rule_reproposed` counts it. That count is the CLAUDE.md "waste event".
     - The match counts as a re-proposal at a word-set Jaccard of at least 0.75 after lower-casing and dropping
       punctuation. This threshold is UNCONFIRMED.
2. **Lesson length: no change.** The owner asked to raise the "80 character" cap on lesson text. That figure was my
   mistake: `clip(text, 80)` in `LessonsGraph.add_node` caps at 80 *tokens*, and `store.clip` multiplies by 4, so a
   lesson already keeps about 320 characters (two or three sentences). The only short character cap on this path is
   the 40-character snake_case skill name, which item 3 replaces.
3. **`skill_names: "use_case"`** (E008: `"snake"`, a one-word snake_case name).
   - **Naming.** A curated skill is named by its use case in at most 10 words, e.g. "when blocks must be pushed onto
     matching targets". Names are compared after lower-casing and dropping punctuation, so the curator can reuse a
     name to update a skill. A name over 10 words is dropped and logged.
   - **Base skills.** The three base skills get use-case names in this mode. Their files and texts are unchanged.

## Why it should work

RHAE squares the action ratio, and paying twice for a fact is the main loss term. A refuted rule that sits as one
`[refuted]` line among the rules is easy to miss, so the agent tests it again and pays for it
again. A separate block, labelled "do not propose again", and a note at the moment the agent re-proposes the rule
should cut those repeat tests. Use-case names let the curator pick the skill for the situation from its name alone.

## Prediction

Committed before the run.

- **Local (tests, no model):**
  - With both switches off, behaviour and tool schemas are byte-identical to E008: the existing prime suites pass
    against the copy.
  - With them on:
    - an act that refutes a hypothesis writes it to `wrong_rules.json` at once;
    - a re-proposal gets the note and the count;
    - the pinned block shows it;
    - the curator's `refuted` lessons land in the rulebook, not the graph;
    - a skill named by its use case is stored, found and loaded by that name.
- **Kaggle (not run; owner's call):** vs arm A on the 5 dev games, fewer re-proposals of refuted rules per game
  (counted in both arms from the act events), and no fewer levels won.

## Kill criterion

Committed before the run. Binding.

> Local: kill as a bug if the switch-off path changes any existing suite result, or if a refuted hypothesis is lost
> (in neither the rulebook nor the graph). Kaggle (when it runs): kill if the re-proposal count is not lower than arm
> A, or if levels won drop by 2 or more over the 5 games. If neither happens and RHAE is within noise, INCONCLUSIVE.

## Measurement

Evaluation systems used:
- **Tests:** `tests/unit/test_prime_wrong_rulebook.py` (new) covers the store, the routing, the act note, the pinned
  block, recall, the curator and the use-case skill names. The existing prime suites run
  against the copy (`PRIME_DIR=Sarbloh-Experimentation`).
- **Kaggle (later):** `kaggle_push.py experimentation` with `EXPERIMENT_OVERRIDES` from `config.yaml`. The results are
  `prime_run/results.json` stats (`wrong_rule_reproposed`, `wrong_rules`) and `act` events, plus per-game RHAE via
  `eval/official_score.py`.

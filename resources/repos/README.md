# resources/repos/ — third-party clones

**Gitignored.** Clone here for reference. Never vendor into `sarbloh/`.

| Repo | Why | Licence | Cloned |
| --- | --- | --- | --- |
| `avo/` | NVIDIA AVO — agent loop, lineage, scoring, knowledge-base structure. Read `src/avo/loop.py`, `session.py`, `lineage.py`, `knowledge.py`. | check `avo/LICENSE` before borrowing | 2026-08 |

## Rules

1. A clone is read, not imported. If a mechanism is worth having, it is reimplemented in `sarbloh/` under this
   project's licence, with the source credited in the module README and in the paper.
2. Licence checked and recorded in the table **before** any code is borrowed. Prize eligibility depends on
   every third-party component carrying a licence that permits public sharing.
3. Nested `.git` directories never enter this repository's history.

Suggested additions: the Milestone #1 open-source entries (Tufa Labs "The Duck", Reki, forge) and the official
`arcprize/arc-agi` toolkit.

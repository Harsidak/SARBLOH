# Mistakes log

Append-only. Anything that cost more than an hour. Newest first.

The entry that matters is the **cause**, not the symptom. "Run crashed" is not an entry. "Reward function failed
silently because the Kaggle environment swallows exceptions in the scoring callback" is an entry.

---

## 2026-08 — Tests moved to the archive

**Cost:** repository verifiability, for roughly six weeks.

**What happened:** fifteen test files were moved to `_archive/tests_2026-08-10/` rather than fixed. Four tests
remained live against an 830 KB agent codebase.

**Cause:** tests were treated as blocking work rather than as the thing that makes work cumulative. Under
schedule pressure the cheapest-looking move was to remove the blocker.

**Rule adopted:** a broken test is fixed or deleted with a recorded reason. Never archived. (`CLAUDE.md` §4.6)

---

## 2026-08 — Results recorded without attribution

**Cost:** three months of experimental results are not reconstructable.

**What happened:** fourteen result files accumulated in `scores/` — `battle2`, `champion`, `memo`,
`llmagent_route` — with no mapping from any of them to the configuration or hypothesis that produced it.

**Cause:** logging was treated as a write-up step to be done later rather than as part of the run.

**Rule adopted:** the ledger row is written by the run. A run that is not logged did not happen. (`CLAUDE.md` §4.1)

---

## 2026-09-21 — Wall-clock ceiling assumed, not verified

**Cost:** low, caught early.

**What happened:** planning proceeded on a 9-hour Kaggle ceiling. The ARC Prize testing policy states 12 hours.

**Cause:** a working figure carried forward across sessions without being checked against the source.

**Rule adopted:** competition constants live in `CLAUDE.md` §2 with a verification date, and anything unverified
is marked UNCONFIRMED rather than used silently.

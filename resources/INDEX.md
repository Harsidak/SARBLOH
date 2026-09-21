# resources/ — intake

Everything the project reads but did not author. Four bins:

| Directory | Holds | Tracked in git |
| --- | --- | --- |
| `official/` | Dated snapshots of the ARC-AGI-3 specification, rules and toolkit docs | Yes |
| `papers/` | PDFs and extracted text | Yes (PDFs may be large — check before committing) |
| `repos/` | Third-party clones | **No** — gitignored, never vendored |
| `web/` | Scraped blog posts, write-ups, leaderboard pages, as markdown | Yes |

## The intake rule

**Every item carries a sibling `.meta.yaml`.** No orphan PDFs, no un-sourced markdown. Copy
`_TEMPLATE.meta.yaml`.

An artefact without provenance cannot be cited in the paper and cannot be re-fetched when it changes. A blog
post scraped in July and a blog post scraped in October are different documents.

## Why `official/` is dated and tracked

ARC Prize changes its documentation during the competition. A dated snapshot lets you diff what changed and when,
and settles arguments about what the rules said at the time a decision was made. Re-snapshot when anything
material moves and keep both.

## Index

| Path | What | Retrieved |
| --- | --- | --- |
| `official/ARC-AGI-3-SPEC.md` | Environment, toolkit and scoring spec | 2026-09-21 |
| `official/COMPETITION-RULES.md` | Dates, prizes, constraints, eligibility | 2026-09-21 |
| `papers/` | See `papers/README.md` for the inherited set | 2026-07 onward |

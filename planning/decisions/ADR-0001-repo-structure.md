# ADR-0001 — Repository structure and the verified-only rule

- **Status:** accepted
- **Date:** 2026-09-21
- **Supersedes:** the flat `ARC_AGI EXPERMENTATIONS` layout

## Context

The pre-existing directory held 787 entries with no version control, three monolithic agent files totalling
roughly 830 KB at the root, fifteen tests in the archive, and fourteen unattributed result files. The working
evaluation spine (`eval/`) was sound. Two deadlines are live: Milestone #2 on 30 Sep and final submission on
2 Nov.

## Decision

1. Reorganise into a package (`sarbloh/`) with one-way dependencies, keeping `eval/` intact as the scoring spine.
2. Freeze the monolithic agents in `sarbloh/legacy/`, extracted component by component, each extraction pinned by
   a test before the move. **No refactoring during reorganisation.**
3. Make the architecture's central rule structural rather than conventional: `parkh` is a separate module and
   both `memory` promotion and `jugat` planning are gated on it. The rule cannot be bypassed by accident because
   the dependency direction forbids it.
4. Make experiment discipline structural: `experiments/E###/` requires `hypothesis.md` before code and
   `verdict.md` after, and `history/ledger.jsonl` is written by the run.
5. Vendored third-party repositories (`avo/`) move to `resources/repos/` and are gitignored, never committed into
   the tree.

## Consequences

**Good.** The ablation the paper needs falls out of the ledger rather than being reconstructed. Licence exposure
from vendored code is eliminated. Dev and submission share one code path via `Arcade`'s OFFLINE / COMPETITION
modes.

**Costly.** Roughly two days of non-scoring work nine days before Milestone #2. Accepted because the alternative
is entering the final six weeks unable to say which change moved the number — which is the exact failure the
August result set already demonstrates.

**Risk accepted.** Freezing rather than decomposing `legacy/` means the working agent is temporarily
unreachable from new code. Mitigated by extracting behind pinning tests, and by the Phase 1 gate on 24 Sep.

## Alternatives rejected

- *Refactor while reorganising.* Reorganisation and refactoring in one pass is how a working agent is lost under
  deadline.
- *Leave `scores/` as the history.* Unattributed results cannot serve as baselines; preserving them as evidence
  of work is the most they support.

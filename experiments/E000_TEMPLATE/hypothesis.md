# E000 — <short name>

- **Date opened:** YYYY-MM-DD
- **Axis varied:** <one of: substrate / clock / perception / consolidation / retrieval / hypotheses / action policy / verification / budget / adaptation>
- **Baseline:** <experiment id or `history/baselines.json` key this is measured against>
- **Split:** tune | heldout   *(report numbers come from heldout only)*

## Mechanism

What is being changed, concretely, in which module. Name the functions.

## Why it should work

The causal story from the change to the RHAE number. If this cannot be stated in three sentences without the
word "better", the experiment is not ready.

## Prediction

Committed before the run. A number and a direction.

> Expected heldout RHAE delta: **+X.X** points vs `<baseline>`.

## Kill criterion

Committed before the run. Binding.

> Kill if <specific, observable condition> after <specific budget>.

## Measurement

- Metric: heldout mean RHAE via `eval/official_score.py`
- Secondary: actions to first certified model; re-derivation events; wall-clock per environment
- Seeds: <n>, reported with variance. A single-seed delta under 2 points is not a result.

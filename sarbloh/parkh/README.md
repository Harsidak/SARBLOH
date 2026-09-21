# sarbloh.parkh — Verification

*Parkh* — to assay, to test metal for purity. The gate the whole architecture turns on.

Responsibilities:
- Replay every recorded transition against a candidate `step()`. Exact match or fail.
- Return a binary verdict by default.
- Detect and report *which* transition broke, so `worldmodel` knows whether to indict the rule or the state
  representation.

Forbidden: partial credit, unless an experiment is explicitly testing partial-credit gating and says so in its
`hypothesis.md`. Softening this gate is the easiest way to make public-set numbers rise and private-set numbers
fall.

Planned units: `backtest.py`, `diagnosis.py`

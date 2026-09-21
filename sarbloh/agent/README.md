# sarbloh.agent — The loop

Owns the observe -> retrieve -> induce -> certify -> plan -> commit cycle and everything about *when* to act.

Responsibilities:
- Commitment policy: how much of a validated plan prefix to commit before re-verifying.
- Halt-on-misprediction: abandon the remaining plan at the first divergence from prediction.
- Stagnation supervision: detect unproductive cycles and redirect.
- Budget governor: hard wall-clock and token accounting with graceful degradation, plus early abort on
  environments showing no modelling progress.

The governor is not optional polish. A timeout scores zero; degrading gracefully scores something.

Planned units: `loop.py`, `commitment.py`, `supervisor.py`, `budget.py`

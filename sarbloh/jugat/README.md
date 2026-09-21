# sarbloh.jugat — Planning and action selection

*Jugat* — the method. Search inside the certified simulator, where actions are free.

Responsibilities:
- Breadth-first / A* search over the certified model to reach the goal hypothesis.
- Information-gain action selection: when hypotheses disagree, choose the action whose outcome most
  discriminates between them. This is the RHAE-optimal exploration criterion — maximise bits per committed action.
- Macro-action induction: compress repeated verified sequences into named options.
- Estimate the human action baseline so plans can be scored against the actual metric, not step count.

Forbidden: searching against an uncertified model.

Planned units: `search.py`, `infogain.py`, `macros.py`, `baseline.py`

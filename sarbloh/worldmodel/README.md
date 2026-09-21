# sarbloh.worldmodel — Executable transition-function induction

Induces and maintains an executable `step(state, action) -> state` that reproduces observed transitions.

ARC-AGI-3 environments are discrete, deterministic and fully observable within a frame. A correct `step()` is
therefore a *perfect* simulator, not an approximation, which is why symbolic induction works here and learned
latent dynamics do not — the sample budget is two orders of magnitude short for the latter.

Responsibilities:
- Propose and revise `step()` from the transition record.
- Maintain k competing hypotheses rather than committing to one (premature commitment is a named failure mode
  in the literature).
- Refactor toward the shortest faithful form. Compression is the generalisation prior.
- Maintain an explicit, versioned goal hypothesis.

Forbidden: being consulted for planning before `parkh` certifies it.

Planned units: `induction.py`, `hypotheses.py`, `refactor.py`, `goal.py`

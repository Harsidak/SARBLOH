# sarbloh.memory — Three-clock memory with explicit promotion

The differentiator. Memory is not one store; it is three, on different update clocks.

| Clock | Holds | Updates |
| --- | --- | --- |
| frame | transient state, current position, active objects | every frame |
| level | mechanics established for this environment | on certified discovery |
| environment | procedures and priors reusable across environments in one run | on level completion |

Also owns the falsified-hypothesis log: rules that were refuted are recorded so they are never re-proposed.
Re-derivation of a refuted rule is a measurable waste event and the counter for it lives here.

Forbidden: promoting anything to the level or environment clock that `parkh` has not certified. A single
undifferentiated store lets transient state overwrite stable mechanics, which is the failure this module exists
to prevent.

Planned units: `clocks.py`, `promotion.py`, `eviction.py`, `falsified_log.py`

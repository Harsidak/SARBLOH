# sarbloh.harness — Integration

`Arcade` setup, environment construction, recording, and the Kaggle submission entry point.

One code path only. `OFFLINE` mode against `eval/real_games/` and `COMPETITION` mode against the private set
use the same agent loop, differing only in `Arcade` configuration. A dev-only loop that diverges from the
submission loop is how a submission fails on constraints nobody tested.

Forbidden: containing agent logic.

Planned units: `arcade.py`, `kaggle.py`, `recording.py`

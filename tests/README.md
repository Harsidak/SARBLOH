# tests/

| Directory | Scope |
| --- | --- |
| `unit/` | Single functions. Fast, no environment. |
| `component/` | One module against a fixed fixture. Pins behaviour extracted from `sarbloh/legacy/`. |
| `regression/` | Full-loop runs against recorded traces. Catches score regressions. |

Reclaimed from `_archive/tests_2026-08-10/`: `test_goalmodel`, `test_griddsl`, `test_livelock`,
`test_clickplanner`, `test_synth`, `test_progress`, `test_stategraph`, `test_execplanner`, `test_llm_repair`,
`test_components`, `test_metrics`, `test_churn`.

**A broken test is fixed or deleted with a recorded reason. It is never archived.** Fifteen tests went to
`_archive/` in August 2026 and the repository stopped being verifiable. That does not happen again.

```bash
python -m pytest tests/ -q
python -m pytest tests/component -q -k goalmodel
```

# results.json schema

Emitted by `eval/official_score.py`. One file per experiment run.

```json
{
  "experiment": "E000_template",
  "git_sha": "abc1234",
  "config_hash": "sha256:...",
  "timestamp": "2026-09-21T00:00:00Z",
  "split": "heldout",
  "seeds": [0, 1, 2],
  "per_environment": [
    {
      "env_id": "ls20",
      "levels_total": 8,
      "levels_completed": 8,
      "agent_actions": 412,
      "human_baseline_actions": 190,
      "rhae": 21.3,
      "wallclock_s": 640,
      "certified_at_action": 87,
      "rederivation_events": 4
    }
  ],
  "aggregate": {
    "mean_rhae": 0.0,
    "environments_solved": 0,
    "total_wallclock_s": 0
  }
}
```

`certified_at_action` and `rederivation_events` are SARBLOH-specific instrumentation, not part of the official
scorecard. They are the diagnostics the memory thesis is tested on: a memory change that does not reduce
re-derivation events has not done its job, whatever the RHAE number says.

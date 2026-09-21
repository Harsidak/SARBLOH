# experiments/

One directory per experiment: `E###_short_name/`. Numbers are never reused.

Required contents:

| File | Written | Contains |
| --- | --- | --- |
| `hypothesis.md` | **before any code** | mechanism, prediction, kill criterion, which axis it varies |
| `config.yaml` | before the run | the full config; its hash goes in the ledger |
| `run.py` | before the run | thin driver; real logic lives in `sarbloh/` |
| `results.json` | by the run | scores per split, emitted by `eval/official_score.py` |
| `verdict.md` | after the run | kept / killed / inconclusive, and why |

## The discipline

Hypothesis first. An experiment whose hypothesis is written after the result is a rationalisation, and it cannot
be defended in a paper.

Kill criteria are binding. When an experiment hits the criterion its `hypothesis.md` declared, it is killed and
`verdict.md` says so. Extending a dead experiment because it feels close is the single most expensive habit
available in a six-week campaign.

Negative results are logged with the same weight as positive ones. The ablation is the paper; an ablation with
only the wins in it is not an ablation.

Copy `E000_TEMPLATE/` to start.

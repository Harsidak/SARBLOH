# ADR-0002 — Prime Agent as the stage-2 harness (replaces the AVO plan)

- **Status:** accepted (owner decision)
- **Date:** 2026-09-27
- **Supersedes:** the `Sarbloh-AVO/` plan (NVIDIA AVO); amends ADR-0001 point 5 for this one repository

## Context

The owner chose Prime Intellect's Prime Agent (arXiv 2608.23552, github PrimeIntellect-ai/prime-agent, MIT) over NVIDIA
AVO. The paper reports ARC-AGI-3 Best@1 from 30% to 95.5% with Opus 5 through the harness alone (public set,
self-reported, UNCONFIRMED). Its ARC harness is not in the repository: the release is a general TypeScript
coding agent (daemon, worker, TUI) plus a Python kernel runtime. Kaggle is offline, Python-first, one GPU.

## Decision

1. Vendor the upstream Python kernel (`prime-agent-runtime/src/rlm/`) unchanged into `Sarbloh-Prime/rlm/`. It is
   stdlib-only, so it runs offline; `mcp*.py` and `skill.py` are dropped (lazy imports, unused).
2. Port the TypeScript host to Python (`Sarbloh-Prime/prime/`) rather than ship Node to Kaggle. Keep what the paper
   says matters: one `ipython` tool, persistent kernel state, autonomous continuation with budgets, compaction with the
   full transcript on disk, the Continual Harness injected every turn, `rlm.spawn` children and agent messaging.
   Drop the daemon, TUI, heartbeats, MCP and model switching.
3. Expose the game as a REPL module (`arc`) over typed host requests. The host owns the environment, counts
   actions, keeps the lossless transition record, and refuses `arc.step` from subagents (least privilege: the paper's
   Factorio exploit shows why).
4. Model: Gemma-4-31B-IT-NVFP4 on vLLM 0.19.0 plus a transformers 5.5.0 overlay. If the tool-call parser fails, the
   agent runs in fenced-code mode instead of failing the run.
5. The TypeScript tree is deleted from `Sarbloh-Prime/`; the full clone stays at `resources/repos/prime-agent` for
   reference.

## Consequences

**Good.** One Python code path, locally (llama.cpp) and on Kaggle (vLLM). The strategy is the model's, as in the paper;
our memory and verification thesis enters only as prompt guidance and the lossless `arc.transitions()` record, so later
experiments can test it against this baseline.

**Bad.** The port can drift from upstream behaviour; every divergence is listed in the `prime/session.py` docstring.
The paper's number comes from frontier models; a 31B open model may not use the harness well (the paper notes models
not trained on it underuse it). E003 measures only plumbing.

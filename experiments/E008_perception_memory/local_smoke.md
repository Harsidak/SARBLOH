# E008 local plumbing smoke (2026-10-01)

**Plumbing only. A 4B score is not evidence about the 27B.** Not a scored run, no ledger row.

- Model: Qwen3.5-4B Q4_K_M + mmproj (`scripts/serve_llm.ps1 -Vision`), llama.cpp, 16k context.
- Command: `uv run python Sarbloh-Arc/prime/run.py --games ls20 --max-actions 30 --minutes 12 --vision --ctx 16384 --out runs/e008_local_smoke`
- Code: working tree on top of f117ca4 (uncommitted E008 changes). Suites: `tests/run_all.py` 23/23 green, 1528 checks.

## What worked (from `runs/e008_local_smoke/games/ls20-*/transcript.jsonl`, `report.md`)
- 7 `act` calls, 9 actions, 0 act argument errors; every act was followed by exactly one observation with one image
  (7 images sent; the server accepted every image request).
- The agent wrote its memory through `act`: 12 hypothesis events (5 proposed, 6 verified, 1 refuted), 2 goal changes,
  a plan on every act. Every level it played has a hypothesis.
- Compaction ran once (11.8k tokens, 16 s); the memory block stayed pinned (largest 2,526 chars).
- The curator ran after the compaction (9 s): 2 open questions, lessons graph 2 rules + 1 goal + 2 shapes, 2 skills
  written (`hud_tracking`, `hypothesis_testing`) and loaded with `planning`.
- Largest observation 6,091 chars: the level-start board (as designed; after an act it is capped at 800 tokens).
- Repeated (state, action) share 0.0.

## What did not
- **The run stalled for 10 h inside one LLM request** (03:59 -> 14:01), then hit the soft deadline: the 8th act was refused
  with "the run is stopping". All agent work happened in the first 1.5 min. Cause UNCONFIRMED: most likely the PC slept
  while llama.cpp held the request (the 900 s request timeout is a socket timeout and does not fire across a suspend).
  Not an agent bug as far as the transcript shows; on Kaggle the watchdog covers a stuck server.
- No level cleared, no recall used, no promotion at a level-up (none happened). The level-up path is covered by
  `tests/unit/test_prime_tools.py` (forced level-up: promotion, erasure, curator) only.

## Next
Owner: commit, update `banwait13/sarblohagent`, push `kaggle/experimental/002_prime.ipynb` (arm A, 17 dev games). The
first cell to read is the vLLM start: did `qwen_fp8_vision` pass the image smoke test, or did the chain fall back to
`qwen_fp8` (that is the image kill criterion)?

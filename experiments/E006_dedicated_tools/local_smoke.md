# E006 local smoke run (2026-09-29): plumbing only, not scored

The model was Qwen3.5-4B Q4_K_M on llama.cpp with a 32k context and no thinking. The game was ls20, limited to 3 levels, 150 actions and 30 min.
The output is in `runs/prime_local_E006_ls20/` (gitignored). A 4B score says nothing about Gemma-4-31B. There is no ledger row.

## Result
- 0 of 7 levels cleared, 43 actions, 73 turns, 3 compactions. The run ended at the time limit.
- Tool calls: ipython 45, act 22 (19 ran, mean batch 2.3 actions), plan 6. Never used: remember, recall,
  delegate, message, reset_level, `wm.register`.
- Every one of the 43 actions has a stated `expect`. The trace captured all five sections.

## What the run showed
1. Understanding failed. The model never worked out that the 9/12 block is the player, moving 5 px per step. It kept
   calling actions "colour swaps", although every `act` result said "colour 9 5x3 moved (+0,-5)".
   The goal was restated 6 times and never became correct.
2. It re-derived what `act` had already reported: about 10 ipython cells rebuilt per-step pixel diffs by hand.
3. It guessed the API: `arc.act`, `obs.legal`, the `Observation` used as an array, `await arc.show`. That caused
   10 cell errors.
4. It typed a grid out as a literal twice. The repetition loop broke the tool-call JSON.
5. The upstream base prompt still taught `rlm.spawn`, `agent_message` and `rlm.harness.*` next to the new
   tools, so the prompt taught two ways to do the same thing.
6. The trace reported "repeated pairs 0", but that is not a real zero. ls20's step bar at y=61..62 changes on every step, so
   `state_key` never repeats. NOT BUILT: mask the pixels that change on every step (a counter) before keying.

## Fixed after the run
- Items 3–5 are fixed in the prompt: a trimmed dedicated base prompt, "tools are not Python functions", "load grids
  from ts", "one act per reply" and "read the act result first".
- Fixes for the review findings (workflow wf_8f9830fb-7fe, adversarially verified):
  - one act/reset per model reply;
  - Planner string arguments;
  - the world model: deep copy, the plan loop, nested numpy equality, bad returns, certification voided by a
    misprediction or a partial check, and numpy in events;
  - `summarize_change` pairs only nearby moves;
  - child `recall` reads the root's memories;
  - short `recall` queries;
  - the history guard;
  - the toolset reported after a fenced fallback;
  - the local experiment label;
  - `sft_levels`: the level switches after the tool round, older transcripts fall back to the recording, there is
    no phantom level and user messages are merged;
  - the trace also counts REPL memory writes, fenced cells, "n/a" repeats without step events, and a call column.

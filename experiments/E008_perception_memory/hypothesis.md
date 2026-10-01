# E008: autonomous Prime agent with per-transition perception and agent-owned memory

- **Date opened:** 2026-10-01
- **Axis varied:** perception (image + ASCII + segmentation pushed after every transition) and consolidation (a
  7-part memory the agent writes, plus a cross-game lessons graph kept by a hidden curator). The action interface is
  cut to 3 tools.
- **Model:** Qwen3.8-27B (not Flash-Next). Serving: `qwen_fp8` profile (official `Qwen/Qwen3.8-27B-FP8`, Kaggle
  `fumiyauchiyama/qwen3-8-27b-fp8-hf-snapshot`, vLLM 0.19, prefix caching, 65k window) with images switched on.
  The NVIDIA NVFP4 build (`nvidia/Qwen3.8-27B-NVFP4`, downloaded and hash-checked 2026-10-01) is a later arm, see
  "Arms".
- **Baseline:** E005 (Prime, `ipython` toolset, Gemma-4-31B): 1 of 76 levels on 10 games. Reference bar, not a
  baseline for this run: the Duck on Qwen3.8 NVFP4 with our vLLM, 6 RHAE (owner-reported 2026-09-30).
- **Split:** tune (the 17 dev games). The eight held-out games are never run or inspected by us; a held-out number
  only comes from the owner's Kaggle submission.

## Mechanism

The agent is **autonomous**: one long session per game. It keeps calling tools until the game is won, lost past
recovery, or out of budget. There is no outer "one step per call" loop like the Duck's.

**1. Perception is pushed, never pulled.** The agent has no `observe`, `transitions` or `diff` call and never sees a raw
numeric grid. Every action is a state transition. When a transition happens, the host builds the new state's
observation at once and it goes into the context in the same turn:
- `agent/vision.py`: the last frame of the transition as a PNG (ARC 16-colour map, nearest-neighbour ×4 = 256×256 px),
  sent as an `image_url` part in a user message right after the `act` tool result. Tool messages cannot carry images.
  Only the newest image stays in context; older ones are replaced with `[image of step N dropped]`.
- `agent/intuition.py`, which gives two text views of the same frame:
  - **ASCII**, one letter per colour (`WwgGcBMPRbSYOrNp`) with a legend. The whole board is shown only at level start.
    After that, only the changed region is shown, with a 3-cell margin and row/column labels.
  - **Segmentation**: 4-connected objects with id, colour, shape hash (translation-invariant), size, bbox, boundary
    corners, containment (children) and adjacency. HUD bars along the frame edge are flagged.
  - **Change line** per transition, matched by shape hash: `obj 4 moved dx=0 dy=-5 · obj 9 vanished · 2 cells recoloured`.
- Written by us, following the format of Tufa's `vision_context.py` / `segmentation.py` / `grid_utils.py`. Their
  code is not copied: the licence is UNCONFIRMED (the bundle has no top-level LICENSE).

**2. Three tools.**
- `ipython`: think and compute. Read-only `scene` in the REPL (`scene.objects`, `scene.ascii(r0,c0,r1,c1)`,
  `scene.history(n)`: past scenes as objects, not raw grids). It cannot spend actions.
- `act(actions, plan, hypotheses?, findings?, goal?)`: 1–5 actions. `1` up, `2` down, `3` left, `4` right, `5` space,
  `6 r c` click, `7` undo, `reset` restarts the level. `plan` is required. Hypotheses carry a status set by the
  agent (`proposed` / `verified` / `refuted`) and evidence step ids. The host records them and judges nothing. There
  is no `expect` field and no host prediction check.
  The batch runs one action at a time. Each step's change line goes in the result. The image and the scene
  are for the state after the last step. The batch stops early at a level-up or a GAME_OVER.
- `recall(query, scope?)`: searches memory and skills (timeline by step or level, hypotheses, findings, lessons
  graph, skills). Free.

**3. Memory** (`prime/memory/`, one file per sub-memory under the run dir):

| # | Memory | Written by | Cleared at level-up | Cleared at new game |
| --- | --- | --- | --- | --- |
| 1 | Timeline (every step, change line, act fields) | host | no | archived |
| 2 | Plan (replaced on each act) | agent | yes | yes |
| 3 | Hypotheses (status set by the agent) | agent | yes, after promotion | yes |
| 5 | Open questions (from the last 20–25 actions) | curator | yes | yes |
| 6 | Level findings | agent | yes, after promotion | yes |
| 7 | Goal (always in context) | agent; host appends "confirmed: won level N" | no | yes |
| 8 | Lessons graph (shapes by hash, rules, goal patterns, skills; edges moves / blocks / wins-when / appears-in-level / contradicts) | hidden curator | never | never (kept across games) |

(#4 Expectations was dropped with `expect`.) At a level-up, the curator promotes items the agent marked `verified`
into the lessons graph, then memories 2, 3, 5 and 6 are erased.

**4. Context, rebuilt every turn in this order:** system + tools (~2000 tok) · goal (150) · plan (200) · ≤3 skills (600)
· hypotheses (400) · findings (300) · lessons subgraph for shapes now on screen (400) · top 3 open questions (100) ·
observation: change lines + scene or crop + image (800) · recent turns (~20k). Compaction and auto-refine run as
usual. Auto-refine is the curator: it turns lessons into short, high-level skills and picks which ≤3 to load.

**5. Skills:** three knowledge files, not how-to guides: what a world model is, planning, and first-principles
questions about game dynamics.

**Removed:** the `plan`, `remember`, `delegate`, `message` and `reset_level` tools, `expect`, `was_right`,
`agent/Planner.py`, `runtime/skills/worldmodel.py`, and `arc.observe/transitions/diff` access for the agent.
Subagents are off (`max_depth: 0`). Every mechanism has a config switch, so arms are config diffs.

## Why it should work

In E005/E006 the agent spent most of its turns pulling and parsing state (observe, print grids, diff), and whatever
it learned lived only in the scrolling transcript, which compaction then cut. Pushing a pre-segmented observation plus an image
after each transition removes the pulling. The object-level change line turns each action into a readable cause and
effect. Pinning goal, plan, hypotheses and findings at the top of every turn keeps the agent's beliefs in view after
compaction, and the lessons graph carries shape→rule knowledge to later levels and games. The result should be fewer
repeated (state, action) pairs and fewer blind actions per level, and RHAE is driven by exactly that.

## Prediction

Committed before the run. Single seed, so a small delta is not a result.
- **Local plumbing** (Qwen3.5-4B + mmproj, ls20, 30 actions): every `act` result is followed by exactly one image
  message and a scene or crop. The rendered context blocks stay under their caps on every turn. Nothing crashes. No score claim.
- **Kaggle** (Qwen3.8-27B-FP8, 17 dev games, 400 actions/game):
  - at least **4 levels** cleared in total (E005: 1 of 76 on 10 games);
  - the repeated-(state, action) share under **10%** of actions;
  - at least one lessons-graph item promoted and later recalled or loaded in a different level.

## Kill criterion

Binding.
- **Kill the design** if the Kaggle 17-game run clears **≤1 level**, or if in more than half of the levels played the
  agent never records a hypothesis (the memory is not being used).
- **Kill the image channel** (fall back to text only, recorded as its own result) if vLLM does not start with
  `--limit-mm-per-prompt '{"image": 1}'` on the FP8 build, or if a turn with the image is more than 2× slower than
  the same turn without it.
- More than 20% of `act` calls failing on argument errors means the schema is wrong. Fix it before scoring; do not score around it.

## Arms (config diffs of one code path)

| Arm | Change | When |
| --- | --- | --- |
| A | full design, `qwen_fp8` + image | first |
| B | A without image (`perception.image: false`) | only if A clears ≥ 2 levels, to show what the image adds |
| C | A without lessons graph (`memory.lessons: false`) | same condition. This is the memory-thesis ablation |
| D | A on `nvidia/Qwen3.8-27B-NVFP4` | after the owner uploads it and a vLLM start test passes (needs a newer vLLM than 0.19: UNCONFIRMED) |

## Measurement

- `trace.py` report per game: levels, actions per level, repeated (state, action) share, resets, hypotheses per
  level with their status changes, findings, goal changes, lessons promoted and reused, context block sizes per turn,
  image tokens per turn, turns per action.
- Wall clock per game and per turn (prefix-cache hit rate: the pinned blocks change each turn, so expect far below
  E007's 0.63).
- Ledger row after the Kaggle run. Arms B and C are reported against A on the same games.

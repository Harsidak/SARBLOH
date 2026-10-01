# E008 plan (BUILT 2026-10-01, not yet run on Kaggle)

Read `hypothesis.md` first. Steps 0-7 are built and their suites are green (`tests/run_all.py`: 23/23). Order is
bottom-up: every step has its own test before the next one starts. `rlm/` stays unchanged. No commits by Claude: the
owner commits and updates `sarblohagent`.

## As built: where it differs from the plan below
1. **Observation placement.** The memory block (goal, plan, skills, hypotheses, findings, lessons, open questions) is
   pinned after the system prompt. The observation is not pinned there: it is pushed as the user message right after
   each act result, so the newest state is the last thing the agent reads (owner: "immediately provide the next state").
2. **System prompt + tools ~2.6k tokens** (prompt ~1.77k, tool schemas ~0.83k), over the 2k target.
3. **Level-start observation ~1.5k tokens** (full 64x64 letter board + every object), over the 800 cap by design;
   after an act it is capped at 800 (objects in the changed region + crops).
4. **PNG encoder is stdlib (zlib)**, not PIL: nothing to install offline.
5. **Promotion at level-up:** verified hypotheses and findings -> `rule` nodes, refuted hypotheses -> `refuted` nodes
   (so they are not proposed again), the goal -> a `goal` node; each linked to the shapes it names as "obj N".
6. **REPL:** `arc` is a stub that raises a pointer to `scene`; the host also refuses `arc.*` requests in E008.
7. **vLLM:** a vision profile must pass an image smoke test (one PNG request) or the chain moves to text-only; the
   watchdog never falls back across image support.
8. **Launcher** (`kaggle/experimental/002_prime.ipynb`, `scripts/kaggle_settings.json`, both gitignored): 17 dev games,
   concurrency 6, 5 h budget, 400 actions per game, FP8 snapshot attached. Pace and KV fit are UNCONFIRMED.
9. **Deleted:** `prime/agent/Planner.py`, `prime/runtime/skills/worldmodel.py`, E006's `dedicated` tool schemas, handlers
   and prompt family, and the E006 version of `tests/unit/test_prime_tools.py` (rewritten for E008). All are in git
   history at f117ca4; `prompts_backup_20260930.py` stays.

## Step 0: serving (Qwen3.8-27B-FP8 with images)
- `prime/llm/qwen.py`:
  - add profile `qwen_fp8_vision` = `qwen_fp8` with `--limit-mm-per-prompt '{"image": 1, "video": 0}'`;
  - set `profile_chain = ["qwen_fp8_vision", "qwen_fp8"]`, so a failed image start falls back to text only.
  - The uncommitted `profile_chain=["qwen_flash"]` change is replaced. **Owner: commit or drop it first.**
- `prime/llm/spec.py`: a `vision: bool` field per profile, which `run.py` passes to the agent (image off when the
  served profile has no image).
- `prime/config.py`: `"model": "qwen"`, new keys `perception.{image, upscale, ascii, segmentation, crop_margin}` and
  `memory.{lessons, curator}`.
- Test: `tests/unit/test_llm_profiles.py` checks the flags. On Kaggle, a startup test of `qwen_fp8_vision` plus one
  image request (arm A's first cell).

## Step 1: perception (`prime/agent/vision.py`, `prime/agent/intuition.py`)
- `vision.py`:
  - `render_png(grid, upscale=4) -> bytes`, `data_url(grid) -> str`, `image_message(grid, step) -> dict`;
  - the colour map is the ARC standard palette;
  - PIL is already a dependency (UNCONFIRMED on Kaggle's image: check in step 0).
- `intuition.py`:
  - `ascii(grid, box=None)`: letter per colour, legend, row/column labels;
  - `segment(grid) -> list[Obj]` (id, colour, hash, size, bbox, corners, children, adjacent, hud);
  - `change(prev_objs, objs, prev, grid) -> str` (moved / appeared / vanished / recoloured, matched by hash);
  - `crop(prev, grid, margin=3)`: ASCII of the changed box;
  - `scene(grid, prev=None, level_start=False) -> Scene` with `.text()` capped at 800 tokens.
  - Our own code, based on the existing `runtime/skills/perception.py` plus Tufa's output format. No Tufa code copied.
- Test: `tests/unit/test_perception_views.py`, on hand-made grids:
  - a moved object gives `dx, dy`;
  - containment and adjacency are right;
  - colours 10–15 keep columns aligned;
  - the crop has a 3-cell margin;
  - the PNG is 256×256 and decodes.

## Step 2: memory (`prime/memory/`)
- `store.py`: one JSON/JSONL file per sub-memory in `runs/<run>/memory/<game>/`. The lessons graph is in
  `runs/<run>/memory/lessons.json` (all games).
- `timeline.py` (host append), `working.py` (plan / hypotheses / findings / goal / open questions, with caps), and
  `lessons.py` (graph: nodes shape, rule, goal-pattern, skill; edges moves, blocks, wins-when, appears-in-level,
  contradicts; `subgraph(hashes)`).
- `lifecycle.py`:
  - `on_level_up`: promote the items the agent marked `verified`, erase 2/3/5/6, append the goal confirmation;
  - `on_new_game`: archive the timeline, erase everything except the lessons graph.
- Test: `tests/unit/test_prime_memory.py` covers the lifecycle and the caps (a block over its cap is trimmed, oldest
  first).

## Step 3: tools (`prime/agent/tools.py`)
- `toolset: "e008"` gives `ipython` (read-only text), `act` and `recall`. The E006 `dedicated` toolset is removed.
- `act` handler in `agent.py`:
  - parse the actions (`reset` → action 0 under the existing reset rules);
  - write plan, hypotheses, findings and goal to memory;
  - run the actions one by one through `ArcHost._step`, collecting each step's change line from `intuition.change`;
  - stop at a level-up or a GAME_OVER;
  - return the text result, then append one user message with the image and the scene (a level-start scene after a
    level-up).
- `recall(query, scope)`: keyword search over the timeline (by step or level), hypotheses, findings, lessons and skills.
  At most 1500 tokens back.
- REPL:
  - `arc.observe/transitions/diff` are refused for the agent;
  - a read-only `scene` object is injected after every act;
  - `runtime/skills/worldmodel.py` is deleted and its import removed from `kernel.py` BOOT_CODE.
- **Deletions, named:** `prime/agent/Planner.py`, `prime/runtime/skills/worldmodel.py`, the `PLAN/REMEMBER/DELEGATE/
  MESSAGE/RESET` tool schemas. `prompts_backup_20260930.py` stays.
- Test: update `tests/unit/test_prime_tools.py`:
  - the schema has 3 tools;
  - `act` with a bad id fails without spending;
  - a 3-action batch gives 3 change lines, 1 image message and 1 scene;
  - the batch stops at a level-up.

## Step 4: context builder (`prime/agent/context.py`)
- `build(memory, skills, scene, recent) -> messages`, in the hypothesis order and caps. The pinned blocks go into
  one user message placed before the recent turns. It is rebuilt every turn and never stored in the transcript.
- Only the newest image is kept; older image parts are replaced with a text stub (vLLM allows 1 image per request).
- Compaction (`compaction.py`) runs on the recent turns only. The pinned blocks are memory, so it never summarises them.
- Test: `tests/unit/test_prime_context.py` covers the order, the caps, one image at most, and survival after
  compaction.

## Step 5: skills and curator
- `prime/agent/skills/world_model.md`, `planning.md`, `first_principles.md`: knowledge, each ≤200 tokens.
- Curator = the existing auto-refine (`refine.py`) with a new prompt. It runs at each level-up, after each compaction,
  and every 25 turns. It:
  - writes open questions from the last 25 timeline steps;
  - promotes verified items into the lessons graph;
  - writes or updates skills from lessons;
  - picks the ≤3 skills to load, matched to the shapes on screen and the level stage.
  - The agent never sees the curator's turns.
- Test: `tests/unit/test_prime_curator.py`, with a fake LLM: the curator output is parsed into memory and a bad output
  is ignored.

## Step 6: prompt (`prime/agent/prompts.py`)
- New system prompt of about 2000 tokens:
  - it covers what the observation is, the 3 tools, the action mapping, and that the agent owns plan, hypotheses and
    goal;
  - it says: never ask for the grid; it is pushed after every action;
  - reasoning and `ipython` are free, and only actions count.

## Step 7: trace and run
- `trace.py` adds hypotheses and statuses per level, promotions, recalls, block sizes, image tokens, and the repeated
  (state, action) share.
- Local plumbing: Qwen3.5-4B with `-Vision` (mmproj), ls20, 30 actions. The check is plumbing only.
- `uv run python tests/run_all.py`: all suites green. Then E008 `local_smoke.md`.
- Hand-off to the owner: commit, update `sarblohagent`, push `kaggle/experimental/002_prime.ipynb` with `model: qwen`,
  `toolset: e008`, games = dev 17. Claude does not push the kernel.

## Open decisions (defaults used unless the owner says otherwise)
1. **Per-transition delivery.** A 5-action batch gets 5 change lines, but only one image and scene, for the final
   state. The alternative is one image per step: 5× the image tokens, and vLLM would need `image: 5`.
2. **`reset` as an `act` action.** The owner's mapping has no reset, but GAME_OVER needs one.
3. **Upscale ×4 (256 px).** Tufa uses ×16. At ×4, Qwen's vision encoder sees about 64 image tokens (UNCONFIRMED; the
   processor may resize).
4. **Subagents off.**
5. **Prefix cache.** The pinned blocks sit above the recent turns, so each turn re-prefills about 22k tokens (about
   2.5 s at E007's 8.5k tok/s). Accepted for now. The alternative is to put the pinned blocks after the recent turns
   (cache-friendly, but not the agreed order).

## Risks
- The image processor or chat template on vLLM 0.19 with FP8 is untested. Step 0's single-image request settles it.
- 17 games × 400 actions inside the 12 h budget has not been measured. `concurrency: 8` is as in E005.
- The Tufa licence is UNCONFIRMED. We copy their format, not their code.

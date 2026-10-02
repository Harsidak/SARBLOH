# Phase 1 — Research notes: perception for ARC-AGI-3

Scope: how should code turn an ARC-AGI-3 frame (or a screenshot of it) into text that a
language-model reader can *interpret*, not just read?

## A. Facts about the environment (verified by running the official engine locally)

- Toolkit `arc-agi` 0.9.9 + `arcengine` 0.9.3 (Python >= 3.12) run games fully offline.
- Game sources found vendored in public repos:
  - 25 official games: github.com/BDR-Pro/arc-prize-2026-arc-agi-3 (`arc_games/`)
  - 30 synthetic ARC3-compatible games, 210 levels: github.com/Felix561/arc3-synthetic-games
  - ~30 community games: github.com/theredbluepill/arc-interactive
- Frame = 64x64 int grid, 16-colour fixed palette (index 0 white ... 15 purple).
  Palette hex map from `arc_agi.rendering.COLOR_MAP`.
- **Our two screenshots identified exactly (4096/4096 cells):** game 1 = `ls20` (level 1),
  game 2 = `r11l` (level 1). r11l only accepts ACTION6 (click), so my earlier "press UP" was wrong.
  The white strip at the left of screenshot 2 is UI chrome; the right column is game content
  (arc_eye's trim rule picked the wrong side).
- Engine gives ground truth for free:
  - deterministic transitions; `copy.deepcopy(game)` works -> counterfactual "what would each action do"
  - `set_level(i)` -> any level reachable without solving earlier ones
  - `current_level` sprites: (name/tag, x, y, pixels, layer, visible) -> exact object instances
  - `_get_valid_clickable_actions()` -> ground-truth clickable objects
  - walls in ls20 are 5x5 tile sprites -> the "5x5 lattice" we inferred is literally how the designer built it
- Implication: screenshot -> palette grid is **solved** (100% on both screenshots once the palette is known).
  The open problem is the *interpretation layer*, which is what the experiments target.

## B. Official benchmark facts (ARC-AGI-3 technical report, arXiv 2603.24621)

- Games use only Core Knowledge priors: objectness, geometry/topology (symmetry, rotation,
  inside/outside, holes), basic physics, agentness. No numbers, letters, real-world clip-art.
- Scoring RHAE: per level min(1, human_actions / ai_actions)^2, later levels weighted more.
  => every exploratory action costs score. Perception that needs probing actions has a real price.
- Frontier LLMs (Mar 2026) score < 1%. Cited limitation (Duke): naive rolling window of 64x64
  frames exhausts context -> representation size matters, not only accuracy.

## C. What competitors use for perception

| System | Perception | Lesson |
|---|---|---|
| StochasticGoose (1st, preview 2025) | CNN on raw grid predicts *which actions change the frame* | perception = learned action-effect filter; no semantics |
| Blind Squirrel (2nd, preview) | same-colour connected components = "buttons"; valid-action filter; state graph | components as click targets |
| Graph explorer (3rd, arXiv 2512.24156) | segments frame, salience-prioritised actions, directed state graph | 30/52 levels without learning |
| Tufa "Duck" (1st, Milestone 1, 2026) | rendered image + raw ASCII grid + zoom/segmentation tool; LLM writes code in REPL; "no-impact" detector for HUD-only changes | multi-view; let the reader zoom |
| Reki / forge (2nd/3rd, Milestone 1) | labelled rendered images to Gemma-4-31B; numpy click heuristics favour small rare-colour button-like shapes | vision LLM + priors |
| BDR-Pro (open) | volatility mask (cells changing in >=20% of frames = clock/HUD), diff-bbox mover detection, action->delta votes for avatar | temporal statistics separate agent / clock / scenery |

Pattern: winners either avoid semantics (CNN / graph search) or hand the LLM raw pixels.
Nobody publishes a perception layer that outputs an *interpretation* (roles, correspondences,
testable predictions). That gap is the target.

## D. Relevant research

- Xu et al. 2023, "LLMs and the ARC: object-based representations" (arXiv 2305.18354):
  GPT-4 solved 13/50 with text grids; object-based (ARGA) representation nearly doubled solves.
  Sequential text order strongly affects object identification. -> objects > grids.
- "Stuck in the Matrix" (arXiv 2510.20198): LLM spatial accuracy drops avg 42.7% (up to 84%)
  as grid size grows; transformation tasks collapse to 0% on large grids. -> 64x64 is too big;
  shrink the board the reader must reason over.
- Grid Spatial Understanding (arXiv 2603.17333): set/text-sentence encodings beat plain
  coordinate matrices; models miscount structures. -> describe, don't make the reader count.
- Learning to Draw ASCII (arXiv 2604.14641): accurate ASCII maps help reasoning; reading is
  easier than writing. -> provide a *correct* small map; never ask the reader to rebuild one.
- Contingency-aware exploration (Choi et al., arXiv 1811.01483): the part of the screen predictable
  from your own actions = the self. -> agentness is an interventional fact, not a visual one.
- EMPA / theory-based RL (Tsividis et al., arXiv 2107.12544): human-like learning by inferring
  object types, interactions and goals as a generative theory; minutes of play. -> perception
  should output a *theory* with types/roles, updated by evidence.
- WorldCoder (arXiv 2402.12275): world model as code, optimism under uncertainty.
- CompressARC (Liao, arXiv 2512.06104): solving ARC by minimum description length, no pretraining.
  -> the shortest description of a frame tends to expose its generative parts.
- OO-MDP (Diuk 2008), Schema Networks (Kansky 2017): object-relational state makes dynamics simple.
- Gestalt grouping (common fate, similarity, proximity, closure): objects = what moves together.

## E. Working thesis going into Phase 2

A frame alone under-determines meaning ("which thing is me?", "what is the goal?").
Interpretation needs three ingredients the current arc_eye lacks:
1. the game's own building blocks (sprites/tiles), not raw connected components;
2. relations that carry goals in ARC games (same shape / same colour / scaled / rotated copies,
   containment, alignment);
3. interventional evidence (what responds to which action) — ideally from transitions that
   happen anyway during play, so it costs no extra actions.

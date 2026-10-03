# E119 — rebuild the game in arcengine from observations, then plan in the rebuilt copy

- **Date opened:** 2026-10-03
- **Axis varied:** substrate (world-model language) + verification
- **Baseline:** current Sarbloh-Experimentation agent (LLM-written generic-Python `step()`), same model, same games
- **Split:** heldout (`eval/splits/heldout_split.json`: ar25 cd82 ft09 g50t m0r0 s5i5 sk48 vc33). Dev games may be
  used to build and debug the probe; the reported number comes from heldout only.
- **Status:** written 2026-10-03 at the owner's order. Not run. Legality question (below) open.

## The problem

The agent already writes a world model in Python and plans in it. It writes that model from scratch, in generic
Python, with no knowledge of how ARC-AGI-3 games are actually built. So every game starts from a prior over
"any physics", and the agent pays environment actions to narrow it.

The 25 public games in `eval/real_games/` are MIT-licensed Python (~70.7k lines total), every one a subclass of
`arcengine.ARCBaseGame` built from the same parts: `Sprite(pixels, collidable, tags)`, `Level`, `Camera`,
`GameAction`. UNCONFIRMED but likely: the private games are written the same way, on the same engine, by the same
designers.

## Hypothesis

If the agent writes its world model **as an `ARCBaseGame` subclass in arcengine's own API**, using the public
games as examples of how designers write games, then:

1. the first frame of a level is close to the `Level` and sprite data, so the layout is recovered with zero actions;
2. the remaining unknown shrinks to a small block of step logic written in the designers' own vocabulary;
3. a candidate that replays every recorded transition exactly in the real engine is a perfect simulator, and a
   BFS/A* inside it gives the optimal path, which can score up to the 1.15 per-level cap.

The prior shrinks from "any physics" to "how these designers write games in this engine". That should cut the
exploration actions, which are the RHAE loss term the current harness has not closed.

## Information firewall (non-negotiable)

The agent may learn about a game **only** from what `step()` returns: frames, state, levels completed.

Allowed:
- a clean, vendored copy of `arcengine` inside the agent dataset, importable in the REPL kernel;
- the public game sources as examples, and a LoRA trained on them and on mutations of them (later stage).

Forbidden, and disqualifying:
- reading any file under the competition game directory (game `.py`, level data, metadata);
- any reference to the live game object or its process memory from the kernel;
- any channel carrying information about the hidden game other than observations.

Enforcement: `harness/game/arc_host.py` stays the only game access and passes frames + action space only.
A component test in `tests/component/` asserts that the kernel namespace holds no env/game object and that no
kernel file access touches the game directory. The test result goes in `verdict.md` as evidence for reviewers.

Before any score is claimed in the paper: ask ARC Prize / the Kaggle forum in writing whether a pre-installed
arcengine in the agent sandbox, with games rebuilt from observations only, is permitted. Record the answer here.

## Mechanism

Probe stage only (no harness change yet):

1. **Step 0 — structure count (no model).** For each of the 25 sources, count lines of engine boilerplate
   (imports, `Sprite`/`Level` literals, rendering) vs custom game logic (`step`/action handling, win checks,
   counters). Report the median custom-logic size and how many distinct logic patterns recur.
2. **Decompilation probe.** For each heldout game: record a trajectory with the current agent (frames + actions).
   Give the model the 17 dev-game sources as examples, the engine API, and the trajectory. Ask for an
   `ARCBaseGame` subclass. Instantiate it in the real (vendored) engine, replay the recorded actions, compare frames
   exactly. Up to 5 repair rounds, each fed the first mismatching transition. Run with the 27B, and once with the
   strongest model available as an upper bound.
3. **Zero-action layout check.** From the first frame of each heldout level only, ask for the `Level` + sprite
   definitions; render them in the engine; compare to the frame pixel for pixel.

Later stages, only if the probe passes: `rebuild()` REPL skill in `harness/runtime/skills/`, plan in the certified
copy, commit with halt-on-first-misprediction; then a mutation corpus (recombined public sources, each runnable and
labelled with its source) and a LoRA via `Alignment/lora_sft.py`.

## Prediction

Committed before the run:
- Step 0: median custom logic under 300 lines per game, with recurring patterns across games.
- Probe (strong model): at least 3 of 8 heldout games rebuilt to an exact replay of the recorded trajectory.
- Probe (27B, no LoRA): at least 1 of 8.
- Layout check: at least 80% of heldout first frames rendered pixel-exact.

## Kill criterion

Binding. Kill if, after the probe:
- the strong model rebuilds fewer than 2 of 8 heldout games exactly, or
- the layout check is under 50% pixel-exact,
because then the engine-language prior does not transfer to unseen games and the LLM-written generic `step()`
stays primary.

## Measurement

- Exact-replay count per heldout game, repair rounds used, first mismatching transition per failure.
- Lines of generated logic vs the true source's custom-logic lines.
- Wall clock and tokens per rebuild (it must fit the 12 h Kaggle budget across the private set).
- Only if a harness stage runs: heldout RHAE via `eval/official_score.py`, actions to first certified model,
  per-level agent/human action ratio.

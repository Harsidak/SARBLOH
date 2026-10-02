# Round 2 research — vision science + VLM failure modes

Constraint from Lt.: market already does (a) 10x scaled screenshot, (b) current+previous diff image,
(c) colour-coded segmentation. New experiments must go beyond these, single screenshot, no frame diff.

## Why VLMs miss what humans get at a glance

- Vision language models are blind (Rahmanzadehgervi et al., arXiv 2407.06581, ACCV 2024):
  4 SOTA VLMs average 58% on trivially easy geometry (overlaps, intersections, counting circles).
  Two findings that drive design:
  1. failures concentrate where primitives are CLOSE or overlapping; adding space -> near-perfect.
  2. linear probes show the vision encoder HAS the information; the language decoder fails to
     turn it into words. => do the decoding in code, hand the model words + spaced visuals.
- V* / SEAL (arXiv 2312.14135): MLLMs miss small details in crowded/high-res images; a guided
  zoom-in search fixes it. => small icons (HUD glyphs) need dedicated enlarged views.
- Set-of-Mark (arXiv 2310.11441): numbered marks on regions make objects "speakable"; GPT-4V
  coordinate grounding 25.7% -> SoM far higher. Mark placement matters. (Already market-adjacent.)
- Visual Sketchpad (arXiv 2406.09403): letting models draw auxiliary lines/boxes gives +12.7% math,
  +8.6% vision. => auxiliary constructions (lines between things, traced paths) carry reasoning.
- Whiteboard-of-Thought (arXiv 2406.14562): text-only CoT on ASCII art 21.6% vs 66.0% when the model
  sees a rendered image; "visual reasoning tasks demand visuals". => never make the model rebuild a
  picture from text grids.
- Grid text studies (arXiv 2510.20198, 2603.17333): accuracy collapses with grid size; sentence
  descriptions beat coordinate dumps.

## Human vision science that is NOT used in the ARC-AGI-3 market

- Ullman, Visual Routines (1984): relations like inside/outside, connectedness, "same curve",
  reachability are NOT pre-attentive; humans compute them with serial routines: indexing, bounded
  activation (colouring/flood fill), boundary tracing, curve tracing, marking. VLMs are bad at
  exactly these. => run the routines in code and show their results.
- Treisman feature integration / pop-out + Bruce & Tsotsos AIM (NeurIPS 2005): saliency =
  self-information (statistical rarity) of local features. => rank what matters by surprise.
- Marr / vision as inverse graphics: perception recovers the generative description of the scene.
- Biederman recognition-by-components: shapes recognised via a small vocabulary of parts (geons);
  naming shapes ("ring", "plus", "L") is how humans compress them into words.
- Gibson affordances: we perceive what we can DO (walkable, blocked, clickable), not pixels.
- Theory-based RL / EMPA (arXiv 2107.12544): humans infer object TYPES and ROLES from a few
  observations using strong priors about agents, goals, obstacles.

## Gap

Market pipelines give the model pixels (scaled, diffed, coloured). None precompute the human
"unconscious" layer: routines (reachability, enclosure, tracing), rarity ranking, correspondences
with the transform that relates them, imagined outcomes, or role hypotheses. The decoding failure
(BlindTest) says this layer is exactly what the model cannot do itself.

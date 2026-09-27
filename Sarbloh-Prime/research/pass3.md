# Pass 3: main components, their roots, and the code

**Builds on:** `pass1.md` (what AVO is: AVO-paper for kernels and AVO-ARC from the blog) and `pass2.md` (how it
works). Pass 2 found the reusable core, three pieces the NVIDIA text leaves partly unspecified:
1. an **edit-evaluate-diagnose loop with a binary correctness gate**;
2. **two-level memory**: the clean committed lineage plus the messy working history;
3. a **conditional supervisor** that fires on stagnation.

It also chose seven main-component terms for this pass. The domain-only kernel terms are deliberately left at
their one-line meanings.

**What this pass is:** for each main term, go to *its* source paper or code, the same way we treat ARC, learn
just enough to understand AVO's choice, and stop. There is no recursion beyond one hop. Then check the community
repo against the paper, and end with what this means for SARBLOH. **Everything in §4 is a hypothesis, not a
decision.**

**New resources snapshotted today (2026-09-26), all indexed in `resources/INDEX.md`:**
- `papers/arxiv/`: 2603.24517 AVO, 2506.13131 AlphaEvolve, 2512.24077 LoongFlow, 2601.16175 TTT-Discover,
  2607.28287 Tycho, 2607.15439 Rodionov ablation, and 2603.05451 FA4 (domain only).
- `web/2026-09-26_nvidia_avo-arc-agi-3-blog.md`
- `repos/vista` (MIT, 900aa33) and `repos/tycho` (Apache-2.0, f68912a). The existing `repos/avo` is now labelled
  correctly as a **community** reproduction.

---

## 1. The seven main terms, one hop deep

### T1. Variation operator and LLM-driven evolution: what AVO replaces

| System | How `Vary` works | Who controls the loop | What we take from it |
|---|---|---|---|
| **FunSearch** (Nature 2024) | LLM writes one program from sampled parents; islands of programs | Framework | The original "LLM as mutation" idea |
| **AlphaEvolve** (2506.13131, DeepMind) | *"orchestrates an autonomous pipeline of LLMs"*; the evolutionary database uses MAP-Elites and islands; the prompt sampler picks parents plus "inspirations"; one or more evaluators | Framework | Evaluators can be a *cascade*: cheap checks first, expensive ones only for survivors |
| **LoongFlow** (2512.24077, Baidu) | Fixed **Plan → Execute → Summarize** per step; MAP-Elites + islands + **adaptive Boltzmann selection**; a "hybrid evolutionary memory" to avoid stagnation. Code: `baidu-baige/LoongFlow` | Framework, with a fixed script | Stagnation handled by *diversity niches* instead of a supervisor |
| **TTT-Discover** (2601.16175, Stanford + NVIDIA) | RL **at test time**: the LLM's weights are trained on this one problem, 512 samples per step, PUCT for choosing states | Framework plus a learned policy | Uses an **open model (gpt-oss-120b)** for about a few hundred dollars per problem. It is the only one in this family that is not a frontier API model |
| **AVO** | `Agent(P, K, f)`: one agent decides everything | **The agent** | The agent chooses what to read, change and test |

**MAP-Elites**, in one line: keep a grid indexed by *behaviour descriptors* (for example program length or
speed) and store the best solution in each cell, which keeps diversity alive. AVO ran a single lineage, so it has
**no diversity mechanism except the supervisor**.

**What this means for us:**
- The design space runs from *framework controls everything* (AlphaEvolve) to *agent controls everything* (AVO).
- A 27B local model is much weaker than the agent NVIDIA used. The right point for us may be **in between**: a
  framework-controlled loop with an agent inside each step. That is **UNCONFIRMED**, and it is exactly what the
  RL lesson in CLAUDE.md §0 warns about: do not copy the cluster-scale method.

### T2. Deep agents and coding agents

AVO cites SWE-bench, SWE-agent, OpenHands, Claude Code and Codex. What they share is an LLM plus tools (edit, shell,
search) plus a loop that keeps going until done, with the conversation as working memory. SWE-agent's lesson is
the **agent-computer interface (ACI)**: *what* the tools return and how compactly matters as much as the model.
For ARC, VISTA is the ACI (T6). Our Duck-based setup already has this shape, with Qwen, a Python sandbox and a
`step_env` tool, so **we do not need to build a coding agent from scratch**.

### T3. Lineage and the commit gate

In the paper, the lineage is a git history of committed versions with scores. The gate is *correct AND ≥ best so
far*. Failures stay in the agent's history, not in the lineage. Its roots are **elitist** evolution (never lose
the best) plus **(1+1) evolution strategies**, which keep one parent and accept the child if it is at least as
good. So AVO's population policy is the oldest and simplest one there is. **All the novelty is in the operator.**

In SARBLOH terms, the gate is our `parkh`: a binary verdict, no partial credit. The "matches" clause matters
because refactors on a plateau enable later jumps (pass 2, §4.4).

### T4. Persistent memory across context limits

The paper only says "conversation history", and the blog says it survives beyond one context. The mechanism is
unpublished. The closest published mechanism is **VISTA's**, and NVIDIA says it borrowed VISTA's interface. From
`repos/vista` and the VISTA site:
- **Lossless frame archive.** Every returned frame is stored with its (turn, frame index) and fetched on demand
  (`inspect`, `read_pixels`, `history`). The context window is *not* the store. **This is exactly our `surat`
  contract** (CLAUDE.md §3: "lossless frames outside context, retrieved on request at frame/region/pixel").
- **Two notes files:**
  - `GUIDE.md`: *"concise, durable, revisable game understanding"*, i.e. rules across levels;
  - `WORKING.md`: a scratchpad for the current level.
  These are close to our level and environment clocks, but without our "only certified facts get promoted" rule.
- **Compaction handoff.** Before the context is compacted, a hook (`claude/compact_hook.py`) *blocks* compaction
  until the player writes a **continuation checkpoint**. A fresh context then resumes from that checkpoint, the
  notes and the current frame (`claude/recovery.py`).
- The instruction *"Before each `play`, briefly state what you expect to see. Afterward, briefly state all
  visible changes, expected or not."* is a **prediction-then-check** habit at the level of every action: a soft,
  in-prompt version of verification.

### T5. The supervisor and stagnation detection

- The paper's trigger is **NOT STATED**. Its output is several directions, based on a review of the whole
  trajectory. LoongFlow handles the same problem with diversity niches; AlphaEvolve with islands.
- The community repo's choice (see §2): fire after **3 consecutive steps without a new best**, with the output
  being text from a separate "supervisor" agent that may not edit code.
- VISTA has **no supervisor**. It does have recovery prompts for runtime failures and compaction, which are about
  continuity, not strategy.
- For ARC, "stagnation" has a natural measurable signal we already track: **actions spent with no level progress
  and no new certified fact**, plus the falsified-hypothesis log that catches *re-proposing a refuted rule*.
  That second signal is the paper's "unproductive cycle" made concrete. **UNCONFIRMED** that it helps; it is an
  experiment.

### T6. The VISTA direct-interaction interface: what AVO-ARC borrowed

Paper: VISTA, Han, Hu, Qiu, Wu, He (MIT, Kaiming He's group), August 2026, site https://vista-research.github.io/,
code `repos/vista` (MIT). No arXiv ID was found.

- **Loop:** observe the current visual → reason, using memory as needed → execute **one** action → observe.
- Observation: a 512×512 PNG (8× upscale plus grid lines). **NVIDIA replaced it with an exact 64×64 text grid.**
  This matters for us: text-only is what our Qwen-class models handle best, and it removes the need for a
  vision tower. It is **UNCONFIRMED** that text beats images for Qwen 3.8 on ARC.
- Scores:
  - Opus 5.0: **100 RHAE, 7,542 actions** against 17,135 human actions.
  - GPT-5.6 Sol: **98.27 or 99** (the site and README differ), 10,063 actions.
- There is no world-model layer.

### T7. Tycho and the Rodionov ablation: the world-model route NVIDIA skipped

This is the most important evidence for *our* thesis, and it is two-sided.

**Tycho** (2607.28287, code `repos/tycho`, Apache-2.0):
- Games are formalised as *parameterised rendered deterministic Moore machines*: state machines whose output
  depends only on the state, rendered to a grid. Tycho separates *actionable* frames from animation,
  level-complete and game-over frames.
- Four orchestration policies were compared on Opus 4.8 with matched budgets. The best is **actor-requested
  delegation** to a model-builder, at **88.49**.
- **Automatic repair on verification failure** produced *more accurate simulators* but **only 83.07**.
  In their words: *"transition match shows whether a simulator reproduces observed dynamics, not whether it has
  identified the objective or whether consulting it improves the next action."*
- With Opus 5: **100.00 RHAE, 6,641 actions**, which is almost identical to AVO-ARC's 6,624 *with no world
  model*.

**Rodionov** (2607.15439, the author of our `ewm-baseline1`):
- Four variants: textual, flexible executable world model, plus simplification, plus exact replay verification.
- **Model capability gains exceed variant differences.**
- Textual beats a flexible executable world model at gpt-5.5.
- The full **verification** variant **ranks first throughout** but *"uses substantially more resources"*, and
  *"succeeds at lower effort"*.

**What this means for SARBLOH:**
- At frontier scale, the world model is *optional*: VISTA, AVO-ARC and textual Rodionov all reach about 100.
- The one hint that points our way: **verification helps most at lower effort**, and a local 27B is "lower
  effort" by a wide margin.
- Tycho adds a warning that fits our `parkh` design: **certifying the dynamics is not certifying the goal.**
  `parkh` passing must not be read as "we know what to do".
- All of this is public-set evidence with post-release models, so it is **UNCONFIRMED** for the private set.

---

## 2. The community repo (`repos/avo`) checked against the paper

Owner instruction: it is reference only and must not be copied, because it may contain mistakes. It is Apache-2.0,
about 4k lines of Python, has kernel, Metal and 2048 targets, and has **no ARC target**. Where it had to *invent*
something the paper leaves out, the invention is listed here so we can decide for ourselves.

| Topic | Paper says | Community repo does | Risk / our view |
|---|---|---|---|
| Memory across steps | One conversation history across the whole run | **A fresh agent session per step**; memory = `NOTES.md` plus the lineage | A real divergence. Failed attempts survive only if the agent wrote them down. It is arguably *more* robust than one giant context, but it is not the paper |
| Commit gate | Correct, and matches or improves the best committed version | `correct and primary >= best` with **no tolerance** (`config.py:137`) | With a noisy `f` (timing), a regression can pass by luck and a true tie can fail. For ARC our `f` would be deterministic (replay), so this matters less |
| Scalar score | Probably the geomean (figures); not stated in §3.2 | `primary` defaults to the geomean of the metrics | Consistent with the figures |
| Supervisor trigger | Stalls **or** unproductive cycles; rule not stated | Only "3 consecutive steps without a new best" (`loop.py:108`, `run.py:328`) | Cycle detection is missing. The window of 3 is a guess |
| Supervisor output | Several directions from a review of the whole trajectory | A separate agent writes several directions; it may not edit (the tree is reverted afterwards) | Matches the paper's text |
| Who runs `f` for the commit | The agent calls `f` and commits | The agent may call `f`, but the **framework re-evaluates** before accepting | A good anti-self-deception choice; we should keep the same principle |
| Population | Single lineage | Single lineage | Same |

**Verdict:** a faithful skeleton of the *kernel* AVO, with three guessed parts: memory, tolerance and trigger.
It has nothing on AVO-ARC. For us it is a useful checklist, not a base.

---

## 3. What AVO really is, in first principles

Strip away the kernels and the brand, and AVO is four rules:
1. **Let the agent own the inner loop.** It chooses what to read, try and measure; the framework does not script
   the step.
2. **Commit only verified progress.** A binary correctness gate plus no regression, and failures are remembered
   but never committed.
3. **Keep memory outside the context**, so the work survives the context window: a lineage (clean), notes and
   history (messy), and an archive of raw evidence.
4. **Watch the trajectory, not the step.** A supervisor steps in only when progress stalls or loops.

Rules 2 and 3 are **already SARBLOH's thesis** (`parkh`, `memory`, `surat`). Rules 1 and 4 are what AVO adds on
top of it.

---

## 4. How this could map onto SARBLOH: hypotheses for the planning step

These are **not decisions**. They are the options the planning step (step 2 of the goal) should choose between,
with the owner. Each one is a hypothesis with a cheap test.

**Option A: AVO-ARC as our in-game agent**, the way NVIDIA used it.
- The Duck-style direct-interaction agent (Qwen, text grid) plus the VISTA-style memory that NVIDIA says it
  borrowed:
  - a lossless frame archive with tools;
  - `GUIDE.md` / `WORKING.md`;
  - compaction checkpoints;
  - "predict before you act".
- Plus the **conditional supervisor**, which fires on actions spent without progress, or on re-proposing a
  falsified rule.
- It is the closest match to what NVIDIA did and the smallest change from the Duck.
- **Risk:** every piece of NVIDIA's evidence used Opus 5. Whether a 27B model can keep its own notes well is
  **UNCONFIRMED**. That is the first thing to measure.

**Option B: AVO as the world-model inducer inside the game.**
- The lineage is our certified `step()` versions.
- `f` is the `parkh` replay vector: one entry per recorded transition, all zero if any replay fails.
- A version is committed if it matches or improves. The supervisor fires when induction stalls.
- This is AVO-paper's mechanism applied to `worldmodel` + `parkh`, and it happens entirely *off* the action
  budget: reasoning is free, acting is not.
- **Risk:** Tycho's warning (accurate dynamics are not the goal) and Rodionov's resource cost. It needs a
  wall-clock governor to fit in 12 h.

**Option C: AVO offline, to evolve our agent itself.**
- Claude Code acts as the operator on this PC. `x` is our agent's prompts, config or code; `f` is RHAE on the
  **tune** split; the lineage is git. Kaggle never sees AVO; it sees only the best committed agent.
- This matches the paper most literally.
- **Risks:**
  - `f` is expensive: a real score needs the 27B on the RTX 6000 (a Kaggle run), and local 4B scores are not
    evidence (CLAUDE.md §6).
  - It is also an overfitting machine unless the held-out split is respected (§4.2) and public-set-only gains
    are rejected (§4.3).

The options can be combined; A and B fit together naturally. **The owner's direction is needed before planning
picks one.**

## 5. Open questions for the owner, raised by the reading

1. Which AVO do we mean for SARBLOH: the agent **system** (A), the **operator** applied inside the game (B), or
   the operator applied to our own code offline (C)?
2. Do we accept the text-grid observation (NVIDIA's choice), or keep images available? The Qwen 3.8 and Gemma 4
   downloads are both image+text models.
3. For the supervisor on a local model, should it be the same model with a different prompt (cheap), or a
   second model (VRAM cost)?

## 6. Stopping point

The recursion ends here on purpose. The next-hop terms that could be chased, and why they are not:
- the MAP-Elites paper and islands: only relevant if we leave single-lineage;
- the PUCT formula: only relevant with tree search, which is `jugat`'s existing backlog;
- Moore-machine theory: only if we formalise games Tycho's way;
- FA4's internals: domain only.

Each can be opened later as a *named* follow-up, never as an open-ended loop.

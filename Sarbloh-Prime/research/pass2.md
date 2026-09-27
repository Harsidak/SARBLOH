# Pass 2: the careful read

**Builds on:** `pass1.md`. There we learned that "AVO" is two things: **AVO-paper** (evolving GPU kernels) and
**AVO-ARC** (playing ARC-AGI-3, blog only). We also left seven questions, Q1–Q7.

**What this pass is:** every section of the paper and every paragraph of the blog, read in order. The answers
come from the text itself, with quotes. Anything the text does not say is marked **NOT STATED**, so we never
fill a gap with a guess and then forget that we guessed. Terms get a one-line plain meaning here; pass 3 decides
which of them deserve a deeper dive.

---

## Part A: the paper, section by section

### §1 Introduction: the problem they attack

In FunSearch- and AlphaEvolve-style systems the LLM *"produces a single output per invocation, with no ability to
proactively consult reference materials, test its changes, interpret feedback, or revise its approach before
committing a candidate."* That is fine for small programs. It is a hard ceiling for code that is already
hand-tuned to the limit, where each further gain needs *"studying hardware documentation, analyzing profiler
output ... diagnosing correctness failures, and revising strategy based on accumulated experience."*

Their bet is that "deep agents" (LLMs with planning, memory and tools, like Claude Code, Codex, SWE-agent and
OpenHands) can already do that multi-step work, so the agent should *be* the variation step.

### §2.1 Background: evolution and the variation operator

Two equations define everything.

**(1) The generic evolutionary loop:**
```
P_{t+1} = Update( P_t, (x_{t+1}, f(x_{t+1})) )        x_{t+1} = Vary(P_t)
```
- `P` is the **population**: a set of (solution, score) pairs.
- `f` is the **scoring function**.
- `Update` adds the new solution, *"possibly pruning low-score members to maintain a bounded archive"*.
- `Vary` is the **variation operator**: whatever turns old solutions into a new one. Classically that means
  *mutation*, a random small change to one parent, and *crossover*, mixing two parents.

**(2) How prior LLM systems implement Vary:**
```
Vary(P_t) = Generate( Sample(P_t) )
```
- **Sample** is fixed code that picks parents *"guided by score-based and diversity-based heuristics"*. For
  example:
  - AlphaEvolve uses an **island-based database inspired by MAP-Elites**, meaning separate sub-populations plus
    a grid that keeps the best solution per "niche" of behaviour.
  - LoongFlow uses MAP-Elites with **Boltzmann selection**: pick parents with probability ∝ exp(score/T).
  - TTT-Discover uses a **PUCT** rule, the AlphaGo tree-search formula that balances value against visit count.
- **Generate** is the LLM writing a candidate.
  - LoongFlow gives it a fixed **Plan → Execute → Summarize** script.
  - TTT-Discover even *trains* the LLM during the search (test-time gradient updates).
  - In every case, *"the sampling strategy, evaluation protocol, population management, and the order of
    operations are all determined by the framework, not by the LLM."*

**A key sentence for us:** *"AVO is orthogonal to the choice of population structure ... In this paper we study the
single-lineage setting to isolate the effect of the operator itself."* So the paper has **no population**,
just one chain `x0 → x1 → … → x40`, and never tests AVO with islands or archives.

### §2.2 Background: attention kernels on Blackwell (domain only)

This section is needed to follow §5 and not needed for ARC. The terms in plain words:

| Term | Plain meaning |
|---|---|
| **Attention** | `O = softmax(QKᵀ/√d)·V`, the core Transformer operation. |
| **FlashAttention / tiling** | Process the keys in blocks so the huge N×N score matrix is never stored. |
| **Online softmax** | Compute softmax block by block, keeping a running **row maximum** and **row sum**. When a new block raises the maximum, the partial output already accumulated must be **rescaled**. |
| **Warp / warp group** | A warp is 32 GPU threads that execute together. A warp group is several warps given one job. |
| **Warp specialization** | Different warp groups do different jobs at the same time: MMA (matrix multiply), softmax, correction (the rescaling), and load/epilogue (data movement through **TMA**, the Tensor Memory Accelerator). |
| **Dual Q-stage** | Two query tiles are in flight at once, so one group's idle time is covered by the other tile. |
| **Causal masking** | A token may only attend to earlier tokens. Some key blocks are fully masked (skip), some fully visible, and some partial. |

The point for us: FA4 is *already* expert-tuned. The paper chose the hardest possible target to show the agent can
go beyond experts.

### §3.1 Formulation: answers Q5

```
Vary(P_t) = Agent(P_t, K, f)                                   (eq. 4)
```
- `P_t = {(x_1, f(x_1)), …, (x_t, f(x_t))}` is the **full lineage**. The agent sees *all* past versions and
  their scores, not a sample.
- `K` is the **knowledge base**: *"CUDA programming guides, PTX ISA documentation, Blackwell architecture
  specifications, and existing kernel implementations including FlashAttention-4 source code."* It is
  documents plus reference code. PTX is NVIDIA's GPU assembly-like language.
- `f` is **vector-valued**: `f(x) = (f_1(x), …, f_n(x))`, with one entry per benchmark configuration. Here
  n = 4 sequence lengths × {causal, non-causal}. Each entry is TFLOPS.
- **Correctness gate (answers Q5):** *"A candidate x_i that fails correctness is assigned zero score (i.e.,
  f_j(x_i) = 0) regardless of throughput."* Correctness is checked numerically against a reference
  implementation, and it is all-or-nothing. **This is exactly our `parkh` idea:** a binary verdict, with no
  partial credit for a fast but wrong kernel.

### §3.2 Anatomy of one variation step: answers Q1 and Q2

**Q1: what happens inside one step.** One step, going from `P_t` to `x_{t+1}`, is *"an autonomous agent
loop"*, and *"a single step may involve numerous internal actions."* The paper observes the agent doing this:
1. It **examines multiple prior versions** in `P_t` and compares their profiling characteristics to find
   bottlenecks.
2. It **consults K** to understand the hardware constraints.
3. It implements a candidate optimisation.
4. It **calls f itself** to test the result.
5. If the candidate fails correctness *or* does not improve, it **diagnoses, revises and repeats**. This is
   the *"edit-evaluate-diagnose cycle"*.
6. It stops when it *"commits a satisfactory x_{t+1}"*.

The strategy also changes over the run: *"early steps may focus on structural changes informed by reference
implementations in K, while later steps can shift toward micro-architectural tuning guided by profiling
feedback."*

**When does the step stop?** When the agent itself judges the candidate satisfactory. There is no fixed
iteration budget per step: **NOT STATED.**

**Q2: the commit rule.** *"we persist a new committed version only when it passes correctness checks and matches
or improves the benchmark score relative to the best committed version so far; unsuccessful intermediate
attempts remain part of the agent's internal search trajectory but are not added to the committed lineage."*
- "Improves" is measured against the **best committed version so far**, not the previous version.
- **Which aggregate is "the benchmark score" is NOT STATED** in §3.2. However, Figures 5 and 6 plot *"running-best
  geometric mean throughput across all configurations"* and mark "new best" on the geomean. So the
  **geometric mean of the vector `f`** is the likely scalar. That is an inference, marked **UNCONFIRMED**.
- **"Matches" is allowed.** A version can be committed with no gain. Figure 5 confirms that many versions sit
  on plateaus. It also means noise can let a slightly worse version in, if "matches" has any tolerance, and the
  tolerance is **NOT STATED**.
- Failed attempts are **kept in the agent's memory** (its "internal search trajectory") but **not in P**. So
  there are two memories: the committed lineage (clean) and the agent's own history (messy, includes failures).

### §3.3 Continuous evolution and the supervisor: answers Q4

- Every committed version is a **git commit** with its score, giving *"full state continuity"*.
- Two named failure modes of long autonomous runs:
  1. **Stalling:** *"the agent may stall when it exhausts its current line of exploration"*.
  2. **Unproductive cycles:** *"it may enter unproductive cycles of edits that repeatedly fail to improve
     scores"*.
- The fix is a **self-supervision mechanism** that *"detects these scenarios and intervenes. Once triggered, the
  mechanism reviews the overall evolutionary trajectory and steers the search toward several candidate
  optimization directions."*
- **Q4 answer:** the trigger rule (how many steps count as stalling, or how a cycle is detected) is **NOT STATED**.
  The output is *"several candidate optimization directions"*: a list of directions, not a single hint. The
  supervisor looks at **the whole trajectory**, not just the last step.
- Division of labour: *"the main agent autonomously decided when to attempt new optimizations, when to revisit
  earlier approaches in P_t, and when to shift strategy, while the supervisor maintained forward progress by
  intervening during periods of stagnation."* The supervisor is **conditional**: it only speaks when there is
  a problem.

### §4.1 Setup: answers Q3

**Q3: what persistent memory is.** *"It maintains persistent memory through its conversation history, which
accumulates the full context of prior edits, compiler outputs, profiling results, and reasoning across the
evolutionary process."* So in the paper, memory = **one long conversation history across all steps**. Three
things are **NOT STATED**:
- how that survives 7 days of context-window limits (summarisation? compaction? retrieval?);
- whether it is one session or many;
- how big the context is.

The blog (Part B) says memory *"carries forward ... accumulated reasoning, allowing the agent to resume from
the current state rather than repeatedly reconstructing the search"*, which implies it is designed to outlive
a single context. The mechanism is still unpublished.

Other setup facts:
- The agent is *"internally-developed general-purpose coding agent powered by frontier LLMs"*; the model is not
  named.
- Tools: code editing, shell, file navigation, documentation retrieval.
- **"No task-specific modifications are made to the agent for kernel optimization"**: the same agent is used
  for general software engineering. Only K and f are task-specific. This is the "generality" claim the blog
  builds on.
- Hardware: B200, CUDA 13.1, PyTorch 2.10.0.
- Benchmarking: FA4's own timing script; 10 runs, averaged. The same setup is used both during evolution and
  for the final report. (This is good practice: the agent optimised the exact metric that is reported. It
  also means there is no held-out benchmark for the kernel.)

### §4.2–4.3 Results

The numbers are in pass 1. Two careful-reading notes:
- **Non-causal at short sequences:** AVO is *"within measurement noise of both baselines"*. The headline
  "+10.5%" is a best case against FA4 on causal masking.
- **GQA transfer:** they *"prompted the AVO agent to adapt the evolved MHA kernel"*. That is one agent task,
  not a new evolution run. It shows the *result* transfers; it is not a test of the *search*.

### §4.4 Evolution trajectory

- 40 committed versions out of 500+ directions tried, so **~92% of explored directions never became a commit**.
  Both the correctness gate and the "must not regress" rule throw away most of the work.
- **Jumps:**
  - v8: QK–PV interleaving plus bitmask causal masking;
  - v13: single-pass softmax;
  - v20: branchless rescaling plus a lighter fence;
  - v30: correction/MMA overlap;
  - v33: register rebalancing.
- **Plateaus:** between the jumps, versions *"refine implementation details without measurably changing
  performance"*. That is why "matches" must be allowed: refactors that enable a *later* jump would otherwise be
  rejected.
- **Diminishing returns:** v1–v20 give coarse gains; v21–v40 give fine ones.
- Caveat stated by the authors: the figures show only the **committed** sequence, not the internal search tree.

### §5 Agent-discovered optimisations

Three ablations, each comparing a version with the one just before it:

| Change | Versions | Non-causal | Causal |
|---|---|---|---|
| Branchless accumulator rescaling | v19→v20 | +8.1% | +1.6% |
| Correction/MMA pipeline overlap | v29→v30 | +1.1% | +0.4% |
| Register rebalancing (192/80/48 → 184/88/56) | v32→v33 | +2.1% | 0% |

What each did, in plain words:
- **Branchless rescaling.** It used to be "check whether any thread needs rescaling; if not, skip". Now it
  always computes the factor and uses 1.0 when not needed. The branch cost more than the wasted multiply, and
  removing it allowed a cheaper memory fence. **The lesson: a check that saves work can cost more than the
  work.**
- **Pipeline overlap.** The correction warp used to wait for *both* tiles' GEMMs to finish. Now it starts on
  tile 1 while tile 2's GEMM runs. **The lesson: turn a serial dependency into a pipeline.**
- **Register rebalancing.** Profiling showed the correction group spilling registers to slow memory while the
  softmax group had spare registers. Moving 8 registers per group fixed it. **The lesson: measure, then move
  resources to the bottleneck.**

§5.4 argues that each change needed reasoning about *several* hardware subsystems at once, which is not simple
parameter tuning.

**Careful-reading critique:** these are single before/after comparisons, on the same benchmark the agent
optimised, and each is the step that the agent *chose to keep*. They show the edits are sensible. They do **not**
show that AVO beats AlphaEvolve-style variation, because no such comparison exists in the paper.

### §6 Conclusion

The authors claim generality beyond kernels, *"engineering or scientific domains that demand extended autonomous
exploration."* The blog is their follow-up evidence for that claim.

### Appendix A

The same comparison against the cuDNN and FA4 numbers *published* in the FA4 paper, in case their own hardware
differs. The conclusions are unchanged: +1.4% to +8.8%.

---

## Part B: the blog, read carefully (AVO-ARC)

### What the blog says AVO *is*

*"AVO is a general-purpose coding agent system developed by NVIDIA ... Its distinguishing focus is sustained
autonomous operation across long horizons."* So by August, "AVO" names the **agent system**, not only the
evolutionary operator. The evolution framing appears only in the kernel recap.

### Q7: did the ARC version keep the evolutionary part?

*"For ARC-AGI-3, we connected the same general-purpose agent to a different task interface. The underlying agent
remains the same; only the environment-specific tools and evaluation change."* There is **no mention** of a
lineage, of committed versions or of `f` for ARC. The two mechanisms the blog stresses are:
1. **Persistent memory:** *"carries forward prior implementations, evaluation results, compiler and profiler
   outputs, and accumulated reasoning, allowing the agent to resume from the current state rather than
   repeatedly reconstructing the search."*
2. **The supervisor:** *"monitors the broader trajectory for stagnation or repeated unproductive cycles and can
   redirect the main agent toward alternative strategies when needed."*

**Answer:** as far as the text goes, AVO-ARC = **long-horizon agent + persistent memory + supervisor + ARC
tools**. Whether any lineage or commit structure was used inside ARC (for example, versions of the agent's game
notes) is **NOT STATED**.

### The ARC task interface

- *"we adopted the direct-interaction design principles described by VISTA and reimplemented the task interface
  independently."* VISTA's interface (pass 3 digs in) is: observe → reason and use memory → one action →
  observe; every frame archived; `GUIDE.md` (durable rules) and `WORKING.md` (scratchpad).
- *"Rather than centering our ARC-AGI-3 system on explicit programmatic world-model construction, as explored by
  Tycho..."* They chose **not** to build a world-model layer, because they wanted to test the *general* agent.
- Observation: *"text-only modality: each observation was supplied as an exact 64 x 64 text grid"*. VISTA used a
  512×512 PNG instead.
- *"the agent received the available actions without descriptions of the game's rules or goals"*.

### The generalisation argument (the core idea, for us)

The loop is the same in both domains:
1. Build hypotheses from incomplete evidence.
2. Act through an external interface.
3. Observe the consequences.
4. Preserve useful state.
5. Revise the model.
6. Recover from wrong assumptions.
7. Keep going over a long horizon.

*"What transfers is not domain knowledge, but the machinery for sustained autonomous progress."*

### Numbers and their honest caveats (the blog's own words)

- 100.00 RHAE, 183/183 levels, **6,624 actions** against VISTA's 7,542 with the same model. *"This should not be
  interpreted as a controlled ablation."*
- The memory system *"may matter"*, but *"this experiment does not isolate its individual contribution"*.
- Opus 5 on its own gets about 30% (ARC Prize's figure, "High" reasoning), but the settings differ, so *"should
  not be interpreted as a direct measurement of the performance contribution of AVO"*.
- GPT-5.6 Sol on a subset: faster in wall-clock time; Opus used fewer actions.
- Public set only; semi-private and private sets were not tested.
- Author note: Jean-Francois Puget (a Kaggle grandmaster) co-wrote the ARC post.

---

## Answers to the pass 1 questions (summary)

| Q | Answer | Status |
|---|---|---|
| Q1 step anatomy | Read lineage + K → implement → call f → diagnose/revise loop → commit when satisfied | Stated. No per-step budget stated |
| Q2 commit rule | Correct **and** ≥ best committed so far; failures stay in agent memory, not in P | Stated. The scalar is probably the geomean (**UNCONFIRMED**); tie tolerance not stated |
| Q3 persistent memory | Paper: the conversation history across the whole run. Blog: survives beyond one context | Mechanism **NOT STATED** |
| Q4 supervisor | Triggered by stagnation or unproductive cycles; reviews the whole trajectory; proposes several directions | Trigger rule **NOT STATED** |
| Q5 vector f | One entry per benchmark config; all zero if incorrect | Stated |
| Q6 mapping to ARC | Not in the paper. See pass 3 | Our design question |
| Q7 evolution in ARC? | No lineage or commit mentioned; memory + supervisor + VISTA-style tools + text grid | **NOT STATED** beyond that |

## What pass 2 changes about pass 1

- Pass 1 treated "AVO" as the evolutionary operator. After the careful read, the **reusable core is smaller and
  more general**: three pieces we can build and test separately:
  1. an **edit-evaluate-diagnose loop** with a **binary correctness gate**;
  2. **two-level memory**: a clean committed record plus a messy working history;
  3. a **conditional supervisor** triggered by stagnation.
- The paper leaves four things unspecified, and any rebuild (including the community repo) must **invent** them:
  - the supervisor trigger;
  - the memory mechanism across context limits;
  - the tie tolerance;
  - the per-step stop rule.
- Every invented part is a hypothesis for us to test, not something to copy.

## Terms that carry the main components → pass 3

The main-component terms get a deeper dive in pass 3. The domain-only ones do not, to avoid the endless loop.

**Main components (deep dive):**
1. variation operator / evolutionary search (FunSearch, AlphaEvolve, MAP-Elites, LoongFlow, TTT-Discover);
2. deep agents / coding agents;
3. lineage and the commit gate;
4. persistent memory across context limits;
5. the supervisor (stagnation detection);
6. the VISTA direct-interaction interface;
7. Tycho's world-model route, which NVIDIA did not take.

**Domain only (one-line meanings above are enough):** warp specialization, TMA, online softmax, PTX, register
spilling, memory fences.

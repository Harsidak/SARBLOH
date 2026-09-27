# Pass 1: the bird's-eye view

**What this pass is:** the first read. It covers only the abstract, introduction, results and conclusion, and
skips the methods on purpose. The goal is to know *what* AVO is and *whether it matters to us* before learning
*how* it works. Pass 2 reads everything carefully and builds on this file. Pass 3 chases the main terms and code.

**Sources (both official NVIDIA, both snapshotted):**
1. The paper: *AVO: Agentic Variation Operators for Autonomous Evolutionary Search*, Chen, Ye, Xu et al., NVIDIA,
   arXiv 2603.24517 v1, 25 Mar 2026, 12 pages, only one version.
   Local copy: `resources/papers/arxiv/2603.24517_avo-agentic-variation-operators.pdf`.
2. The blog post: *NVIDIA AVO Reaches 100% on ARC-AGI-3*, NVIDIA Technical Blog, 21 Aug 2026, by Terry Chen, Eva
   Zhu, Zhifan Ye, Jean-Francois Puget and Humphrey Shi.
   Local copy: `resources/web/2026-09-26_nvidia_avo-arc-agi-3-blog.md`.

> **The most important finding of pass 1:** the paper never mentions ARC. It is about making **GPU attention
> kernels** faster. The ARC-AGI-3 result exists **only in the blog post**, five months later, with no paper,
> no code and no ablations. So "NVIDIA AVO" means two related things, and we must keep them apart:
>
> | | **AVO-paper** (March) | **AVO-ARC** (August blog) |
> |---|---|---|
> | Task | Optimise a CUDA program against a benchmark | Play ARC-AGI-3 games |
> | What is evolved | Source code, 40 committed versions | Nothing is "evolved" in the paper's sense; the agent plays |
> | Evidence | Paper, figures, one ablation table | A blog post, one public-set score, no ablations |
> | Model | "Internal agent powered by frontier LLMs", unnamed | Claude Opus 5 (plus a small GPT-5.6 Sol trial) |
> | Code released | No | No |

---

## 1. In one sentence

**The paper:** stop using the LLM as a one-shot code generator inside an evolutionary algorithm, and let a
full coding agent (one that plans, uses tools, tests and remembers) *be* the step that makes the next version.
**The blog:** the same agent machinery (persistent memory plus a supervisor that unsticks it) plays ARC-AGI-3
well enough to clear every public level.

## 2. The idea in plain words

Evolutionary search is a loop:

```
keep a population of solutions, each with a score
repeat:
    make a new candidate from existing ones   <- the "variation operator", Vary
    score it
    add it to the population
```

Older LLM systems (FunSearch, AlphaEvolve) split `Vary` into two fixed parts:
- **Sample:** the *framework* picks some parent solutions using hand-written rules.
- **Generate:** the LLM is shown the parents and writes *one* new candidate in *one* shot.

The LLM cannot look anything up, test its idea, read the error or try again before handing the candidate back.

AVO removes the split. `Vary` becomes one autonomous agent run that gets:
- **P:** every previous solution and its score, called the *lineage*;
- **K:** a folder of domain documents, the *knowledge base*;
- **f:** the scoring program, which it can call as often as it likes.

The agent decides for itself what to read, what to change and when to test. It commits a new version only when
the new version is correct and at least as good as the best so far.

```
Old:  Vary(P) = Generate( Sample(P) )     framework picks parents, LLM writes once
AVO:  Vary(P) = Agent( P, K, f )          the agent does everything, as many times as it needs
```

## 3. What they claim (results section, numbers verbatim)

**Kernel work (paper):**
- 7 days, no human intervention, **500+ optimisation directions explored, 40 committed versions**.
- Multi-head attention (MHA) on a B200 reaches up to **1668 TFLOPS** in BF16.
  - That beats cuDNN by up to **+3.5%** and FlashAttention-4 by up to **+10.5%**.
  - On causal masking it wins everywhere: **+0.4% to +3.5%** over cuDNN and **+5.0% to +10.5%** over FA4.
  - On non-causal masking it gains **+1.8% to +2.4%** only at long sequences, and is *within noise* at short ones.
- Transfer to grouped-query attention (GQA) took **about 30 minutes** of extra agent time.
  - That gives up to **+7.0%** over cuDNN and **+9.3%** over FA4.
- Progress comes in **jumps, not a smooth climb**, with the big jumps at versions 8, 13, 20, 30 and 33.
  - Versions v1–v20 give the largest gains; v21–v40 give small ones that compound.

**ARC-AGI-3 (blog):**
- **100.00 RHAE** on the **public set**: all 25 games, all 183 levels.
- **6,624 environment actions.** VISTA, using the same Opus 5 model, took **7,542**, so AVO used about 12% fewer.
- Observation: **text only**, an exact 64×64 grid as text, with no images. The agent was told which actions
  exist but nothing about the rules or goals.
- Interface: they reimplemented **VISTA's direct-interaction design** themselves. They explicitly chose **not**
  to build an explicit programmatic world model (the Tycho approach).
- They name two mechanisms that matter over long horizons:
  - **Persistent memory:** carries forward prior implementations, results and reasoning.
  - **A supervisor:** watches for stagnation or unproductive cycles and redirects the agent.
- ARC Prize reports Claude Opus 5 on its own at about 30%. NVIDIA themselves warn this gap is **not an
  ablation**: the reasoning settings and the whole setup differ.

## 4. The five Cs (a standard first-pass checklist)

| C | Answer |
|---|---|
| **Category** | A systems and method paper: a new kind of evolutionary operator, shown on one case study. The blog is a result announcement, not a paper. |
| **Context** | It builds on LLM-driven evolution (FunSearch, AlphaEvolve, LoongFlow, TTT-Discover) and on "deep agents" (SWE-agent, OpenHands, Claude Code, Codex). The ARC side builds on VISTA and contrasts with Tycho. |
| **Correctness** (first impression) | The kernel numbers are careful: 10 repeats, FA4's own timing script, and an appendix against FA4's published numbers. The *mechanism* is thin: one lineage, no comparison against AlphaEvolve-style variation with the same model and budget, no supervisor-on/off ablation. The ARC claim is on the **contaminated public set**, unreviewed, and with a model released after the games were public. |
| **Contributions** | (1) Naming and formalising `Vary = Agent(P, K, f)`. (2) SOTA MHA kernels on B200. (3) An analysis showing the agent's edits are real hardware reasoning. (4) Blog only: the same machinery transfers to ARC-AGI-3. |
| **Clarity** | The paper is short and readable. The agent itself ("internally-developed", unnamed LLM) is a **black box**, and so is the supervisor's trigger rule. This is the biggest gap for anyone trying to rebuild it. |

## 5. What AVO does *not* tell us (important for SARBLOH)

1. **Everything ran on frontier API models with no compute limit.** We have Qwen-class 27–31B open weights, one
   RTX PRO 6000, 12 h and no internet. Pass 1 cannot say whether any of it survives that drop. **UNCONFIRMED.**
2. **The ARC run's cost** (tokens, wall-clock, number of supervisor interventions) is not reported.
3. **No private-set number.** NVIDIA's own editor's note says so.
4. **Which part helps is unknown.** Memory, supervisor, text grid and backend all changed together.

## 6. Why it still matters to us

The blog's core claim is our thesis written by someone else: *"long-horizon capability is a property of the
full system. Memory determines what survives..."* and *"designed to carry useful understanding forward and
reduce repeated exploration"*. CLAUDE.md says the same: *"the binding constraint is memory and verification,
not model intelligence."*

AVO is therefore a candidate architecture for the **agent loop** (`sarbloh/agent` plus `memory`). It does not
replace our world model: NVIDIA explicitly *skipped* the world-model layer.

## 7. Questions to carry into pass 2

- Q1. What exactly is inside one variation step (Section 3.2)? What does the agent see, and when does it stop?
- Q2. What is the commit rule, exactly? "Matches or improves": improves on what, the whole score vector or an
  average?
- Q3. What is "persistent memory" concretely? One conversation for 7 days? Files? Both?
- Q4. What triggers the supervisor, and what does it output?
- Q5. What does `f` look like when it is a *vector* of scores, and how does a failed correctness check zero it?
- Q6. How do the kernel-world terms (lineage, K, f, commit) map onto an ARC game, where "committing" costs a
  real action?
- Q7. Did the ARC version keep the *evolutionary* part at all, or only the agent (memory plus supervisor)?

# SARBLOH — Agent Operating Spec

You are working inside SARBLOH, a competition and research repository targeting **ARC-AGI-3** (ARC Prize 2026).
This file is the contract. Read it fully before acting.

---

## 0. Operating mode — read this first

**Iteration beats theory.** The loop is: **push to Kaggle → read results → refine the architecture → repeat.**
A shipped run that scores 0 but produces a trace teaches more than a week of design documents. When you have to
choose between thinking harder and running an experiment, run it.

### The five standing rules
1. **Everything is wrong until proven.** That includes this file, the papers, the leaderboard claims, other
   teams' write-ups, our own designs, and the model's own explanation of its behaviour. A claim becomes a fact
   only when a run, a test or an official source confirms it. Mark anything unproven as UNCONFIRMED.
2. **Know the environment before the method.** Kaggle's hardware, runtime, offline rules and the ARC-AGI-3 game
   dynamics are facts to verify, not assume (§2). If a fact is missing, go and collect it, snapshot it into
   `resources/` with provenance, and record it here.
3. **Experiment constantly.** Every change is an experiment with a hypothesis, a run and a ledger row (§4). Small
   and fast beats big and careful. Several cheap experiments in flight are better than one perfect one.
4. **Method follows hardware, not taste.** See the RL lesson below.
5. **Keep resources and logs current as you go:** papers, repos, blog posts and leaderboards go into
   `resources/`. Plans go into `planning/`, mistakes into `planning/MISTAKES.md`, and runs into the ledger and the
   experiment directories. Do it as part of the work, not afterwards.

### The RL lesson
The literature makes a strong case that RL is the secret ingredient for advanced agentic systems. In a previous
Kaggle competition RL was **forced onto a problem where Kaggle's
hardware constraints made it suboptimal**, and **LoRA SFT gave the best results.**
Consequences for this repo:
- RL is a hypothesis like any other. It has to beat a simpler baseline **inside Kaggle's compute and
  wall-clock budget** in a measured run before it is adopted.
- If we train at all, the default is **LoRA SFT on our own verified trajectories**. RL is attempted only after
  SFT is working and there is spare compute budget.
- Choose methods by what the hardware can run in 12 hours offline on one GPU, not by what the papers do
  on a cluster.

### Where code lives right now
Owner decisions of 2026-09-25, in two stages. They replace the earlier single-notebook `agent.py` plan.

**Stage 1: lock the baseline.** `kaggle/main.ipynb` is the one submission notebook. It runs **Tufa Labs' Duck
bundle unchanged** (`jeroencottaar/taaf-kaggle-source-share`). Only the serving layer is ours: a vLLM profile chain
(`fast` → `fast_nospec` → `duck`, where the last is the exact Duck flags, and each profile must pass a tool-call
smoke test), a throughput probe, a watchdog and guaranteed teardown. Every knob is in its `CONFIG` cell. The gate
is to score at least the Duck's ~4.5 offline (E001). Nothing else is built on until this passes.

**Stage 2: our agent.** It lives in the `sarbloh/` package and ships as the MIT-0 dataset
`banwait13/sarbloh-agent` (staged in `kaggle/datasets/sarbloh-agent/` by `scripts/kaggle_push.py`). The code
already written for it is parked, not in use: a ported harness in `sarbloh/harness/{games,runner,vllm,kaggle,local}.py`,
the vendored Duck in `sarbloh/agent/duck/` and the launcher `kaggle/experimental/001_sarbloh.ipynb`. Experiments are
chosen from `experiments/BACKLOG.md`.

**Stage 2, revised (owner decision 2026-09-27): Prime Agent replaces the AVO plan.** `Sarbloh/` implements
Prime Intellect's Prime Agent (arXiv 2608.23552, MIT) for ARC-AGI-3: `Sarbloh/rlm/` is the upstream Python kernel,
vendored unchanged; `Sarbloh/harness/` is our Python port of the TypeScript host loop (tool loop, autonomous mode,
compaction, Continual Harness digest, `rlm.spawn` subagents) plus the `arc` REPL module and a Gemma 4 vLLM launcher.
The model is Gemma-4-31B-IT-NVFP4 (`banwait13/sarblohmodels`). Kaggle imports the code from `banwait13/sarblohagent`,
a dataset mirrored from this git repo, which the owner updates after pushing. The launcher is
`kaggle/experimental/002_prime.ipynb` (`scripts/kaggle_push.py prime`). First gate: E003, a 1-hour harness smoke test.

- **Docker is deferred, not forbidden.** It comes later, together with production-level integration.
- **Offline wheels:** `kaggle/wheels.ipynb` runs with internet on and builds a vLLM wheelhouse plus `requirements.lock`
  in the same layout as driessmit's. `arc-agi` always comes from the competition mount.
- The logging rules of §4 apply to every run: ledger row, experiment directory and mistakes entry.

### Model choice (working recommendation; UNCONFIRMED until measured)
| Role | Model | Why |
| --- | --- | --- |
| **Primary** | `Qwen/Qwen3.8-27B` (Apache-2.0, image+text, `qwen3_5` arch, updated 2026-08-14) | Direct successor to the model behind the only verified open-weight winner (the Duck, Qwen 3.6 27B). Same size, same arch family, so it is a drop-in swap in the Duck harness. `nvidia/Qwen3.8-27B-NVFP4` exists for Blackwell. |
| **Control** | `Qwen/Qwen3.6-27B-FP8` | Reproduces the Duck exactly. Every other model is measured against it. |
| **Challenger** | Gemma-4-31B | Used by Milestone #1 places 2 and 3. |
| **LoRA SFT target** | Whichever of the above wins | Fine-tune on our verified trajectories if and when training is attempted. |

The first model experiment is Qwen 3.6 FP8 vs Qwen 3.8 in the **same harness, on the same split**. Verify on the
hardware first that the model fits in VRAM with a usable context and reaches a usable tokens/s. The 14 Aug run
measured 18 tok/s without fast kernels, and that throughput consumed the entire wall clock.

---

## 1. Mission

Three outcomes, in priority order.

1. **Score.** Beat the open-weight field on ARC-AGI-3 under Kaggle's offline constraints.
2. **Evidence.** Produce a clean ablation showing the gain comes from the *memory architecture*, not the backbone model.
3. **Paper.** A submission to the ARC Prize paper track built from the experiment ledger, not written retroactively.

The thesis the repo exists to defend: **on an interactive benchmark scored by action efficiency, the binding constraint is memory and verification, not model intelligence.**

---

## 2. Hard competition facts

Verified against official sources on 2026-09-21. Re-verify anything marked UNCONFIRMED before relying on it.
Full snapshots live in `resources/official/`.

### Dates
| Gate | Date |
| --- | --- |
| ARC-AGI-3 Milestone Prize #2 (open-source) | 30 Sep 2026 |
| Final submission | **2 Nov 2026** |
| Paper track deadline | 8 Nov 2026 |
| Results announced | 4 Dec 2026 |

### Environment
- Observation: grid up to **64x64**, cell values **integers 0-15**, origin `(0,0)` top-left, `(x,y)` ordering.
- Actions: `GameAction` — ACTION1-ACTION4 directional, ACTION5 game-specific, ACTION6 complex (carries coordinate `data`). **Verified 2026-09-24** against the installed SDK (`arc-agi 0.9.9`, `arcengine 0.9.3`): `list(GameAction)` = `RESET, ACTION1 … ACTION7`. The enum lives in `arcengine`, not `arc_agi`. **UNCONFIRMED:** that ACTION7 is the undo action the docs mention — check its semantics in a game before relying on it.
- `GameState`: WIN, GAME_OVER (plus in-progress states — verify against SDK).
- Observation payload: `FrameDataRaw` carrying `state`, `levels_completed`, and frame data.
- Environments are a series of levels. Early levels teach mechanics; later levels require composing them.

### Toolkit
- Package `arc-agi`, version **0.9.9** (2026-06-10), Python **>=3.12**, **MIT** licence.
- Entry point `Arcade`: `make()`, `get_environments()`, `get_scorecard()`, `create_scorecard()`, `close_scorecard()`, `listen_and_serve()`.
- `EnvironmentWrapper`: `observation_space`, `action_space`, `info`, `reset()`, `step(action, data, reasoning)`.
- Operation modes: **NORMAL** (local + API), **ONLINE** (API only), **OFFLINE** (local only), **COMPETITION**.
- Recordings written as JSONL.

**Consequence you must exploit:** `OFFLINE` mode drives local environments through the same interface as `COMPETITION`. Development against `eval/real_games/` and the real submission share one code path. Never write a second, dev-only agent loop.

### Scoring — RHAE
Relative Human Action Efficiency. The ratio of human to agent action count, **squared**, per level, weighted by level index, capped at 1.15x.

```latex
\text{RHAE}_{\text{level}} = \min\left(\left(\frac{a_{\text{human}}}{a_{\text{agent}}}\right)^2,\ 1.15\right)
```

Human baseline is the upper-median best human action count (third-best completer). It already includes human mechanics-discovery overhead.

**Two rules follow and they govern every design decision in this repo:**
1. **Reasoning is free, acting is not.** Internal deliberation, retrieval from the frame store, and simulated rollouts cost nothing against the metric. Only committed environment actions count. Push work out of the environment and into the model.
2. **Exploration cost dominates.** Paying twice for a fact already established is the primary loss term, and it is a memory failure by definition.

### Competition constraints
- **No internet access during Kaggle evaluation.** No API models. Local open weights only.
- **GPU notebook <= 9 hours run-time** (Kaggle competition page "Code Requirements", read via the Kaggle API
  2026-10-25). This overrides the 12 hours in the ARC Prize testing policy, which this file wrongly adopted on
  2026-09-21. Whether Kaggle enforces it on the Phase B rerun is UNCONFIRMED: versions with 10 h and 11.5 h Submit
  budgets completed, but their actual run-times are unknown (see `planning/MISTAKES.md` 2026-10-23).
  **Owner decision 2026-10-28:** the Experimentation notebook's Submit budget is 12 h. The three failed Submits
  ("A system error") were Kaggle errors, not the time limit, and 10 h runs complete.
- **GPU (partly confirmed 2026-09-24, [docs.arcprize.org/arc-prize-2026](https://docs.arcprize.org/arc-prize-2026)):**
  the notebook options are T4 x2, P100 and CPU, plus **"Nvidia RTX 6000 (g4-standard-48)", which is reserved for
  ARC-AGI-3**. Do not burn it on early iteration. **Measured 2026-09-25** (`machine_shape: NvidiaRtxPro6000`,
  kernel `banwait13/sarbloh-main` v1): NVIDIA RTX PRO 6000 Blackwell, **97,887 MiB** VRAM, driver 580.159.04,
  CUDA 13.0; 176 GB RAM; 20 GB `/kaggle/working`; Python 3.12.13. Qwen3.6-27B-FP8 on vLLM 0.19 at 65k context
  leaves 45.3 GiB KV cache (174k tokens). **UNCONFIRMED:** the weekly quota, and whether the Phase B rerun gets
  the same card.
- Submission is in two phases: **Phase A** "Save & Run All" validates the notebook, then **Phase B**
  "Submit to Competition" reruns it on the hidden games.
- **All authored code must be open-sourced under a permissive licence** to be prize-eligible. Third-party code needs a licence permitting public sharing.
- Private evaluation environments are **out of distribution** — ARC Prize states limited mechanical overlap with the public set. Nothing about specific public mechanics can be memorised in advance.

### The field
| System | Model | Score | Verified |
| --- | --- | --- | --- |
| NVIDIA AVO | Opus 5 / GPT-5.6 Sol | 100.00 RHAE, public set | No |
| Schema (Impossible Research) | Opus 4.8 + Fable 5 | 98.98 RHAE, public set | No |
| VISTA | Opus 5.0 | 100.00 RHAE, public set | No |
| Executable World Models | GPT-5.5 High | 58.12 RHAE, 15/25 solved | Public set |
| **Tufa Labs "The Duck"** | **Qwen 3.6 27B FP8, local** | **Milestone #1, 1st** | **Yes, Kaggle offline** |

The last row is the comparison class. The rows above it used unbounded frontier inference on the contaminated public set. Do not cite them as targets; cite them as sources of mechanism.

---

## 3. Architecture

### Principle: act on a good-enough model, do not wait for proof (owner decision 2026-10-07)
The games are deterministic, but the private games are new and their rules are unknown. Proving a rule before acting
costs wall clock and tokens, and the clock is what runs out. The model may spend extra actions on level 1 and 2 to
learn the mechanics and the goal by trying; later levels carry most of the weight and are where it pays back.
- **No verification gate.** Nothing in the harness may refuse, hold back or delay an action because a rule is
  "unverified" or "uncertified". A wrong guess is corrected by the next frame, not prevented in advance.
- **The agent writes code and acts.** The target loop (WorldCoder / Tufa style): read the 64x64 grid, write a Python
  `step()` model of the game, search it (BFS/DFS or anything else) for a path to the goal, and send the moves. When
  the game disagrees, fix the model and continue.
- **One tool, not two.** The plan is to merge the read-only Python tool and the action tool, so the agent can send
  moves from inside its code. Fewer tools, less friction, fewer turns.
- **Many moves per call.** One call may send 50 or more moves; a cap that forces a turn every few moves wastes latency.
- **Behaviour is fixed by training, not by gates.** Bad action habits are corrected through LoRA SFT (and RL only
  if it beats SFT) on our own traces, not by harness rules.
- Reasoning is still free and acting is not (§2): the model should still simulate in code before it acts, it just
  does not need proof.

Every loop that does not reach an action costs zero RHAE, but a run that times out scores zero. The time budget
binds: the whole notebook must finish inside **9 hours** (Kaggle's GPU notebook limit, §2).

### Older module plan (the deleted `sarbloh/` package; history only)
The surat / memory / worldmodel / parkh / jugat split, with the `parkh` certification gate, is retired by the
principle above. Ideas worth keeping from it: lossless frames outside the context window, a refuted-rule log so a
falsified rule is not re-proposed, and a budget governor that aborts a game showing no progress.

---

## 4. Working rules

These are not style preferences. Violating them destroys the paper.

1. **Ledger obligation.** Every scored run appends one row to `history/ledger.jsonl` and one line to `history/LEDGER.md`: experiment id, hypothesis, config hash, git SHA, split, RHAE, delta vs baseline, verdict. A run that is not logged did not happen. No retroactive logging.
2. **Held-out protocol.** The split in `eval/splits/heldout_split.json` is frozen. Tune on the tune set. Report on the held-out set. **A gain that does not survive the split is noise and must be recorded as such, not quietly dropped.**
3. **No public-set tuning.** The public environments are contaminated and mechanically unrepresentative of the private set. Any change justified only by a public-set gain is a liability.
4. **Hypothesis before code.** Every experiment starts as `experiments/E###_name/hypothesis.md` stating the mechanism, the predicted effect, and the kill criterion. Then the code. Not the reverse.
5. **Kill criteria are binding.** If an experiment hits its stated kill criterion, it is killed and `verdict.md` records why. Extending a dead experiment is how six weeks disappear.
6. **Tests do not go to the archive.** If a test breaks, fix the test or fix the code. Moving it to an archive folder is forbidden; that is how this repo lost verifiability in August 2026.
7. **Mistakes log.** Anything that cost more than an hour goes in `planning/MISTAKES.md` with the cause, not just the symptom. Append-only.
8. **Resources carry provenance.** Every item under `resources/` has a sibling `.meta.yaml` with source URL, retrieval date, and one line on why it is here. No orphan PDFs.
9. **Never vendor third-party code into the package.** Clones live under `resources/repos/` and are gitignored. Licence compatibility is checked before any code is borrowed.
   **Exception (owner decision, 2026-09-25):** the Duck agent is vendored at `sarbloh/agent/duck/` as the control arm. It is
   copied from `resources/repos/duck-kaggle-bundle`, with imports rewritten and one Windows patch marked `# sarbloh:`. Its
   licence is **UNCONFIRMED** and must be resolved before any prize submission.
   **Exception (owner decision, 2026-09-27):** Prime Agent is vendored at `Sarbloh/` (MIT, commit 2d24ad4; clone in
   `resources/repos/prime-agent`). `rlm/` is upstream code, unchanged; `harness/` is ours. `Sarbloh/rlm/LICENSE` (upstream MIT) must stay; the repo
   is MIT-0 (root `LICENSE`; the rules require a public-domain-equivalent licence, owner review 2026-11-02) and `NOTICE` lists the third-party parts (owner decision 2026-10-26).
10. **Verify SDK facts against the installed SDK**, not against this file. Where they disagree, the SDK wins and this file gets corrected in the same commit.

---

## 5. Repo map

```
SARBLOH/
  CLAUDE.md           this file
  kaggle/             Kaggle workspace: notebooks tracked; weights/staging/third-party ignored (see §5 kaggle/)
  Sarbloh/            Prime Agent for ARC-AGI-3 (was Sarbloh-Arc/; its prime/ became harness/ on 2026-10-02; the old
                      sarbloh/ package was deleted in 078d816): rlm/ (upstream kernel, MIT); harness/ (ours): run.py, config.py, trace.py,
                      agent/ (agent.py loop + prompts, compaction, refine; E008: vision.py, intuition.py, context.py,
                      curator.py, skills/*.md), memory/ (E008: timeline, working memory, lessons graph, lifecycle),
                      runtime/ (REPL kernel + skills/arc.py, skills/scene.py),
                      game/ (arc_host.py, the only game access; games.py, the
                      ArcGame/Arcade wrapper moved from the deleted sarbloh/harness/ on 2026-10-02), llm/ (client.py + server.py
                      + bench.py, model-free; gemma.py / qwen.py hold everything model-specific; config `model` picks one)
  Sarbloh-Experimentation/  copy of Sarbloh (2026-10-01) where daily experiments run; launcher
                      kaggle/experimental/sarbloh_experimentation.ipynb (`kaggle_push.py experimentation`), 5 public
                      dev games; daily verdict log experiments_<date>.md at the root (owner deletes it)
  Alignment/          LoRA SFT + LoRA RL (E107, E108), torch+transformers+peft, no TRL: lora_sft.py (JSONL chat/tool
                      conversations or Prime sft_levels.jsonl rows -> assistant-only loss masks -> LoRA -> result card
                      with PASS/FAIL checks), lora_rl.py (GRPO on the same prompts, verifiable tool_match reward or a
                      module:function, KL to the adapter-disabled reference, init_adapter for SFT -> RL),
                      data/smoke3.jsonl (3 hand-written Prime-style traces), alignment.ipynb (THE Kaggle notebook for
                      both pipelines; edited directly; `kaggle_push.py alignment` refreshes its generated %%writefile
                      source cells from the .py files and pushes the file as is)
  eval/               scoring spine: official_score, bench, scoreboard, real_games, splits
  Writing/            the arXiv preprint (AAAI-27 style, \nocopyright): main.tex + sections/, references.bib, figures/
                      and data/ built by Writing/scripts/ (collect_data, extract_curves, make_perception_figure,
                      plot_results); `latexmk -pdf main.tex` in Writing/ -> output/main.pdf
  tests/              unit / component / regression
  experiments/        one dir per experiment, E### prefixed
  history/            LEDGER.md, ledger.jsonl, baselines.json
  planning/           ROADMAP.md, MISTAKES.md, decisions/ADR-*.md
  resources/          official/ papers/ repos/ web/ + INDEX.md
  scripts/            local dev tooling: serve_llm.ps1, llm_smoke.py
  tools/llama.cpp/    local inference server binaries, gitignored
  runs/               traces, gitignored
```

`README.md` at the root is the only README in the repository. Per-directory rules live here, not in scattered
READMEs. Third-party clones and downloaded model cards keep their own READMEs.

### kaggle/: the Kaggle workspace (tracked since 2026-10-01, except weights, staging and third-party code)
```
kaggle/
  main.ipynb                 THE submission notebook (stage 1: Duck bundle + our vLLM launcher)
  wheels.ipynb               internet ON: builds the offline vLLM wheelhouse + requirements.lock
  experimental/              launchers not yet submitted (001_sarbloh.ipynb: sarbloh harness, stage 2)
  reference/                 TufaLabs.ipynb, DuckQwen.ipynb (public Duck launchers, read-only)
  archive/                   v-o-i-d*.ipynb, pre-reorg Jul-Aug 2026 (still tracked from before the ignore)
  datasets/                  what gets uploaded as Kaggle datasets
    sarbloh-agent/           LICENSE (MIT-0) + dataset-metadata.json; sarbloh/ rebuilt by kaggle_push.py
    models/local/            GGUFs for the RTX 4050 rig (models.json lists them)
    models/cloud/            models.json: RTX Pro 6000 models and where they live on Kaggle (no weights in the repo)
```
- **Still ignored:** `.push/`, `datasets/models/`, `datasets/Taaf agent/`, `datasets/sarbloh-agent/sarbloh/`, `*.whl`
  under `datasets/`, and `reference/` (third-party notebooks, licence UNCONFIRMED). The run prints `CONFIG` and the ledger row. Copy a notebook into `experiments/E###_*/` when a run is logged.
- The **first cell prints the environment**: `nvidia-smi`, the Python and `arc_agi` versions, `list(GameAction)`,
  the model path and the config. Any UNCONFIRMED fact it settles is recorded in §2.
- All knobs go in one `CONFIG` dict at the top, and every mechanism can be switched off from it.
- The run prints a `LEDGER_ROW` JSON line and writes `sarbloh_summary.json`. Each run links to `experiments/E###_*/`.
- Do not iterate on the RTX 6000 when a T4 or CPU run answers the question.

### eval/: the scoring spine
`official_score.py` is the only scorer whose output goes in the ledger. `bench.py` is the benchmark driver,
`scoreboard.py` shows standings, `components.py` does component evaluation, and `agent_loader.py` resolves the agent.
`real_games/` holds the local environment replicas and `games_index.json`, `splits/` the frozen split, `suites/`
the component suites, and `probes/` the diagnostic probes. A number quoted without its split is not a result.
Public-game RHAE is a smoke test, never evidence.

### experiments/
One `E###_short_name/` per experiment; numbers are never reused. Copy `E000_TEMPLATE/`. Contents, in order:
`hypothesis.md` (**before any code**: mechanism, prediction, kill criterion, the axis it varies), `config.yaml`
(its hash goes in the ledger), `run.py` (a thin driver), `results.json` (written by the run) and `verdict.md`
(kept, killed or inconclusive, and why). Negative results carry the same weight as positive ones.
`BACKLOG.md` is the catalogue; a directory is created only when an experiment is opened.

### history/
`LEDGER.md` is human-readable, append-only, newest first. `ledger.jsonl` has one row per scored run, and the plots
and the paper read it. `baselines.json` holds named reference points. `legacy_scores/` holds pre-reorg results,
which are **unattributed**.

### tests/
`unit/` covers single functions, `component/` covers one module against a fixture (pinning behaviour from
`legacy/`), and `regression/` runs full loops against recorded traces. The suites are scripts, not pytest; run
`uv run python tests/run_all.py [--only <name>]`. A broken test is fixed, or deleted with a recorded reason. It is
never archived.

### resources/
Four folders: `official/` for dated spec and rules snapshots, `papers/` for PDFs (planning-phase arXiv set in
`papers/arxiv/`), `repos/` for third-party clones (gitignored, read and never imported) and `web/` for scraped
pages named `YYYY-MM-DD_domain_slug.md`, with the substance kept and numbers verbatim. Every item has a
`.meta.yaml` (template: `_TEMPLATE.meta.yaml`). **`resources/INDEX.md` is the catalogue**: every paper with its
arXiv ID and why it is there, and every repo with its licence and commit. Record a clone's licence there before
borrowing any code from it.

### kaggle/datasets/models/local/ and tools/llama.cpp/
Each model gets its own folder, and nothing in it is tracked. Current: `qwen3.5-4b-gguf/` holds
`Qwen_Qwen3.5-4B-Q4_K_M.gguf` (3.0 GB, from bartowski's quantisation of Qwen/Qwen3.5-4B, Apache-2.0) and
`mmproj-…-f16.gguf` (0.7 GB vision projector, used with `-Vision`). Both files are SHA256-verified against HF. It is
the same `qwen3_5` family and chat template as the 27B targets. Rig: RTX 4050 Laptop, 6 GB, 32 GB RAM, CUDA 13.4.
At Q4, 4B fits comfortably, 9B only at short context, and 27B needs CPU offload (one-off checks only).
- Add a GGUF: `uv run hf download <repo>-GGUF <file>.gguf --local-dir kaggle\datasets\models\local\<name>`. On a slow link use
  `curl -L -C - -o <file> https://huggingface.co/<repo>/resolve/main/<file>`, which resumes.
- Add full weights for LoRA: `uv run hf download <repo> --local-dir kaggle\datasets\models\local\<name>-hf` (needs `--group train`).
- llama.cpp: the Windows CUDA 13.4 build, version in `tools/llama.cpp/VERSION`. To update, unzip the
  `llama-bNNNN-bin-win-cuda-13.4-x64.zip` and `cudart-*.zip` release assets into that folder.

## 6. Commands

Local environment is **uv**, Python 3.12 (`.python-version`), all deps in `pyproject.toml`. Verified 2026-09-24.

```bash
uv sync                                          # base env: arc-agi, notebook, llm client, dev tools
uv sync --group train                            # + torch cu128, transformers, peft (LoRA SFT); ~3 GB
uv run python tests/run_all.py                   # full suite (16 suites; suites are scripts, not pytest)
uv run python eval/bench.py --label <name> --seeds 0-2 --split dev --agent-file <path>
uv run python eval/scoreboard.py                 # current standings
uv run python eval/official_score.py <scorecard.json>   # official RHAE
```

Local LLM (the stand-in for vLLM on Kaggle; both speak the OpenAI API, so only the base URL changes):

```powershell
.\scripts\serve_llm.ps1                          # llama.cpp server, Qwen3.5-4B Q4_K_M, http://127.0.0.1:8080/v1
uv run python scripts/llm_smoke.py               # endpoint up? answer correct? tok/s?
```

Measured 2026-09-25 (llama.cpp b11157, RTX 4050): server up in about 6 s, smoke test correct, **45 tok/s**, 3.4 of 6.1 GB
VRAM at a 16k context. Most of the 476 tokens are thinking. PowerShell scripts must stay **ASCII-only**: Windows
PowerShell 5.1 reads BOM-less UTF-8 as ANSI, and a single em dash breaks parsing.

Agent runs, locally and on Kaggle, use the same `sarbloh.harness` code path:

```bash
uv run python -m sarbloh.harness.local --games ls20 --max-actions 30      # local llama.cpp, OFFLINE games
uv run python scripts/kaggle_push.py kernel                  # push kaggle/main.ipynb (starts a run)
uv run python scripts/kaggle_push.py dataset -m "..."        # stage 2: build + upload sarbloh-agent dataset
uv run python scripts/kaggle_push.py status | output         # poll / download outputs into runs/
```
Alignment (E107 SFT, E108 RL). The repo env has no torch. Tests run on CPU with a tiny random qwen3_5 model; the Kaggle
run is a T4 with internet on, and the result cards are in `sft_out/` and `rl_out/` of the kernel output. Edit
`Alignment/alignment.ipynb` directly (configs, flags); library code goes in `Alignment/*.py`, never in the generated cells:
```bash
uv run --with torch --with transformers --with peft --with matplotlib python tests/unit/test_alignment_sft.py
uv run --with torch --with transformers --with peft --with matplotlib python tests/unit/test_alignment_rl.py
uv run python scripts/kaggle_push.py alignment   # sync source cells + push + run; then alignment-status | alignment-output
```
Kaggle credentials go in the gitignored repo `.env` (`KAGGLE_API_TOKEN`, or `KAGGLE_USERNAME` + `KAGGLE_KEY`), which
`kaggle_push.py` loads; `~/.kaggle/kaggle.json` also works. The `username` is set in `scripts/kaggle_settings.json`.
Measured 2026-09-25 locally with Qwen3.5-4B: the plumbing works end to end (tool calls parsed, sandbox runs). The 4B
model then loops in its reasoning and never calls `step_env`, so local runs test plumbing only.

Models live in `kaggle/datasets/models/` (gitignored, see §5); the Jupyter kernel is "SARBLOH (py3.12, uv)".
Local runs are for plumbing and debugging only — a 4B model's score is not evidence about the 27B. Scored runs are
on Kaggle.

---

*Last verified against official sources: 2026-09-21. Corrections go in the same commit as the code that found them.*

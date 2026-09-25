# resources/ — catalogue

Everything the project reads but did not author. Rules are in `CLAUDE.md` §4.8–4.9 and §5. Every item has a
`.meta.yaml` beside it (template `_TEMPLATE.meta.yaml`). Claims from any source here are UNCONFIRMED until a run of
ours reproduces them.

| Folder | Holds | In git |
| --- | --- | --- |
| `official/` | Dated snapshots of the ARC-AGI-3 spec and competition rules | yes |
| `papers/` | PDFs and extracted text; the planning-phase arXiv set is in `papers/arxiv/` | no (gitignored) |
| `repos/` | Third-party clones, shallow, read and never imported | no |
| `web/` | Scraped pages as `YYYY-MM-DD_domain_slug.md`, substance kept, numbers verbatim | no |

## Papers

### Inherited (`papers/`, moved from `Docs/`)
| File | Paper | Read for |
| --- | --- | --- |
| `ARC-AGI-3 A New Challenge for Frontier Agentic Intelligence.pdf` | Official technical report, arXiv 2603.24621 | RHAE, human baselines, the four named failure modes |
| `2605.05138v2.pdf` | Executable World Models for ARC-AGI-3 (Rodionov), arXiv 2605.05138 | Closest published method to `worldmodel` + `parkh`. 58.12 RHAE with GPT-5.5; premature commitment is its main failure; read its open-problems section first. Code: `repos/ewm-baseline1` |
| `WorldCoder.pdf` | WorldCoder, arXiv 2402.12275 | Inducing programmatic world models from interaction |
| `ARC-AGI-3 Competition Research.pdf` | Competition research notes (not a paper) | No `.meta.yaml` yet |
| `VIGA.pdf` | Unknown | No `.meta.yaml` yet; identify or delete |
| `Robotic muscle memory …pdf`, `capMem …pdf` | Memory-architecture inspiration | No `.meta.yaml` yet |

Extracted text (`_challenge_paper.txt`, `_comp_research.txt`, `_ewm_paper.txt`) sits beside the PDFs. `research/`
holds earlier notes (`PAPERS_INDEX.md`, `Plan.md`, `Schema.md`, the comprehensive research write-up) and is the
source of the arXiv list below. The two `Docker Desktop Installer*.exe` files in `papers/` do not belong there.

### Planning-phase set (`papers/arxiv/`, downloaded 2026-09-24)
| Priority | arXiv | Paper | Read for |
| --- | --- | --- | --- |
| Must | 2512.24156 | Graph-Based Exploration for ARC-AGI-3 (Rudakov, Shock, Cowley) | Training-free, 3rd in the preview; state graph plus salience-ranked actions. Exploration-cost baseline. Code: `repos/graph-explore-3rd` |
| Must | 2510.04542 | Code World Models for General Game Playing (DeepMind) | Code as the world model, plus MCTS. `worldmodel`, `jugat` |
| Should | 2405.15383 | GIF-MCTS: code world models via MCTS | Search over world-model programs |
| Should | 2503.04412 | AB-MCTS, "Wider or Deeper?" (Sakana) | Explore-versus-refine search. Code: `repos/treequest` |
| Should | 2505.01081 | MADIL: MDL program synthesis for ARC | MDL pressure on induced models |
| Should | 2606.22813 | Active Inference as a Test-Time Scaling Law | Information-gain action selection |
| Should | 2510.04871 | Tiny Recursive Models (TRM) | Tiny nets and augmentation. Code: `repos/trm` |
| Memory | 2006.08381 | DreamCoder | Library learning, the procedure clock |
| Memory | 2310.19791 | LILO | Library compression (cited by EWM) |
| Memory | 2506.11058 | Refactoring Codebases through Library Design | Refactor pressure (cited by EWM) |
| Background | 2603.06590 | ARC-AGI-2 Technical Report | The previous year's winning methods |
| Background | 2402.03507 | Neural networks for abstraction and reasoning | Neural ARC approaches |
| Background | 1911.01547 | On the Measure of Intelligence (Chollet) | Skill-acquisition efficiency, the idea behind RHAE |
| Background | 1803.10122 | World Models (Ha & Schmidhuber) | Learned world models |
| Only if RL | 2509.09675 | CDE: Curiosity-Driven Exploration | RL exploration bonuses; only if RL beats LoRA SFT (§0) |

File names are `<id>_<slug>.pdf`. Every `verified:` field is `false` until someone has read the paper.

## Repos (`repos/`, shallow clones, cloned 2026-09-24/25)

Check the licence here **before** borrowing any code; prize eligibility depends on it.

| Folder | Source | Commit | Size | Licence | Why |
| --- | --- | --- | --- | --- | --- |
| `duck-harness` | Tufalabs/duck-harness | 7652836 | 3.0 GB | **UNCONFIRMED**: no LICENSE file; `ARC3-Inference/pyproject.toml` has an MIT classifier only | Milestone #1 winner. B000; fork and reproduce (E002). Includes the 25×20 run and viewer |
| `duck-repro-sonpham` | sonpham-org/arc-3 | b3c67be | 40 MB | **UNCONFIRMED** (same as above) | Instrumented Duck fork: GCP spot kit, run logs, reproduction matrix. Useful for E002/E003 |
| `ewm-baseline1` | astroseger/arc-3-agents-baseline1 | ef104f1 | 3.6 MB | MIT | Executable World Models agent (58.12 RHAE, frontier model) |
| `graph-explore-3rd` | dolphin-in-a-coma/arc-agi-3-just-explore | c2d9831 | 2.3 MB | MIT | Rudakov graph exploration, training-free |
| `arc3-kaggle-starter` | arcprize/ARC-AGI-3-Kaggle-Starter | eeb1535 | 147 KB | none found | Official Kaggle submission flow (`make submit`) |
| `arc3-agents-official` | arcprize/ARC-AGI-3-Agents | 4743e7d | 1.1 MB | MIT | Official agent templates and loop |
| `arc3-benchmarking-official` | arcprize/arc-agi-3-benchmarking | 4ca00e6 | 1.6 MB | MIT | Official benchmarking harness |
| `arc-agi-toolkit` | arcprize/arc-agi | f12822c | 803 KB | MIT | The `arc-agi` SDK source; §4.10 says to verify SDK facts here |
| `symbolica-arc3-agents` | symbolica-ai/ARC-AGI-3-Agents | c540041 | 971 MB | MIT | Symbolica's agent fork (cited by the technical report) |
| `driessmit-arc3` | DriesSmit/ARC3-solution | a6e77bb | 163 KB | none found | Preview-era solution (cited by the technical report); uses submodules, not fetched |
| `wd13ca-arc3-agents` | wd13ca/ARC-AGI-3-Agents | 641c00e | 1.9 MB | MIT | Preview-era agents (cited by the technical report) |
| `treequest` | SakanaAI/treequest | 96047d7 | 3.8 MB | Apache-2.0 | AB-MCTS implementation |
| `trm` | SamsungSAILMontreal/TinyRecursiveModels | c011037 | 13 MB | MIT | TRM |
| `avo` | NVIDIA AVO | f6dad9e | 2.2 MB | Apache-2.0 | Agent loop, lineage, knowledge base. Read `src/avo/loop.py`, `session.py`, `lineage.py`, `knowledge.py` |

Not cloned (Kaggle notebooks only; download with the Kaggle CLI once `kaggle.json` is set up):
- Reki, 2nd place: https://www.kaggle.com/code/ruichardliu/milestone1-2nd-solution
- forge, 3rd place: https://www.kaggle.com/code/mbmmurad/arc-agi-3-lb-0-86-3rd-place-candidate-milestone
- Duck, 1st place (Kaggle version): https://www.kaggle.com/code/jeroencottaar/tufa-labs-duck-harness-june-30-milestone-winner

`tufa-labs/arc-agi-3`, named in `papers/research/PAPERS_INDEX.md`, does not exist.

## Web (`web/`)
| File | What |
| --- | --- |
| `2026-09-24_kaggle-arc3_hardware-and-models.md` | Kaggle hardware and model availability for ARC-AGI-3 |
| `2026-09-25_arcprize_milestone-1-results.md` | Milestone #1: all three winners, methods and links |
| `2026-09-25_tufalabs_duck-harness.md` | Duck technical summary; benchmark mean 1.6002 ± 0.4475 (25 games × 20) |

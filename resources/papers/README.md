# resources/papers/

Inherited set, moved from `Docs/`. Each needs a `.meta.yaml` written before it can be cited.

| File | What | Meta written |
| --- | --- | --- |
| `ARC-AGI-3 A New Challenge for Frontier Agentic Intelligence.pdf` | Official technical report | [ ] |
| `2605.05138v2.pdf` | Executable World Models for ARC-AGI-3 in the Era of Coding Agents | [ ] |
| `ARC-AGI-3 Competition Research.pdf` | Competition research notes | [ ] |
| `WorldCoder.pdf` | Program-synthesis world models | [ ] |
| `VIGA.pdf` | — | [ ] |

Extracted plain text (`_challenge_paper.txt`, `_comp_research.txt`, `_ewm_paper.txt`) sits beside the PDFs.

## What to read for what

- **Executable World Models** — the closest published method to `sarbloh.worldmodel` + `sarbloh.parkh`. Reports
  58.12 mean RHAE with GPT-5.5 High, 15/25 solved. Names premature commitment as its main failure mode and
  excludes its own earlier ablations as unsound. Read its open-problems section before designing experiments.
- **Technical report** — the source for RHAE, human baselines, and the four named agent failure modes.
- **WorldCoder** — prior art for inducing programmatic world models from interaction; relevant to the MDL
  refactor pressure in `worldmodel`.

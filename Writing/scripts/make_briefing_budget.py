"""The briefing-budget figure: the tr87 first-frame briefing under a single 30-line cap and under the per-kind share.

Both boxes come from the same Briefing object built by the agent's own code (Sarbloh/harness). "Single cap" lists
every fact in the engine's order and stops after 30 fact and guess lines, which is what the line cap did before
Briefing.lines() shared the lines between fact kinds (experiments/E116_x8_perception/hypothesis.md, 2026-10-03).
"Shared" is Briefing.lines(30), the policy the agent runs now. Writes Writing/figures/perception/tr87/*.txt and
prints the per-kind counts. Run: uv run python Writing/scripts/make_briefing_budget.py
"""

import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "Sarbloh"))

from harness.agent import perception  # noqa: E402
from harness.game.games import build_games, make_arcade  # noqa: E402

OUT = REPO / "Writing/figures/perception/tr87"
LIMIT = 30


def single_cap(b: perception.Briefing, limit: int) -> list[str]:
    """Every fact kind in turn, then the guesses, cut after ``limit`` lines: no kind is held back."""
    body = [f for k in perception.FACT_KINDS for f in b.facts.get(k, [])]
    body = [f"- {f}" for f in body] + [f"- {g}" for g in b.guesses]
    return [f"BRIEFING ({b.controls})"] + body[:limit]


def main() -> None:
    rec = tempfile.mkdtemp(prefix="sarbloh_rec_", dir=os.environ.get("TEMP"))
    arcade = make_arcade("offline", environments_dir=str(REPO / "eval/real_games"), recordings_dir=rec)
    (g,) = build_games(arcade, "offline", only=["tr87"])
    st = g.start()
    rows = [list(r) for r in st.grid]
    objs, _ = perception.Tracker().observe(rows)
    b = perception.brief(rows, objs, st.available_actions)
    OUT.mkdir(parents=True, exist_ok=True)
    counts = {k: len(b.facts.get(k, [])) for k in perception.FACT_KINDS}
    print("facts per kind", counts, "guesses", len(b.guesses))
    before = single_cap(b, LIMIT)
    after = b.lines(LIMIT)
    (OUT / "single_cap.txt").write_text("\n".join(before), encoding="utf-8")
    (OUT / "shared.txt").write_text("\n".join(after), encoding="utf-8")
    for name, lines in (("single cap", before), ("shared", after)):
        n_align = sum(1 for x in lines if x in {f"- {f}" for f in b.facts.get("align", [])})
        print(name, "lines", len(lines), "align lines", n_align)


if __name__ == "__main__":
    main()

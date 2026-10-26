"""The perception figure: one real first frame shown three ways, by the agent's own code (Sarbloh/harness).

For each game: raw.txt (the integer grid a model would get as text), plain.png (the plain frame, what the
best offline run sent), map.png (the labelled map the briefing draws), briefing.txt (the MEASURED / GUESSES text),
objects.txt (the object list with stable ids). Run: uv run python Writing/scripts/make_perception_figure.py ls20 r11l
"""

import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "Sarbloh"))

from harness.agent import perception  # noqa: E402
from harness.game.games import build_games, make_arcade  # noqa: E402

OUT = REPO / "Writing/figures/perception"


def main(games: list[str]) -> None:
    rec = tempfile.mkdtemp(prefix="sarbloh_rec_", dir=os.environ.get("TEMP"))
    arcade = make_arcade("offline", environments_dir=str(REPO / "eval/real_games"), recordings_dir=rec)
    for g in build_games(arcade, "offline", only=games):
        st = g.start()
        rows = [list(r) for r in st.grid]
        name = g.game_id.split("-")[0]
        d = OUT / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "raw.txt").write_text("\n".join(" ".join(f"{v:2d}" for v in r) for r in rows), encoding="utf-8")
        (d / "plain.png").write_bytes(perception.render(rows, cell=10))
        tracker = perception.Tracker()
        objs, _ = tracker.observe(rows)
        b = perception.brief(rows, objs, st.available_actions)
        (d / "map.png").write_bytes(perception.picture(b, cell=10))
        (d / "briefing.txt").write_text("\n".join(b.lines(20)), encoding="utf-8")
        (d / "objects.txt").write_text("\n".join(o.row() for o in objs), encoding="utf-8")
        (d / "ascii.txt").write_text(perception.ascii(rows), encoding="utf-8")
        print(name, "actions", st.available_actions, "objects", len(objs), "briefing lines", len(b.lines()))


if __name__ == "__main__":
    main(sys.argv[1:] or ["ls20", "r11l"])

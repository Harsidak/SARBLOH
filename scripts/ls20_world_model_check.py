"""Check scripts/ls20_world_model.py against the real ls20 engine.

For every level: read the parameters out of the real level (this is the cheat a test is allowed, the agent must
read them off the frame), then
  1. play random action lists in both and report the first move where they disagree;
  2. plan a solution with the world model and replay it in the real game.

    uv run python scripts/ls20_world_model_check.py [--walks 200] [--length 60]
"""

from __future__ import annotations

import argparse
import importlib.util
import random
import sys
from pathlib import Path

from arcengine import ActionInput, GameAction

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import ls20_world_model as wm  # noqa: E402

SPEC = importlib.util.spec_from_file_location("ls20_game", ROOT / "eval/real_games/ls20/9607627b/ls20.py")
GAME = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GAME)

CHANGER_TAGS = {"ttfwljgohq": "shape", "soyhouuebz": "color", "rhsxkxzdjz": "rotation"}
ACTS = {1: GameAction.ACTION1, 2: GameAction.ACTION2, 3: GameAction.ACTION3, 4: GameAction.ACTION4}


def fresh(level: int):
    g = GAME.Ls20()
    g.set_level(level)
    return g


def params_from_game(g) -> wm.Ls20Params:
    lvl = g.current_level
    tagged = lambda t: lvl.get_sprites_by_tag(t)  # noqa: E731
    goals = [
        wm.Goal(s.x, s.y, wm.Key(g.ldxlnycps[i], g.tnkekoeuk[g.yjdexjsoa[i]], g.dhksvilbb[g.ehwheiwsk[i]]))
        for i, s in enumerate(g.plrpelhym)
    ]
    movers = {id(m._sprite) for m in g.wsoslqeku}
    changers = {
        (s.x, s.y): kind for tag, kind in CHANGER_TAGS.items() for s in tagged(tag) if id(s) not in movers
    }
    moving = []
    for m in g.wsoslqeku:
        t = m.bfdcztirdu
        track = frozenset(
            (t.x + i, t.y + j) for j in range(t.height) for i in range(t.width) if int(t.pixels[j, i]) >= 0
        )
        kind = next(k for tag, k in CHANGER_TAGS.items() if tag in (m._sprite.tags or []))
        moving.append(wm.MovingChanger(m._sprite.x, m._sprite.y, kind, track))
    pushers = [wm.Pusher(h.sprite.x, h.sprite.y, h.width, h.height, h.dx, h.dy) for h in g.hasivfwip]
    ui = g._step_counter_ui
    return wm.Ls20Params(
        player=(g.gudziatsk.x, g.gudziatsk.y),
        start_key=key_of(g),
        goals=goals,
        walls={(s.x, s.y) for s in tagged("ihdgageizm")},
        changers=changers,
        refills={(s.x, s.y) for s in tagged("npxgalaybz")},
        pushers=pushers,
        moving_changers=moving,
        budget=ui.osgviligwp,
        cost_per_move=ui.efipnixsvl,
        lives=g.aqygnziho,
        cell=g.gisrhqpee,
        free_match_move=g.level_index == 0,
    )


def key_of(g) -> wm.Key:
    return wm.Key(g.fwckfzsyc, g.tnkekoeuk[g.hiaauhahz], g.dhksvilbb[g.cklxociuu])


def observed(g, start_level: int) -> tuple:
    if g.level_index != start_level:
        return ("won",)
    return (g.gudziatsk.x, g.gudziatsk.y, key_of(g), g._step_counter_ui.current_steps, g.aqygnziho)


def predicted(s: wm.State) -> tuple:
    if s.won:
        return ("won",)
    return (s.x, s.y, s.key, s.budget, s.lives)


def act(g, a: int) -> None:
    g.perform_action(ActionInput(id=ACTS[a]), raw=True)


def check_level(level: int, walks: int, length: int, rng: random.Random) -> tuple[int, int, str]:
    p = params_from_game(fresh(level))
    bad = 0
    first = ""
    for _ in range(walks):
        g = fresh(level)
        s = wm.initial_state(p)
        for n, a in enumerate(rng.choices([1, 2, 3, 4], k=length)):
            act(g, a)
            s = wm.step(p, s, a)
            if s.lost or g.aqygnziho == 0:
                break
            if observed(g, level) != predicted(s):
                bad += 1
                first = first or f"move {n + 1} ({wm.ACTION_NAMES[a]}): game {observed(g, level)} model {predicted(s)}"
                break
            if s.won:
                break
    plan = wm.plan(p)
    if plan is None:
        verdict = "no plan found"
    else:
        g = fresh(level)
        for a in plan:
            act(g, a)
        verdict = f"plan of {len(plan)} moves {'WINS' if g.level_index != level or g._state.name == 'WIN' else 'FAILS'} in the real game"
    return bad, walks, verdict + (f"; first mismatch: {first}" if first else "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--walks", type=int, default=200)
    ap.add_argument("--length", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    for level in range(len(GAME.levels)):
        bad, walks, verdict = check_level(level, args.walks, args.length, rng)
        print(f"level {level + 1}: {walks - bad}/{walks} random walks match; {verdict}")


if __name__ == "__main__":
    main()

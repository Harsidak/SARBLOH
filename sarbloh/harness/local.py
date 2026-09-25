"""Local runner: the Kaggle code path against eval/real_games and a local OpenAI-compatible server.

    uv run python -m sarbloh.harness.local --games ls20 --concurrency 1 --max-actions 30
    uv run python -m sarbloh.harness.local --split dev --set llm.context_window=16384

Plumbing and debugging only (CLAUDE.md section 6): a 4B local model's score says nothing about the 27B.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from sarbloh.config import apply_llm_env, default_config, merge

REPO = Path(__file__).resolve().parents[2]


def _parse_set(items: list[str]) -> dict[str, Any]:
    """--set a.b=value (value parsed as JSON when possible)."""
    out: dict[str, Any] = {}
    for item in items:
        key, _, raw = item.partition("=")
        try:
            value: Any = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        node = out
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="*", help="game ids or prefixes (default: the split)")
    ap.add_argument("--split", choices=["dev", "heldout", "all"], default="dev")
    ap.add_argument("--env-dir", default=str(REPO / "eval" / "real_games"))
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--model-id", default="local")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--max-actions", type=int, default=None)
    ap.add_argument("--max-runtime", type=float, default=1800.0, help="seconds per game")
    ap.add_argument("--budget", type=float, default=None, help="total wall-clock seconds")
    ap.add_argument("--label", default="local")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. llm.context_window=16384")
    args = ap.parse_args()

    games = args.games
    if not games and args.split != "all":
        games = json.loads((REPO / "eval" / "splits" / "heldout_split.json").read_text())[args.split]
    overrides = {
        "label": args.label,
        "games": games,
        "concurrency": args.concurrency,
        "max_actions_per_game": args.max_actions,
        "max_runtime_s_per_game": args.max_runtime,
        "llm": {"base_url": args.base_url, "model_id": args.model_id, "context_window": 16384},
    }
    config = merge(merge(default_config(), overrides), _parse_set(args.set))
    apply_llm_env(config)

    from sarbloh.harness.games import build_games, make_arcade
    from sarbloh.harness.runner import run_games

    arcade = make_arcade("offline", environments_dir=args.env_dir)
    game_list = build_games(arcade, "offline", only=config["games"])
    run_dir = REPO / "runs" / f"{time.strftime('%Y%m%d_%H%M%S')}_{args.label}"
    soft_end = time.time() + args.budget if args.budget else None
    run_games(game_list, config, run_dir, soft_end_epoch=soft_end)
    print(f"[local] results in {run_dir}")


if __name__ == "__main__":
    main()

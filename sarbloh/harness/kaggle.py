"""Kaggle entry point. The notebook imports this from the attached sarbloh dataset and calls ``main(overrides)``.

Flow: print environment -> start vLLM -> build games (COMPETITION gateway on a rerun, OFFLINE env files on a
Save & Run) -> play under the notebook deadline -> write submission.parquet (offline) and results.json -> stop vLLM.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

from sarbloh.config import apply_llm_env, config_hash, default_config, merge

COMPETITION_DIR = Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-3")
WORKING_DIR = Path("/kaggle/working")


def is_competition_rerun() -> bool:
    return os.environ.get("KAGGLE_IS_COMPETITION_RERUN", "").strip().lower() in {"1", "true"}


def print_environment(config: dict[str, Any]) -> None:
    """CLAUDE.md section 5: the first thing a run prints. Settles the UNCONFIRMED hardware facts."""
    print("=" * 88)
    try:
        print(subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=30).stdout)
    except Exception as exc:  # noqa: BLE001
        print(f"nvidia-smi unavailable: {exc!r}")
    import arc_agi
    import arcengine

    print(f"python {sys.version.split()[0]} | arc_agi {getattr(arc_agi, '__version__', '?')} | "
          f"GameAction {[a.name for a in arcengine.GameAction]}")
    bundle = _bundle_info()
    print(f"sarbloh bundle: {json.dumps(bundle)}")
    print(f"rerun={is_competition_rerun()} config_hash={config_hash(config)}")
    print(json.dumps(config, indent=1, default=str))
    print("=" * 88, flush=True)


def _bundle_info() -> dict[str, Any]:
    marker = Path(__file__).resolve().parents[2] / "sarbloh-bundle.json"
    return json.loads(marker.read_text(encoding="utf-8")) if marker.exists() else {"source": "local checkout"}


def _wait_for_gateway(base_url: str, timeout_s: float = 600.0) -> None:
    deadline = time.monotonic() + timeout_s
    last = ""
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}api/games", timeout=10) as resp:
                if resp.status < 500:
                    return
        except Exception as exc:  # noqa: BLE001
            last = repr(exc)
        time.sleep(5)
    raise RuntimeError(f"Kaggle gateway did not become ready: {last}")


def main(overrides: dict[str, Any] | None = None, notebook_start_epoch: float | None = None) -> dict[str, Any]:
    start = notebook_start_epoch or time.time()
    config = merge(default_config(), overrides)
    rerun = is_competition_rerun()
    os.environ["ONLY_RESET_LEVELS"] = "true"
    os.environ.setdefault("RECORDINGS_DIR", str(WORKING_DIR / "recordings"))
    WORKING_DIR.mkdir(parents=True, exist_ok=True)
    print_environment(config)

    from sarbloh.harness.vllm import VllmServer

    server = VllmServer(config, WORKING_DIR)
    summary: dict[str, Any] = {}
    try:
        server.start()
        config["llm"]["base_url"] = server.base_url
        config["llm"]["model_id"] = config["vllm"]["served_model_name"]
        config["vllm"]["active_profile"] = server.active_profile
        apply_llm_env(config)
        if config["vllm"].get("watchdog"):
            server.start_watchdog()

        from sarbloh.harness.games import build_games, make_arcade
        from sarbloh.harness.runner import run_games

        if rerun:
            os.environ.setdefault("ARC_API_KEY", "test-key-123")
            os.environ.setdefault("ARC_BASE_URL", "http://gateway:8001/")
            _wait_for_gateway(os.environ["ARC_BASE_URL"])
            arcade = make_arcade("competition", base_url=os.environ["ARC_BASE_URL"])
            games = build_games(arcade, "competition")
        else:
            env_dir = str(COMPETITION_DIR / "environment_files")
            arcade = make_arcade("offline", environments_dir=env_dir)
            games = build_games(arcade, "offline", only=config["games"])

        soft_end = start + config["notebook_budget_s"] - config["teardown_reserve_s"]
        summary = run_games(games, config, WORKING_DIR / "run", soft_end_epoch=soft_end)
        summary["vllm_profile"] = server.active_profile
        summary["vllm_restarts"] = server.restarts
        (WORKING_DIR / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
        if not rerun:
            # Save & Run is not scored but must leave a valid submission file behind.
            import pandas as pd

            pd.DataFrame([["1_0", "1", True, 1]], columns=["row_id", "game_id", "end_of_game", "score"]).to_parquet(
                WORKING_DIR / "submission.parquet", index=False
            )
    finally:
        server.stop()
    return summary

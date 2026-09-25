"""Run a list of games concurrently under one wall-clock deadline and write results.json.

Replaces taaf's Benchmark.run + HarnessSolver._run_games: games start serially (Arcade.make is not assumed
thread-safe), then play on a pool of ``concurrency`` threads. At ``soft_end_epoch`` a stop event is set; the
agent stops at its next check and every game is finished, which in COMPETITION mode closes the shared
scorecard. A game left unfinished scores nothing, so ``finish`` is guaranteed in ``finally``.
"""

from __future__ import annotations

import json
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from sarbloh.agent.session import play_game
from sarbloh.config import config_hash
from sarbloh.harness.games import ArcGame


def run_games(
    games: list[ArcGame],
    config: dict[str, Any],
    run_dir: Path,
    soft_end_epoch: float | None = None,
    drain_timeout_s: float = 120.0,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    stop_event = threading.Event()

    def soft_remaining() -> float | None:
        return None if soft_end_epoch is None else max(0.0, soft_end_epoch - time.time())

    started: list[ArcGame] = []
    for game in games:
        try:
            game.start()
            started.append(game)
        except Exception as exc:  # noqa: BLE001 - one broken env must not sink the run
            print(f"[start-failed] {game.env_name}: {type(exc).__name__}: {exc}", flush=True)
    print(f"[runner] started {len(started)}/{len(games)} games, concurrency={config['concurrency']}", flush=True)

    pool = ThreadPoolExecutor(max_workers=max(1, int(config["concurrency"])), thread_name_prefix="game")
    futures = [pool.submit(play_game, g, config, run_dir, stop_event, soft_remaining) for g in started]
    status_every = 300.0
    last_status = time.time()
    try:
        while True:
            timeout = 5.0
            done, pending = wait(futures, timeout=timeout)
            if not pending:
                break
            if soft_end_epoch is not None and time.time() >= soft_end_epoch and not stop_event.is_set():
                print("[runner] soft deadline reached, stopping games", flush=True)
                stop_event.set()
                wait(futures, timeout=drain_timeout_s)
                break
            if time.time() - last_status >= status_every:
                last_status = time.time()
                _print_status(started, t0)
    except KeyboardInterrupt:
        stop_event.set()
        wait(futures, timeout=drain_timeout_s)
    finally:
        stop_event.set()
        for g in started:
            g.finish("cancelled")
        pool.shutdown(wait=False, cancel_futures=True)

    summary = summarize(started, config, time.time() - t0)
    (run_dir / "results.json").write_text(
        json.dumps({"summary": summary, "config": config, "runs": [g.run.to_json() for g in started if g.run]}, indent=1),
        encoding="utf-8",
    )
    print("[runner] " + json.dumps(summary), flush=True)
    return summary


def _print_status(games: list[ArcGame], t0: float) -> None:
    playing = [g for g in games if g.run and g.run.state == "playing"]
    levels = sum(g.run.levels_completed for g in games if g.run)
    actions = sum(len(g.run.history) for g in games if g.run)
    print(f"[status] t={time.time() - t0:.0f}s playing={len(playing)} levels={levels} actions={actions}", flush=True)


def summarize(games: list[ArcGame], config: dict[str, Any], wall_s: float) -> dict[str, Any]:
    runs = [g.run for g in games if g.run is not None]
    scores = [r.final_score or 0.0 for r in runs]
    return {
        "label": config.get("label"),
        "config_hash": config_hash(config),
        "games": len(runs),
        "mean_score": round(statistics.mean(scores), 4) if scores else 0.0,  # 0-100 scale, mean over games
        "median_score": round(statistics.median(scores), 4) if scores else 0.0,
        "levels_completed": sum(r.levels_completed for r in runs),
        "levels_total": sum(r.number_of_levels for r in runs),
        "actions": sum(len(r.history) for r in runs),
        "states": {s: sum(1 for r in runs if r.state == s) for s in sorted({r.state for r in runs})},
        "wall_s": round(wall_s, 1),
        "per_game": {r.game_id: round(r.final_score or 0.0, 3) for r in runs},
    }

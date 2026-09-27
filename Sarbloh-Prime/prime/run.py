"""Run Prime Agent on ARC-AGI-3 games: one root AgentSession per game, games in parallel, one deadline.

    main(overrides)                       # Kaggle: vLLM up, games (COMPETITION on a rerun, OFFLINE otherwise)
    python -m prime.run --local ...       # local: an already running OpenAI-compatible server (llama.cpp)

Writes <working>/prime_run/{results.json, summary.json, games/<game_id>/...} and prints a LEDGER_ROW line.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

PRIME_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PRIME_ROOT.parent
for _p in (PRIME_ROOT, REPO_ROOT):  # rlm/ + prime/ live in Sarbloh-Prime; sarbloh/ (games) at the repo root
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# pyrefly: ignore [missing-import]
from prime import prompts
# pyrefly: ignore [missing-import]
from prime.config import DEFAULT, config_hash, merge

COMPETITION_DIR = Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-3")


def is_competition_rerun() -> bool:
    return os.environ.get("KAGGLE_IS_COMPETITION_RERUN", "").strip().lower() in {"1", "true"}


def play_game(game: Any, cfg: dict[str, Any], llm: Any, run_dir: Path, stop_event: threading.Event,
              soft_end: float) -> dict[str, Any]:
    from prime.arc_host import ArcHost
    from prime.session import AgentSession

    game_dir = run_dir / "games" / game.game_id
    deadline = min(soft_end, time.time() + cfg["game_wall_s"])
    session_ref: dict[str, Any] = {}
    host = ArcHost(game, cfg["max_actions_per_game"], should_stop=lambda: stop_event.is_set() or time.time() >= deadline,
                   tokens_spent=lambda: session_ref["s"].tokens_spent() if "s" in session_ref else 0)
    task = prompts.ARC_TASK.format(game_id=game.game_id, win_levels=game.number_of_levels,
                                   max_actions=cfg["max_actions_per_game"],
                                   minutes=int(max(0, deadline - time.time()) // 60))
    session = AgentSession(cfg=cfg["agent"], llm=llm, name=game.game_id.split("-")[0], session_dir=game_dir,
                           task=task, arc=host, deadline=deadline, stop_event=stop_event,
                           global_harness_dir=run_dir / "global_harness")
    session_ref["s"] = session
    t0 = time.time()
    try:
        session.run()
    finally:
        game.finish("cancelled" if stop_event.is_set() and not host.finished else None)
    stats = {**session.stats, "end_reason": session.end_reason, "wall_s": round(time.time() - t0, 1),
             "tokens_total": session.tokens_spent(), "kernel_restarts": session.kernel.restarts}
    (game_dir / "session.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    return stats


def run_games(games: list[Any], cfg: dict[str, Any], llm: Any, run_dir: Path, soft_end: float) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "global_harness").mkdir(exist_ok=True)
    t0 = time.time()
    stop_event = threading.Event()
    started = []
    for game in games:
        try:
            game.start()
            started.append(game)
        except Exception as exc:  # noqa: BLE001 - one broken env must not sink the run
            print(f"[start-failed] {game.env_name}: {type(exc).__name__}: {exc}", flush=True)
    print(f"[runner] started {len(started)}/{len(games)} games, concurrency={cfg['concurrency']}", flush=True)
    sessions: dict[str, dict[str, Any]] = {}

    def job(g: Any) -> None:
        try:
            sessions[g.game_id] = play_game(g, cfg, llm, run_dir, stop_event, soft_end)
        except Exception:  # noqa: BLE001
            sessions[g.game_id] = {"end_reason": "crash", "traceback": traceback.format_exc()[-3000:]}
            g.finish("crashed")

    pool = ThreadPoolExecutor(max_workers=max(1, int(cfg["concurrency"])), thread_name_prefix="game")
    futures = [pool.submit(job, g) for g in started]
    last_status = time.time()
    try:
        while True:
            _, pending = wait(futures, timeout=5.0)
            if not pending:
                break
            if time.time() >= soft_end and not stop_event.is_set():
                print("[runner] soft deadline reached, stopping games", flush=True)
                stop_event.set()
                wait(futures, timeout=180)
                break
            if time.time() - last_status >= 120:
                last_status = time.time()
                _status(started, t0, llm)
    finally:
        stop_event.set()
        for g in started:
            g.finish("cancelled")
        pool.shutdown(wait=False, cancel_futures=True)
    summary = summarize(started, sessions, cfg, llm, time.time() - t0)
    (run_dir / "results.json").write_text(json.dumps(
        {"summary": summary, "config": cfg, "sessions": sessions,
         "runs": [g.run.to_json() for g in started if g.run]}, indent=1, default=str), encoding="utf-8")
    return summary


def _status(games: list[Any], t0: float, llm: Any) -> None:
    playing = sum(1 for g in games if g.run and g.run.state == "playing")
    levels = sum(g.run.levels_completed for g in games if g.run)
    actions = sum(len(g.run.history) for g in games if g.run)
    print(f"[status] t={time.time() - t0:.0f}s playing={playing} levels={levels} actions={actions} "
          f"llm={llm.usage.to_json()}", flush=True)


def summarize(games: list[Any], sessions: dict[str, dict[str, Any]], cfg: dict[str, Any], llm: Any,
              wall_s: float) -> dict[str, Any]:
    runs = [g.run for g in games if g.run is not None]
    scores = [r.final_score or 0.0 for r in runs]
    agg = lambda k: sum(int(s.get(k) or 0) for s in sessions.values())
    return {
        "experiment": cfg["experiment"],
        "config_hash": config_hash(cfg),
        "games": len(runs),
        "mean_score": round(statistics.mean(scores), 4) if scores else 0.0,
        "levels_completed": sum(r.levels_completed for r in runs),
        "levels_total": sum(r.number_of_levels for r in runs),
        "actions": sum(len(r.history) for r in runs),
        "states": {s: sum(1 for r in runs if r.state == s) for s in sorted({r.state for r in runs})},
        "end_reasons": {k: sum(1 for s in sessions.values() if s.get("end_reason") == k)
                        for k in sorted({str(s.get("end_reason")) for s in sessions.values()})},
        "turns": agg("turns"), "tool_calls": agg("tool_calls"), "cell_errors": agg("cell_errors"),
        "native_calls": agg("native_calls"), "fenced_calls": agg("fenced_calls"), "compactions": agg("compactions"),
        "continuations": agg("continuations"), "children": agg("children"),
        "llm": llm.usage.to_json(),
        "wall_s": round(wall_s, 1),
        "per_game": {r.game_id: {"score": round(r.final_score or 0.0, 3), "levels": r.levels_completed,
                                 "of": r.number_of_levels, "actions": len(r.history), "state": r.state}
                     for r in runs},
    }


def _print_env(cfg: dict[str, Any]) -> None:
    print("=" * 100)
    try:
        print(subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=30).stdout)
    except Exception as exc:  # noqa: BLE001
        print(f"nvidia-smi unavailable: {exc!r}")
    import arcengine

    print(f"python {sys.version.split()[0]} | GameAction {[a.name for a in arcengine.GameAction]}")
    print(f"rerun={is_competition_rerun()} config_hash={config_hash(cfg)}")
    print(json.dumps(cfg, indent=1, default=str))
    print("=" * 100, flush=True)


def main(overrides: dict[str, Any] | None = None, notebook_start: float | None = None,
         working_dir: Path = Path("/kaggle/working")) -> dict[str, Any]:
    """Kaggle entry point."""
    start = notebook_start or time.time()
    cfg = merge(DEFAULT, overrides)
    rerun = is_competition_rerun()
    os.environ["ONLY_RESET_LEVELS"] = "true"
    os.environ.setdefault("RECORDINGS_DIR", str(working_dir / "recordings"))
    working_dir.mkdir(parents=True, exist_ok=True)
    _print_env(cfg)

    from prime.llm import LLM
    from prime.vllm import VllmServer
    from sarbloh.harness.games import build_games, make_arcade

    server = VllmServer(cfg["vllm"], working_dir)
    summary: dict[str, Any] = {}
    try:
        server.start()
        cfg["llm"]["base_url"] = server.base_url
        cfg["llm"]["model"] = cfg["vllm"]["served_model_name"]
        cfg["agent"]["tool_mode"] = server.tool_mode
        from prime.vllm import PROFILES

        max_len = int(PROFILES[server.profile]["--max-model-len"])  # fit context handling to the profile that started
        cfg["agent"]["max_tokens_per_turn"] = min(cfg["agent"]["max_tokens_per_turn"], max_len // 4)
        cfg["agent"]["compact_at_tokens"] = min(cfg["agent"]["compact_at_tokens"],
                                                max_len - cfg["agent"]["max_tokens_per_turn"] - 4096)
        if cfg["vllm"]["bench"] and not rerun:
            try:
                server.throughput_probe()
            except Exception as exc:  # noqa: BLE001
                print(f"[vllm] throughput probe failed: {exc!r}", flush=True)
        if cfg["vllm"]["watchdog"]:
            server.start_watchdog()
        llm = LLM(cfg["llm"])
        if rerun:
            os.environ.setdefault("ARC_API_KEY", "test-key-123")
            os.environ.setdefault("ARC_BASE_URL", "http://gateway:8001/")
            _wait_for_gateway(os.environ["ARC_BASE_URL"])
            arcade = make_arcade("competition", base_url=os.environ["ARC_BASE_URL"])
            games = build_games(arcade, "competition")
        else:
            arcade = make_arcade("offline", environments_dir=str(COMPETITION_DIR / "environment_files"))
            games = build_games(arcade, "offline", only=cfg["games"])
        soft_end = start + cfg["notebook_budget_s"] - cfg["teardown_reserve_s"]
        summary = run_games(games, cfg, llm, working_dir / "prime_run", soft_end)
        summary.update({"vllm_profile": server.profile, "tool_mode": server.tool_mode,
                        "vllm_attempts": server.attempts, "vllm_restarts": server.restarts,
                        "throughput": server.bench})
        (working_dir / "prime_run" / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    finally:
        server.stop()
        if not rerun:
            import pandas as pd  # Save & Run is not scored but must leave a valid submission file

            pd.DataFrame([["1_0", "1", True, 1]], columns=["row_id", "game_id", "end_of_game", "score"]).to_parquet(
                working_dir / "submission.parquet", index=False)
    print("LEDGER_ROW " + json.dumps({k: summary.get(k) for k in (
        "experiment", "config_hash", "vllm_profile", "tool_mode", "throughput", "games", "mean_score",
        "levels_completed", "levels_total", "actions", "turns", "tool_calls", "native_calls", "fenced_calls",
        "wall_s")}), flush=True)
    return summary


def _wait_for_gateway(base_url: str, timeout_s: float = 600.0) -> None:
    import urllib.request

    deadline, last = time.monotonic() + timeout_s, ""
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}api/games", timeout=10) as resp:
                if resp.status < 500:
                    return
        except Exception as exc:  # noqa: BLE001
            last = repr(exc)
        time.sleep(5)
    raise RuntimeError(f"Kaggle gateway did not become ready: {last}")


def local() -> None:
    """Local plumbing run against an already running server, OFFLINE games from eval/real_games."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="+", default=["ls20"])
    ap.add_argument("--env-dir", default=str(REPO_ROOT / "eval" / "real_games"))
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--model", default="local")
    ap.add_argument("--minutes", type=float, default=10)
    ap.add_argument("--max-actions", type=int, default=30)
    ap.add_argument("--tool-mode", default="native", choices=["native", "fenced"])
    ap.add_argument("--no-thinking", action="store_true")
    ap.add_argument("--out", default=str(REPO_ROOT / "runs" / "prime_local"))
    a = ap.parse_args()
    from prime.llm import LLM
    from sarbloh.harness.games import build_games, make_arcade

    cfg = merge(DEFAULT, {
        "games": a.games, "concurrency": len(a.games), "game_wall_s": a.minutes * 60,
        "max_actions_per_game": a.max_actions, "notebook_budget_s": a.minutes * 60 + 60, "teardown_reserve_s": 0,
        "llm": {"base_url": a.base_url, "model": a.model, "top_k": None,
                "chat_template_kwargs": {"enable_thinking": not a.no_thinking}},
        "agent": {"tool_mode": a.tool_mode, "compact_at_tokens": 12000, "max_tokens_per_turn": 4096},
    })
    os.environ["ONLY_RESET_LEVELS"] = "true"
    arcade = make_arcade("offline", environments_dir=a.env_dir)
    games = build_games(arcade, "offline", only=a.games)
    summary = run_games(games, cfg, LLM(cfg["llm"]), Path(a.out), time.time() + a.minutes * 60)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    local()

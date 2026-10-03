"""Run Prime Agent on ARC-AGI-3 games: one root AgentSession per game, games in parallel, one deadline.

    main(overrides)                       # Kaggle: model server up, games (COMPETITION on a rerun, OFFLINE otherwise)
    python -m harness.run --local ...       # local: an already running OpenAI-compatible server (llama.cpp)

Writes <working>/prime_run/{results.json, summary.json, trace.md, games/<game_id>/..., recordings/...} and prints a
LEDGER_ROW line. ``harness.trace`` turns the transcripts and the SDK recordings into a per-game step map (steps.md).
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
for _p in (PRIME_ROOT, REPO_ROOT):  # rlm/ + harness/ live in Sarbloh; the games now live in harness/game/games.py
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# pyrefly: ignore [missing-import]
from harness import trace
from harness.agent import prompts
from harness.config import build_config, config_hash

COMPETITION_DIR = Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-3")


def is_competition_rerun() -> bool:
    return os.environ.get("KAGGLE_IS_COMPETITION_RERUN", "").strip().lower() in {"1", "true"}


def fit_context(agent: dict[str, Any], window: int) -> None:
    """Scale the compaction settings to the served window. At 128k they stay as configured (16384 reserve, 16000
    kept, compact above 100000); smaller windows get at most a quarter each, and a turn's output never exceeds the
    reserve."""
    agent["context_window"] = window
    comp = agent["compaction"]
    comp["reserve_tokens"] = min(comp["reserve_tokens"], window // 4)
    comp["keep_recent_tokens"] = min(comp["keep_recent_tokens"], window // 4)
    if comp.get("trigger_tokens") and comp["trigger_tokens"] < 2 * comp["keep_recent_tokens"]:
        raise ValueError(f"compaction.trigger_tokens {comp['trigger_tokens']} < 2x keep_recent_tokens "
                         f"{comp['keep_recent_tokens']}: every compaction would leave the context near the trigger")
    agent["max_tokens_per_turn"] = min(agent["max_tokens_per_turn"], comp["reserve_tokens"])


def play_game(game: Any, cfg: dict[str, Any], llm: Any, run_dir: Path, stop_event: threading.Event,
              soft_end: float, scheduler: Any = None) -> dict[str, Any]:
    from harness.game.arc_host import ArcHost
    from harness.agent.agent import AgentSession

    game_dir = run_dir / "games" / game.game_id
    deadline = min(soft_end, time.time() + cfg["game_wall_s"])
    if scheduler is not None:  # this game's LLM calls run only while it holds a scheduler slot
        from harness.scheduler import ScheduledLLM

        llm = ScheduledLLM(llm, scheduler, game.game_id, levels_won=lambda: game.state.levels_completed,
                           n_levels=game.number_of_levels,
                           should_stop=lambda: stop_event.is_set() or time.time() >= deadline)
    session_ref: dict[str, Any] = {}
    host = ArcHost(game, cfg["max_actions_per_game"], should_stop=lambda: stop_event.is_set() or time.time() >= deadline,
                   tokens_spent=lambda: session_ref["s"].tokens_spent() if "s" in session_ref else 0,
                   stop_after_levels=cfg.get("stop_after_levels"))
    task = prompts.task_message(game_id=game.game_id, max_actions=cfg["max_actions_per_game"],
                                minutes=int(max(0, deadline - time.time()) // 60))
    session = AgentSession(cfg=cfg["agent"], llm=llm, name=game.game_id.split("-")[0], session_dir=game_dir,
                           task=task, arc=host, deadline=deadline, stop_event=stop_event,
                           memory_root=run_dir / "memory")
    session_ref["s"] = session
    t0 = time.time()
    try:
        session.run()
    finally:
        if scheduler is not None:
            llm.close()
        game.finish("cancelled" if stop_event.is_set() and not host.finished else None)
    if game.scorecard:  # the SDK's own scorecard for this game (offline: one per game)
        (game_dir / "scorecard.json").write_text(json.dumps(game.scorecard, indent=1), encoding="utf-8")
    stats = {**session.stats, "end_reason": session.end_reason, "wall_s": round(time.time() - t0, 1),
             "tokens_total": session.tokens_spent(), "kernel_restarts": session.kernel.restarts,
             "recording": game.recording_path}
    (game_dir / "session.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    return stats


def git_sha() -> str:
    """The code version: $PRIME_GIT_SHA, else `git rev-parse HEAD` (+ "-dirty"), else "unknown" (a dataset copy)."""
    if os.environ.get("PRIME_GIT_SHA"):
        return os.environ["PRIME_GIT_SHA"]
    try:
        sha = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True,
                             timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "Sarbloh"], cwd=REPO_ROOT,
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return (sha + ("-dirty" if dirty else "")) if sha else "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def run_games(games: list[Any], cfg: dict[str, Any], llm: Any, run_dir: Path, soft_end: float) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    stop_event = threading.Event()
    started = []
    for game in games:
        try:
            game.start()
            started.append(game)
        except Exception as exc:  # noqa: BLE001 - one broken env must not sink the run
            print(f"[start-failed] {game.env_name}: {type(exc).__name__}: {exc}", flush=True)
    sched_cfg = cfg.get("scheduler") or {}
    scheduler = None
    workers = max(1, int(cfg["concurrency"]))
    if sched_cfg.get("enabled"):  # every game alive at once; the scheduler decides who is on the GPU
        from harness.scheduler import PriorityScheduler

        scheduler = PriorityScheduler(int(sched_cfg.get("slots", 6)), int(sched_cfg.get("quantum_calls", 4)),
                                      float(sched_cfg.get("token_scale", 80000.0)),
                                      log_path=run_dir / "scheduler.jsonl")
        workers = max(1, len(started))
    print(f"[runner] started {len(started)}/{len(games)} games, concurrency={workers}"
          + (f", scheduler slots={scheduler.slots}" if scheduler else ""), flush=True)
    sessions: dict[str, dict[str, Any]] = {}

    def job(g: Any) -> None:
        try:
            sessions[g.game_id] = play_game(g, cfg, llm, run_dir, stop_event, soft_end, scheduler)
        except Exception:  # noqa: BLE001
            sessions[g.game_id] = {"end_reason": "crash", "traceback": traceback.format_exc()[-3000:]}
            g.finish("crashed")

    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="game")
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
                if scheduler is not None:
                    print(f"[scheduler] {scheduler.status()}", flush=True)
    finally:
        stop_event.set()
        if scheduler is not None:
            scheduler.close()
        for g in started:
            g.finish("cancelled")
        pool.shutdown(wait=False, cancel_futures=True)
    summary = summarize(started, sessions, cfg, llm, time.time() - t0)
    if scheduler is not None:
        summary["scheduler"] = scheduler.stats()
    (run_dir / "results.json").write_text(json.dumps(
        {"summary": summary, "config": cfg, "sessions": sessions,
         "runs": [g.run.to_json() for g in started if g.run]}, indent=1, default=str), encoding="utf-8")
    try:
        print(f"[trace] step maps: {trace.build_run(run_dir)}", flush=True)
    except Exception as exc:  # noqa: BLE001 - the trace is a report; it must not fail the run
        print(f"[trace] failed: {type(exc).__name__}: {exc}", flush=True)
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
        "git_sha": git_sha(),
        "model": cfg["llm"].get("model"),
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
        "continuations": agg("continuations"),
        # memory and perception use
        "act_calls": agg("act_calls"), "act_arg_errors": agg("act_arg_errors"), "recalls": agg("recalls"),
        "hypothesis_events": agg("hypothesis_events"), "promotions": agg("promotions"),
        "curator_runs": agg("curator_runs"), "curator_errors": agg("curator_errors"),
        "skills_written": agg("skills_written"), "images_sent": agg("images_sent"),
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
    cfg = build_config(overrides)
    rerun = is_competition_rerun()
    os.environ["ONLY_RESET_LEVELS"] = "true"
    run_dir = working_dir / "prime_run"
    os.environ.setdefault("RECORDINGS_DIR", str(run_dir / "recordings"))
    record = bool(cfg.get("record")) and not rerun  # rerun outputs are never seen; skip the disk writes
    working_dir.mkdir(parents=True, exist_ok=True)
    _print_env(cfg)

    from harness.llm import bench
    from harness.llm.client import LLM
    from harness.llm.spec import get_spec
    from harness.llm.server import LlmServer
    from harness.game.games import build_games, make_arcade

    spec = get_spec(cfg["model"])
    soft_end = start + cfg["notebook_budget_s"] - cfg["teardown_reserve_s"]
    server = LlmServer(spec, cfg["server"], working_dir, deadline=soft_end)
    summary: dict[str, Any] = {}
    try:
        server.start()
        cfg["llm"]["base_url"] = server.base_url
        cfg["llm"]["model"] = spec.served_model_name
        if server.tool_mode != "native":
            print(f"[server] WARNING: tool mode {server.tool_mode!r}: the agent needs native tool calls to act, so "
                  "this run can only answer in ```python blocks and will not move", flush=True)
        cfg["agent"]["vision"] = server.vision  # images only when the served profile passed the image smoke test
        fit_context(cfg["agent"], server.max_model_len)  # fit context handling to the profile that started
        if cfg["server"]["bench"] and not rerun:
            try:
                bench.quick(server)
            except Exception as exc:  # noqa: BLE001
                print(f"[server] throughput probe failed: {exc!r}", flush=True)
        if cfg["server"]["watchdog"]:
            server.start_watchdog()
        llm = LLM(cfg["llm"], spec=spec, gate=server.gate)
        if rerun:
            os.environ.setdefault("ARC_API_KEY", "test-key-123")
            os.environ.setdefault("ARC_BASE_URL", "http://gateway:8001/")
            _wait_for_gateway(os.environ["ARC_BASE_URL"])
            arcade = make_arcade("competition", base_url=os.environ["ARC_BASE_URL"],
                                 recordings_dir=os.environ["RECORDINGS_DIR"])
            games = build_games(arcade, "competition", record=record)
        else:
            arcade = make_arcade("offline", environments_dir=str(COMPETITION_DIR / "environment_files"),
                                 recordings_dir=os.environ["RECORDINGS_DIR"])
            games = build_games(arcade, "offline", only=cfg["games"], record=record)
        summary = run_games(games, cfg, llm, run_dir, soft_end)
        summary.update({"server_profile": server.profile, "tool_mode": server.tool_mode, "vision": server.vision,
                        "image_probe": server.image_probe,
                        "server_attempts": server.attempts, "server_restarts": server.restarts,
                        "server_freezes": server.freezes,
                        "throughput": server.bench})
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    finally:
        server.stop()
        if not rerun:
            import pandas as pd  # Save & Run is not scored but must leave a valid submission file

            pd.DataFrame([["1_0", "1", True, 1]], columns=["row_id", "game_id", "end_of_game", "score"]).to_parquet(
                working_dir / "submission.parquet", index=False)
    print("LEDGER_ROW " + json.dumps({k: summary.get(k) for k in (
        "experiment", "config_hash", "git_sha", "model", "server_profile", "tool_mode", "vision", "throughput",
        "games",
        "mean_score", "levels_completed", "levels_total", "actions", "turns", "tool_calls", "native_calls",
        "fenced_calls", "wall_s")}), flush=True)
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
    ap.add_argument("--no-thinking", action="store_true")
    ap.add_argument("--experiment", default="sarbloh_experimentation_local")
    ap.add_argument("--ctx", type=int, default=16384, help="context window of the local server (llama.cpp -c)")
    ap.add_argument("--out", default=str(REPO_ROOT / "runs" / "prime_local"))
    ap.add_argument("--vision", action="store_true", help="send the picture (the server must take images, e.g. "
                    "serve_llm.ps1 -Vision)")
    ap.add_argument("--levels", type=int, default=None, help="end each game after this many levels")
    ap.add_argument("--max-tokens", type=int, default=4096, help="output tokens per turn")
    a = ap.parse_args()
    from harness.llm.client import LLM
    from harness.game.games import build_games, make_arcade

    cfg = build_config({
        "experiment": a.experiment, "stop_after_levels": a.levels,
        "games": a.games, "concurrency": len(a.games), "game_wall_s": a.minutes * 60,
        "max_actions_per_game": a.max_actions, "notebook_budget_s": a.minutes * 60 + 60, "teardown_reserve_s": 0,
        "llm": {"base_url": a.base_url, "model": a.model, "top_k": None,
                "chat_template_kwargs": {"enable_thinking": not a.no_thinking}},
        "agent": {"max_tokens_per_turn": a.max_tokens, "vision": a.vision,
                  "compaction": {"reserve_tokens": 4608, "keep_recent_tokens": 4000}},
    })
    cfg["agent"]["context_window"] = a.ctx
    os.environ["ONLY_RESET_LEVELS"] = "true"
    arcade = make_arcade("offline", environments_dir=a.env_dir, recordings_dir=str(Path(a.out) / "recordings"))
    games = build_games(arcade, "offline", only=a.games, record=bool(cfg["record"]))
    summary = run_games(games, cfg, LLM(cfg["llm"]), Path(a.out), time.time() + a.minutes * 60)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    local()

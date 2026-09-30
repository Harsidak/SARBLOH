"""E007 serving benchmark: the same workloads on every profile, so tokens/s numbers compare like for like.

- ``decode``: N streams x 512 forced tokens (ignore_eos); per-stream prompts differ so nothing is shared.
- ``cold_prefill``: N fresh ~24k-token prompts at once, 1 output token each: prefill throughput, and the E006
  pattern (the third freeze came on ten cold 40k-token prefills right after a restart).
- ``agent_like``: N sessions x T turns of an append-only conversation (shared system prompt, per-session grid
  text, a grid delta per turn): time per turn, prefix-cache hit rate, spec-decode acceptance.
- ``tool_check``: the dedicated-tool schema shape (enum string, list of strings); records whether the raw parser
  output needed ``spec.fix_tool_arguments``.
- ``stress``: agent-like load through the real client and gate while the engine is frozen on purpose (SIGSTOP on
  the engine processes); the watchdog must detect it, dump stacks, restart, and the clients must finish.

Counters come from vLLM's /metrics (names logged by the watchdog's ``metrics_keys`` event); a missing counter
reads as None, never as 0.
"""

from __future__ import annotations

import json
import os
import random
import signal
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from prime.llm.client import LLM, ServerGate
from prime.llm.vllm import VllmServer, log, request_json

SYSTEM = ("You are an agent playing a grid game. Frames are 64x64 grids of integers 0-15, origin top-left, (x, y). "
          "Actions: 1 up, 2 down, 3 left, 4 right, 5 use, 6 click x y. Think about the objects and their rules, "
          "then act with the tools. Keep notes short. ") * 40  # ~2k tokens, identical for every session


def grid_text(seed: int, size: int = 64) -> str:
    rng = random.Random(seed)
    rows = []
    for _ in range(size):
        rows.append(" ".join(str(rng.choice((0, 0, 0, 0, 1, 3, 5, 8, 11))) for _ in range(size)))
    return "\n".join(rows)


def _delta(m0: dict | None, m1: dict | None, key: str) -> float | None:
    if m0 is None or m1 is None or key not in m1:
        return None
    return m1[key] - m0.get(key, 0.0)


def _ratio(a: float | None, b: float | None) -> float | None:
    return None if a is None or not b else round(a / b, 3)


def _post(server: VllmServer, body: dict[str, Any], timeout: float = 900) -> tuple[dict, float]:
    t = time.time()
    out = request_json(f"{server.base_url}/chat/completions", {"model": server.spec.served_model_name, **body},
                       timeout=timeout)
    return out, time.time() - t


def decode(server: VllmServer, streams: int, max_tokens: int = 512) -> dict[str, Any]:
    def one(i: int) -> int:
        out, _ = _post(server, {"temperature": 0.7, "max_tokens": max_tokens, "ignore_eos": True,
                                "chat_template_kwargs": {"enable_thinking": False},
                                "messages": [{"role": "user", "content": f"[{i}] Describe a grid puzzle game."}]})
        return (out.get("usage") or {}).get("completion_tokens", 0)

    m0, t = server.metrics(), time.time()
    with ThreadPoolExecutor(streams) as pool:
        total = sum(pool.map(one, range(streams)))
    wall, m1 = time.time() - t, server.metrics()
    return {"streams": streams, "tok_s": round(total / max(1e-6, wall), 1),
            "per_stream_tok_s": round(total / max(1e-6, wall) / streams, 1), "wall_s": round(wall, 1),
            "spec_accept": _ratio(_delta(m0, m1, "vllm:spec_decode_num_accepted_tokens_total"),
                                  _delta(m0, m1, "vllm:spec_decode_num_draft_tokens_total"))}


def cold_prefill(server: VllmServer, streams: int, grids: int = 3) -> dict[str, Any]:
    def one(i: int) -> tuple[int, float]:
        text = "\n\n".join(grid_text(10_000 + 97 * i + g) for g in range(grids))
        out, s = _post(server, {"temperature": 0.0, "max_tokens": 1, "chat_template_kwargs": {"enable_thinking": False},
                                "messages": [{"role": "user", "content": f"[{i}]\n{text}\nSay ok."}]})
        return (out.get("usage") or {}).get("prompt_tokens", 0), s

    t = time.time()
    with ThreadPoolExecutor(streams) as pool:
        rows = list(pool.map(one, range(streams)))
    wall = time.time() - t
    tokens = sum(r[0] for r in rows)
    return {"streams": streams, "prompt_tokens_each": rows[0][0], "prefill_tok_s": round(tokens / max(1e-6, wall)),
            "max_latency_s": round(max(r[1] for r in rows), 1), "wall_s": round(wall, 1)}


def _session(chat: Any, sid: int, turns: int, max_tokens: int, thinking: bool) -> list[float]:
    msgs: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM},
                                  {"role": "user", "content": f"Game {sid}. Frame 0:\n{grid_text(sid)}\nWhat do you see?"}]
    times = []
    for turn in range(1, turns + 1):
        t = time.time()
        content = chat(msgs, max_tokens, thinking)
        times.append(time.time() - t)
        msgs.append({"role": "assistant", "content": content})
        delta = grid_text(sid * 1000 + turn, size=24)
        msgs.append({"role": "user", "content": f"Turn {turn}: you pressed 4. Changed region:\n{delta}\nNext?"})
    return times


def agent_like(server: VllmServer, sessions: int, turns: int = 5, max_tokens: int = 256,
               thinking: bool = False) -> dict[str, Any]:
    def chat(msgs: list, max_tokens: int, thinking: bool) -> str:
        out, _ = _post(server, {"temperature": 0.6, "max_tokens": max_tokens, "messages": msgs,
                                "chat_template_kwargs": {"enable_thinking": thinking}})
        return out["choices"][0]["message"].get("content") or "ok"

    m0, t = server.metrics(), time.time()
    with ThreadPoolExecutor(sessions) as pool:
        per = list(pool.map(lambda s: _session(chat, s, turns, max_tokens, thinking), range(sessions)))
    wall, m1 = time.time() - t, server.metrics()
    first = [p[0] for p in per]
    later = [x for p in per for x in p[1:]]
    return {"sessions": sessions, "turns": turns, "wall_s": round(wall, 1),
            "turn1_mean_s": round(sum(first) / len(first), 2),
            "later_turn_mean_s": round(sum(later) / max(1, len(later)), 2),
            "prefix_hit_rate": _ratio(_delta(m0, m1, "vllm:prefix_cache_hits_total"),
                                      _delta(m0, m1, "vllm:prefix_cache_queries_total")),
            "prompt_tokens": _delta(m0, m1, "vllm:prompt_tokens_total"),
            "gen_tok_s": _ratio(_delta(m0, m1, "vllm:generation_tokens_total"), wall),
            "spec_accept": _ratio(_delta(m0, m1, "vllm:spec_decode_num_accepted_tokens_total"),
                                  _delta(m0, m1, "vllm:spec_decode_num_draft_tokens_total"))}


TOOLS = [
    {"type": "function", "function": {"name": "plan", "description": "Update the plan.", "parameters": {
        "type": "object", "required": ["goal", "phase"], "properties": {
            "goal": {"type": "string"}, "phase": {"type": "string", "enum": ["explore", "exploit", "verify"]}}}}},
    {"type": "function", "function": {"name": "act", "description": "Take up to 5 actions.", "parameters": {
        "type": "object", "required": ["actions", "expect"], "properties": {
            "actions": {"type": "array", "items": {"type": "string"}}, "expect": {"type": "string"}}}}},
]


def tool_check(server: VllmServer) -> dict[str, Any]:
    body: dict[str, Any] = {"temperature": 0.0, "max_tokens": 2048, "tools": TOOLS, "tool_choice": "auto",
                            "messages": [{"role": "user", "content": "Call the plan tool with goal 'find the exit' "
                                                                     "and phase explore. Only call the tool."}]}
    if server.spec.smoke_template_kwargs:
        body["chat_template_kwargs"] = dict(server.spec.smoke_template_kwargs)
    out, s = _post(server, body, timeout=600)
    raw = list(out["choices"][0]["message"].get("tool_calls") or [])
    fixed = server.spec.fix_tool_calls(raw)
    args = [c["function"]["arguments"] for c in raw]
    return {"seconds": round(s, 1), "calls": [c["function"]["name"] for c in raw], "raw_arguments": args,
            "fixed_arguments": [c["function"]["arguments"] for c in fixed],
            "repair_needed": args != [c["function"]["arguments"] for c in fixed]}


def quick(server: VllmServer, concurrency: int = 8) -> dict[str, Any]:
    """The throughput probe run.py prints before the games (same keys as before E007)."""
    one, many = decode(server, 1), decode(server, concurrency)
    server.bench = {"single_stream_tok_s": one["tok_s"], "aggregate_tok_s": many["tok_s"],
                    "concurrency": concurrency, "spec_accept": many["spec_accept"]}
    log(f"throughput: {server.bench}")
    return server.bench


def profile_bench(server: VllmServer, streams: int = 10) -> dict[str, Any]:
    row: dict[str, Any] = {"profile": server.profile, "tool_mode": server.tool_mode,
                           "flags": server.spec.profiles[server.profile]["flags"],
                           "model_dataset": server.spec.profiles[server.profile]["model_dataset"],
                           "startup_s": server.startup_s}
    for name, fn in (("tool_check", lambda: tool_check(server)),
                     ("decode_1", lambda: decode(server, 1)),
                     ("decode_8", lambda: decode(server, 8)),  # E006's probe ran 8 streams: like for like
                     (f"decode_{streams}", lambda: decode(server, streams)),
                     ("cold_prefill", lambda: cold_prefill(server, streams)),
                     ("agent_like", lambda: agent_like(server, streams)),
                     ("agent_like_thinking", lambda: agent_like(server, streams, turns=3, max_tokens=1024,
                                                                thinking=True))):
        try:
            row[name] = fn()
        except Exception as exc:  # noqa: BLE001 - one workload failing must not lose the others
            row[name] = {"error": repr(exc)[:500]}
        log(f"{server.profile} {name}: {json.dumps(row[name], default=str)[:400]}")
    return row


# --- freeze injection ---------------------------------------------------------------------------------------
def engine_pids(server: VllmServer) -> list[int]:
    """Every process in the server's group except the API server itself (the engine core and its workers)."""
    if server.process is None:
        return []
    out = subprocess.run(["pgrep", "-g", str(os.getpgid(server.process.pid))], capture_output=True, text=True).stdout
    return [int(p) for p in out.split() if p.strip() and int(p) != server.process.pid]


def stress(server: VllmServer, llm_cfg: dict[str, Any], minutes: float = 12.0, sessions: int = 10,
           freeze_at_s: float = 120.0, turns: int = 6) -> dict[str, Any]:
    """Agent-like sessions through ``LLM`` + the server's gate, with the watchdog on, freezing the engine once."""
    llm = LLM(llm_cfg, spec=server.spec, gate=server.gate)
    end = time.time() + minutes * 60
    stats = {"ok": 0, "failed": 0, "errors": [], "latencies": []}
    lock = threading.Lock()

    def chat(msgs: list, max_tokens: int, thinking: bool) -> str:
        t = time.time()
        try:
            reply = llm.chat(msgs, max_tokens=max_tokens, timeout_s=max(30.0, min(900.0, end - time.time())),
                             thinking=thinking)
        except Exception as exc:  # noqa: BLE001
            with lock:
                stats["failed"] += 1
                stats["errors"].append(repr(exc)[:300])
            return "error"
        with lock:
            stats["ok"] += 1
            stats["latencies"].append(time.time() - t)
        return reply.content or "ok"

    def worker(sid: int) -> None:
        while time.time() < end - 60:
            _session(chat, 100 + sid, turns, 256, False)
            sid += sessions

    server.start_watchdog()
    froze_at = None
    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(sessions)]
    for th in threads:
        th.start()
    time.sleep(freeze_at_s)
    pids = engine_pids(server)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGSTOP)
        except OSError:
            pass
    froze_at = time.time()
    log(f"stress: SIGSTOP engine pids {pids}")
    for th in threads:
        th.join(timeout=max(0.0, end - time.time()) + 120)
    events = []
    if server.events_path.exists():
        events = [json.loads(l) for l in server.events_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    after = [e for e in events if e["t"] >= froze_at - 1]
    detected = next((e for e in after if e["event"] == "freeze_detected"), None)
    freeze = next((e for e in after if e["event"] == "freeze"), None)
    ready = next((e for e in after if e["event"] == "ready"), None)
    lat = sorted(stats["latencies"])
    return {"minutes": minutes, "sessions": sessions, "sigstop_pids": pids, "ok": stats["ok"],
            "failed": stats["failed"], "errors": stats["errors"][:10],
            "p50_s": round(lat[len(lat) // 2], 1) if lat else None, "max_s": round(lat[-1], 1) if lat else None,
            "detect_s": round(detected["t"] - froze_at) if detected else None,
            "recovered_s": round(ready["t"] - froze_at) if ready else None,
            "stack_dump": bool(freeze and freeze.get("stacks")), "restarts": server.restarts,
            "llm_failures": llm.usage.failures}


def run(model_name: str, profiles: list[str], vllm_cfg: dict[str, Any], llm_cfg: dict[str, Any], out_dir: Path,
        deadline: float, streams: int = 10, stress_minutes: float = 12.0) -> list[dict[str, Any]]:
    """Bench each profile on a fresh server, then stress the fastest native-tool profile. Rows go to
    ``out_dir/vllm_bench.json`` as they finish, so a crash keeps what was measured."""
    from prime.llm.spec import get_spec

    spec = get_spec(model_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "vllm_bench.json"
    rows: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []

    def save() -> None:
        path.write_text(json.dumps(rows, indent=1, default=str), encoding="utf-8")

    for profile in profiles:
        if time.time() > deadline - 900:
            rows.append({"model": model_name, "profile": profile, "skipped": "time"})
            save()
            continue
        server = VllmServer(spec, vllm_cfg, out_dir, deadline=deadline)
        try:
            server.start(chain=[profile])
            rows.append({"model": model_name, **profile_bench(server, streams)})
        except Exception as exc:  # noqa: BLE001
            rows.append({"model": model_name, "profile": profile, "error": repr(exc)[:800],
                         "attempts": server.attempts})
        finally:
            server.stop()
        save()
    ranked = [r for r in rows if r.get("model") == model_name and r.get("tool_mode") == "native"
              and isinstance(r.get(f"decode_{streams}"), dict) and "tok_s" in r[f"decode_{streams}"]]
    ranked.sort(key=lambda r: -r[f"decode_{streams}"]["tok_s"])
    if ranked and stress_minutes > 0 and time.time() < deadline - (stress_minutes * 60 + 900):
        best = ranked[0]["profile"]
        server = VllmServer(spec, {**vllm_cfg, "max_inflight": streams}, out_dir, deadline=deadline)
        try:
            server.start(chain=[best])
            cfg = {**spec.llm, **llm_cfg, "base_url": server.base_url, "model": spec.served_model_name}
            rows.append({"model": model_name, "profile": best,
                         "stress": stress(server, cfg, stress_minutes, sessions=streams)})
        except Exception as exc:  # noqa: BLE001
            rows.append({"model": model_name, "profile": best, "stress": {"error": repr(exc)[:800]}})
        finally:
            server.stop()
        save()
    return rows

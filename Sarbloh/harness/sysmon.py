"""Host memory: a snapshot for the logs, and the kernel's out-of-memory choice.

The 2026-10-04 run lost the model server twice to an exit with no Python stack (SIGKILL, most likely the kernel's OOM
killer: the Flash-Next PLE table alone is ~102 GB of pinned host RAM against 176 GB; UNCONFIRMED, nothing logged RAM).
So the server is marked never to be picked (``oom_score_adj`` -1000) and each game's REPL kernel is marked to be picked
first (+500): a dead REPL kernel is restarted by ``Kernel`` and costs one game its Python state, a dead server stalls
every game for minutes. The watchdog logs ``snapshot`` every minute, so the next crash shows what was growing.

Linux only (``/proc``); elsewhere every function is a no-op that returns an empty result.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

PROC = Path("/proc")
SERVER_OOM_SCORE_ADJ = -1000
KERNEL_OOM_SCORE_ADJ = 500


def set_oom_score_adj(pid: int, value: int) -> bool:
    """Lowering it needs root (Kaggle runs as root); False when it could not be written."""
    try:
        (PROC / str(pid) / "oom_score_adj").write_text(str(int(value)))
        return True
    except OSError:
        return False


def _ppids() -> dict[int, int]:
    out = {}
    for d in PROC.iterdir() if PROC.is_dir() else []:
        if d.name.isdigit():
            try:
                stat = (d / "stat").read_text()
                out[int(d.name)] = int(stat[stat.rindex(")") + 2:].split()[1])
            except (OSError, ValueError, IndexError):
                pass
    return out


def tree(pid: int) -> list[int]:
    """``pid`` and every process below it."""
    children: dict[int, list[int]] = {}
    for child, parent in _ppids().items():
        children.setdefault(parent, []).append(child)
    out, todo = [], [pid]
    while todo:
        p = todo.pop()
        out.append(p)
        todo.extend(children.get(p, []))
    return out


def protect_tree(pid: int, value: int = SERVER_OOM_SCORE_ADJ) -> int:
    """Sets ``value`` on ``pid`` and every process below it (the server's workers start after the launch). Returns how
    many were set."""
    return sum(set_oom_score_adj(p, value) for p in tree(pid))


def _rss_kb(pid: int) -> int:
    try:
        for line in (PROC / str(pid) / "status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return 0


def _name(pid: int) -> str:
    try:
        return (PROC / str(pid) / "comm").read_text().strip()
    except OSError:
        return "?"


def meminfo() -> dict[str, float]:
    """GB from /proc/meminfo: total, available, swap free, page cache, shared memory."""
    keys = {"MemTotal": "total_gb", "MemAvailable": "available_gb", "SwapFree": "swap_free_gb", "Cached": "cached_gb",
            "Shmem": "shmem_gb"}
    out: dict[str, float] = {}
    try:
        for line in (PROC / "meminfo").read_text().splitlines():
            k, _, rest = line.partition(":")
            if k in keys:
                out[keys[k]] = round(int(rest.split()[0]) / 1048576, 1)
    except (OSError, ValueError, IndexError):
        pass
    return out


def gpu() -> dict[str, Any]:
    try:
        line = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=20).stdout.strip().splitlines()[0]
        used, total = (int(x) for x in line.split(","))
        return {"vram_used_mib": used, "vram_total_mib": total}
    except Exception:  # noqa: BLE001 - no GPU (local tests)
        return {}


def snapshot(server_pid: int | None = None, top: int = 4) -> dict[str, Any]:
    """RAM and VRAM, the server tree's resident memory, this process's, and the largest processes."""
    if not PROC.is_dir():
        return {}
    out: dict[str, Any] = {**meminfo(), **gpu(), "harness_rss_gb": round(_rss_kb(os.getpid()) / 1048576, 1)}
    if server_pid:
        out["server_rss_gb"] = round(sum(_rss_kb(p) for p in tree(server_pid)) / 1048576, 1)
    pids = [int(d.name) for d in PROC.iterdir() if d.name.isdigit()]
    big = sorted(((_rss_kb(p), p) for p in pids), reverse=True)[:top]
    out["top"] = [{"pid": p, "name": _name(p), "rss_gb": round(kb / 1048576, 1)} for kb, p in big]
    return out

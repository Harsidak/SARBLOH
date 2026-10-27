"""Play a game through the real Sarbloh-Experimentation harness with replies written in a script, and export the SFT rows.

    uv run python Alignment/authoring/play_script.py Alignment/authoring/scripts/ls20_a.py [--out runs/authored/ls20_a]

The script is a Python file that defines ``GAME`` (an id prefix such as "ls20"), ``TURNS`` (a list of replies) and
optionally ``CURATOR`` ({level number: curator JSON dict} for the level-up reviews) and ``ENV_DIR`` (a folder of game
versions, default ``eval/real_games``). A reply is a dict with ``think`` (the reasoning) and ``code`` (one ipython
cell; a list gives several calls in one reply), or ``text`` with no code (the reply ends the turn).

Everything else is the harness as it runs on Kaggle (config of ``kaggle/experimental/sarbloh_experimentation.ipynb``:
Flash-Next, the vision prompt, append-only context, stubbed old code, the curator): the kernel really runs the code,
``act()`` really moves the local game, and ``harness/trace.py`` builds ``sft_levels.jsonl`` from the transcript and
checks every request against the digest the agent logged. So the rows are exactly what the model would be sent.

When the script runs out before the game ends, the run stops and ``pending.md`` in the output folder shows what the
model would see next (the messages after the last reply), so the next reply can be written. ``view.md`` has every
request's new messages in order.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
HARNESS_ROOT = REPO / "Sarbloh-Experimentation"
for p in (HARNESS_ROOT, REPO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from harness import run as hrun  # noqa: E402
from harness import trace  # noqa: E402
from harness.config import build_config  # noqa: E402
from harness.llm.client import Reply  # noqa: E402

BUDGET_S = 40680.0   # the notebook's 11.3 h: the minutes in the task message and the status lines match a Kaggle run


class ScriptExhausted(RuntimeError):
    pass


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    out = []
    for p in content or []:
        if isinstance(p, dict) and p.get("type") == "text":
            out.append(p.get("text", ""))
        elif isinstance(p, dict):
            out.append("[picture]")
    return "\n".join(out)


def _render(msgs: list[dict[str, Any]]) -> str:
    out = []
    for m in msgs:
        role = m.get("role")
        if role == "assistant":
            calls = m.get("tool_calls") or []
            out.append(f"### assistant ({len(calls)} calls)")
            continue
        out.append(f"### {role}\n{_text(m.get('content'))}")
    return "\n\n".join(out)


class ScriptedLLM:
    """Stands in for ``harness.llm.client.LLM``: answers the agent from ``turns`` and the curator from ``curator``."""

    def __init__(self, turns: list[dict[str, Any]], curator: dict[int, dict[str, Any]], out: Path,
                 stop: threading.Event, model: str) -> None:
        self.turns, self.curator, self.out, self.stop = list(turns), dict(curator), out, stop
        self.i = 0
        self.model = model
        self.cfg = {"model": model}
        from harness.llm.client import Usage

        self.usage = Usage()
        self.levels_reviewed: list[int] = []
        (out / "view.md").write_text("", encoding="utf-8")

    def _tail(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        last = max((i for i, m in enumerate(messages) if m.get("role") == "assistant"), default=0)
        return messages[last + 1:] if last else messages[1:]

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, max_tokens: int | None = None,
             timeout_s: float | None = None, thinking: bool | None = None,
             template_kwargs: dict[str, Any] | None = None) -> Reply:
        system = messages[0].get("content") if messages else ""
        if not tools and isinstance(system, str) and system.startswith("You review the memory"):
            return self._curate(messages)
        tail = self._tail(messages)
        with (self.out / "view.md").open("a", encoding="utf-8") as fh:
            fh.write(f"\n\n## request for reply {self.i + 1}\n\n{_render(tail)}\n")
        if self.i >= len(self.turns):
            (self.out / "pending.md").write_text(f"# next reply: {self.i + 1}\n\n{_render(tail)}\n", encoding="utf-8")
            self.stop.set()
            raise ScriptExhausted(f"script has {len(self.turns)} replies")
        t = self.turns[self.i]
        self.i += 1
        codes = t.get("code")
        codes = [codes] if isinstance(codes, str) else list(codes or [])
        calls = [{"id": f"call_{uuid.UUID(int=(self.i << 16) + k).hex[-24:]}", "type": "function",
                  "function": {"name": "ipython", "arguments": json.dumps({"code": c.strip("\n") + "\n"})}}
                 for k, c in enumerate(codes)]
        think = t.get("think", "").strip()
        content = t.get("text", "") if not calls else "\n\n"
        tokens = (len(think) + sum(len(c) for c in codes) + len(content)) // 4
        self.usage.add(0, tokens)
        return Reply(content=content, reasoning=think, tool_calls=calls, finish_reason="tool_calls" if calls else "stop",
                     prompt_tokens=0, completion_tokens=tokens)

    def _curate(self, messages: list[dict[str, Any]]) -> Reply:
        user = _text(messages[-1].get("content"))
        level_up = "<trigger>level_up</trigger>" in user
        out: dict[str, Any] = {"open_questions": [], "lessons": [], "skills": [], "load_skills": []}
        if level_up:
            n = len(self.levels_reviewed) + 1
            self.levels_reviewed.append(n)
            out = {**out, **self.curator.get(n, {})}
        self.usage.add(0, 0)
        return Reply(content=json.dumps(out), reasoning="", tool_calls=[], finish_reason="stop", prompt_tokens=0,
                     completion_tokens=0)


def load_script(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(f"authored_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def kaggle_config() -> dict[str, Any]:
    """The Kaggle notebook's config (model, budget, scheduler off: one game), vision on as on the Kaggle run."""
    cfg = build_config({
        "model": "qwen_flash_next", "experiment": "authored", "teardown_reserve_s": 300.0, "record": True,
        "tracing": True, "notebook_budget_s": BUDGET_S, "game_wall_s": BUDGET_S, "concurrency": 1,
        "scheduler": {"enabled": False},
        "agent": {"limits": {"max_turns": 1000000, "max_output_tokens": 10000000000}},
    })
    cfg["agent"]["vision"] = True
    hrun.fit_context(cfg["agent"], 139264)   # the SGLang window of the Kaggle profile
    return cfg


def play(script: Path, out: Path) -> dict[str, Any]:
    mod = load_script(script)
    from harness.game.games import build_games, make_arcade

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    os.environ["ONLY_RESET_LEVELS"] = "true"
    cfg = kaggle_config()
    env_dir = str(getattr(mod, "ENV_DIR", REPO / "eval" / "real_games"))
    arcade = make_arcade("offline", environments_dir=env_dir, recordings_dir=str(out / "recordings"))
    games = build_games(arcade, "offline", only=[mod.GAME], record=True)
    game = games[0]
    stop = threading.Event()
    llm = ScriptedLLM(mod.TURNS, getattr(mod, "CURATOR", {}), out, stop, cfg["llm"]["model"])
    import harness.agent.agent as agent_mod

    real_sleep = agent_mod.time.sleep
    agent_mod.time.sleep = lambda s: None if stop.is_set() else real_sleep(s)   # no 10 s wait after the script ends
    try:
        game.start()
        t0 = time.time()
        stats = hrun.play_game(game, cfg, llm, out, stop, time.time() + BUDGET_S, None)
    finally:
        agent_mod.time.sleep = real_sleep
        game.finish("cancelled")
    summary = hrun.summarize([game], {game.game_id: stats}, cfg, llm, time.time() - t0)
    (out / "results.json").write_text(json.dumps({"summary": summary, "config": cfg, "sessions": {game.game_id: stats},
                                                  "runs": [game.run.to_json()]}, indent=1, default=str),
                                      encoding="utf-8")
    trace.build_run(out)
    run = game.run.to_json()
    return {"game": game.game_id, "levels": run.get("levels_completed"), "of": run.get("number_of_levels"),
            "actions_per_level": run.get("actions_per_level"), "baseline": run.get("baseline_actions"),
            "replies_used": llm.i, "replies": len(mod.TURNS), "end": stats.get("end_reason"),
            "cell_errors": stats.get("cell_errors"), "pending": (out / "pending.md").exists()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("script")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    script = Path(a.script).resolve()
    out = Path(a.out) if a.out else REPO / "runs" / "authored" / script.stem
    print(json.dumps(play(script, out.resolve()), indent=1))


if __name__ == "__main__":
    main()

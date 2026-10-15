"""Expected prefix-cache hit rate of the current harness, with no model and no GPU.

    cd Sarbloh-Experimentation
    uv run --with tokenizers --with jinja2 python -m harness.prefix_cache --turns 150 --turns-per-level 17

The real ``AgentSession`` plays a real offline game (eval/real_games) against a scripted stand-in for the model: each
reply is filler thinking of a sampled length plus an ipython call that writes a function, sets the memory variables,
prints part of ``observe()`` and usually sends a few moves with ``await act``. Everything the harness does to the context
(stored replies, hidden code, tool outputs, observations and pictures, turn-yield notes, drains, trims, level-up resets,
curator calls) is the harness's own code. A level-up is forced every ``--turns-per-level`` turns (random moves never win
one), through the same path a real level-up takes (``_level_up_text``, then ``_reset_if_level_up``).

Every request is rendered with the model's own chat template (preserve_thinking on, the server default) and tokenized
with its tokenizer; each picture becomes the number of image tokens the Qwen vision processor gives it. The server's
radix cache is modelled as every earlier sequence of the game (its prompt plus the reply it generated, assuming the
model wrote exactly what the template renders back). A request reuses its longest common prefix with any of them:
- ``page``: cut to the 64-token page (``--page-size 64``);
- ``mamba``: on this hybrid model a prefix is reusable only where a linear-attention state was kept. With
  ``--mamba-radix-cache-strategy extra_buffer --mamba-track-interval 64`` that is the end of an earlier prompt and every
  64 tokens of an earlier reply. UNCONFIRMED: whether SGLang also keeps states inside a prefill (it would only help).

Not modelled: eviction (one game alone; the server pool was 1M tokens and 60 linear-attention states for 10 running
games), the model writing text the template renders differently (token boundaries, whitespace), and other games.
Writes ``prefix_cache.md`` and ``prefix_cache.jsonl`` (one row per request) in ``--out``.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import random
import re
import statistics
import struct
import sys
import threading
import time
from pathlib import Path
from typing import Any

PRIME_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PRIME_ROOT.parent
for _p in (PRIME_ROOT, REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402

from harness.agent import prompts  # noqa: E402
from harness.config import build_config  # noqa: E402
from harness.llm.client import Reply  # noqa: E402

MODEL_DIR = REPO_ROOT / "kaggle" / "datasets" / "models" / "cloud" / "qwen3.8-27b-mtp-nvfp4"
PAGE = 64
WORDS = ("the object block moves cell row column left right up down wall goal target player key door counter bar "
         "grid colour shape path search state frame level rule test move probe check because so if then maybe "
         "not yet this that green blue red yellow grey black orange region hole fill push slide stop changed same "
         "different count steps plan model function returns list BFS reach next first after before should could "
         "would think try again observe act result value position offset 3 5 7 12 24 x y r c").split()


# --- rendering: the model's chat template and tokenizer -------------------------------------------------------
class Renderer:
    def __init__(self, model_dir: Path) -> None:
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        from tokenizers import Tokenizer

        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        env.filters["tojson"] = lambda x, indent=None, **_: json.dumps(x, ensure_ascii=False, indent=indent)

        def _raise(msg: str) -> None:
            raise ValueError(msg)

        env.globals["raise_exception"] = _raise
        self.template = env.from_string((model_dir / "chat_template.jinja").read_text(encoding="utf-8"))
        self.tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.pad = self.tok.token_to_id("<|image_pad|>")

    def text(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None,
             kwargs: dict[str, Any] | None) -> tuple[str, list[bytes]]:
        """The prompt text as the server renders it, and the pictures in order. Tool-call arguments become dicts, as
        SGLang does before it applies the template."""
        msgs, pngs = [], []
        for m in messages:
            m = dict(m)
            if m.get("tool_calls"):
                calls = []
                for tc in m["tool_calls"]:
                    fn = dict(tc.get("function") or {})
                    if isinstance(fn.get("arguments"), str):
                        try:
                            fn["arguments"] = json.loads(fn["arguments"])
                        except ValueError:
                            pass
                    calls.append({**tc, "function": fn})
                m["tool_calls"] = calls
            if isinstance(m.get("content"), list):
                for p in m["content"]:
                    if p.get("type") == "image_url":
                        url = p["image_url"]["url"]
                        pngs.append(base64.b64decode(url.split(",", 1)[1]) if url.startswith("data:") else b"")
            msgs.append(m)
        kw = {"preserve_thinking": True, **(kwargs or {})}
        return self.template.render(messages=msgs, tools=tools, add_generation_prompt=True, **kw), pngs

    def ids(self, text: str, pngs: list[bytes]) -> np.ndarray:
        """Token ids; each image pad becomes the picture's image tokens, as ids unique to that picture (the server
        keys pictures by a hash of their bytes, so the same picture matches)."""
        out: list[int] = []
        k = 0
        for t in self.tok.encode(text, add_special_tokens=False).ids:
            if t == self.pad and k < len(pngs):
                h = hash(pngs[k]) & 0x3FFFFFFF
                out += [-(h + i) - 1 for i in range(image_tokens(pngs[k]))]
                k += 1
            else:
                out.append(t)
        return np.asarray(out, dtype=np.int64)

    def count(self, text: str) -> int:
        return len(self.tok.encode(text, add_special_tokens=False).ids)


def image_tokens(png: bytes) -> int:
    """Qwen3-VL: patch 16, merge 2, so one token per 32x32 pixels after rounding each side to a multiple of 32."""
    if len(png) < 24:
        return 0
    w, h = struct.unpack(">II", png[16:24])
    return max(1, round(h / 32)) * max(1, round(w / 32))


# --- the radix cache --------------------------------------------------------------------------------------------
def lcp(a: np.ndarray, b: np.ndarray) -> int:
    n = min(len(a), len(b))
    if n == 0:
        return 0
    diff = np.nonzero(a[:n] != b[:n])[0]
    return int(diff[0]) if len(diff) else n


class Cache:
    """Every earlier sequence (prompt + generated reply) of the game; hits by the two rules in the module docstring."""

    def __init__(self) -> None:
        self.seqs: list[tuple[np.ndarray, int]] = []   # (prompt + reply ids, prompt length)

    def hit(self, ids: np.ndarray) -> tuple[int, int, int]:
        """(page hit, mamba hit, index of the sequence matched) for a new prompt."""
        best_page, best_mamba, src = 0, 0, -1
        limit = len(ids) - 1   # at least one token is always computed
        for j, (seq, plen) in enumerate(self.seqs):
            n = min(lcp(ids, seq), limit)
            page = n // PAGE * PAGE
            if n >= plen:            # into the reply: a state every 64 tokens of it
                mamba = page
            elif n >= plen // PAGE * PAGE:
                mamba = plen // PAGE * PAGE
            else:
                mamba = 0            # inside an earlier prompt: no state there (strict rule)
            if page > best_page:
                best_page, src = page, j
            best_mamba = max(best_mamba, mamba)
        return best_page, best_mamba, src

    def add(self, prompt: np.ndarray, reply: np.ndarray) -> None:
        self.seqs.append((np.concatenate([prompt, reply]), len(prompt)))


# --- the stand-in model -----------------------------------------------------------------------------------------
class ScriptedLLM:
    """Plays the model: sampled thinking, then one ipython call. Records what every request costs."""

    def __init__(self, renderer: Renderer, rng: random.Random, a: argparse.Namespace) -> None:
        self.r, self.rng, self.a = renderer, rng, a
        self.cache = Cache()
        self.rows: list[dict[str, Any]] = []
        self.session: Any = None
        self.turn = 0
        self.events: list[str] = []      # rewrites since the last request (drain, trim, reset), read from the log
        self.prev = ""                   # the last prompt text plus its reply, to split what a request adds

    def _thinking(self) -> str:
        a = self.a
        n = int(min(a.max_thinking, max(40, self.rng.lognormvariate(math.log(a.thinking_median), a.thinking_sigma))))
        words = [self.rng.choice(WORDS) for _ in range(int(n / 1.25))]
        lines, k = [], 0
        while k < len(words):
            step = self.rng.randint(8, 30)
            lines.append(" ".join(words[k:k + step]).capitalize() + ".")
            k += step
        return "\n".join(lines)

    def _code(self) -> str:
        t, rng = self.turn, self.rng
        moves = [str(rng.randint(1, 4)) for _ in range(rng.randint(1, 6))]
        body = "\n".join(f"    # step {i}: check the cell ahead and the counter\n    out.append(moves[{i % len(moves)}])"
                         for i in range(rng.randint(2, 10)))
        code = [f"def plan_{t}(obs, moves={moves!r}):",
                f'    """Turn {t}: pick the next moves from the board."""',
                "    out = []", body, "    return out[:len(moves)]",
                f'GOAL = "reach the target with the player block (turn {t})"',
                f'PLAN = "probe the moves, then search a path (turn {t})"',
                "o = observe()", f"print(str(o)[:{rng.randint(200, self.a.print_chars)}])"]
        if rng.random() >= self.a.no_act_share:
            code += ["try:", f"    r = await act(plan_{t}(o))",
                     "    if 'GAME_OVER' in r.state:", '        await act(["reset"])',
                     "except RuntimeError as e:", "    print('refused:', e)", '    await act(["reset"])']
        return "\n".join(code)

    def _parts(self, text: str, pngs: list[bytes]) -> dict[str, int]:
        """What this request adds after the last one, in tokens, by kind: the code note that replaced the last
        reply's code, tool outputs, user messages (observations, notes) and their pictures, and template glue."""
        n = 0
        for x, y in zip(self.prev, text):
            if x != y:
                break
            n += 1
        delta = text[n:]
        out = {"code_note": 0, "tool_output": 0, "user_text": 0, "pictures": 0, "glue": 0}
        code_at = self.prev.rfind("<parameter=code>\n")
        if code_at < 0 or n < code_at:
            out["rewrite"] = self.r.count(delta)
            return out
        end = delta.find("\n</parameter>")
        out["code_note"] = self.r.count(delta[:max(end, 0)])
        rest = delta[max(end, 0):]
        for body in re.findall(r"<tool_response>\n(.*?)\n</tool_response>", rest, re.S):
            out["tool_output"] += self.r.count(body)
        pad = "<|vision_start|><|image_pad|><|vision_end|>"
        k_seen = rest.count(pad)
        for body in re.findall(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", rest, re.S):
            if "<tool_response>" not in body[:20]:
                out["user_text"] += self.r.count(body.replace(pad, ""))
        out["pictures"] = sum(image_tokens(p) for p in pngs[len(pngs) - k_seen:]) if k_seen else 0
        out["glue"] = max(0, self.r.count(rest.replace(pad, "")) - out["tool_output"] - out["user_text"])
        return out

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
             max_tokens: int | None = None, timeout_s: float | None = None, thinking: bool | None = None,
             template_kwargs: dict[str, Any] | None = None) -> Reply:
        text, pngs = self.r.text(messages, tools, template_kwargs)
        ids = self.r.ids(text, pngs)
        if self.a.prewarm_system and not self.rows and tools:   # one request of the system prompt alone, at the start
            head = text[:text.index("<|im_end|>\n") + len("<|im_end|>\n")]
            self.cache.add(self.r.ids(head, []), np.zeros(0, dtype=np.int64))
        page, mamba, src = self.cache.hit(ids)
        curator = not tools
        parts = {} if curator else self._parts(text, pngs)
        if curator:
            reasoning, content, calls = "Short review.", "{}", []
            gen = "Short review.\n</think>\n\n{}<|im_end|>"
        else:
            self.turn += 1
            reasoning, content = self._thinking(), ""
            code = self._code()
            calls = [{"id": f"call_{self.turn:05d}", "type": "function",
                      "function": {"name": "ipython", "arguments": json.dumps({"code": code})}}]
            gen = (f"{reasoning}\n</think>\n\n<tool_call>\n<function=ipython>\n<parameter=code>\n{code}\n"
                   f"</parameter>\n</function>\n</tool_call><|im_end|>")
            if self.turn % self.a.turns_per_level == 0 and self.session is not None and not self.a.no_level_reset:   # a level-up after this reply
                s = self.session
                self.session._level_up_text = prompts.event_message(
                    "level_up", levels_done=self.turn // self.a.turns_per_level, win_levels=7,
                    level=self.turn // self.a.turns_per_level + 1, moves_left=s.arc.budget_left, minutes_left=20)
        reply_ids = np.asarray(self.r.tok.encode(gen, add_special_tokens=False).ids, dtype=np.int64)
        self.cache.add(ids, reply_ids)
        self.rows.append({"i": len(self.rows), "turn": self.turn, "curator": curator, "prompt": len(ids),
                          "hit_page": page, "hit_mamba": mamba, "src": src, "reply": len(reply_ids),
                          "images": len(pngs), "image_tokens": int((ids < 0).sum()), "after": self.events,
                          "added": parts})
        if not curator:
            self.prev = text + gen
        self.events = []
        return Reply(content=content, reasoning=reasoning, tool_calls=calls, finish_reason="tool_calls" if calls
                     else "stop", prompt_tokens=len(ids), completion_tokens=len(reply_ids))


# --- the run ----------------------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--game", default="ls20")
    ap.add_argument("--turns", type=int, default=150)
    ap.add_argument("--turns-per-level", type=int, default=17)
    ap.add_argument("--thinking-median", type=float, default=700, help="thinking tokens per reply, median")
    ap.add_argument("--thinking-sigma", type=float, default=1.1, help="lognormal sigma (1.1: p90 about 4x the median)")
    ap.add_argument("--max-thinking", type=int, default=11000)
    ap.add_argument("--no-act-share", type=float, default=0.3, help="share of replies that make no move")
    ap.add_argument("--print-chars", type=int, default=2500, help="most characters of observe() a cell prints")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prewarm-system", action="store_true",
                    help="what if: the system prompt is sent alone once, so a linear-attention state sits at its end")
    ap.add_argument("--no-level-reset", action="store_true",
                    help="what if: a level-up only appends its message (no context reset)")
    ap.add_argument("--model-dir", default=str(MODEL_DIR))
    ap.add_argument("--env-dir", default=str(REPO_ROOT / "eval" / "real_games"))
    ap.add_argument("--out", default=str(REPO_ROOT / "runs" / "prefix_cache"))
    ap.add_argument("--set", action="append", default=[], help="agent config override, e.g. hide_code=false")
    a = ap.parse_args()

    from harness.agent.agent import AgentSession
    from harness.game.arc_host import ArcHost
    from harness.game.games import build_games, make_arcade

    out = Path(a.out) / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    cfg = build_config({"model": "qwen_flash_next", "max_actions_per_game": 100000})
    agent = cfg["agent"]
    agent["vision"] = True
    for kv in a.set:   # dotted keys reach nested dicts: perception.briefing=false, compaction.trigger_tokens=120000
        k, v = kv.split("=", 1)
        d, *path = [agent] + k.split(".")
        for key in path[:-1]:
            d = d[key]
        d[path[-1]] = json.loads(v)
    os.environ["ONLY_RESET_LEVELS"] = "true"
    arcade = make_arcade("offline", environments_dir=a.env_dir, recordings_dir=str(out / "recordings"))
    game = build_games(arcade, "offline", only=[a.game], record=False)[0]
    game.start()

    llm = ScriptedLLM(Renderer(Path(a.model_dir)), random.Random(a.seed), a)
    stop = threading.Event()
    host = ArcHost(game, cfg["max_actions_per_game"], should_stop=stop.is_set, tokens_spent=lambda: 0)
    task = prompts.task_message(game_id=game.game_id, win_levels=game.number_of_levels,
                                max_actions=cfg["max_actions_per_game"], minutes=27)
    agent["limits"] = {**agent["limits"], "max_turns": a.turns}
    session = AgentSession(cfg=agent, llm=llm, name=a.game, session_dir=out / "game", task=task, arc=host,
                           deadline=time.time() + 6 * 3600, stop_event=stop, memory_root=out / "memory", tracing=True)
    llm.session = session
    log_event = session._log_event

    def spy(obj: dict[str, Any]) -> None:   # rewrites, so each request knows what came before it
        if obj.get("event") in ("drain", "context_trim", "context_reset", "turn_yield"):
            llm.events.append(obj["event"] if obj["event"] != "context_reset" else f"reset:{obj.get('reason')}")
        log_event(obj)

    session._log_event = spy
    t0 = time.time()
    session.run()
    game.finish(None)

    system_text, _ = llm.r.text([{"role": "system", "content": session._system_prompt()},
                                 {"role": "user", "content": "x"}], session.tools, {"reasoning_effort": "xhigh"})
    report(llm, a, cfg, out, time.time() - t0, session, system_text)


def _added(rows: list[dict[str, Any]]) -> list[str]:
    """Mean tokens a steady request adds, by kind, for requests after a move and after no move."""
    out = []
    for label, sel in (("after a move", lambda r: r["added"].get("pictures")), ("after no move",
                                                                                lambda r: not r["added"].get("pictures"))):
        g = [r["added"] for r in rows if r.get("added") and "rewrite" not in r["added"] and sel(r)]
        if g:
            keys = ("code_note", "tool_output", "user_text", "pictures", "glue")
            mean = {k: statistics.mean(x.get(k, 0) for x in g) for k in keys}
            out.append(f"- new tokens {label} ({len(g)} requests), mean: " + ", ".join(f"{k} {v:.0f}" for k, v in
                                                                                    mean.items())
                       + f"; total {sum(mean.values()):.0f}")
    return out


def _share(rows: list[dict[str, Any]], key: str) -> float:
    p = sum(r["prompt"] for r in rows)
    return sum(r[key] for r in rows) / p if p else 0.0


def report(llm: ScriptedLLM, a: argparse.Namespace, cfg: dict[str, Any], out: Path, wall: float, session: Any,
           system_text: str) -> None:
    rows = llm.rows
    with (out / "prefix_cache.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    agent_rows = [r for r in rows if not r["curator"]]
    first = agent_rows[0]["prompt"] if agent_rows else 0
    sys_tokens = llm.r.count(system_text.split("<|im_start|>user")[0])

    def kind(r: dict[str, Any]) -> str:
        if r["curator"]:
            return "curator"
        if r["i"] == 0:
            return "first request"
        ev = r["after"]
        if any(e.startswith("reset") for e in ev):
            return "after level-up reset"
        if "drain" in ev or "context_trim" in ev:
            return "after drain/trim"
        return "steady (append-only)"

    groups: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(kind(r), []).append(r)
    total_p = sum(r["prompt"] for r in rows)
    steady = groups.get("steady (append-only)", [])
    new_steady = [r["prompt"] - r["hit_mamba"] for r in steady]
    st = session.stats
    lines = [
        "# Prefix cache, current harness (simulated)", "",
        f"game {a.game}, {len(agent_rows)} agent requests + {len(rows) - len(agent_rows)} curator, level-up every "
        f"{a.turns_per_level} turns, thinking median {a.thinking_median:.0f} tokens (sigma {a.thinking_sigma}), "
        f"{a.no_act_share:.0%} of replies without a move, seed {a.seed}; wall {wall:.0f} s", "",
        f"what-ifs: prewarm_system={a.prewarm_system} no_level_reset={a.no_level_reset} set={a.set}", "",
        f"hide_code={cfg['agent']['hide_code']} append_only={cfg['agent']['append_only']} "
        f"memory.in_context={cfg['agent']['memory']['in_context']} drains={st['drains']} trims={st['context_trims']} "
        f"resets={st['context_resets']} turn_yields={st['turn_yields']} code_stubs={st['code_stubs']}", "",
        "## Hit rate (cached prompt tokens / prompt tokens)", "",
        "| | page rule | mamba rule (expected on the server) |", "|---|---|---|",
        f"| all requests | {_share(rows, 'hit_page'):.3f} | **{_share(rows, 'hit_mamba'):.3f}** |",
        f"| agent requests only | {_share(agent_rows, 'hit_page'):.3f} | {_share(agent_rows, 'hit_mamba'):.3f} |",
        f"| steady requests only | {_share(steady, 'hit_page'):.3f} | {_share(steady, 'hit_mamba'):.3f} |", "",
        "## Where the missed tokens go (mamba rule)", "",
        "| request kind | requests | prompt tokens | missed tokens | share of all missed | hit rate |",
        "|---|---|---|---|---|---|"]
    missed_all = sum(r["prompt"] - r["hit_mamba"] for r in rows) or 1
    for k, g in sorted(groups.items(), key=lambda kv: -sum(r["prompt"] - r["hit_mamba"] for r in kv[1])):
        p = sum(r["prompt"] for r in g)
        m = sum(r["prompt"] - r["hit_mamba"] for r in g)
        lines.append(f"| {k} | {len(g)} | {p} | {m} | {m / missed_all:.2f} | {1 - m / p if p else 0:.3f} |")
    prompts_ = [r["prompt"] for r in agent_rows]
    lines += ["", "## Prompt size (tokens, real tokenizer)", "",
              f"- system prompt + tool schemas + reasoning-effort line: **{sys_tokens}**",
              f"- first request (system + task + first full view + picture): **{first}**",
              f"- one picture: {max((r['image_tokens'] // max(r['images'], 1) for r in rows), default=0)} tokens",
              f"- agent prompt: median {statistics.median(prompts_):.0f}, mean {statistics.mean(prompts_):.0f}, "
              f"max {max(prompts_)}" if prompts_ else "",
              f"- new tokens per steady request: median {statistics.median(new_steady):.0f}, "
              f"mean {statistics.mean(new_steady):.0f}" if new_steady else "",
              *_added(steady),
              f"- reply tokens: median {statistics.median([r['reply'] for r in agent_rows]):.0f}"
              if agent_rows else "", ""]
    (out / "prefix_cache.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"[prefix_cache] wrote {out}")


if __name__ == "__main__":
    main()

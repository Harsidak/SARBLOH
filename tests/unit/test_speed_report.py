# --- path shim: these suites live in tests/ but import from the repo root ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_PRIME = _os.environ.get("PRIME_DIR", "Sarbloh-Experimentation")   # speed.py is in the experimentation copy only
for _p in (_ROOT, _os.path.join(_ROOT, _PRIME)):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end shim ---
"""harness/speed.py on a hand-built run (one game, three turns, one level-up, an SGLang log):
- turn wall time runs from the last event before the reply; cells, acts, errors and level-ups attach by turn number;
- a cut-off reply, a reply with no tool call and a turn with a cell error are wasted; a turn with no act is idle;
- each turn is matched to its server line (same prompt length, output up to 3 tokens longer, entry inside the turn),
  giving cached tokens, prefill, decode and the wait outside the server; an unmatched line is another call;
- the cleared level is timed from the game start and counted inside the window; speed.md/json/turns are written.

Run: uv run python tests/unit/test_speed_report.py
"""
import json
import tempfile
from pathlib import Path

from harness import speed

FAILS, CHECKS = [], [0]


def check(name, ok, detail=""):
    CHECKS[0] += 1
    if not ok:
        FAILS.append(f"{name}: {detail}")


T = 1000.0
EVENTS = [
    {"t": T, "event": "system_prompt"},
    {"t": T + 0.5, "event": "message", "role": "user", "content": "task"},
    # turn 1: 20 s, one cell that acts twice and clears level 1
    {"t": T + 20.5, "event": "message", "role": "assistant", "content": "", "reasoning": "x" * 300,
     "tool_calls": [{"id": "a"}], "usage": [5000, 800], "finish": "tool_calls", "turn": 1},
    {"t": T + 21.0, "event": "act", "turn": 1, "done": 2},
    {"t": T + 21.0, "event": "level_up", "turn": 1, "level": 1},
    {"t": T + 21.5, "event": "cell", "turn": 1, "status": "ok", "duration_s": 1.0},
    {"t": T + 22.0, "event": "curator", "duration_s": 4.0},
    # turn 2: cut off at the cap, 60 s
    {"t": T + 82.0, "event": "message", "role": "assistant", "content": "", "reasoning": "y" * 900,
     "usage": [7000, 4096], "finish": "length", "turn": 2},
    # turn 3: a cell error, no actions, 10 s
    {"t": T + 92.0, "event": "message", "role": "assistant", "content": "", "tool_calls": [{"id": "b"}],
     "usage": [9000, 300], "finish": "tool_calls", "turn": 3},
    {"t": T + 93.0, "event": "cell", "turn": 3, "status": "error", "duration_s": 0.5},
    {"t": T + 93.0, "event": "end"},
]
LOG = "\n".join([
    "[x] ReqTimeStats(rid=a, input_len=5000, cached_input_len=0, output_len=801, attempts=0, type=unified): "
    f"queue_duration=500.00ms, initial_prefill_elapsed=1000.00ms, post_prefill_elapsed=12000.00ms, "
    f"forward_duration=13000.00ms, entry_time={T + 3}",
    "[x] ReqTimeStats(rid=b, input_len=7000, cached_input_len=5000, output_len=4096, attempts=0, type=unified): "
    f"queue_duration=0.00ms, initial_prefill_elapsed=500.00ms, post_prefill_elapsed=59000.00ms, "
    f"forward_duration=59500.00ms, entry_time={T + 22.2}",
    "[x] ReqTimeStats(rid=c, input_len=9000, cached_input_len=7000, output_len=302, attempts=0, type=unified): "
    f"queue_duration=0.00ms, initial_prefill_elapsed=200.00ms, post_prefill_elapsed=9000.00ms, "
    f"forward_duration=9200.00ms, entry_time={T + 82.3}",
    "[x] ReqTimeStats(rid=d, input_len=3000, cached_input_len=0, output_len=500, attempts=0, type=unified): "
    f"queue_duration=0.00ms, initial_prefill_elapsed=300.00ms, post_prefill_elapsed=3700.00ms, "
    f"forward_duration=4000.00ms, entry_time={T + 18}",
])

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    run = root / "prime_run"
    (run / "games" / "g1").mkdir(parents=True)
    (run / "games" / "g1" / "transcript.jsonl").write_text("\n".join(json.dumps(e) for e in EVENTS), encoding="utf-8")
    (root / "llm-server.log").write_text(LOG, encoding="utf-8")
    (run / "results.json").write_text(json.dumps({
        "config": {"max_tokens_per_turn": 4096},
        "runs": [{"game_id": "g1", "levels_completed": 1, "number_of_levels": 3, "baseline_actions": [4, 9, 9],
                  "actions_per_level": [2, 0, 0], "actions": 2}]}), encoding="utf-8")

    out = speed.build(root, window_min=1.0)   # the folder above prime_run is found too
    a = json.loads((run / "speed.json").read_text(encoding="utf-8"))
    turns = [json.loads(l) for l in (run / "speed_turns.jsonl").read_text(encoding="utf-8").splitlines()]
    h = a["headline"]
    check("report written", out == run / "speed.md" and "# Speed report" in out.read_text(encoding="utf-8"))
    check("three turns", len(turns) == 3, len(turns))
    t1, t2, t3 = turns
    check("turn 1 wall from the user message", t1["wall_s"] == 20.0, t1["wall_s"])
    check("turn 2 wall from the curator event", t2["wall_s"] == 60.0, t2["wall_s"])
    check("turn 1 acts, cell time, level-up", (t1["actions"], t1["cell_s"], t1["levels_up"]) == (2, 1.0, 1), t1)
    check("turn 2 is on level 1", t2["level"] == 1, t2["level"])
    check("turn 1 not wasted", not t1["wasted"] and not t1["idle"], t1)
    check("turn 2 cut off and wasted", t2["cut_off"] and t2["wasted"] and not t2["no_tool"], t2)
    check("turn 3 cell error wasted and idle", t3["cell_errors"] == 1 and t3["wasted"] and t3["idle"], t3)
    check("turn 1 matched (+1 stop token)", t1.get("decode_s") == 12.0 and t1.get("prefill_s") == 1.0, t1)
    check("turn 1 wait outside the server", t1.get("wait_s") == 6.5, t1.get("wait_s"))
    check("turn 2 cached tokens", t2.get("cached") == 5000, t2.get("cached"))
    check("turn 3 matched (+2)", t3.get("decode_s") == 9.0, t3)
    check("one other request", h["other_requests"] == 1 and h["other_requests_s"] == 4.0, h)
    check("reasoning share", t1["reasoning_share"] > 0.5 and t3["reasoning_share"] < 0.5, (t1, t3))
    check("level timed and in window", a["levels"][0]["cleared_at_min"] == 0.3 and h["levels_in_1_min"] == 1,
          a["levels"])
    check("level rhae", a["levels"][0]["rhae"] == 1.15 and a["levels"][0]["human"] == 4, a["levels"][0])
    check("shares", abs(h["share_wasted"] - 2 / 3) < 0.01 and h["share_cut_off"] == 0.333, h)
    check("cache hit", h["cache_hit"] == round(12000 / 21000, 3), h["cache_hit"])
    check("curator time", h["share_curator"] is not None and h["share_curator"] > 0, h["share_curator"])

    # a run without a server log still reports the turns
    (root / "llm-server.log").unlink()
    a2 = speed.analyze(root, 1.0)
    check("no log: turns, no server fields", len(a2["turns"]) == 3 and "decode_s" not in a2["turns"][0]
          and a2["headline"]["share_decode"] is None, a2["headline"])
    # --vs adds the comparison columns
    text = speed.render(a2, a)
    check("compare columns", "| other | change |" in text and "compared with" in text)

print(f"speed report: {CHECKS[0] - len(FAILS)}/{CHECKS[0]} checks passed")
for f in FAILS:
    print("FAIL", f)
raise SystemExit(1 if FAILS else 0)

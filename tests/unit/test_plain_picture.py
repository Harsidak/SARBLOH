# --- path shim: these suites live in tests/ but import from the repo root ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_PRIME = _os.environ.get("PRIME_DIR", "Sarbloh-Experimentation")
for _p in (_ROOT, _os.path.join(_ROOT, _PRIME)):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end shim ---
"""The plain board after the map: with config perception.plain_picture (and the briefing on) a picture message
carries two images, the map then the labelled plain frame, both saved under frames/; off, or when the briefing failed,
one image. The trace gives each image its own file. The system prompt says so only when it is on. Only the newest
message keeps its images (both). The server smoke test sets images_per_prompt and run.py turns the option off when the
server takes one image only."""
import inspect
import tempfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from harness import config, run, trace
from harness.agent import agent, context, perception
from harness.llm import server

FAILS, CHECKS = [], [0]


def ok(name, cond, detail=""):
    CHECKS[0] += 1
    print(("PASS " if cond else "FAIL ") + name + (f"  {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


rows = [[0] * 8 for _ in range(8)]
rows[3][3] = 8


def fake(plain, brief_ok=True):
    events = []
    return SimpleNamespace(
        _frame=lambda: rows, perception={"cell": 4, "plain_picture": plain},
        _briefing=lambda r: object() if brief_ok else None, stats=Counter(), _frames=0,
        session_dir=Path(tempfile.mkdtemp()), _log_event=events.append)


orig_picture = perception.picture
perception.picture = lambda b, cell: b"MAP"   # the briefing map, stubbed: its drawing has its own tests
try:
    s = fake(True)
    pngs, names = agent.AgentSession._picture(s)
    ok("two_pictures", pngs[0] == b"MAP" and pngs[1].startswith(b"\x89PNG") and len(pngs) == 2, str(pngs[:1]))
    ok("both_saved", names == ["frames/img_00001.png", "frames/img_00001_plain.png"]
       and all((s.session_dir / n).exists() for n in names) and s.stats["images_sent"] == 2, str(names))
    s = fake(False)
    pngs, names = agent.AgentSession._picture(s)
    ok("off_map_only", pngs == [b"MAP"] and names == ["frames/img_00001.png"])
    s = fake(True, brief_ok=False)
    pngs, names = agent.AgentSession._picture(s)
    ok("briefing_failed_plain_only", len(pngs) == 1 and pngs[0].startswith(b"\x89PNG"))
finally:
    perception.picture = orig_picture

msg = perception.image_message("state", [b"A", b"B"], 7)
kinds = [p["type"] for p in msg["content"]]
ok("message_map_label_plain", kinds == ["text", "image_url", "text", "image_url"]
   and msg["content"][2]["text"] == perception.PLAIN_LABEL, str(kinds))
ok("message_one_png_unchanged", [p["type"] for p in perception.image_message("s", b"A", 1)["content"]]
   == ["text", "image_url"])
agent.AgentSession._name_images(msg, ["frames/img_1.png", "frames/img_1_plain.png"])
stored = trace._stored(dict(msg), "run")
urls = [p["image_url"]["url"] for p in stored["content"] if p["type"] == "image_url"]
ok("trace_each_image_its_file", urls == ["run/frames/img_1.png", "run/frames/img_1_plain.png"], str(urls))
old = perception.image_message("old", [b"A", b"B"], 1)
kept = context.images([old, msg], vision=True)
ok("newest_keeps_both", sum(p["type"] == "image_url" for p in kept[1]["content"]) == 2
   and not any(p["type"] == "image_url" for p in kept[0]["content"]))

hist = [{"role": "user", "content": "task"}, {**old, "_kind": "picture"}, {"role": "assistant", "content": "x"},
        {"role": "tool", "content": "t"}, {"role": "assistant", "content": "y"}, {**msg, "_kind": "picture"}]
ok("drain_keeps_newest_pair", all(sum(p.get("type") == "image_url" for m in context.drop_turns(hist, cut)
                                      if isinstance(m["content"], list) for p in m["content"]) == 2 for cut in (1, 2, 4)))

cfg = config.build_config({"agent": {"vision": True}})["agent"]
ok("config_default_on", cfg["perception"]["plain_picture"] is True)
on = agent.system_prompt(cfg)
cfg_off = {**cfg, "perception": {**cfg["perception"], "plain_picture": False}}
ok("prompt_note_when_on", "A second picture comes right after the map" in on)
ok("prompt_no_note_when_off", "A second picture" not in agent.system_prompt(cfg_off))
cfg_nobrief = {**cfg, "perception": {**cfg["perception"], "briefing": False}}
ok("prompt_no_note_without_briefing", "A second picture" not in agent.system_prompt(cfg_nobrief))

ok("smoke_sets_images_per_prompt", "images_per_prompt = 2" in inspect.getsource(server.LLMServer.image_smoke)
   if hasattr(server, "LLMServer") else "images_per_prompt = 2" in inspect.getsource(server))
ok("run_turns_off_for_one_image", "images_per_prompt" in inspect.getsource(run) and '"plain_picture"] = False'
   in inspect.getsource(run))

print("================ SUMMARY ================")
print(f"CHECKS: {CHECKS[0]}")
print(f"HARD FAILS: {len(FAILS)} -> {FAILS}")
raise SystemExit(1 if FAILS else 0)

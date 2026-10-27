# --- path shim: these suites live in tests/ but import from the repo root ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_PRIME = _os.environ.get("PRIME_DIR", "Sarbloh-Experimentation")   # E110 exists only in the copy so far
for _p in (_ROOT, _os.path.join(_ROOT, _PRIME)):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end shim ---
"""E110 prompts (experiments/E110_fast_thinking_prompts): with `agent.prompt_version` "e008" the E008 text is
unchanged; with "e110" the time paragraph, the code-over-thinking rules, the HUD and solvable rules and the level-up
carry-over are in, placeholders filled, and the edits compose with E021 and E022. The agent counts length cut-offs
and sends the E110 cut-off and level-up texts only with the switch on."""
import re
import sys

from harness.agent import prompts
from harness.config import DEFAULT, build_config

FAILS, CHECKS = [], [0]


def ok(name, cond, detail=""):
    CHECKS[0] += 1
    print(("PASS " if cond else "FAIL ") + name + (f"  {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


KW = dict(game_id="ls20", win_levels=7, act_max=5, cwd="/x")
base = prompts.e008_system(**KW)
ok("default_is_e008", DEFAULT["agent"]["prompt_version"] == "e008"
   and build_config({})["agent"]["prompt_version"] == "e008")
ok("e008_explicit_identical", prompts.e008_system(**KW, prompt_version="e008") == base)
ok("e008_has_no_e110_text", "## Time" not in base and "Never read, count or compare" not in base)

e110 = prompts.e008_system(**KW, prompt_version="e110", max_tokens=16384)
ok("time_paragraph", "## Time" in e110 and "longer than 16384 tokens is cut" in e110, e110[:0])
ok("code_over_thinking", "Never read, count or compare cells in your thinking" in e110)
ok("bfs_batch_line", "send the path in one act of up to 5 actions" in e110)
ok("hud_line", "do not click its segments" in e110)
ok("solvable_line", "The game can be solved." in e110)
ok("level_up_step", "keep the mechanics you verified; do not test them again" in e110
   and "Read the lessons in your memory, find what is new, then continue." not in e110)
ok("old_ipython_rule_replaced", "Use ipython for anything you would count or compare by eye" not in e110)
ok("no_unfilled_placeholders", not re.search(r"\{(act_max|cut|game_id|win_levels)\}", e110))
ok("rest_unchanged", all(line in e110 for line in base.splitlines()
                         if line and not any(old.splitlines()[0] in line for old, _ in prompts.E110_EDITS)))
ok("cut_follows_config", "longer than 8192 tokens" in prompts.e008_system(**KW, prompt_version="e110", max_tokens=8192))

both = prompts.e008_system(**KW, prompt_version="e110", goal_versioning=True, goal_lock_after=1, level_review=True)
ok("composes_with_e021_e022", "## Time" in both and prompts.E021_LOCK_LINE.format(n=1) in both
   and prompts.E022_REVIEW_TEXT.strip() in both and all(new in both for _, new in prompts.E021_GOAL_LINES))
try:
    prompts.e008_system(**KW, prompt_version="e999")
    ok("unknown_version_raises", False)
except ValueError:
    ok("unknown_version_raises", True)

src = open(_os.path.join(_ROOT, _PRIME, "harness", "agent", "agent.py"), encoding="utf-8").read()
ok("agent_counts_cutoffs", 'self.stats["length_cutoffs"] += 1' in src and '"length_cutoffs": 0' in src)
ok("agent_e110_texts_gated", "prompts.E110_CUT if self.e008 and self.prompt_version == \"e110\"" in src
   and "prompts.E110_LEVEL_UP if self.prompt_version == \"e110\" else \"\"" in src)
ok("agent_passes_switch", "prompt_version=self.prompt_version" in src)
ok("texts_are_plain", "{" not in prompts.E110_CUT and "{" not in prompts.E110_LEVEL_UP)

print("\n================ SUMMARY ================")
print(f"CHECKS: {CHECKS[0]}")
print(f"HARD FAILS: {len(FAILS)} -> {FAILS}")
sys.exit(1 if FAILS else 0)

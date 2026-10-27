# --- path shim: these suites live in tests/ but import from the repo root ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_PRIME = _os.environ.get("PRIME_DIR", "Sarbloh")   # PRIME_DIR=Sarbloh-Experimentation tests the copy
for _p in (_ROOT, _os.path.join(_ROOT, _PRIME), _os.path.join(_ROOT, _PRIME, "harness", "runtime", "skills")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end shim ---
"""Pins the Prime Agent port (`Sarbloh/harness/`) to upstream (commit 2d24ad4). See
experiments/E005_prime_fidelity/audit.md.

WHY THIS FILE EXISTS. E003/E004 were read as model failures while the port itself was not faithful: prompts
labelled "verbatim" had lines missing, and compaction kept only the task message. Text that claims to be upstream
is compared with the upstream clone (skipped with a warning when `resources/repos/prime-agent` is absent), and the
compaction cut-point rules are checked on hand-built conversations.
Sarbloh-Experimentation (E118) dropped the upstream one-tool REPL arm: no `refine.py` (digest, auto-refine), no `arc`
REPL module and no base prompt, so those checks are skipped there; compaction, wire format and host checks still run.
"""
import json
import re
import sys
from pathlib import Path

from harness.agent import compaction as C
from harness.agent import agent as S
try:
    from harness.agent import refine as R
except ImportError:   # E118 copy: no Continual Harness digest or auto-refine
    R = None

FAILS, WARNS, CHECKS = [], [], [0]
# Sarbloh keeps upstream's summary prompts verbatim; the experimentation copy writes its own game note.
UPSTREAM_SUMMARY = hasattr(C, "UPDATE_SUMMARIZATION_PROMPT")
UP = Path(_ROOT) / "resources" / "repos" / "prime-agent" / "packages" / "coding-agent" / "src" / "core"


def ok(name, cond, detail=""):
    CHECKS[0] += 1
    print(("PASS " if cond else "FAIL ") + name + (f"  {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


def ts_const(path, name):
    """A TypeScript template-literal constant, with \\` unescaped."""
    src = path.read_text(encoding="utf-8")
    m = re.search(r"const " + name + r" = `((?:[^`\\]|\\.)*)`;", src, re.S)
    return m.group(1).replace("\\`", "`") if m else None


# --- 1. verbatim upstream text ---------------------------------------------------------------------------
if UP.exists():
    pairs = [
        *([(UP / "refinement" / "refinement.ts", "AUTO_REFINE_REVIEW_SYSTEM_PROMPT",
            R.AUTO_REFINE_REVIEW_SYSTEM_PROMPT)] if R else []),
        *([(UP / "compaction" / "compaction.ts", "SUMMARIZATION_PROMPT", C.SUMMARIZATION_PROMPT),
           (UP / "compaction" / "compaction.ts", "UPDATE_SUMMARIZATION_PROMPT", C.UPDATE_SUMMARIZATION_PROMPT),
           (UP / "compaction" / "compaction.ts", "TURN_PREFIX_SUMMARIZATION_PROMPT",
            C.TURN_PREFIX_SUMMARIZATION_PROMPT),
           (UP / "compaction" / "utils.ts", "SUMMARIZATION_SYSTEM_PROMPT", C.SUMMARIZATION_SYSTEM_PROMPT)]
          if UPSTREAM_SUMMARY else []),
        (UP / "messages.ts", "COMPACTION_SUMMARY_PREFIX", C.COMPACTION_SUMMARY_PREFIX),
        *([(UP / "messages.ts", "HARNESS_DIGEST_PREFIX", R.HARNESS_DIGEST_PREFIX)] if R else []),
    ]
    for path, name, ours in pairs:
        up = ts_const(path, name)
        ok(f"verbatim_{name}", up is not None and up == ours, "differs from upstream")
    # Base system prompt: the owner rewrote it in plain English on 2026-10-02 (sub-agent text removed the same day), so
    # only the upstream LONG_RUNNING_WORK lines are still pinned verbatim.
    from harness.agent import prompts as P
    ts_all = "\n".join(p.read_text(encoding="utf-8") for p in UP.parent.rglob("*.ts"))
    ts_all = re.sub(r"""(["'`])\s*\+\s*\n\s*(["'`])""", "", ts_all).replace("\\`", "`").replace('\\"', '"')
    if hasattr(P, "base_prompt"):
        text = P.base_prompt(cwd="/x", transcript="/t")
        drift = [ln[:80] for ln in P.LONG_RUNNING_WORK.splitlines() if ln not in ts_all]
        ok("long_running_work_verbatim", not drift and P.LONG_RUNNING_WORK in text, str(drift))
        ok("base_prompt_no_subagent_text", not any(w in text for w in ("rlm.spawn", "agent_message", "child agent",
                                                                           "Delegating to sub-agents", "agent depth")))
    if R:
        # REFINEMENT_SYSTEM_PROMPT is upstream minus the subagent component (subagents removed 2026-10-03).
        refine_text = R.REFINEMENT_SYSTEM_PROMPT + " ".join(R._DIGEST_HEADER)
        ok("refine_prompt_no_subagent_text", not any(w in refine_text for w in ("subagent", "rlm.spawn",
                                                                                 "agent_message")))
    comp_src = (UP / "compaction" / "compaction.ts").read_text(encoding="utf-8")
    if UPSTREAM_SUMMARY:
        ok("verbatim_KERNEL_PERSIST_SUMMARY_NOTE", C.KERNEL_PERSIST_SUMMARY_NOTE in comp_src)
    else:   # the experimentation copy: a short game note that replaces the old one; kernel names listed in code
        ok("game_summary_prompt", "handover note" in C.SUMMARIZATION_PROMPT and "{words}" in C.SUMMARIZATION_PROMPT
           and "PRESERVE" not in C.PREVIOUS_SUMMARY_RULE)
    ok("upstream_compaction_defaults", "reserveTokens: 16384" in comp_src and "keepRecentTokens: 20000" in comp_src)
else:
    WARNS.append("upstream_clone_missing_verbatim_checks_skipped")


# --- 2. compaction cut point -----------------------------------------------------------------------------
def asst(i):
    return {"role": "assistant", "content": f"step {i}", "tool_calls": [
        {"id": f"c{i}", "type": "function", "function": {"name": "ipython", "arguments": json.dumps({"code": f"x={i}"})}}]}


def tool(i):
    return {"role": "tool", "tool_call_id": f"c{i}", "content": "y" * 800}


msgs = [{"role": "user", "_kind": C.DIGEST_KIND, "content": "digest"}, {"role": "user", "content": "TASK"}]
for i in range(30):
    msgs += [asst(i), tool(i)]
cut, turn_start, split = C.find_cut_point(msgs, 0, 2000)
ok("cut_never_at_tool_result", msgs[cut]["role"] != "tool")
ok("long_autonomous_turn_is_split", split and turn_start == 1)
kept_tokens = sum(C.estimate_tokens(m) for m in msgs[cut:])
biggest = max(C.estimate_tokens(m) for m in msgs)
# Upstream cuts at the first user/assistant message at or after the point where the walk reaches the budget, so up
# to one message less than the budget is kept.
ok("keeps_about_keep_recent_tokens", 2000 - biggest <= kept_tokens <= 2000 + biggest, str(kept_tokens))
prep = C.prepare(msgs, 2000, None, 10_000)
ok("digest_never_summarized", all(m.get("_kind") != C.DIGEST_KIND for m in prep.to_summarize + prep.turn_prefix))
ok("nothing_to_summarize_returns_none", C.prepare(msgs[:4], 10**6, None, 100) is None)
head = C.head_message("S", "")
prep2 = C.prepare([head, *msgs[cut:]], 500, "S", 10_000)
ok("old_head_passed_as_previous_summary",
   prep2 is not None and all(m.get("_kind") != C.COMPACTION_KIND for m in prep2.to_summarize + prep2.turn_prefix))
ok("tool_result_truncated_for_summarizer", len(C.truncate_for_summary("z" * 10_000, 2000)) <= 2000)
# trigger_tokens lowers upstream's `window - reserve` threshold, never raises it
ok("trigger_tokens_lowers_threshold", C.should_compact(41_000, 131_072, 16_384, 40_000)
   and not C.should_compact(41_000, 131_072, 16_384) and not C.should_compact(39_000, 131_072, 16_384, 40_000))
ok("trigger_tokens_never_raises_threshold", C.should_compact(12_000, 16_384, 4_608, 40_000))
from harness import run as RUN  # noqa: E402
from harness.config import DEFAULT  # noqa: E402
for w in (131_072, 65_536, 32_768):
    agent = json.loads(json.dumps(DEFAULT["agent"]))
    try:
        RUN.fit_context(agent, w)
        ok(f"fit_context_{w}", agent["compaction"]["keep_recent_tokens"] * 2 <= agent["compaction"]["trigger_tokens"])
    except ValueError as e:
        ok(f"fit_context_{w}", False, str(e))

# --- 3. wire format --------------------------------------------------------------------------------------
wire = S._wire(msgs[:5] + [{"role": "tool", "tool_call_id": "c1", "content": "e", "_error": True}])
ok("private_keys_never_sent", all(not k.startswith("_") for m in wire for k in m))
ok("consecutive_user_messages_merged", wire[0]["role"] == "user" and "TASK" in wire[0]["content"]
   and wire[1]["role"] == "assistant")
from harness.agent import context as CTX  # noqa: E402
if hasattr(CTX, "reasoning"):
    # Thinking goes back with the tool calls of the current chain only (Gemma's template rule), never as `_reasoning`.
    chain = [{"role": "user", "content": "obs"}, {**asst(0), "_reasoning": "old"}, tool(0),
             {"role": "user", "content": "obs2"}, {**asst(1), "_reasoning": "new"}, tool(1)]
    sent = CTX.build("S", chain)
    replies = [m for m in sent if m["role"] == "assistant"]
    ok("reasoning_sent_in_current_chain", replies[1].get("reasoning_content") == "new")
    ok("reasoning_dropped_after_user_message", "reasoning_content" not in replies[0] and "reasoning" not in replies[0])
    ok("reasoning_private_key_never_sent", all("_reasoning" not in m for m in sent))
    ok("reasoning_logged_once", "_reasoning" not in CTX.loggable(chain[1]))

# --- 4. transitions are dicts that explain themselves (the `arc` REPL module; not in the E118 copy) ----------
try:
    import arc  # noqa: E402
except ImportError:
    arc = None
if arc is not None:
    t = arc.Transition({"action": 1, "after": [[1]]})
    ok("transition_attribute_access", t.after == [[1]])
    try:
        t[0]
        ok("transition_tuple_index_explained", False)
    except KeyError as e:
        ok("transition_tuple_index_explained", "t['before']" in str(e))

# --- 5. RESET is only arc.reset() (E005 smoke: `for a in range(5): step(a)` spent 100/150 actions on resets) ----
try:
    from harness.game.arc_host import ArcHost
    from harness.game.games import build_games, make_arcade
    game = build_games(make_arcade("offline", environments_dir=str(Path(_ROOT) / "eval" / "real_games")), "offline",
                       only=["ls20"])[0]
    game.start()
    host = ArcHost(game, 10, should_stop=lambda: False, tokens_spent=lambda: 0)
    ok("available_actions_hide_reset", 0 not in host.handle({"type": "arc.observe"})["available_actions"])
    try:
        host.handle({"type": "arc.step", "action": 0})
        ok("step_0_refused", False)
    except ValueError as e:
        ok("step_0_refused", ("arc.reset()" in str(e) or '"reset"' in str(e)) and game.action_count == 0)
    try:
        host.handle({"type": "arc.step", "action": 6, "x": 1, "y": 1})
        ok("illegal_action_error_hides_reset", False)
    except ValueError as e:
        ok("illegal_action_error_hides_reset", "legal: [1," in str(e), str(e))
    try:
        host.handle({"type": "arc.reset"})
        ok("reset_at_level_start_refused", False)
    except ValueError as e:
        ok("reset_at_level_start_refused", "already at its start" in str(e) and game.action_count == 0)
    host.handle({"type": "arc.step", "action": 1})
    host.handle({"type": "arc.reset"})
    ok("reset_request_spends_one_action", game.action_count == 2 and host.transitions[-1]["action"] == 0)
    try:
        host.handle({"type": "arc.reset"})
        ok("second_reset_in_a_row_refused", False)
    except ValueError:
        ok("second_reset_in_a_row_refused", game.action_count == 2)
    game.finish()
except Exception as e:  # noqa: BLE001
    WARNS.append(f"arc_host_reset_checks_skipped: {type(e).__name__}: {e}")
if arc is not None:
    try:
        arc.step(0).send(None)
        ok("kernel_step_0_refused", False)
    except arc.ArcError as e:
        ok("kernel_step_0_refused", "arc.reset()" in str(e))
    ok("diff_accepts_observations", arc.diff(type("O", (), {"grid": [[1, 2]]})(), [[1, 3]]) == [(1, 0, 2, 3)])

print("\n================ SUMMARY ================")
print(f"CHECKS: {CHECKS[0]}")
print(f"HARD FAILS: {len(FAILS)} -> {FAILS}")
print(f"WARNINGS  : {len(WARNS)} -> {WARNS}")
sys.exit(1 if FAILS else 0)

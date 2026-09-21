# --- path shim: these suites live in tests/ but import the agent from the repo root ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, 'eval')):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end shim ---
"""test_llm_repair.py -- the LLM output contract and the bounded repair loop.

The property under test, stated once so every case can be read against it:

    NOTHING the LLM emits is trusted until it has (a) parsed under a strict
    reader, (b) compiled under a restricted namespace with no import and no
    dunder access, and (c) passed CandidateVerifier. When any of those three
    refuses, the EXACT reason is fed back to the model and it gets a bounded
    number of further attempts -- never unbounded, never zero.

Why this file exists. The old path was `_extract_json` -> `find("{") ... rfind("}")`
-> `compile_python` -> silent `return None`. Three distinct kinds of garbage were
swallowed without a word:

  * JSON whose "code" string carries LITERAL newlines. This is not an edge case,
    it is what an instruct model does most of the time when you ask it for a
    multi-line function inside a JSON field, and `json.loads` rejects it
    ("Invalid control character"). Every such proposal was thrown away.
  * A bare ```python block with no JSON at all -- a correct answer to a coding
    prompt, discarded because the reader only knew how to find braces.
  * A model that got one detail wrong got no second chance, because the error
    was printed to a log nobody parses and never shown to the thing that could
    fix it.

The loop is UNSCORED -- it takes no environment action -- so under RHAE, which
squares the action ratio, it is free. That is the whole reason it is a repair
loop and not a ReAct loop.

Run:  PYTHONIOENCODING=utf-8 python test_llm_repair.py
Never delete a case.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ARC_NO_LLM", "1")

from my_agent import (                                    # noqa: E402
    extract_candidate, compile_python_checked, compile_python,
    WorldModel, WorldModelManager, Transition, StateEncoder,
    LLMProposalStats, LLM_STATS, LLM_REPAIR_ROUNDS, VERIFY_PROMOTE_THRESHOLD,
)

FAILS = []
CHECKS = [0]


def check(name, got, want):
    CHECKS[0] += 1
    if got != want:
        FAILS.append(f"{name}: got {got!r}, want {want!r}")
        print(f"  FAIL {name}: got {got!r}, want {want!r}")
    else:
        print(f"  ok   {name}")


def truthy(name, got):
    check(name, bool(got), True)


def section(t):
    print(f"\n--- {t} ---")


GOOD_CODE = ("def transition(state_grid, action_str):\n"
             "    g = state_grid.copy()\n"
             "    g[0, 0] = 1\n"
             "    return g\n")


# ============================================================
section("1. the shapes a cooperative model emits")
# ============================================================

wm, why = extract_candidate(
    '{"hypothesis": "gravity", "program": '
    '[{"action": "ACTION1", "op": "gravity", "args": {"direction": "down"}}]}')
check("clean_json_program_parses", isinstance(wm, WorldModel), True)
check("clean_json_program_source", wm.source if wm else None, "llm-dsl")
check("clean_json_program_no_complaint", why, "")

wm, why = extract_candidate(
    '{"hypothesis": "h", "code": ' + repr(GOOD_CODE).replace("'", '"') + '}')
check("clean_json_code_parses", isinstance(wm, WorldModel), True)
check("clean_json_code_source", wm.source if wm else None, "llm-code")

wm, why = extract_candidate(
    "Sure! Here is my answer.\n\n```json\n"
    '{"hypothesis": "h", "program": [{"action": "*", "op": "identity", "args": {}}]}'
    "\n```\nHope that helps!")
check("fenced_json_with_prose_parses", isinstance(wm, WorldModel), True)
check("fenced_json_with_prose_source", wm.source if wm else None, "llm-dsl")


# ============================================================
section("2. the shapes that used to be thrown away")
# ============================================================
# THE case. Literal newlines inside a JSON string are invalid JSON, and it is
# what the model emits when asked for a function in a "code" field. The reader
# must recover the code rather than lose the whole proposal.
raw_bad_json = ('{"hypothesis": "copy the grid",\n'
                ' "code": "def transition(state_grid, action_str):\n'
                '    return state_grid.copy()"}')
wm, why = extract_candidate(raw_bad_json)
check("literal_newlines_in_code_field_recovered", isinstance(wm, WorldModel), True)
check("literal_newlines_source", wm.source if wm else None, "llm-code")
truthy("literal_newlines_kept_the_body",
       wm is not None and "state_grid.copy()" in wm.spec)
# Proof this is not a test of nothing: the OLD reader really does lose it.
import json as _json                                       # noqa: E402
_old_lost = False
try:
    _json.loads(raw_bad_json)
except Exception:
    s, e = raw_bad_json.find("{"), raw_bad_json.rfind("}")
    try:
        _json.loads(raw_bad_json[s:e + 1])
    except Exception:
        _old_lost = True
check("old_reader_really_lost_that_one", _old_lost, True)

# A bare python block is a correct answer to a coding prompt.
wm, why = extract_candidate("Here you go:\n\n```python\n" + GOOD_CODE + "```\n")
check("bare_python_fence_accepted", isinstance(wm, WorldModel), True)
check("bare_python_fence_source", wm.source if wm else None, "llm-code")

# No fence, no JSON, just the function.
wm, why = extract_candidate("I think the rule is simple.\n\n" + GOOD_CODE)
check("unfenced_function_accepted", isinstance(wm, WorldModel), True)

# find("{")/rfind("}") spans two unrelated objects and yields nonsense; a
# balanced scan must pick the real one.
wm, why = extract_candidate(
    'Consider {a, b} first. Then:\n'
    '{"hypothesis": "h", "program": [{"action": "*", "op": "identity", "args": {}}]}\n'
    'and note {c, d}.')
check("brace_scanner_beats_first_to_last", isinstance(wm, WorldModel), True)

# Two fenced blocks: pick the one that actually defines transition.
wm, why = extract_candidate(
    "```python\nx = 1\n```\nand the real one:\n```python\n" + GOOD_CODE + "```")
truthy("picks_the_block_defining_transition",
       wm is not None and "def transition" in wm.spec)


# ============================================================
section("3. garbage is refused, with a reason")
# ============================================================

for name, text in [
    ("empty", ""),
    ("pure_prose", "I am not sure what the rule is, sorry."),
    ("json_without_either_key", '{"hypothesis": "h"}'),
    ("program_not_a_list", '{"hypothesis": "h", "program": "gravity down"}'),
    ("program_empty", '{"hypothesis": "h", "program": []}'),
    ("syntax_error", "```python\ndef transition(g, a)\n    return g\n```"),
    ("wrong_function_name", "```python\ndef step(g, a):\n    return g\n```"),
]:
    wm, why = extract_candidate(text)
    check(f"refused_{name}", wm, None)
    truthy(f"refused_{name}_has_a_reason", isinstance(why, str) and len(why) > 0)

# The reason has to be USABLE by the model, not just non-empty: it must name
# the thing that went wrong. A reason that says nothing repairs nothing.
_, why = extract_candidate("```python\ndef transition(g, a)\n    return g\n```")
truthy("syntax_reason_says_syntax", "syntax" in why.lower())
_, why = extract_candidate("```python\ndef step(g, a):\n    return g\n```")
truthy("naming_reason_says_transition", "transition" in why.lower())


# ============================================================
section("4. hardening: no imports, no dunders, no exec at parse time")
# ============================================================
# compile_python restricts __builtins__, so `open(...)` fails at CALL time --
# after the candidate has been accepted and handed to the verifier. Refusing at
# PARSE time means untrusted code with an escape hatch never reaches exec at all.

for name, body in [
    ("import", "import os\ndef transition(g, a):\n    return g\n"),
    ("from_import", "from os import system\ndef transition(g, a):\n    return g\n"),
    ("dunder_attr", "def transition(g, a):\n    return g.__class__.__bases__\n"),
    ("dunder_name", "def transition(g, a):\n    return __import__('os')\n"),
]:
    wm, why = extract_candidate("```python\n" + body + "```")
    check(f"refused_unsafe_{name}", wm, None)
    truthy(f"refused_unsafe_{name}_has_a_reason", bool(why))

# ...and the same refusal holds at the compile boundary, so a candidate that
# reached compile another way is still stopped.
fn, err = compile_python_checked("import os\ndef transition(g, a):\n    return g\n")
check("compile_refuses_import", fn, None)
truthy("compile_refuses_import_reason", bool(err))

fn, err = compile_python_checked(GOOD_CODE)
truthy("compile_accepts_good_code", callable(fn))
check("compile_good_code_no_error", err, "")
out = fn(np.zeros((3, 3), dtype=int), "ACTION1")
check("compiled_code_runs", int(out[0, 0]), 1)

fn, err = compile_python_checked("def transition(g, a)\n    return g\n")
check("compile_refuses_syntax_error", fn, None)
truthy("compile_syntax_error_reason", "syntax" in err.lower())

# The old single-return entry point must keep working for existing callers.
check("legacy_compile_python_still_returns_none", compile_python("nonsense ("), None)
truthy("legacy_compile_python_still_compiles", callable(compile_python(GOOD_CODE)))


# ============================================================
section("5. the bounded repair loop")
# ============================================================


class FakeLLM:
    """Scripted generator. Records every prompt so we can assert the feedback
    actually reached the model."""

    def __init__(self, replies):
        self.replies, self.prompts = list(replies), []

    def generate(self, prompt, max_new_tokens=None, temperature=0.2):
        self.prompts.append(prompt)
        return self.replies.pop(0) if self.replies else ""

    # Borrowed from LocalLLM below, unbound, so the code under test is the
    # SHIPPING proposal path and prompt -- not a copy that could drift from it.
    propose_world_model = None
    _synth_prompt = None


from my_agent import LocalLLM                              # noqa: E402
FakeLLM.propose_world_model = LocalLLM.propose_world_model
FakeLLM._synth_prompt = LocalLLM._synth_prompt

# The fixture world: ACTION1 shifts the scene down one row, ACTION2 shifts it
# right one column. Chosen so that RIGHT and WRONG are separable by the
# verifier -- an all-identity world would accept the identity rule AND leave
# nothing for a wrong rule to get wrong, which tests neither arm.
RIGHT_JSON = ('{"hypothesis": "each action shifts the scene", "program": ['
              '{"action": "ACTION1", "op": "translate", "args": {"dr": 1, "dc": 0}},'
              '{"action": "ACTION2", "op": "translate", "args": {"dr": 0, "dc": 1}}]}')
WRONG_JSON = ('{"hypothesis": "nothing ever moves", "program": '
              '[{"action": "*", "op": "identity", "args": {}}]}')


def shift_log(n=24):
    from my_agent import GridDSL
    ts, g = [], np.zeros((10, 10), dtype=int)
    g[2, 3], g[4, 4], g[5, 2] = 5, 5, 3       # sparse scene on a 0 background
    for i in range(n):
        action = f"ACTION{1 + i % 2}"
        nxt = (GridDSL.translate(g, 0, dr=1, dc=0) if action == "ACTION1"
               else GridDSL.translate(g, 0, dr=0, dc=1))
        if not nxt.any():                     # scene walked off the board
            nxt = np.zeros((10, 10), dtype=int)
            nxt[2, 3], nxt[4, 4], nxt[5, 2] = 5, 5, 3
        ts.append(Transition(prev=g.copy(), action=action, next=nxt.copy(),
                             reward=0.0, diff_encoding="", timestamp=float(i)))
        g = nxt
    return ts


LOG = shift_log()


def run_loop(replies):
    wmm = WorldModelManager(llm=FakeLLM(replies), encoder=StateEncoder())
    wmm.synthesize_with_llm("obs", lambda: LOG)
    return wmm, wmm.llm


# The fixture is only worth anything if the two arms really differ. Assert it
# directly, so a silently-degenerate world cannot make section 5 pass by
# accident (see the clamping-toy trap in memory/verifier-last5-gate).
_probe = WorldModelManager(llm=FakeLLM([]), encoder=StateEncoder())
_right_fn = _probe._compile(WorldModel("h", "llm-dsl", _json.loads(RIGHT_JSON)["program"]))
_wrong_fn = _probe._compile(WorldModel("h", "llm-dsl", _json.loads(WRONG_JSON)["program"]))
truthy("fixture_right_model_verifies",
       _probe.verify(_right_fn, LOG) >= VERIFY_PROMOTE_THRESHOLD)
truthy("fixture_wrong_model_does_not",
       _probe.verify(_wrong_fn, LOG) < VERIFY_PROMOTE_THRESHOLD)


# 5a. Garbage first, good second: the second attempt must happen.
wmm, llm = run_loop(["total nonsense, no code here", RIGHT_JSON])
check("repair_retries_after_garbage", len(llm.prompts), 2)
truthy("repair_can_promote_on_round_two", wmm.active_model is not None)

# 5b. The feedback must be IN the retry prompt. A loop that re-asks the same
# question is not a repair loop, it is a resample.
truthy("retry_prompt_differs_from_first", llm.prompts[1] != llm.prompts[0])
truthy("retry_prompt_carries_the_reason",
       any(w in llm.prompts[1].lower() for w in ("previous", "invalid", "error",
                                                 "rejected", "attempt")))

# 5c. Always-garbage terminates, and terminates at the declared bound.
wmm, llm = run_loop(["garbage"] * 10)
check("repair_is_bounded", len(llm.prompts), LLM_REPAIR_ROUNDS + 1)
check("repair_promotes_nothing_on_garbage", wmm.active_model, None)

# 5d. An empty generation means the budget is gone. Burning the remaining
# rounds on a model that cannot answer is the one failure the ledger cannot
# catch, because generate() returns "" for free and the loop would spin.
wmm, llm = run_loop(["", RIGHT_JSON])
check("empty_generation_stops_immediately", len(llm.prompts), 1)

# 5e. First try correct: no repair round is spent.
wmm, llm = run_loop([RIGHT_JSON, RIGHT_JSON])
check("no_repair_when_first_attempt_works", len(llm.prompts), 1)
truthy("first_attempt_promoted", wmm.active_model is not None)

# 5f. A candidate that parses and compiles but FAILS the verifier is still a
# repair opportunity -- that is the round trip the trust boundary exists for.
wmm, llm = run_loop([WRONG_JSON, RIGHT_JSON])
check("verifier_rejection_triggers_repair", len(llm.prompts), 2)
truthy("recovered_after_verifier_rejection", wmm.active_model is not None)
truthy("verifier_reason_reached_the_model",
       any(c.isdigit() for c in llm.prompts[1]))   # the accuracy is quoted back

# 5g. The trust boundary is NOT bypassed by the loop: a wrong model alone is
# never promoted, however many rounds it gets.
wmm, llm = run_loop([WRONG_JSON] * 10)
check("wrong_model_never_promoted", wmm.active_model, None)


# ============================================================
section("6. the accept rate is measured, not assumed")
# ============================================================
# "Is the LLM producing code that works?" is an empirical question, and on
# Kaggle the log is the only instrument. If nothing counts, nobody knows.

st = LLMProposalStats()
st.attempt()
st.attempt()
st.accepted_on(1)
st.rejected("syntax error")
check("stats_counts_attempts", st.attempts, 2)
check("stats_counts_accepts", st.accepted, 1)
check("stats_rate_is_a_fraction", round(st.rate(), 3), 0.5)
truthy("stats_report_is_printable", isinstance(st.report(), str) and st.report())
truthy("stats_report_names_the_failure", "syntax" in st.report().lower())
check("stats_rate_is_zero_when_unused", LLMProposalStats().rate(), 0.0)
truthy("stats_report_survives_zero_attempts",
       isinstance(LLMProposalStats().report(), str))
truthy("module_exposes_a_shared_counter", isinstance(LLM_STATS, LLMProposalStats))

# The shipping loop must actually feed it.
before = LLM_STATS.attempts
run_loop([RIGHT_JSON])
truthy("shipping_loop_records_attempts", LLM_STATS.attempts > before)


# ============================================================
section("7. the watchdog must outlast the loop it is watching")
# ============================================================
# SYNTH_MAX_INFLIGHT_S was a flat 540s, sized when one synthesis meant ONE
# generate call. The repair loop makes a HEALTHY synthesis up to
# LLM_REPAIR_ROUNDS + 1 generations, so a fixed bound would orphan a good final
# round -- on Kaggle, silently, and nowhere else. The constant is now derived;
# this case is what keeps someone raising LLM_REPAIR_ROUNDS from undoing that.

from my_agent import (                                     # noqa: E402
    SYNTH_MAX_INFLIGHT_S, LLM_GEN_MAX_TIME_S, LLM_SESSION_BUDGET_S,
)

worst_case = (LLM_REPAIR_ROUNDS + 1) * LLM_GEN_MAX_TIME_S
truthy("watchdog_outlasts_a_full_repair_loop", SYNTH_MAX_INFLIGHT_S > worst_case)
truthy("watchdog_leaves_room_for_the_generate_lock",
       SYNTH_MAX_INFLIGHT_S >= worst_case + LLM_GEN_MAX_TIME_S)
# ...and the session ledger must still be the thing that ends the spending, not
# the watchdog: one synthesis may not be able to consume the whole budget.
truthy("one_synthesis_cannot_drain_the_session_budget",
       worst_case < LLM_SESSION_BUDGET_S / 4)


print()
print("=" * 70)
print(f"CHECKS: {CHECKS[0]}   HARD FAILS: {len(FAILS)}")
for f in FAILS:
    print("  " + f)
print("=" * 70)
sys.exit(1 if FAILS else 0)

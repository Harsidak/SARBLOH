"""Continual Harness refinement and digest, ported from upstream ``core/refinement/refinement.ts`` and the
auto-refine and digest code in ``core/agent-session.ts`` (commit 2d24ad4).

- Auto /refine: every ``turnInterval`` assistant turns (default 25) and after each compaction, rate-limited by a
  cooldown (default 20 min), a review call (the gate) decides whether the trajectory holds evidence worth
  persisting; if so a planner call returns JSON Create/Update/Delete edits, which the host validates and applies
  to the local harness file and records as a refinement event. Upstream runs this between turns in headless
  autonomous runs (``serializedRefine``); so do we. The model is told what changed with an ``[auto-refinement]``
  notice. Both system prompts are verbatim.
- Digest: the harness state rendered for the model (upstream formatHarnessStateForPrompt), relevance-ranked
  against the last messages, delivered as a ``[harness-digest]`` message on the first turn and on every
  compaction head. It is not re-rendered into the system prompt each turn (upstream keeps that prefix stable).
Differences: query terms are split on letters and digits only (upstream also bigrams CJK text); the request is
fitted to the context window by halving the conversation on overflow (upstream sizes it from the window).
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from prime.compaction import serialize_conversation

REFINEMENT_SYSTEM_PROMPT = """You are Prime Agent's /refine continual harness subsystem.

Your job is to improve the editable continual harness state from the current trajectory.
This is similar in spirit to context compaction, but instead of summarizing the
conversation you emit precise Create, Update, or Delete edits to reusable state.
The continual harness is the persistent, editable set of prompt notes, memories,
skills, and subagent specs that lets Prime Agent improve reusable behavior
outside the token history.
Use "continual harness" for that persistent artifact layer; keep "RLM" for the
runtime, Python REPL kernel, and native call interface that executes those artifacts.

Continual harness components:
- prompt: supplemental prompt notes only. The base system prompt is immutable and MUST NOT be rewritten.
- memory: durable facts, decisions, failures, preferences, and outcomes.
- skill: installed Python REPL skill. Skill create/update edits MUST include a `reference` object with `{"type":"python"}`, a Python import, and a callable or call pattern; they also MUST include an `arguments` object describing accepted inputs, required fields, defaults, and constraints. Use `{}` for `arguments` only when the Python callable truly needs no external inputs. Include the RLM-native call form `await <skill_import>(...)`.
- subagent: reusable delegation specs, including purpose, instructions, and when to invoke. Include the RLM-native call form: compose a concise task prompt and spawn with `handle = await rlm.spawn("sub-task", name="worker")`; admission returns immediately with `rlm_child_id`, `name`, `session_dir`, and `model`, never the child's answer. Results arrive only through explicit `agent_message` replies or files; children reply with `await agent_message.send(message, receiver_role="parent")`. Use `await rlm.list_subagents()` to recover direct child handles and `await agent_message.send(..., receiver_role="child", receiver_name=handle.name)` for follow-ups. Do not invent wrappers like `run_subagent(...)`.

Scope and persistence policy:
- The default editable continual harness store is local to the current Prime Agent session. Use it for session-specific progress, active task state, current-run coordination notes, temporary blockers, and project facts that should not affect other sessions.
- A caller may explicitly request global refinement. Global edits must be stable cross-session lessons, durable user preferences, reusable skills/subagents, or tool/environment facts that should affect future sessions.
- Entry ids in the harness overview may carry a display-only `local:` or `global:` prefix. Always use the bare id (no prefix) in edits.
- All edits in one refinement apply only to the requested scope's store. During a local refinement, global entries are read-only context: never propose update or delete edits for them; create a local entry instead when a session-specific override is genuinely needed.
- Project/workspace-specific lessons may be persisted globally only when the title, path, or content explicitly names the project/workspace and the lesson is likely to be reused in future sessions for that project. Prefer local edits when the lesson only belongs in the current conversation.
- Use memory for declarative facts and preferences, skill for repeatable procedures exposed as Python calls, prompt for narrow behavioral policy addendums, and subagent for reusable delegation roles.
- Create or update the smallest relevant component: repeated delegation roles should become subagent specs, repeated procedures should become skills, durable facts/preferences should become memories, and narrow behavioral policies should become prompt addendums.
- When an edit is persisted, include metadata such as `{"scope":"local"}` or `{"scope":"global"}` when that helps future review understand the intended blast radius.

Use the trajectory, current continual harness state, and prior refinement history. Prefer
small evidence-backed edits. If prior refinements caused issues, rollback or
replace the faulty editable entries. Never edit source files directly. Output
JSON only with this exact shape:

{
  "summary": "one sentence",
  "rationale": "why these edits are justified by trajectory evidence",
  "expectedOutcome": "what should improve and how to validate it",
  "edits": [
    {
      "action": "create|update|delete",
      "kind": "prompt|memory|skill|subagent",
      "id": "stable id for update/delete, optional for create",
      "title": "required for create/update except delete",
      "content": "required for create/update except delete",
      "path": "optional grouping path",
      "reference": {"type": "python", "import": "package.module", "callable": "function_name", "call_pattern": "await function_name(...)"},
      "arguments": {"name": {"type": "string", "required": true, "description": "accepted input"}},
      "metadata": {},
      "reason": "why this edit is useful"
    }
  ]
}"""

AUTO_REFINE_REVIEW_SYSTEM_PROMPT = """You are Prime Agent's automatic /refine review gate.

Decide whether this checkpoint should run /refine. Auto /refine writes local continual harness state by default, so approve when the trajectory contains evidence useful to this session's future turns.
Reject one-off noise, unsupported hypotheses, and transient tool outputs. Ask for global refinement only for durable cross-session lessons or explicitly project-qualified lessons likely to be reused in future sessions.

Return JSON only:
{
  "shouldRefine": true|false,
  "rationale": "short reason",
  "instructions": "optional concise instructions for /refine if shouldRefine is true"
}"""

LOCAL_SCOPE = ("Requested refinement scope: local. Prefer local continual harness edits for current task progress, "
               "temporary blockers, current-run coordination, and project facts that are not clearly reusable across "
               "Prime Agent sessions. Global entries in the overview are read-only context: do not propose update or "
               "delete edits for them; create a local entry instead if an override is needed.")
REVIEW_CLOSING = ("Return shouldRefine=true when the trajectory contains evidence useful to this session's future "
                  "turns. Prefer local harness edits for current task progress, temporary blockers, and current-run "
                  "coordination. Ask for global refinement only for durable cross-session lessons or explicitly "
                  "project-qualified facts likely to be reused in future sessions.")
REVIEW_CONVERSATION_CHARS = 40_000   # upstream: serializeConversation(...).slice(-40_000)
PLAN_CONVERSATION_CHARS = 80_000     # upstream: .slice(-80_000)
_KINDS = ("prompt", "memory", "skill", "subagent")

HARNESS_DIGEST_PREFIX = ("[harness-digest]\n\nThe persistent memories produced across this session so far:\n\n"
                         "<harness_state>\n")
HARNESS_DIGEST_SUFFIX = "\n</harness_state>"
_DIGEST_HEADER = [
    "# Continual Harness State",
    "",
    "Local continual harness entries belong to this Prime Agent session. Global continual harness entries persist "
    "across Prime Agent sessions.",
    "The continual harness entries below are compact summaries, not full descriptions. Use them as routing/context "
    "hints; inspect or refine the underlying continual harness entry only when detail matters.",
    "Default to local continual harness refinement for current task progress, temporary blockers, and session "
    "coordination. Use global continual harness refinement only for stable cross-session lessons, durable user "
    "preferences, reusable skills/subagents, or explicitly project-qualified facts.",
    "Use these continual harness prompt notes, memories, skills, and subagent specs when they are relevant. The base "
    "system prompt is immutable; prompt entries below are supplemental notes only.",
    "",
    "When to refine the continual harness: after a repeated failure, a reusable tactic emerges, a repeated delegation "
    "role should become a subagent spec, a repeated procedure should become a skill, a durable fact/preference should "
    "become a memory, a narrow behavioral policy should become a prompt addendum, a user corrects behavior that "
    "should persist locally or globally, validation shows a continual harness entry is wrong, or a "
    "skill/subagent/memory/prompt note should be created, updated, deleted, or rolled back. Keep continual harness "
    "edits small and evidence-backed.",
    "",
    "Call contract: read each installed Python skill's SKILL.md and call its documented module function in the "
    "Python REPL; do not assume a `.run` entrypoint. Use `<skill_import> ...` in shell when a CLI exists. Continual "
    "harness skill entries are Python REPL skills with an explicit Python `reference` and `arguments` contract. Spawn "
    "a continual harness subagent spec by composing a concise task prompt and calling `handle = await "
    "rlm.spawn('sub-task', name='worker')`; admission returns immediately with `rlm_child_id`, `name`, "
    "`session_dir`, and `model`, never the child's answer. Results arrive only through explicit `agent_message` "
    "replies or files; children reply with `await agent_message.send(message, receiver_role='parent')`. Use `await "
    "rlm.list_subagents()` to recover direct child handles and `await agent_message.send(..., receiver_role='child', "
    "receiver_name=handle.name)` for follow-ups. Do not invent wrappers such as `call_skill(...)`, "
    "`run_subagent(...)`, or named subagent registries.",
    "",
]


# --- harness state: load and merge ------------------------------------------------------------------------
def load_merged(global_file: Path, local_file: Path) -> tuple[dict[str, dict[str, Any]], list[Any]]:
    """Global state overlaid with local state (upstream mergeHarnessStates): kind -> {id: entry}, refinements."""
    from rlm.harness import HarnessState

    merged: dict[str, dict[str, Any]] = {k: {} for k in _KINDS}
    refinements: list[Any] = []
    for scope, path in (("global", global_file), ("local", local_file)):
        if not path.exists():
            continue
        state = HarnessState(path, scope=scope)
        for kind in _KINDS:
            for eid, entry in state.entries[kind].items():
                entry.scope = scope
                merged[kind][f"{scope}:{eid}" if eid in merged[kind] else eid] = entry
        refinements += state.refinements
    return merged, refinements


def _compact(text: str, n: int) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text if len(text) <= n else f"{text[:max(0, n - 3)]}..."


# --- relevance ranking (upstream harnessQueryTerms / harnessQueryTermIdf / scoreHarnessEntryForQuery) -------
def query_terms(text: str) -> list[str]:
    return list(dict.fromkeys(t for t in re.findall(r"[^\W_]+", (text or "").lower()) if len(t) >= 4))


def build_query_terms(messages: list[dict[str, Any]]) -> dict[str, float]:
    """Terms from the last four user/assistant messages, newest weighted highest (2, 1.5, 1, 1), at most 48."""
    terms: dict[str, float] = {}
    recent = [m for m in messages if m["role"] in ("user", "assistant") and not m.get("_kind")][-4:]
    weight = 2.0
    for m in reversed(recent):
        for t in query_terms(str(m.get("content") or "")):
            if len(terms) >= 48 and t not in terms:
                break
            terms.setdefault(t, weight)
        weight = max(1.0, weight - 0.5)
    return terms


def _fields(e: Any) -> tuple[str, str, str]:
    return (e.title or "").lower(), (e.content or "").lower(), f"{(e.path or '').lower()} {(e.id or '').lower()}"


def _score(e: Any, terms: dict[str, float], idf: dict[str, float]) -> float:
    score = 0.0
    for term, weight in terms.items():
        fields = sum(1 for f in _fields(e) if term in f)
        if fields:
            score += weight * idf.get(term, 1.0) * (1 + (fields - 1) * 0.5)
    return score


def _rank(entries: list[Any], terms: dict[str, float]) -> list[Any]:
    key = lambda e: ((e.path or ""), (e.title or ""), (e.id or ""))
    if not terms:
        return sorted(entries, key=key)
    matches: dict[str, int] = {}
    for e in entries:
        f = _fields(e)
        for t in terms:
            if any(t in x for x in f):
                matches[t] = matches.get(t, 0) + 1
    idf = {t: math.log(1 + len(entries) / n) for t, n in matches.items()}
    return sorted(sorted(entries, key=key), key=lambda e: -_score(e, terms, idf))


# --- digest (upstream formatHarnessStateForPrompt, REPL session without the refine skill) ------------------
def format_digest(merged: dict[str, dict[str, Any]], refinements: list[Any], terms: dict[str, float] | None = None,
                  max_entries: int = 6, max_refinements: int = 5, max_len: int = 180) -> str:
    lines = list(_DIGEST_HEADER)
    total = 0
    for kind in _KINDS:
        entries = _rank(list(merged[kind].values()), terms or {})
        total += len(entries)
        if kind == "subagent" and entries:
            lines.append(f"{kind}: {len(entries)} (invoke a spec by turning it into a concise task prompt and spawning "
                         "with `await rlm.spawn('<task>', name='<worker>')`; admission returns a child handle, never "
                         "the answer)")
        else:
            lines.append(f"{kind}: {len(entries)}")
        if terms and len(entries) > max_entries:
            lines.append("(entries ranked by relevance to the current task; see harness.search)")
        for e in entries[:max_entries]:
            ref = f" ref={_compact(json.dumps(e.reference), max_len)}" if kind == "skill" and e.reference else ""
            args = f" args={_compact(json.dumps(e.arguments), max_len)}" if kind == "skill" and e.arguments else ""
            lines.append(f"- [{e.scope}:{e.id}] {e.title} ({e.path}, v{e.version}){ref}{args}: "
                         f"{_compact(e.content, max_len)}")
        if len(entries) > max_entries:
            lines.append(f"- +{len(entries) - max_entries} more {kind} entries")
        lines.append("")
    if total == 0:
        lines += ["No saved harness entries yet.", ""]
    lines.append(f"recent refinements: {len(refinements)}")
    for ev in refinements[-max_refinements:]:
        changes = ", ".join(ev.changes) if ev.changes else "no applied edits"
        outcome = f"; outcome: {_compact(ev.outcome, max_len)}" if ev.outcome else ""
        lines.append(f"- [{ev.id}] {_compact(ev.trigger, max_len)}: {changes}{outcome}")
    if len(refinements) > max_refinements:
        lines.append(f"- +{len(refinements) - max_refinements} older refinement events")
    return "\n".join(lines).strip()


def digest_block(digest: str) -> str:
    return HARNESS_DIGEST_PREFIX + digest + HARNESS_DIGEST_SUFFIX


# --- refine requests ---------------------------------------------------------------------------------------
def _overview_for_prompt(merged: dict[str, dict[str, Any]]) -> str:
    lines = []
    for kind in _KINDS:
        entries = list(merged[kind].values())
        lines.append(f"{kind}: {len(entries)}")
        for e in entries[:40]:
            ref = f" ref={json.dumps(e.reference)[:240]}" if kind == "skill" and e.reference else ""
            args = f" args={json.dumps(e.arguments)[:240]}" if kind == "skill" and e.arguments else ""
            content = re.sub(r"\s+", " ", e.content or "")[:240]
            lines.append(f"- [{e.scope}:{e.id}] {e.title} ({e.path}, v{e.version}){ref}{args}: {content}")
        if len(entries) > 40:
            lines.append(f"- +{len(entries) - 40} more {kind} entries")
    return "\n".join(lines)


def _history_for_prompt(refinements: list[Any]) -> str:
    if not refinements:
        return "No prior refinement history."
    return "\n\n".join(f"[{ev.id}] {ev.trigger}\n{', '.join('applied ' + c for c in ev.changes)}\n"
                       f"Expected outcome: {ev.outcome}" for ev in refinements[-20:])


def extract_json(text: str) -> dict[str, Any]:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in the reply")
    value = json.loads(text[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("reply JSON is not an object")
    return value


def _ask(llm: Any, system: str, build: Any, conversation: str, max_tokens: int, overflow: type[Exception]) -> dict:
    """One JSON request. Overflow: drop the oldest half of the conversation. Bad JSON: one retry."""
    last: Exception | None = None
    tries = 0
    while tries < 2:
        try:
            reply = llm.chat([{"role": "system", "content": system}, {"role": "user", "content": build(conversation)}],
                             tools=None, max_tokens=max_tokens)
        except overflow:
            if len(conversation) < 2000:
                raise
            conversation = "[Earlier conversation omitted to fit the model context.]\n" + \
                conversation[len(conversation) // 2:]
            continue
        tries += 1
        try:
            return extract_json(reply.content)
        except ValueError as exc:
            last = exc
    raise ValueError(f"unparseable JSON after 2 tries: {last}")


def apply_edits(state: Any, edits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate and apply planner edits to the local store (upstream applyRefinementProposal)."""
    results = []
    for edit in edits if isinstance(edits, list) else []:
        action, kind = edit.get("action"), edit.get("kind")
        eid, title, content = edit.get("id"), edit.get("title"), edit.get("content")
        row: dict[str, Any] = {"action": action, "kind": kind, "id": eid, "title": title}
        try:
            if kind not in _KINDS or action not in ("create", "update", "delete"):
                raise ValueError(f"bad action/kind {action!r}/{kind!r}")
            if action == "delete":
                before = state.get(kind, str(eid))
                if before is None or not state.delete(kind, str(eid)):
                    raise ValueError("entry not found in the local store (global entries are read-only)")
                row.update(title=before.title, content=before.content)
            else:
                extra = {k: edit[k] for k in ("path", "reference", "arguments", "metadata") if edit.get(k)}
                extra["source"] = "refine"
                if kind == "skill":
                    extra.setdefault("arguments", {})
                if action == "create":
                    entry = state.create(kind, str(title), str(content), id=eid or None, **extra)
                else:
                    before = state.get(kind, str(eid))
                    if before is None:
                        raise ValueError("entry not found in the local store (global entries are read-only)")
                    entry = state.update(kind, str(eid), title or before.title,
                                         content if content is not None else before.content, **extra)
                row.update(id=entry.id, version=entry.version, title=entry.title, content=entry.content)
            row["applied"] = True
        except Exception as exc:  # noqa: BLE001 - the error goes into the record, like upstream
            row.update(applied=False, error=f"{type(exc).__name__}: {exc}"[:300])
        results.append(row)
    return results


def notice(summary: str, edits: list[dict[str, Any]]) -> str:
    """Model-facing ``[auto-refinement]`` message (upstream createRefinementNoticeMessage + body format)."""
    lines = [_compact(summary, 180)]
    for e in edits:
        if e.get("applied"):
            lines.append(f"- {e['action']} {e['kind']} [local:{e['id']}] {e.get('title') or e['id']}: "
                         f"{_compact(str(e.get('content') or ''), 180)}")
    return "[auto-refinement]\n\n" + "\n".join(lines)


def auto_refine(*, llm: Any, global_file: Path, local_file: Path, messages: list[dict[str, Any]], reason: str,
                turns_since: int, max_tokens: int, overflow: type[Exception]) -> dict[str, Any]:
    """One auto-refine pass: review gate, then plan + apply. Returns a record; ``notice`` is set when edits applied."""
    from rlm.harness import HarnessState

    merged, refinements = load_merged(global_file, local_file)
    overview, history = _overview_for_prompt(merged), _history_for_prompt(refinements)
    conversation = serialize_conversation([m for m in messages if m.get("_kind") != "harness_digest"])
    record: dict[str, Any] = {"reason": reason, "turns_since": turns_since}

    review = _ask(llm, AUTO_REFINE_REVIEW_SYSTEM_PROMPT, lambda c: "\n\n".join([
        f"<trigger>\n{reason}; {turns_since} assistant turns since last auto-refine review\n</trigger>",
        f"<current_harness_state>\n{overview}\n</current_harness_state>",
        f"<refinement_history>\n{history}\n</refinement_history>",
        f"<conversation>\n{c}\n</conversation>",
        REVIEW_CLOSING,
    ]), conversation[-REVIEW_CONVERSATION_CHARS:], max_tokens, overflow)
    record["review"] = review
    if review.get("shouldRefine") is not True:
        return record

    instructions = ("Automatic refine review triggered by "
                    f"{reason}. Only create/update/delete local harness entries if there is clear evidence that "
                    "should help this session continue. Prefer an empty edits array over speculative or one-off "
                    "memories. Do not promote anything global unless explicitly requested. Reviewer rationale: "
                    f"{review.get('rationale', '')}"
                    + (f"\nReviewer instructions: {review['instructions']}" if review.get("instructions") else ""))
    plan = _ask(llm, REFINEMENT_SYSTEM_PROMPT, lambda c: "\n\n".join([
        f"<current_harness_state>\n{overview}\n</current_harness_state>",
        f"<refinement_history>\n{history}\n</refinement_history>",
        f"<conversation>\n{c}\n</conversation>",
        f"<scope_policy>\n{LOCAL_SCOPE}\n</scope_policy>",
        f"<user_refine_instructions>\n{instructions}\n</user_refine_instructions>",
        "Return only JSON edits. If no useful edit is justified, return an empty edits array with a rationale.",
    ]), conversation[-PLAN_CONVERSATION_CHARS:], max_tokens, overflow)
    record["summary"] = plan.get("summary")
    state = HarnessState(local_file, scope="local")
    record["edits"] = apply_edits(state, plan.get("edits") or [])
    changes = [f"{e['action']} {e['kind']}:{e['id']}" for e in record["edits"] if e.get("applied")]
    if changes:
        summary = str(plan.get("summary") or reason)[:500]
        state.record_refinement(summary, changes, evidence=str(plan.get("rationale") or "")[:1000],
                                outcome=str(plan.get("expectedOutcome") or "")[:500])
        record["notice"] = notice(summary, record["edits"])
    return record

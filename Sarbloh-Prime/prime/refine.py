"""Host-driven Continual Harness refinement (auto /refine), ported from upstream
``packages/coding-agent/src/core/refinement/refinement.ts`` and the auto-refine triggers in ``agent-session.ts``
(commit 2d24ad4).

Upstream, the harness is not only edited by the agent: the host runs an automatic refine review every
``turnInterval`` assistant turns (default 25) and after every compaction, rate-limited by a cooldown (default
20 min). A review call (the gate) decides whether the trajectory holds evidence worth persisting; if so, a
planner call emits JSON Create/Update/Delete edits, which the host validates and applies to the local harness
file, then records as a refinement event. The agent sees the result in its next system prompt (the digest).

E003 ran without this and recorded 0 harness writes. Ported here: both prompts (verbatim except for lines about
features we do not have), the JSON contract, edit validation, application and the refinement record.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

REFINEMENT_SYSTEM_PROMPT = """You are Prime Agent's /refine continual harness subsystem.

Your job is to improve the editable continual harness state from the current trajectory.
This is similar in spirit to context compaction, but instead of summarizing the
conversation you emit precise Create, Update, or Delete edits to reusable state.
The continual harness is the persistent, editable set of prompt notes, memories,
skills, and subagent specs that lets Prime Agent improve reusable behavior
outside the token history.

Continual harness components:
- prompt: supplemental prompt notes only. The base system prompt is immutable and MUST NOT be rewritten.
- memory: durable facts, decisions, failures, preferences, and outcomes.
- skill: installed Python REPL skill. Skill create/update edits MUST include a `reference` object with `{"type":"python"}`, a Python import, and a callable or call pattern; they also MUST include an `arguments` object describing accepted inputs, required fields, defaults, and constraints. Use `{}` for `arguments` only when the Python callable truly needs no external inputs. A skill may reference a helper the agent defined in its REPL: use `"import": "__main__"` and the helper's name as `callable`.
- subagent: reusable delegation specs, including purpose, instructions, and when to invoke. Include the RLM-native call form: compose a concise task prompt and spawn with `handle = await rlm.spawn("sub-task", name="worker")`.

Scope and persistence policy:
- The default editable continual harness store is local to the current Prime Agent session. Use it for session-specific progress, active task state, temporary blockers, and facts that should not affect other sessions.
- Entry ids in the harness overview may carry a display-only `local:` or `global:` prefix. Always use the bare id (no prefix) in edits.
- Use memory for declarative facts and preferences, skill for repeatable procedures exposed as Python calls, prompt for narrow behavioral policy addendums, and subagent for reusable delegation roles.
- Create or update the smallest relevant component: repeated delegation roles should become subagent specs, repeated procedures should become skills, durable facts/preferences should become memories, and narrow behavioral policies should become prompt addendums.

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
Reject one-off noise, unsupported hypotheses, and transient tool outputs.

Return JSON only:
{
  "shouldRefine": true|false,
  "rationale": "short reason",
  "instructions": "optional concise instructions for /refine if shouldRefine is true"
}"""

LOCAL_SCOPE = ("Requested refinement scope: local. Prefer local continual harness edits for current task progress, "
               "temporary blockers, current-run coordination, and facts that are not clearly reusable across sessions.")
_KINDS = ("prompt", "memory", "skill", "subagent")


def serialize(messages: list[dict[str, Any]], max_chars: int) -> str:
    """Conversation as text, newest kept (upstream: serializeConversation(...).slice(-N))."""
    out = []
    for m in messages:
        role = m.get("role", "?")
        text = m.get("content") or ""
        for tc in m.get("tool_calls") or []:
            args = tc.get("function", {}).get("arguments", "")
            try:
                args = json.loads(args).get("code", args)
            except (ValueError, AttributeError):
                pass
            text += f"\n[ipython call]\n{args}"
        out.append(f"[{role}]\n{text}")
    return "\n\n".join(out)[-max_chars:]


def extract_json(text: str) -> dict[str, Any]:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in the reply")
    value = json.loads(text[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("reply JSON is not an object")
    return value


def _history(state: Any, limit: int = 5) -> str:
    rows = [f"- [{r.id}] {r.trigger}: {', '.join(r.changes)}" + (f" -> {r.outcome}" if r.outcome else "")
            for r in state.refinements[-limit:]]
    return "\n".join(rows) or "(none)"


def _ask(llm: Any, system: str, user: str, max_tokens: int) -> dict[str, Any]:
    last: Exception | None = None
    for _ in range(2):
        reply = llm.chat([{"role": "system", "content": system}, {"role": "user", "content": user}],
                         tools=None, max_tokens=max_tokens)
        try:
            return extract_json(reply.content)
        except ValueError as exc:
            last = exc
    raise ValueError(f"unparseable JSON after 2 tries: {last}")


def apply_edits(state: Any, edits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate and apply planner edits (upstream applyRefinementProposal); one bad edit never blocks the rest."""
    results = []
    for edit in edits if isinstance(edits, list) else []:
        action, kind = edit.get("action"), edit.get("kind")
        eid, title, content = edit.get("id"), edit.get("title"), edit.get("content")
        row = {"action": action, "kind": kind, "id": eid, "title": title}
        try:
            if kind not in _KINDS or action not in ("create", "update", "delete"):
                raise ValueError(f"bad action/kind {action!r}/{kind!r}")
            if action == "delete":
                if not state.delete(kind, str(eid)):
                    raise ValueError("entry not found")
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
                        raise ValueError("entry not found")
                    entry = state.update(kind, str(eid), title or before.title,
                                         content if content is not None else before.content, **extra)
                row.update(id=entry.id, version=entry.version)
            row["applied"] = True
        except Exception as exc:  # noqa: BLE001 - the error goes into the record, like upstream
            row.update(applied=False, error=f"{type(exc).__name__}: {exc}"[:300])
        results.append(row)
    return results


def auto_refine(*, llm: Any, harness_file: Path, messages: list[dict[str, Any]], reason: str, turns_since: int,
                conversation_chars: int, max_tokens: int) -> dict[str, Any]:
    """One auto-refine pass: review gate, then plan + apply. Returns a record for the transcript."""
    from rlm.harness import HarnessState

    state = HarnessState(harness_file, scope="local")
    overview = state.overview(max_entries_per_kind=30)
    history = _history(state)
    record: dict[str, Any] = {"reason": reason, "turns_since": turns_since}

    review_user = "\n\n".join([
        f"<trigger>\n{reason}; {turns_since} assistant turns since last auto-refine review\n</trigger>",
        f"<current_harness_state>\n{overview}\n</current_harness_state>",
        f"<refinement_history>\n{history}\n</refinement_history>",
        f"<conversation>\n{serialize(messages, conversation_chars // 2)}\n</conversation>",
        "Return shouldRefine=true when the trajectory contains evidence useful to this session's future turns. "
        "Prefer local harness edits for current task progress, temporary blockers, and current-run coordination.",
    ])
    review = _ask(llm, AUTO_REFINE_REVIEW_SYSTEM_PROMPT, review_user, max_tokens)
    record["review"] = review
    if review.get("shouldRefine") is not True:
        return record

    instructions = (f"Automatic refine review triggered by {reason}. Only create/update/delete local harness entries "
                    "if there is clear evidence that should help this session continue. Prefer an empty edits array "
                    "over speculative or one-off memories. Reviewer rationale: "
                    f"{review.get('rationale', '')}" + (f"\nReviewer instructions: {review['instructions']}"
                                                         if review.get("instructions") else ""))
    plan_user = "\n\n".join([
        f"<current_harness_state>\n{overview}\n</current_harness_state>",
        f"<refinement_history>\n{history}\n</refinement_history>",
        f"<conversation>\n{serialize(messages, conversation_chars)}\n</conversation>",
        f"<scope_policy>\n{LOCAL_SCOPE}\n</scope_policy>",
        f"<user_refine_instructions>\n{instructions}\n</user_refine_instructions>",
        "Return only JSON edits. If no useful edit is justified, return an empty edits array with a rationale.",
    ])
    plan = _ask(llm, REFINEMENT_SYSTEM_PROMPT, plan_user, max_tokens)
    record["summary"] = plan.get("summary")
    record["edits"] = apply_edits(state, plan.get("edits") or [])
    changes = [f"{e['action']} {e['kind']}:{e['id']}" for e in record["edits"] if e.get("applied")]
    if changes:
        state.record_refinement(str(plan.get("summary") or reason)[:500], changes,
                                evidence=str(plan.get("rationale") or "")[:1000],
                                outcome=str(plan.get("expectedOutcome") or "")[:500])
    return record

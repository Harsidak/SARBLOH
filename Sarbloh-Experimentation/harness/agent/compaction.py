"""Context compaction, ported from upstream ``core/compaction/compaction.ts`` and ``core/compaction/utils.ts``
(commit 2d24ad4) to OpenAI-format chat messages.

Kept from upstream:
- Trigger: the context (last usage + chars/4 of the messages after it) passes ``window - reserve_tokens``.
- The newest ``keep_recent_tokens`` of messages stay verbatim. The cut is made only at a user or assistant
  message, never at a tool result. Only the older prefix is summarized.
- The summarizer is a separate call with its own system prompt. The conversation is serialized as text, so the
  model summarizes it instead of continuing it. Tool results are cut to 2000 characters for the summarizer.
- A cut inside a turn adds a second "turn prefix" summary. The newest kept assistant text is passed as a recency
  anchor, so the summary cannot lag behind the kept messages.
- The summary enters the context as a user message with the ``[compaction-summary]`` header; the harness digest
  is attached to it mechanically, never through the summarizer.
Not ported (not used here): file-operation lists, branch summaries, auxiliary-model routing.
Changed: the summary is a short game handover note (at most ``summary_tokens``, about 2k) that replaces the previous
one (passed as ``<previous-summary>``) instead of preserving all of it. The plan, hypotheses and findings are in the
agent's memory block and the kernel's names are listed by the host in code, so the note leaves them out. The token
estimate counts the stored thinking, so the cut point sees what is big.
Added: if the summarizer request itself overflows, the oldest part of the serialized conversation is dropped and
the call is retried (upstream routes to a larger model instead).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any

SUMMARIZATION_SYSTEM_PROMPT = """You write short handover notes. You read part of a conversation in which an agent plays a grid puzzle game: it thinks, runs Python in a kernel, and makes moves with the act tool. Your note lets the agent carry on where it left off. Do not continue the conversation and do not answer anything in it. Write only the note."""

# One note at a time: a later compaction writes a new note that replaces the old one (the old note is input, not
# something to keep whole). The plan, hypotheses and findings live in the agent's memory block, and the kernel's names
# are listed by the host, so the note holds only what neither has.
SUMMARIZATION_PROMPT = """The conversation above is older work that is about to be removed from the agent's context. Write a handover note of at most {words} words to take its place.{previous}

The agent keeps its memory block (goal, plan, hypotheses, findings, lessons), so do not copy any of that into the note. Its Python kernel keeps running too, and the host lists the names defined there, so do not list them either. Write only what the memory block does not hold, under these three headings:

## Recent moves
What the agent did on the board most recently and what happened, with step numbers.

## Tests and results
The experiments it ran in Python or on the board and what each one showed, with exact numbers, positions and colours where they matter. Leave out ideas that the memory block already marks as refuted.

## Where it stopped
What it was doing at the end and what it meant to do next.

Use plain sentences and short bullets. Keep step numbers, coordinates and error messages exact. If a heading has nothing new, write "(nothing)"."""

PREVIOUS_SUMMARY_RULE = """

The note in <previous-summary> was written at the last compaction. Your note replaces it: keep from it only what still matters now, and drop what is finished, out of date or already in the memory block."""

TURN_PREFIX_SUMMARIZATION_PROMPT = """This is the start of the agent's current turn. The turn was too long to keep whole, so only its end stays in the context. In at most {words} words, say what the turn started from and what it found so far, so that the kept end makes sense. Keep step numbers, coordinates and error messages exact."""

COMPACTION_SUMMARY_PREFIX = """[compaction-summary]

The conversation history before this point was compacted into the following summary.
The retained messages below are authoritative; this summary may lag behind them.

<summary>
"""
COMPACTION_SUMMARY_SUFFIX = "\n</summary>"

TOOL_RESULT_MAX_CHARS = 2000
TOOL_RESULT_TAIL_CHARS = 500
RECENT_STATE_ANCHOR_MAX_CHARS = 2000

# Private message keys (never sent to the model): "_kind" marks host messages that are not conversation.
COMPACTION_KIND = "compaction"
DIGEST_KIND = "harness_digest"


def should_compact(context_tokens: int, context_window: int, reserve_tokens: int,
                   trigger_tokens: int | None = None) -> bool:
    """Upstream: context > window - reserve. Ours: also above ``trigger_tokens`` when it is set (smaller models)."""
    limit = context_window - reserve_tokens
    if trigger_tokens:
        limit = min(limit, trigger_tokens)
    return context_window > 0 and context_tokens > limit


def _text(msg: dict[str, Any]) -> str:
    content = msg.get("content") or ""
    if isinstance(content, list):  # multimodal parts
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return str(content)


def estimate_tokens(msg: dict[str, Any], reasoning: bool = True, image_tokens: int = 0) -> int:
    """chars/4, like upstream (conservative). The stored thinking counts too, so the cut point sees what is big;
    ``reasoning=False`` leaves it out (the context estimate adds only the thinking that is actually sent). Each picture
    adds ``image_tokens``."""
    chars = len(_text(msg)) + (len(msg.get("_reasoning") or "") if reasoning else 0)
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        chars += len(fn.get("name") or "") + len(fn.get("arguments") or "")
    content = msg.get("content")
    pictures = sum(1 for p in content if isinstance(p, dict) and p.get("type") == "image_url") \
        if isinstance(content, list) else 0
    return math.ceil(chars / 4) + pictures * image_tokens


def _turn_start(messages: list[dict[str, Any]], index: int, start: int) -> int:
    for i in range(index, start - 1, -1):
        if messages[i]["role"] == "user":
            return i
    return -1


def find_cut_point(messages: list[dict[str, Any]], start: int, keep_recent_tokens: int) -> tuple[int, int, bool]:
    """(first kept index, start of the split turn or -1, is split turn). Walks back from the newest message."""
    cut_points = [i for i in range(start, len(messages)) if messages[i]["role"] in ("user", "assistant")]
    if not cut_points:
        return start, -1, False
    cut = cut_points[0]
    acc = 0
    for i in range(len(messages) - 1, start - 1, -1):
        acc += estimate_tokens(messages[i])
        if acc >= keep_recent_tokens:
            cut = next((c for c in cut_points if c >= i), cut_points[-1])
            break
    if messages[cut]["role"] == "user":
        return cut, -1, False
    turn_start = _turn_start(messages, cut, start)
    return cut, turn_start, turn_start != -1


def truncate_for_summary(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    marker_max = len(f"[... {len(text)} characters truncated; first {max_chars} and last {TOOL_RESULT_TAIL_CHARS} "
                     "kept ...]")
    head = max_chars - TOOL_RESULT_TAIL_CHARS - marker_max - 4
    elided = len(text) - head - TOOL_RESULT_TAIL_CHARS
    return (f"{text[:head]}\n\n[... {elided} characters truncated; first {head} and last {TOOL_RESULT_TAIL_CHARS} "
            f"kept ...]\n\n{text[-TOOL_RESULT_TAIL_CHARS:]}")


def serialize_conversation(messages: list[dict[str, Any]]) -> str:
    """Messages as text for a summarizer (upstream serializeConversation): numbered tool calls, cut tool results."""
    parts: list[str] = []
    index_of: dict[str, int] = {}
    name_of: dict[str, str] = {}
    n = 0
    for msg in messages:
        role = msg["role"]
        if role == "user":
            text = _text(msg)
            if text:
                parts.append(f"[User]: {text}")
        elif role == "assistant":
            text = _text(msg)
            if text:
                parts.append(f"[Assistant]: {text}")
            calls = []
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except ValueError:
                    args = {"arguments": fn.get("arguments")}
                if not isinstance(args, dict):
                    args = {"arguments": args}
                n += 1
                index_of[tc.get("id", "")] = n
                name_of[tc.get("id", "")] = fn.get("name", "ipython")
                arg_text = ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in args.items())
                calls.append(f"#{n} {fn.get('name', 'ipython')}({arg_text})")
            if calls:
                parts.append(f"[Assistant tool calls]: {'; '.join(calls)}")
        elif role == "tool":
            text = _text(msg)
            if text:
                idx = index_of.get(msg.get("tool_call_id", ""))
                suffix = "" if idx is None else f" #{idx}"
                tname = name_of.get(msg.get("tool_call_id", ""), "ipython")
                label = f"[Tool result ({tname}, error){suffix}]" if msg.get("_error") else f"[Tool result ({tname}){suffix}]"
                parts.append(f"{label}: {truncate_for_summary(text, TOOL_RESULT_MAX_CHARS)}")
    return "\n\n".join(parts)


def recent_state_anchor(kept: list[dict[str, Any]]) -> str | None:
    for msg in reversed(kept):
        if msg["role"] == "assistant":
            text = _text(msg).strip()
            if text:
                return text[-RECENT_STATE_ANCHOR_MAX_CHARS:]
    return None


def _words(tokens: int) -> int:
    """A word limit that fits ``tokens``: half a word per token, because digits and coordinates are a token each."""
    return max(50, tokens // 2 // 50 * 50)


def history_summary_prompt(conversation: str, previous_summary: str | None, anchor: str | None,
                           summary_tokens: int = 2048) -> str:
    text = f"<conversation>\n{conversation}\n</conversation>\n\n"
    if previous_summary:
        text += f"<previous-summary>\n{previous_summary}\n</previous-summary>\n\n"
    if anchor:
        text += ("<recent-state-anchor>\nNewest assistant message that stays retained below the summary. The "
                 "conversation to summarize is older than this anchor; the retained messages below are authoritative, "
                 "so treat this anchor, not the conversation above, as the current state.\n\n"
                 f"{anchor}\n</recent-state-anchor>\n\n")
    return text + SUMMARIZATION_PROMPT.format(words=_words(summary_tokens),
                                              previous=PREVIOUS_SUMMARY_RULE if previous_summary else "")


def turn_prefix_prompt(conversation: str, summary_tokens: int = 512) -> str:
    return (f"<conversation>\n{conversation}\n</conversation>\n\n"
            + TURN_PREFIX_SUMMARIZATION_PROMPT.format(words=_words(summary_tokens)))


@dataclass
class Preparation:
    first_kept: int
    to_summarize: list[dict[str, Any]]
    turn_prefix: list[dict[str, Any]]
    is_split: bool
    previous_summary: str | None
    anchor: str | None
    tokens_before: int


def _conversation_only(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Harness digests are regenerated on the new head; the old head is passed as previous_summary.
    return [m for m in messages if m.get("_kind") not in (COMPACTION_KIND, DIGEST_KIND)]


def prepare(messages: list[dict[str, Any]], keep_recent_tokens: int, previous_summary: str | None,
            tokens_before: int) -> Preparation | None:
    """Upstream prepareCompaction. ``messages[0]`` is the previous compaction head, if any."""
    start = 1 if messages and messages[0].get("_kind") == COMPACTION_KIND else 0
    cut, turn_start, split = find_cut_point(messages, start, keep_recent_tokens)
    history_end = turn_start if split else cut
    to_summarize = _conversation_only(messages[start:history_end])
    prefix = _conversation_only(messages[turn_start:cut]) if split else []
    if not to_summarize and not prefix and not previous_summary:
        return None
    return Preparation(first_kept=cut, to_summarize=to_summarize, turn_prefix=prefix, is_split=split,
                       previous_summary=previous_summary, anchor=recent_state_anchor(messages[cut:]),
                       tokens_before=tokens_before)


def _complete(llm: Any, prompt_for: Any, conversation: str, max_tokens: int, overflow: type[Exception]) -> str:
    """One summarizer call. On a context overflow, drop the oldest half of the conversation text and retry."""
    for _ in range(4):
        try:
            reply = llm.chat([{"role": "system", "content": SUMMARIZATION_SYSTEM_PROMPT},
                              {"role": "user", "content": prompt_for(conversation)}],
                             tools=None, max_tokens=max_tokens, thinking=False)
        except overflow:
            conversation = "[Earlier conversation omitted to fit the model context.]\n" + \
                conversation[len(conversation) // 2:]
            continue
        text = re.sub(r"<think>.*?</think>", "", reply.content or "", flags=re.DOTALL).strip()
        if not text:
            raise RuntimeError("the summarizer returned no text")
        return text
    raise RuntimeError("the summarizer request does not fit the context window")


def summarize(llm: Any, prep: Preparation, summary_tokens: int, overflow: type[Exception]) -> str:
    """Upstream compact(): the history note, plus a turn-prefix note for a split turn, at most ``summary_tokens`` in
    all (a quarter of it for the turn prefix). With nothing new to summarize, the previous note stays as it is."""
    prefix_budget = summary_tokens // 4 if prep.is_split and prep.turn_prefix else 0
    budget = summary_tokens - prefix_budget
    if prep.to_summarize:
        history = _complete(llm, lambda c: history_summary_prompt(c, prep.previous_summary, prep.anchor, budget),
                            serialize_conversation(prep.to_summarize), budget, overflow)
    else:
        history = prep.previous_summary or "No prior history."
    if not prefix_budget:
        return history
    prefix = _complete(llm, lambda c: turn_prefix_prompt(c, prefix_budget), serialize_conversation(prep.turn_prefix),
                       prefix_budget, overflow)
    return f"{history}\n\n---\n\n**Turn Context (split turn):**\n\n{prefix}"


def head_message(summary: str, digest_block: str) -> dict[str, Any]:
    """The compaction head as the model sees it (upstream convertToLlm for a compactionSummary)."""
    return {"role": "user", "_kind": COMPACTION_KIND,
            "content": digest_block + COMPACTION_SUMMARY_PREFIX + summary + COMPACTION_SUMMARY_SUFFIX}

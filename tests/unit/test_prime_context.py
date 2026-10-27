# --- path shim: these suites live in tests/ but import from the repo root ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_PRIME = _os.environ.get("PRIME_DIR", "Sarbloh")   # PRIME_DIR=Sarbloh-Experimentation tests the copy
for _p in (_ROOT, _os.path.join(_ROOT, _PRIME)):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end shim ---
"""E008 context builder: the memory block goes right after the system prompt, in the agreed order, with empty blocks
labelled; only the newest image survives (older ones become a stub; all of them with images off); consecutive user
messages are joined, keeping image parts; nothing private reaches the server; the transcript copy has no PNG bytes;
and the memory block is outside ``self.messages``, so compaction cannot summarise it.
On Sarbloh-Experimentation the memory is joined to the front of the newest user message instead (so the prompt before
it stays a prefix of the next one for the server's prefix cache), and the current chain keeps its thinking.
"""
import json
import sys

from harness.agent import compaction, context
try:
    from harness.agent.vision import image_message
except ImportError:   # E116 copy: perception.py builds the message from the text and a PNG
    from harness.agent import perception as _P

    def image_message(g, text, step):
        return _P.image_message(text, _P.render(g, 4), step)

FAILS, CHECKS = [], [0]


def ok(name, cond, detail=""):
    CHECKS[0] += 1
    print(("PASS " if cond else "FAIL ") + name + (f"  {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


g = [[0] * 8 for _ in range(8)]
blocks = {"goal": "reach the door", "plan": "go up", "skills": "### planning\nx", "hypotheses": "h1 [proposed] a",
          "findings": "", "lessons": "- [rule] b", "questions": "- why?"}
pin = context.pinned_message(blocks, "game x | level 1 of 3")
txt = pin["content"]
order = [txt.index(f"## {context.TITLES[k]}") for k in ("goal", "plan", "skills", "hypotheses", "lessons", "questions")]
ok("pinned_order", txt.startswith("[memory]") and txt.index("## Game") < order[0] and order == sorted(order))
ok("pinned_skips_empty_findings", context.TITLES["findings"] not in txt)
empty = context.pinned_message({}, "s")["content"]
ok("pinned_empty_labels", "(none yet: write one" in empty and "## Plan" in empty and "## Lessons" not in empty)

msgs = [
    {"role": "user", "content": "task"},
    image_message(g, "state 0", 0),
    {"role": "assistant", "content": "", "tool_calls": [{"id": "a", "type": "function",
                                                          "function": {"name": "act", "arguments": "{}"}}]},
    {"role": "tool", "tool_call_id": "a", "content": "act: 1 of 1"},
    image_message(g, "state 1", 1),
    {"role": "assistant", "content": "", "tool_calls": [{"id": "b", "type": "function",
                                                          "function": {"name": "act", "arguments": "{}"}}]},
    {"role": "tool", "tool_call_id": "b", "content": "act: 1 of 1", "_error": True},
    image_message(g, "state 2", 2),
]
wire = context.build("SYS", msgs, pin, vision=True)
imgs = [p for m in wire if isinstance(m["content"], list) for p in m["content"] if p.get("type") == "image_url"]
first = wire[1]["content"]
_probe = context.build("S", [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
                             {"role": "user", "content": "c"}], {"role": "user", "content": "[memory] m"})
MEMORY_LAST = _probe[-1]["content"].startswith("[memory]")   # the memory goes with the newest user message
if MEMORY_LAST:
    newest = wire[-1]["content"]
    ok("memory_joined_to_newest_state", json.dumps(wire).count("[memory]") == 1 and newest[0]["text"].startswith(
        "[memory]") and newest[0]["text"].endswith("state 2") and newest[-1]["type"] == "image_url", str(newest)[:300])
    ok("system_then_task_then_first_state", wire[0] == {"role": "system", "content": "SYS"}
       and first[0]["text"] == "task\n\nstate 0" and len(first) == 2 and "dropped" in first[1]["text"], str(first))
    # The chain after the newest user message: the memory stays before it, the chain keeps its thinking, and the
    # request before a tool call is a prefix of the request after it (what the server's prefix cache needs).
    chain = [{"role": "assistant", "content": "", "_reasoning": "think",
              "tool_calls": [{"id": "c", "type": "function", "function": {"name": "ipython", "arguments": "{}"}}]},
             {"role": "tool", "tool_call_id": "c", "content": "out"}]
    longer = context.build("SYS", msgs + chain, pin, vision=True)
    ok("chain_after_memory_keeps_thinking", longer[-3] == wire[-1] and longer[-2].get("reasoning_content") == "think")
    ok("request_is_prefix_of_next_in_chain", longer[:len(wire)] == wire)
    nxt = context.build("SYS", msgs + chain + [image_message(g, "state 3", 3)], pin, vision=True)
    ok("new_state_moves_memory_keeps_older_prefix", nxt[:len(wire) - 1] == wire[:-1]
       and json.dumps(nxt).count("[memory]") == 1 and nxt[-1]["content"][0]["text"].startswith("[memory]")
       and "reasoning_content" not in nxt[-3], str(nxt[-3])[:300])
else:
    ok("system_then_memory", wire[0] == {"role": "system", "content": "SYS"} and wire[1]["role"] == "user"
       and isinstance(first, list) and first[0]["text"].startswith("[memory]"))
    ok("memory_then_task_then_first_state", first[0]["text"].endswith("task\n\nstate 0") and len(first) == 2
       and "dropped" in first[1]["text"], str(first)[-200:])
ok("one_image_newest", len(imgs) == 1 and wire[-1]["content"][-1]["type"] == "image_url")
stubs = json.dumps(wire).count("dropped: only the newest image is kept")
ok("older_images_stubbed", stubs == 2, str(stubs))
ok("no_private_keys", all(not k.startswith("_") for m in wire for k in m))
ok("roles_alternate_users", all(not (a["role"] == b["role"] == "user") for a, b in zip(wire, wire[1:])))
off = context.build("SYS", msgs, pin, vision=False)
ok("vision_off_no_images", "image_url" not in json.dumps(off) and json.dumps(off).count("images are off") == 3)
ok("input_not_mutated", msgs[1]["content"][1]["type"] == "image_url")

merged = context.merge_users([{"role": "user", "content": "a"}, image_message(g, "b", 5), {"role": "user",
                                                                                           "content": "c"}])
ok("merge_keeps_image_part", len(merged) == 1 and [p["type"] for p in merged[0]["content"]]
   == ["text", "image_url", "text"] and merged[0]["content"][0]["text"] == "a\n\nb")
ok("merge_strings_stay_string", context.merge_users([{"role": "user", "content": "a"}, {"role": "user",
                                                                                        "content": "b"}])
   == [{"role": "user", "content": "a\n\nb"}])
log = context.loggable(msgs[4])
ok("loggable_no_bytes", "base64" not in json.dumps(log) and "[png of step 1]" in json.dumps(log)
   and "base64" in json.dumps(msgs[4]))

# Compaction sees only the conversation: the memory message is built per request and is never in the list it cuts.
ok("compaction_text_reads_parts", compaction._text(msgs[1]) == "state 0")
prep = compaction.prepare(msgs, keep_recent_tokens=5, previous_summary=None, tokens_before=10_000)
ok("compaction_never_sees_memory", prep is not None and all("[memory]" not in compaction._text(m)
                                                            for m in prep.to_summarize + msgs[prep.first_kept:]))
ok("compaction_cuts_at_user_or_assistant", msgs[prep.first_kept]["role"] in ("user", "assistant"))

print("\n================ SUMMARY ================")
print(f"CHECKS: {CHECKS[0]}")
print(f"HARD FAILS: {len(FAILS)} -> {FAILS}")
sys.exit(1 if FAILS else 0)

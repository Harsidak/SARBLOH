# --- path shim: these suites live in tests/ but import from the repo root ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
for _p in (_ROOT, _os.path.join(_ROOT, "Alignment")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end shim ---
"""Alignment/lora_sft.py without a GPU: validation, loss masks on the real Qwen3.5 chat template, and one end-to-end run
on a tiny random qwen3_5 model built from the real config (CPU, fp32, ~1 min).

Needs torch, transformers, peft and the Qwen3.5 tokenizer files (downloaded once, ~20 MB; no weights). The repo env
has no torch by default, so run it with:

    uv run --with torch --with transformers --with peft --with matplotlib python tests/unit/test_alignment_sft.py

Without those packages the suite prints SKIP and reports zero checks.
"""
import json
import shutil
import tempfile
from pathlib import Path

FAILS, CHECKS = [], [0]


def ok(name, cond, detail=""):
    CHECKS[0] += 1
    print(("PASS " if cond else "FAIL ") + name + (f"  {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


try:
    import torch  # noqa: F401
    import transformers  # noqa: F401
    import peft  # noqa: F401
except ImportError as exc:
    print(f"SKIP: {exc}. Run with: uv run --with torch --with transformers --with peft python {__file__}")
    print("CHECKS: 0")
    print("HARD FAILS: 0 -> []")
    raise SystemExit(0)

import lora_sft as L  # noqa: E402

DATA = Path(_ROOT) / "Alignment" / "fixtures" / "smoke3.jsonl"
ROWS = [json.loads(x) for x in DATA.read_text(encoding="utf-8").splitlines() if x.strip()]

from huggingface_hub import snapshot_download  # noqa: E402

TOKDIR = snapshot_download("Qwen/Qwen3.5-2B", allow_patterns=["*.json", "*.jinja", "merges.txt", "vocab.json"],
                           ignore_patterns=["model.safetensors*"])
cfg = L.make_config({"data_path": str(DATA)})
tok = L.load_tokenizer(TOKDIR, cfg)


def targets(e):
    segs, cur = [], []
    for i, lab in zip(e.input_ids, e.labels):
        if lab != L.IGNORE:
            cur.append(i)
        elif cur:
            segs.append(cur)
            cur = []
    if cur:
        segs.append(cur)
    return [tok.decode(s) for s in segs]


# --- 1. config and validation ---------------------------------------------------------------------------------
try:
    L.make_config({"lora_rank": 8})
    ok("unknown config key rejected", False)
except KeyError:
    ok("unknown config key rejected", True)

st = L.Stats()
convs = L.load_conversations(DATA, cfg, st, None)
ok("3 smoke conversations load", len(convs) == 3 and not st.errors, str(st.errors))
ok("prime row: system prepended", convs[1]["messages"][0]["role"] == "system")
ok("prime row: JSON-string arguments parsed to dict",
   isinstance(convs[1]["messages"][2]["tool_calls"][0]["function"]["arguments"], dict))
ok("image part replaced by placeholder", st.images_replaced == 1 and "[image]" in convs[1]["messages"][1]["content"])


def rejects(mutate, why):
    row = json.loads(json.dumps(ROWS[0]))
    mutate(row)
    try:
        L.normalize_record(row, 0, cfg, L.Stats())
        ok(f"rejects {why}", False)
    except L.DataError:
        ok(f"rejects {why}", True)


rejects(lambda r: r["messages"][2]["tool_calls"][0]["function"].update(name="shell"), "unknown tool")
rejects(lambda r: r["messages"][2]["tool_calls"][0]["function"].update(arguments="{not json"), "bad JSON arguments")
rejects(lambda r: r["messages"][4]["tool_calls"][0]["function"]["arguments"].pop("expect"), "missing required arg")
rejects(lambda r: r["messages"].insert(1, {"role": "tool", "content": "x"}), "tool result without a call")
rejects(lambda r: r["messages"].insert(2, {"role": "system", "content": "x"}), "late system message")
rejects(lambda r: r.update(messages=[m for m in r["messages"] if m["role"] != "assistant"]), "no assistant turn")
try:
    L.normalize_record(ROWS[1], 0, L.make_config({"images": "error"}), L.Stats())
    ok("images='error' rejects image parts", False)
except L.DataError:
    ok("images='error' rejects image parts", True)

# --- 2. loss masks on the real template -------------------------------------------------------------------------
st = L.Stats()
ex = {c["id"]: L.build_examples(tok, c, cfg, st) for c in convs}
ok("conversation 1 renders whole (prefix-stable)", [e.source for e in ex["smoke-ls20-push"]] == ["smoke-ls20-push#whole"])
ok("conversation 3 falls back to per-turn (second user turn)",
   [e.source for e in ex["smoke-vc33-nudge"]] == ["smoke-vc33-nudge#t2", "smoke-vc33-nudge#t5"])
t1 = targets(ex["smoke-ls20-push"][0])
ok("one target span per assistant turn", len(t1) == 4, str(len(t1)))
ok("every target ends with <|im_end|>", all(t.rstrip("\n").endswith("<|im_end|>") for t in t1))
ok("reasoning is trained, opening <think> is prompt",
   t1[0].startswith("First look at the objects") and "</think>" in t1[0] and "<think>" not in t1[0])
ok("no user/tool text in targets", not any("<tool_response>" in t or "<|im_start|>" in t for t in t1))
ok("tool call rendered in target", "<function=act>" in t1[1] and "<parameter=expect>" in t1[1])
ok("no boundary straddles / tokenisation mismatches", st.boundary_straddles == 0 and st.tokenization_mismatches == 0,
   f"{st.boundary_straddles} {st.tokenization_mismatches}")

# turns=last and train=false
st2 = L.Stats()
last = L.build_examples(tok, convs[0], L.make_config({"turns": "last"}), st2)
ok("turns='last' trains only the final turn", len(targets(last[0])) == 1 and "five more steps" in targets(last[0])[0])
c_masked = json.loads(json.dumps(convs[0]))
c_masked["messages"][2]["train"] = False
ok("train=false keeps the turn as context only", len(targets(L.build_examples(tok, c_masked, cfg, L.Stats())[0])) == 3)

# truncation keeps the targets
st3 = L.Stats()
short = L.build_examples(tok, convs[2], L.make_config({"max_seq_len": 200}), st3)
ok("over-long examples lose context from the left, never targets",
   st3.left_truncated >= 1 and all(e.n_target > 0 and len(e.input_ids) <= 200 for e in short), str(L.asdict(st3)))
st4 = L.Stats()
L.build_examples(tok, convs[0], L.make_config({"max_seq_len": 64}), st4)
ok("targets longer than max_seq_len are dropped and counted", st4.dropped_too_long >= 1)

# generation prompt is the masked prefix
prompt, ref = L.generation_prompt(tok, convs[0], cfg, L.Stats())
ok("generation prompt ends in the open think block", prompt.endswith("<|im_start|>assistant\n<think>\n"))
ok("generation reference is the last target", ref.startswith("The block is at x=4"))
ok("tool_names_in parses Qwen XML and Hermes JSON",
   L.tool_names_in(ref) == ["act"] and L.tool_names_in('<tool_call>\n{"name": "remember", "arguments": {}}\n</tool_call>')
   == ["remember"])

class _GC:  # Qwen3.5's generation_config eos; the tokenizer's own eos is <|endoftext|>
    eos_token_id = None
_stops = L.stop_token_ids(type("M", (), {"generation_config": _GC()})(), tok)
ok("generation stops at <|im_end|> even when the tokenizer eos differs",
   tok.convert_tokens_to_ids("<|im_end|>") in _stops, str(_stops))

# Kaggle ships torchao 0.10, for which peft's is_torchao_available() raises. Simulate it, so the e2e below runs the
# same way it does on Kaggle.
import peft.import_utils as _piu
import peft.tuners.lora.torchao as _plt
def _old_torchao():
    raise ImportError("Found an incompatible version of torchao. Found version 0.10.0")
_piu.is_torchao_available = _plt.is_torchao_available = _old_torchao
L.guard_peft_torchao(log=lambda *_: None)
ok("incompatible torchao is treated as unavailable",
   _piu.is_torchao_available() is False and _plt.is_torchao_available() is False)

# --- 3. end to end on a tiny random qwen3_5 --------------------------------------------------------------------
work = Path(tempfile.mkdtemp(prefix="sft_test_"))
try:
    from transformers import AutoConfig
    mc = AutoConfig.from_pretrained(TOKDIR)
    t = mc.text_config
    t.hidden_size, t.intermediate_size, t.num_hidden_layers = 64, 128, 4
    t.layer_types = ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
    t.num_attention_heads, t.num_key_value_heads, t.head_dim = 2, 1, 32
    t.linear_num_key_heads, t.linear_num_value_heads, t.linear_key_head_dim, t.linear_value_head_dim = 2, 2, 16, 16
    v = mc.vision_config
    v.depth, v.hidden_size, v.intermediate_size, v.num_heads, v.out_hidden_size = 1, 32, 64, 2, 64
    import transformers
    torch.manual_seed(0)
    tiny = getattr(transformers, mc.architectures[0])(mc)
    mdir = work / "tiny-qwen3_5"
    tiny.save_pretrained(mdir)
    for f in Path(TOKDIR).iterdir():
        if f.is_file() and f.name != "config.json":
            shutil.copy(f, mdir / f.name)
    del tiny
    card = L.run({"model": str(mdir), "data_path": str(DATA), "output_dir": str(work / "out"), "max_steps": 30,
                  "lr": 5e-3, "warmup_ratio": 0.0, "lora_r": 8, "lora_alpha": 16, "gen_samples": 1,
                  # a random 64-wide model with a frozen 248k-vocab head moves slowly: this checks that the loss
                  # falls at all; the real-model bar (50% on Qwen3.5-2B) is in the Kaggle smoke notebook's CONFIG
                  "gen_max_new_tokens": 8, "min_loss_drop": 0.02, "gradient_checkpointing": True})
    ok("e2e verdict PASS", card["verdict"] == "PASS",
       json.dumps({k: v for k, v in card["checks"].items() if not v["pass"]}) + (card.get("error") or "")[-1500:])
    ok("e2e loaded the declared class", card.get("load", {}).get("class") == mc.architectures[0])
    ok("e2e LoRA only outside the vision tower",
       card.get("lora", {}).get("target_layers", 0) > 0 and not card["lora"]["trainable_non_lora"])
    ok("e2e target-only logits (chunked LM head)", (card.get("chunked_loss") or {}).get("ok") is True,
       str(card.get("chunked_loss")))
    ok("e2e chunked loss check recorded", "chunked_loss" in card.get("checks", {}), str(card.get("checks", {}).keys()))
    for f in ("result_card.json", "result_card.md", "train.log", "adapter/adapter_config.json", "loss.png"):
        ok(f"e2e wrote {f}", (work / "out" / f).exists())
    ok("e2e generations record whether they stopped", all("stopped" in g for g in card.get("generation_after", [])))
    ok("e2e reloaded adapter reproduces the eval loss", card["checks"].get("reload_matches", {}).get("pass") is True)
    bad = L.run({"model": str(mdir), "data_path": str(work / "missing.jsonl"), "output_dir": str(work / "bad")})
    ok("a failing run still writes a FAIL card", bad["verdict"] == "FAIL" and (work / "bad" / "result_card.md").exists())
finally:
    shutil.rmtree(work, ignore_errors=True)

print(f"CHECKS: {CHECKS[0]}")
print(f"HARD FAILS: {len(FAILS)} -> {FAILS}")
raise SystemExit(1 if FAILS else 0)

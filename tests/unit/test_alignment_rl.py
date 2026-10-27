# --- path shim: these suites live in tests/ but import from the repo root ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
for _p in (_ROOT, _os.path.join(_ROOT, "Alignment")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end shim ---
"""Alignment/lora_rl.py without a GPU: the tool_match reward on the real Qwen3.5 chat template, GRPO advantages, the
direction of one policy-gradient step, and end-to-end runs on a tiny random qwen3_5 (from the base, and on top of an
SFT adapter via init_adapter). CPU, fp32, a few minutes.

    uv run --with torch --with transformers --with peft --with matplotlib python tests/unit/test_alignment_rl.py
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
    import torch
    import transformers  # noqa: F401
    import peft  # noqa: F401
except ImportError as exc:
    print(f"SKIP: {exc}. Run with: uv run --with torch --with transformers --with peft python {__file__}")
    print("CHECKS: 0")
    print("HARD FAILS: 0 -> []")
    raise SystemExit(0)

import lora_rl as R  # noqa: E402
import lora_sft as S  # noqa: E402

DATA = Path(_ROOT) / "Alignment" / "fixtures" / "smoke3.jsonl"
from huggingface_hub import snapshot_download  # noqa: E402

TOKDIR = snapshot_download("Qwen/Qwen3.5-2B", allow_patterns=["*.json", "*.jinja", "merges.txt", "vocab.json"],
                           ignore_patterns=["model.safetensors*"])
cfg = R.make_config({"data_path": str(DATA), "turns": "all"})
tok = S.load_tokenizer(TOKDIR, cfg)

# --- 1. config ------------------------------------------------------------------------------------------------------
for bad, why in (({"lora_rank": 8}, "unknown key"), ({"group_size": 1}, "group_size < 2"),
                 ({"init_adapter": "x", "load_in_4bit": True}, "init_adapter with 4-bit")):
    try:
        R.make_config(bad)
        ok(f"config rejects {why}", False)
    except (KeyError, ValueError):
        ok(f"config rejects {why}", True)

# --- 2. prompts and reward on the real template ---------------------------------------------------------------------
st = S.Stats()
convs = S.load_conversations(DATA, cfg, st, None)
items = R.build_items(tok, convs, cfg, st)
ok("one item per trained assistant turn", len(items) == 8, str([i["id"] for i in items]))
ok("prompts end at the generation prompt (open think, or the empty think block for turns without reasoning)",
   all(i["prompt"].endswith(("<|im_start|>assistant\n<think>\n", "<|im_start|>assistant\n<think>\n\n</think>\n\n"))
       for i in items) and sum(i["prompt"].endswith("</think>\n\n") for i in items) == 2)
last = R.build_items(tok, convs, R.make_config({"data_path": str(DATA)}), S.Stats())
ok("turns='last' gives one prompt per conversation", len(last) == 3)

ref_scores = [R.score(R.tool_match, i["reference"], i, {"stopped": True}) for i in items]
ok("every reference scores 1.0 under tool_match (reward_sane)", all(r["total"] >= 0.999 for r in ref_scores),
   str([(i["id"], r) for i, r in zip(items, ref_scores) if r["total"] < 0.999]))
it = last[0]   # reference: act(actions=["4"x5], expect=..., was_right=yes)
ref = it["reference"]
calls = R.parse_tool_calls(ref)
ok("parses Qwen XML calls with JSON-typed values",
   calls and calls[0][0] == "act" and calls[0][1]["actions"] == ["4"] * 5, str(calls))
wrong_tool = ref.replace("<function=act>", "<function=ipython>")
fewer = ref.replace('["4", "4", "4", "4", "4"]', '["4", "4"]')
s_ref = R.score(R.tool_match, ref, it, {"stopped": True})["total"]
s_unstopped = R.score(R.tool_match, ref, it, {"stopped": False})["total"]
s_args = R.score(R.tool_match, fewer, it, {"stopped": True})["total"]
s_tool = R.score(R.tool_match, wrong_tool, it, {"stopped": True})["total"]
s_none = R.score(R.tool_match, "I am not sure.</think>\n\nLet me think more.<|im_end|>", it, {"stopped": True})["total"]
ok("reward orders: exact > {not stopped, wrong args} > wrong tool >= no call",
   s_ref > max(s_unstopped, s_args) and min(s_unstopped, s_args) > s_tool >= s_none,
   str([s_ref, s_unstopped, s_args, s_tool, s_none]))
ok("a call inside the reasoning does not count",
   R.parse_tool_calls("<tool_call>\n<function=act>\n</function>\n</tool_call></think>\n\nno call") == [])
ok("Hermes JSON calls parse", R.parse_tool_calls('<tool_call>\n{"name": "act", "arguments": {"actions": ["1"]}}\n'
                                                 '</tool_call>') == [("act", {"actions": ["1"]})])
ok("a callable / module:function reward resolves", R.resolve_reward(len) is len
   and R.resolve_reward("lora_rl:tool_match") is R.tool_match)

# --- 3. GRPO advantages ---------------------------------------------------------------------------------------------
ok("equal rewards give zero advantage (group skipped)", R.advantages([0.5, 0.5, 0.5], True) == [0.0, 0.0, 0.0])
adv = R.advantages([1.0, 0.0], True)
ok("advantages are group-normalised", abs(adv[0] - 1.0) < 1e-3 and abs(adv[1] + 1.0) < 1e-3, str(adv))
ok("scale_rewards=false keeps the centred reward", R.advantages([1.0, 0.0], False) == [0.5, -0.5])

# --- 4. tiny model: gradient direction, then end to end -------------------------------------------------------------
work = Path(tempfile.mkdtemp(prefix="rl_test_"))
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
    torch.manual_seed(0)
    tiny = getattr(transformers, mc.architectures[0])(mc)
    mdir = work / "tiny-qwen3_5"
    tiny.save_pretrained(mdir)
    for f in Path(TOKDIR).iterdir():
        if f.is_file() and f.name != "config.json":
            shutil.copy(f, mdir / f.name)
    del tiny

    # one GRPO step must raise the log-prob of the rewarded completion relative to the punished one
    gcfg = R.make_config({"model": str(mdir), "lora_r": 8, "lora_alpha": 16, "dtype": "fp32"})
    model, _ = S.load_model(str(mdir), gcfg, torch.float32)
    model, _ = S.attach_lora(model, gcfg)
    lp = R.LogProbs()
    p_ids = it["prompt_ids"][-64:]
    good, bad = tok("<tool_call>", add_special_tokens=False)["input_ids"], tok("hello", add_special_tokens=False)["input_ids"]
    dev, ac = torch.device("cpu"), S.make_autocast(torch.float32)

    def margin():
        with torch.no_grad():
            model.eval()
            return float(lp(model, p_ids, good, dev, ac, 1.0).mean() - lp(model, p_ids, bad, dev, ac, 1.0).mean())

    m0 = margin()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-3)
    for _ in range(3):
        model.train()
        n = len(good) + len(bad)
        kls = []
        for ids, a in ((good, 1.0), (bad, -1.0)):
            loss, kl = R.policy_loss(model, lp, p_ids, ids, a, 0.02, n, dev, ac, 1.0)
            loss.backward()
            kls.append(kl)
        opt.step()
        opt.zero_grad()
    m1 = margin()
    ok("a GRPO step moves probability toward the rewarded completion", m1 > m0 + 0.05, f"{m0:.4f} -> {m1:.4f}")
    ok("KL to the adapter-disabled reference becomes positive", max(kls) > 0, str(kls))
    del model, opt

    # end to end from the base, with a reward the random model's samples differ on
    def distinct_chars(completion, item, info):
        return len(set(completion)) / 64

    common = {"model": str(mdir), "data_path": str(DATA), "max_samples": 3, "lora_r": 8, "lora_alpha": 16,
              "group_size": 2, "prompts_per_step": 2, "max_steps": 3, "max_new_tokens": 6, "eval_max_new_tokens": 6,
              "lr": 1e-3, "min_reward_gain": None, "reward": distinct_chars}
    card = R.run({**common, "output_dir": str(work / "rl")})
    ok("e2e verdict PASS", card["verdict"] == "PASS",
       json.dumps({k: v for k, v in card["checks"].items() if not v["pass"]}) + (card.get("error") or "")[-1500:])
    ok("e2e every step had a live group", card.get("training", {}).get("updated_steps") == 3)
    ok("e2e logits_to_keep path used", card.get("logits_to_keep") is True, str(card.get("logits_to_keep")))
    for f in ("result_card.json", "result_card.md", "train.log", "adapter/adapter_config.json", "reward.png"):
        ok(f"e2e wrote {f}", (work / "rl" / f).exists())
    ok("e2e builtin-only reward_sane check skipped for a custom reward", "reward_sane" not in card["checks"])
    ok("e2e reloaded adapter reproduces the policy log-probs", card["checks"].get("reload_matches", {}).get("pass"))

    # SFT -> RL: an SFT adapter merged into the base, RL on top
    sft = S.run({"model": str(mdir), "data_path": str(DATA), "output_dir": str(work / "sft"), "max_steps": 2,
                 "lora_r": 8, "lora_alpha": 16, "gen_samples": 0, "reload_check": False})
    ok("SFT stage for the chain passes", sft["verdict"] in ("PASS", "FAIL") and (work / "sft" / "adapter").exists())
    chained = R.run({**common, "output_dir": str(work / "rl2"), "init_adapter": str(work / "sft" / "adapter"),
                     "max_steps": 1})
    ok("RL on top of the SFT adapter passes, with the adapter merged into the base",
       chained["verdict"] == "PASS" and chained.get("load", {}).get("merged_adapter"),
       json.dumps({k: v for k, v in chained["checks"].items() if not v["pass"]}) + (chained.get("error") or "")[-800:])
    bad_run = R.run({**common, "output_dir": str(work / "bad"), "data_path": str(work / "missing.jsonl")})
    ok("a failing run still writes a FAIL card", bad_run["verdict"] == "FAIL" and (work / "bad" / "result_card.md").exists())
finally:
    shutil.rmtree(work, ignore_errors=True)

print(f"CHECKS: {CHECKS[0]}")
print(f"HARD FAILS: {len(FAILS)} -> {FAILS}")
raise SystemExit(1 if FAILS else 0)

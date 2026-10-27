# --- path shim: these suites live in tests/ but import from the repo root ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
for _p in (_ROOT, _os.path.join(_ROOT, "Alignment")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end shim ---
"""Flash-Next (qwen4_exp) long-row training in Alignment/: qwen4_exp_long.py and the lora_sft.py features the Azure run
needs (chunked loss, multi-run data selection, per-row template kwargs, never cutting a row, resumable checkpoints).

Runs on a tiny random qwen4_exp model with every block type, on CPU (fp32) unless noted; the long-sequence memory
check runs only when a GPU is visible. Needs transformers >= 5.19 (qwen4_exp) and the Flash-Next tokenizer files
(downloaded once, ~25 MB; no weights):

    uv run --with torch --with "transformers>=5.19" --with peft --with matplotlib python tests/unit/test_alignment_long.py
"""
import copy
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
    import peft  # noqa: F401
    from transformers import Qwen4ExpForConditionalGeneration  # noqa: F401
except ImportError as exc:
    print(f"SKIP: {exc}. Run with: uv run --with torch --with 'transformers>=5.19' --with peft python {__file__}")
    print("CHECKS: 0")
    print("HARD FAILS: 0 -> []")
    raise SystemExit(0)

import lora_sft as L  # noqa: E402
import qwen4_exp_long as Q  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402

SMOKE = Path(_ROOT) / "Alignment" / "fixtures" / "smoke3.jsonl"
TOKDIR = snapshot_download("Qwen/Qwen3.8-Flash-Next", allow_patterns=["*.json", "*.jinja", "merges.txt", "vocab.json",
                                                                     "tokenizer*"],
                           ignore_patterns=["model.safetensors*"])

# --- 1. sparse attention: same maths as the reference ---------------------------------------------------------
for seed, n, qc, mc in ((0, 300, 64, 128), (1, 1000, 100, 300), (2, 37, 8, 16)):
    r = Q.check_equivalence(seq_len=n, seed=seed, query_chunk=qc, moe_chunk=mc)
    ok(f"QSA path matches the reference logits (L={n}, chunk {qc})",
       r["logit_max_diff"] <= 1e-5 * max(r["logit_scale"], 1.0), json.dumps(r))
    ok(f"QSA path matches the reference attention grads (L={n})",
       r["grad_max_diff"] <= 1e-5 * max(r["grad_scale"], 1e-6) and r["grads_compared"] > 0 and not r["missing_grads"],
       json.dumps(r))

# the check is sensitive: a dense causal attention (budget larger than the row) must differ from the sparse reference
_orig_select = Q.select_tokens


def _dense_wrapper(indexer, h, cos, sin, q_chunk):
    L_ = h.shape[1]
    idx = torch.arange(L_, device=h.device)
    full = idx[None, :].expand(L_, -1)
    return torch.where(full <= idx[:, None], full, -1)[None].expand(h.shape[0], -1, -1)


Q.select_tokens = _dense_wrapper
r = Q.check_equivalence(seq_len=300, seed=0, query_chunk=64, moe_chunk=128)
Q.select_tokens = _orig_select
ok("the equivalence check is sensitive (dense attention differs from QSA)",
   r["logit_max_diff"] > 1e-4 * max(r["logit_scale"], 1.0), json.dumps(r))

# short rows (every query sees fewer blocks than the budget) are dense causal in both
r = Q.check_equivalence(seq_len=40, seed=3, query_chunk=16, moe_chunk=16)
ok("rows shorter than the budget match too", r["logit_max_diff"] <= 1e-5 * max(r["logit_scale"], 1.0), json.dumps(r))

# --- 2. chunked LM-head loss == full-logits loss (value and LoRA gradients) -------------------------------------
torch.manual_seed(0)
tiny_cfg = Q.tiny_config(vocab_size=1024, eos_token_id=1023, bos_token_id=1023)
from transformers import Qwen4ExpForConditionalGeneration  # noqa: E402

model = Qwen4ExpForConditionalGeneration(tiny_cfg).float()
lcfg = L.make_config({"lora_r": 4, "lora_alpha": 8, "gradient_checkpointing": False})
model, linfo = L.attach_lora(model, lcfg)
ok("LoRA lands on attention and DeltaNet projections only (no experts, router, indexer, PLE)",
   linfo["target_layers"] > 0 and not any(k in ("gate", "index_qk_proj", "key_proj", "value_proj")
                                          for k in linfo["target_leaf_counts"]),
   json.dumps(linfo["target_leaf_counts"]))
with torch.no_grad():
    for n, p in model.named_parameters():
        if "lora_B" in n:
            p.normal_(0, 0.02)
ids = torch.randint(0, 1024, (1, 120))
labels = torch.full_like(ids, L.IGNORE)
labels[0, 30:60] = ids[0, 30:60]
labels[0, 90:] = ids[0, 90:]
batch = {"input_ids": ids, "labels": labels, "attention_mask": torch.ones_like(ids)}
ac = L.make_autocast(torch.float32)
Q.enable(query_chunk=32, moe_chunk=64)


def loss_and_grads(fn):
    model.zero_grad(set_to_none=True)
    loss, n, hits = fn(model, batch, ac)
    loss.backward()
    return float(loss), n, hits, {k: p.grad.clone() for k, p in model.named_parameters() if p.grad is not None}


model.train()
ref = loss_and_grads(L.LossFn(None))
chk = L.LossFn(7)
got = loss_and_grads(chk)
ok("chunked loss self-check passes", chk.chunk_ok is True, str(chk.chunk_check))
ok("chunked loss equals full-logits loss", abs(ref[0] - got[0]) <= 1e-4 * abs(ref[0]), f"{ref[0]} vs {got[0]}")
ok("chunked loss counts the same targets and hits", ref[1] == got[1] and ref[2] == got[2],
   f"{ref[1:3]} vs {got[1:3]}")
gd = max(float((ref[3][k] - got[3][k]).abs().max()) for k in ref[3])
gs = max(float(ref[3][k].abs().max()) for k in ref[3])
ok("chunked loss gives the same LoRA gradients", gd <= 1e-4 * gs and set(ref[3]) == set(got[3]), f"{gd} vs {gs}")
Q.disable()
del model

# --- 3. data: several runs, best run per level, row kwargs, no cutting, empty context turns ----------------------
work = Path(tempfile.mkdtemp(prefix="long_test_"))
try:
    base = json.loads(SMOKE.read_text(encoding="utf-8").splitlines()[0])

    def row(game, level, cleared, rhae, actions, effort=None, empty_ctx=False):
        r = copy.deepcopy(base)
        r.update(game_id=game, level=level, level_cleared=cleared, rhae=rhae, actions=actions)
        r.pop("id", None)
        if effort:
            r["template_kwargs"] = {"reasoning_effort": effort}
        if empty_ctx:
            i = next(k for k, m in enumerate(r["messages"]) if m["role"] == "assistant")
            r["messages"][i:i] = [{"role": "assistant", "content": "", "train": False},
                                  {"role": "user", "content": "go on"}]
        return r

    runs = {
        "run_a": [row("g1", 0, True, 0.5, 20, "low"), row("g1", 1, True, 1.0, 9), row("g2", 0, False, 0.0, 30)],
        "run_b": [row("g1", 0, True, 0.9, 12, "low", empty_ctx=True), row("g1", 1, True, 0.4, 15),
                  row("g2", 0, False, 0.0, 25)],
    }
    for name, rows in runs.items():
        (work / name).mkdir()
        (work / name / "sft_levels.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    glob_spec = str(work / "run_*" / "sft_levels.jsonl")
    cfg = L.make_config({"data_path": glob_spec, "select": "best_run"})
    st = L.Stats()
    convs = L.load_conversations(cfg["data_path"], cfg, st, None)
    ids_ = sorted(c["id"] for c in convs)
    ok("a glob reads every run file", st.input_files == 2, str(st.input_files))
    ok("best_run keeps g1/L0 from run_b, g1/L1 from run_a, and every uncleared row",
       ids_ == sorted(["run_b/g1/L0", "run_a/g1/L1", "run_a/g2/L0", "run_b/g2/L0"]) and st.dropped_not_best_run == 2,
       f"{ids_} dropped {st.dropped_not_best_run}")
    ok("an empty assistant turn kept as context (train false) is accepted", st.skipped_conversations == 0,
       str(st.errors[:2]))
    tok = L.load_tokenizer(TOKDIR, cfg)
    g1 = next(c for c in convs if c["id"] == "run_b/g1/L0")
    ok("the row's template kwargs are carried", g1.get("template_kwargs") == {"reasoning_effort": "low"},
       str(g1.get("template_kwargs")))
    ex = L.build_examples(tok, g1, cfg, L.Stats())
    text = tok.decode(ex[0].input_ids)
    ok("the row's reasoning effort reaches the rendered prompt", "Reasoning effort is set to low" in text, text[:300])
    st2 = L.Stats()
    exs = [e for c in convs for e in L.build_examples(tok, c, cfg, st2)]
    ok("Flash-Next template: whole-conversation renders, clean masks",
       st2.whole_renders == len(convs) and st2.per_turn_renders == 0 and st2.boundary_straddles == 0,
       json.dumps({k: v for k, v in vars(st2).items() if k != "errors"}))
    n_max = max(len(e.input_ids) for e in exs)
    st3 = L.Stats()
    cut = L.make_config({"data_path": glob_spec, "max_seq_len": n_max - 1, "overlong": "drop"})
    kept = [e for c in convs for e in L.build_examples(tok, c, cut, st3)]
    ok("overlong drop never cuts: longer rows are skipped and counted",
       st3.dropped_too_long >= 1 and st3.left_truncated == 0 and all(len(e.input_ids) <= n_max - 1 for e in kept),
       f"dropped {st3.dropped_too_long} truncated {st3.left_truncated}")

    # examples cache: written once, reused, invalidated when a data file changes
    cache = work / "cache" / "examples.jsonl"
    ccfg = L.make_config({"data_path": glob_spec, "select": "best_run", "examples_cache": str(cache)})
    key = L.cache_key(ccfg, tok)
    L.save_examples(cache, key, exs, None, st2)
    back = L.load_examples(cache, key)
    ok("examples cache round-trips", back is not None and [e.input_ids for e in back[0]] == [e.input_ids for e in exs]
       and [e.labels for e in back[0]] == [e.labels for e in exs] and back[1] is None)
    (work / "run_a" / "sft_levels.jsonl").write_text(
        (work / "run_a" / "sft_levels.jsonl").read_text(encoding="utf-8") + "\n", encoding="utf-8")
    ok("examples cache is invalidated by a changed data file", L.load_examples(cache, L.cache_key(ccfg, tok)) is None)

    # --- 4. end to end on a tiny random Flash-Next saved with the real tokenizer ---------------------------------
    torch.manual_seed(0)
    mdir = work / "tiny-flash-next"
    tiny = Qwen4ExpForConditionalGeneration(Q.tiny_config(vocab_size=len(tok), eos_token_id=tok.convert_tokens_to_ids(
        "<|im_end|>"), bos_token_id=tok.convert_tokens_to_ids("<|im_end|>")))
    tiny.save_pretrained(mdir)
    for f in Path(TOKDIR).iterdir():
        if f.is_file() and f.name not in ("config.json",) and not f.name.endswith(".safetensors"):
            shutil.copy(f, mdir / f.name)
    del tiny
    common = {"model": str(mdir), "data_path": str(SMOKE), "lora_r": 8, "lora_alpha": 16, "lr": 5e-3,
              "warmup_ratio": 0.0, "dtype": "fp32", "loss_chunk": 64, "overlong": "drop", "max_seq_len": 131072,
              "long_query_chunk": 64, "long_moe_chunk": 256}
    card = L.run({**common, "output_dir": str(work / "e2e"), "max_steps": 12, "gen_samples": 1,
                  "gen_max_new_tokens": 4, "min_loss_drop": 0.01})
    ok("e2e tiny Flash-Next verdict PASS", card["verdict"] == "PASS",
       json.dumps({k: v for k, v in card["checks"].items() if not v["pass"]}) + (card.get("error") or "")[-1500:])
    ok("e2e used the long-context path", card.get("load", {}).get("long_context") is True, str(card.get("load")))
    ok("e2e chunked loss check passed", (card.get("chunked_loss") or {}).get("ok") is True, str(card.get("chunked_loss")))
    ok("e2e generation ran (reference path with cache)", bool(card.get("generation_after")))

    # merge: shard-streamed merge of the trained adapter == base + adapter
    import merge_lora
    rep = merge_lora.merge(mdir, work / "e2e" / "adapter", work / "merged")
    ok("merge places every adapter target", rep["merged"] == rep["targets"] > 0 and rep["max_abs_delta"] > 0,
       json.dumps(rep))
    from peft import PeftModel
    Q.disable()
    x = torch.randint(0, len(tok), (1, 50))
    with torch.no_grad():
        pm = PeftModel.from_pretrained(Qwen4ExpForConditionalGeneration.from_pretrained(mdir), work / "e2e" / "adapter")
        want = pm(input_ids=x, use_cache=False).logits
        bm = Qwen4ExpForConditionalGeneration.from_pretrained(mdir)
        base_logits = bm(input_ids=x, use_cache=False).logits
        got_m = Qwen4ExpForConditionalGeneration.from_pretrained(work / "merged")(input_ids=x, use_cache=False).logits
    dm = float((want - got_m).abs().max())
    db = float((want - base_logits).abs().max())
    ok("merged checkpoint reproduces base + adapter logits", dm <= 1e-4 * max(float(want.abs().max()), 1.0) and db > 10 * dm,
       f"merged diff {dm}, adapter effect {db}")
    del pm, bm
    rep2 = merge_lora.merge(mdir, work / "e2e" / "adapter", work / "touched", touched_only=True)
    ok("touched-only merge writes only the target shards",
       rep2["touched_shards"] == ["model.safetensors"] and (work / "touched" / "model.safetensors").exists(),
       json.dumps(rep2))
    from safetensors.torch import load_file as _lf, save_file as _sf
    qdir = work / "quantised"
    shutil.copytree(mdir, qdir)
    t = _lf(str(qdir / "model.safetensors"))
    victim = next(k for k in t if k.endswith("linear_attn.in_proj_qkv.weight"))
    t[victim[: -len(".weight")] + ".weight_scale"] = torch.ones(1)
    _sf(t, str(qdir / "model.safetensors"), metadata={"format": "pt"})
    try:
        merge_lora.merge(qdir, work / "e2e" / "adapter", work / "qout")
        refused = False
    except SystemExit as exc:
        refused = "quantised" in str(exc)
    ok("merge refuses a quantised target", refused)

    # resume: an interrupted run (checkpoint at step 2 of 4) resumed to the end == one uninterrupted run
    full = L.run({**common, "output_dir": str(work / "full"), "max_steps": 4, "gen_samples": 0, "reload_check": False})
    part = L.run({**common, "output_dir": str(work / "part"), "max_steps": 4, "gen_samples": 0, "reload_check": False,
                  "save_every": 2})
    ok("checkpoint written at save_every", (work / "part" / "checkpoint" / "loop.json").exists())
    # simulate an eviction after step 2: the checkpoint is from step 2; the adapter dir is from the finished run
    shutil.rmtree(work / "part" / "adapter")
    res = L.run({**common, "output_dir": str(work / "part"), "max_steps": 4, "gen_samples": 0, "reload_check": False,
                 "resume": True})
    ok("resume starts from the checkpoint step", res.get("training", {}).get("resumed_from") == 2,
       str(res.get("training", {}).get("resumed_from")))
    ok("resumed history is complete", len(res["training"]["history"]) == 4, str(len(res["training"]["history"])))
    la = full["training"]["history"][-1]["loss"]
    lb = res["training"]["history"][-1]["loss"]
    ok("resumed run ends where the uninterrupted run ends", abs(la - lb) <= 1e-4 * max(abs(la), 1e-6), f"{la} vs {lb}")
    from safetensors.torch import load_file
    wa = load_file(str(work / "full" / "adapter" / "adapter_model.safetensors"))
    wb = load_file(str(work / "part" / "adapter" / "adapter_model.safetensors"))
    dmax = max(float((wa[k] - wb[k]).abs().max()) for k in wa)
    ok("resumed adapter equals the uninterrupted adapter", set(wa) == set(wb) and dmax <= 1e-5, f"max diff {dmax}")
    stop = L.run({**common, "output_dir": str(work / "stop"), "max_steps": 4, "gen_samples": 0, "reload_check": False,
                  "max_train_hours": 1e-9})
    ok("max_train_hours stops cleanly and saves a checkpoint",
       stop["training"]["stopped_early"] and (work / "stop" / "checkpoint" / "loop.json").exists(),
       str({k: stop["training"].get(k) for k in ("steps", "stopped_early")}))
finally:
    shutil.rmtree(work, ignore_errors=True)

# --- 5. GPU: memory grows linearly with row length on the patched path ------------------------------------------
if torch.cuda.is_available():
    torch.manual_seed(0)
    gm = Qwen4ExpForConditionalGeneration(Q.tiny_config(vocab_size=1024, eos_token_id=1023, bos_token_id=1023,
                                                        max_position_embeddings=131072)).to("cuda", torch.bfloat16)
    gm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    gm.train()
    Q.enable(query_chunk=1024, moe_chunk=16384)
    peaks = {}
    for n in (8192, 32768):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        x = torch.randint(0, 1024, (1, n), device="cuda")
        hidden = gm.model(input_ids=x)[0]
        hidden.float().pow(2).mean().backward()
        peaks[n] = torch.cuda.max_memory_allocated() / 2 ** 20
        gm.zero_grad(set_to_none=True)
    Q.disable()
    ratio = peaks[32768] / peaks[8192]
    ok("GPU: 4x longer row -> < 6x peak memory (an L x L mask would be 16x)", ratio < 6,
       f"peaks MiB {json.dumps({k: round(v) for k, v in peaks.items()})}")
    print(f"INFO long-row peaks MiB: {json.dumps({k: round(v) for k, v in peaks.items()})}")

    # --- 6. GPU: split placement (layers over devices, n-gram table kept on CPU) == one device -----------------
    torch.manual_seed(0)
    sdir = Path(tempfile.mkdtemp(prefix="split_test_"))
    try:
        sm = Qwen4ExpForConditionalGeneration(Q.tiny_config(
            vocab_size=1024, eos_token_id=1023, bos_token_id=1023, num_hidden_layers=8, ngram_vocab_size_base=200_003,
            layer_types=["linear_attention"] * 3 + ["full_attention"] + ["linear_attention"] * 3 + ["full_attention"]))
        with torch.no_grad():
            for n, p in sm.named_parameters():
                if (p.dim() == 1 and "norm" in n) or "ngram" in n or ".ple." in n:
                    p.normal_(0, 0.1 if "norm" in n else 0.3)
        sm.save_pretrained(sdir)
        del sm
        sids = torch.randint(0, 1024, (1, 700))
        slab = torch.full_like(sids, L.IGNORE)
        slab[0, 300:] = sids[0, 300:]

        def split_run(over):
            torch.manual_seed(1)
            c = L.make_config({"model": str(sdir), "dtype": "fp32", "lora_r": 4, "lora_alpha": 8, "long_query_chunk": 128,
                               "long_moe_chunk": 256, "experts_implementation": "grouped_mm", **over})
            mm, info = L.load_model(str(sdir), c, torch.float32)
            mm, _ = L.attach_lora(mm, c)
            with torch.no_grad():
                for n, p in mm.named_parameters():
                    if "lora_B" in n:
                        p.fill_(0.01)
            dev = L.input_device(mm)
            bb = {"input_ids": sids.to(dev), "labels": slab.to(dev), "attention_mask": torch.ones_like(sids).to(dev)}
            mm.train()
            lf = L.LossFn(64)
            loss, _, _ = lf(mm, bb, L.make_autocast(torch.float32))
            loss.backward()
            grads = {k: p.grad.detach().cpu() for k, p in mm.named_parameters() if p.grad is not None}
            ngd = [str(p.device) for k, p in mm.named_parameters() if "ngram_embedding" in k]
            return float(loss), grads, info.get("split"), ngd, lf.chunk_ok

        l1, g1, _, _, _ = split_run({})
        l2, g2, plan, ngd, ck = split_run({"device_map": "split", "split_devices": ["cuda:0", "cpu"],
                                          "cpu_modules": ["ngram_embedding"]})
        gd = max(float((g1[k] - g2[k]).abs().max()) for k in g1)
        gs = max(float(v.abs().max()) for v in g1.values())
        ok("GPU: split placement keeps the n-gram table on CPU and uses both devices",
           ngd == ["cpu"] and plan and len(plan["layers_per_device"]) == 2, json.dumps({"plan": plan, "ngram": ngd}))
        ok("GPU: split placement gives the same loss and LoRA grads (TF32 conv tolerance)",
           abs(l1 - l2) <= 1e-4 * abs(l1) and gd <= 5e-3 * gs and set(g1) == set(g2) and ck,
           f"loss {l1} vs {l2}; grad diff {gd} of {gs}")
        sizes = [10] * 48
        ok("split plan: 48 equal layers over 8 devices, contiguous, every device used",
           [L.split_plan(sizes, 0, 8).count(d) for d in range(8)] == [6] * 8, str(L.split_plan(sizes, 0, 8)))
    finally:
        shutil.rmtree(sdir, ignore_errors=True)
else:
    print("SKIP GPU memory and split checks (no CUDA)")

print(f"CHECKS: {CHECKS[0]}")
print(f"HARD FAILS: {len(FAILS)} -> {FAILS}")

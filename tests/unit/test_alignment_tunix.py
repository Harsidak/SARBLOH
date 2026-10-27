# --- path shim: these suites live in tests/ but import from the repo root ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
for _p in (_ROOT, _os.path.join(_ROOT, "Alignment")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end shim ---
"""Alignment/lora_sft_tunix.py without a TPU: config checks, the MaxText model mapping on the real Qwen3.8-27B
config.json, the MaxText arguments, the batch layout (targets shifted by one, loss mask = assistant tokens) and the
batch iterator (step count, fixed order for resumes, the time limit). JAX, MaxText and Tunix are not needed.

    uv run python tests/unit/test_alignment_tunix.py
"""
import json
import tempfile
import time
from pathlib import Path

import lora_sft as L
import lora_sft_tunix as T

FAILS, CHECKS = [], [0]


def ok(name, cond, detail=""):
    CHECKS[0] += 1
    print(("PASS " if cond else "FAIL ") + name + (f"  {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


def raises(fn, exc):
    try:
        fn()
    except exc:
        return True
    return False


# config
ok("backend_default_torch", L.make_config({})["backend"] == "torch")
ok("backend_rejects_unknown", raises(lambda: L.make_config({"backend": "jax"}), ValueError))
ok("tunix_rejects_unknown_key", raises(lambda: T.tunix_config({"tunix": {"fsdpp": 8}}), KeyError))
ok("tunix_merges_defaults", T.tunix_config({"tunix": {"fsdp": 4}})["fsdp"] == 4
   and T.tunix_config({})["export_hf"] is True)
ok("run_fails_before_work_on_bad_tunix_key",
   raises(lambda: L.run({"backend": "tunix", "tunix": {"nope": 1}}), KeyError))

# MaxText model mapping on the real 27B config (config.json only, no weights)
with tempfile.TemporaryDirectory() as d:
    try:
        from huggingface_hub import hf_hub_download
        src = hf_hub_download("Qwen/Qwen3.8-27B", "config.json")
        Path(d, "config.json").write_text(Path(src).read_text(encoding="utf-8"), encoding="utf-8")
        have = True
    except Exception as exc:  # noqa: BLE001 - offline: these checks are skipped, the rest still run
        print(f"SKIP qwen 27B config: {exc}")
        have = False
    if have:
        name, over, hf = T.maxtext_model(d, T.tunix_config({}))
        ok("qwen27b_runs_as_qwen35_9b", name == "qwen3.5-9b" and hf["model_type"] == "qwen3_5")
        ok("qwen27b_dims", over.get("base_emb_dim") == 5120 and over.get("base_num_decoder_layers") == 64
           and over.get("base_num_query_heads") == 24 and over.get("base_num_kv_heads") == 4
           and over.get("base_mlp_dim") == 17408 and over.get("gdn_num_value_heads") == 48
           and over.get("inhomogeneous_layer_cycle_interval") == 4, json.dumps(over))
        ok("qwen27b_lora_mlp_only", T.lora_module_path(name, T.tunix_config({})) == T.QWEN3_5_MLP_LORA)
        args = T.mt_args(name, over, {"lora.enable_lora": True})
        ok("args_override_flag", "override_model_config=True" in args and "base_emb_dim=5120" in args
           and "lora.enable_lora=True" in args and "use_multimodal=False" in args, str(args))
    ok("explicit_name_passes", T.maxtext_model(d, T.tunix_config({"tunix": {"maxtext_model": "gemma4-31b"}}))
       [:2] == ("gemma4-31b", {}) if have else True)
    Path(d, "config.json").write_text(json.dumps({"model_type": "gemma4", "text_config": {"num_hidden_layers": 60}}))
    ok("gemma4_31b_native", T.maxtext_model(d, T.tunix_config({})) == ("gemma4-31b", {}, None))
    ok("gemma4_lora_maxtext_table", T.lora_module_path("gemma4-31b", T.tunix_config({})) == "")
    Path(d, "config.json").write_text(json.dumps({"model_type": "llama"}))
    ok("unknown_arch_raises", raises(lambda: T.maxtext_model(d, T.tunix_config({})), ValueError))
ok("args_no_override_flag_without_overrides", not any(a.startswith("override_model_config") for a in
                                                       T.mt_args("gemma4-31b", {}, {})))

# batch layout
IG = L.IGNORE
e = L.Example([10, 11, 12, 13, 14], [IG, IG, 12, 13, IG], "x", 2)
b = T.make_batch([e], 2, 8, 0)
ok("batch_shapes", all(v.shape == (2, 8) for v in b.values()) and len(b) == 6)
ok("batch_inputs", b["inputs"][0, :4].tolist() == [10, 11, 12, 13] and b["targets"][0, :4].tolist() == [11, 12, 13, 14])
ok("batch_loss_mask", b["targets_segmentation"][0].tolist() == [0, 1, 1, 0, 0, 0, 0, 0],
   str(b["targets_segmentation"][0].tolist()))
ok("batch_mask_hits_targets", [int(t) for t, m in zip(b["targets"][0], b["targets_segmentation"][0]) if m]
   == [12, 13])
ok("batch_segments_positions", b["inputs_segmentation"][0].tolist() == [1, 1, 1, 1, 0, 0, 0, 0]
   and b["inputs_position"][0, :4].tolist() == [0, 1, 2, 3])
ok("batch_padding_row_empty", b["inputs_segmentation"][1].sum() == 0 and b["targets_segmentation"][1].sum() == 0)

cache = Path(_ROOT) / "Alignment" / "data" / "cache" / "examples_authored.jsonl"
if cache.exists():
    with cache.open(encoding="utf-8") as fh:
        fh.readline()
        real = [L.Example(**json.loads(x)) for x, _ in zip(fh, range(8))]
    L_MAX = max(len(x.input_ids) for x in real)
    rb = T.make_batch(real, len(real), L_MAX, 0)
    ok("real_rows_mask_equals_targets", all(int(rb["targets_segmentation"][i].sum()) == x.n_target
                                            for i, x in enumerate(real)))
else:
    print(f"SKIP real rows: {cache} not built")

# iterator
ex = [L.Example([1, 2, 3], [IG, 2, 3], str(i), 2) for i in range(10)]
it = T.Batches(ex, 4, 8, 0, 3, 0, None, None)
got = list(it)
ok("steps_count", it.steps() == 9 and len(got) == 9)
ok("order_repeatable", all((a["inputs"] == c["inputs"]).all() for a, c in zip(got, list(T.Batches(ex, 4, 8, 0, 3, 0,
                                                                                                    None, None)))))
ok("max_steps_cap", len(list(T.Batches(ex, 4, 8, 0, 3, 0, None, 5))) == 5)
dl = T.Batches(ex, 4, 8, 0, 3, 0, time.time() - 1, None)
ok("deadline_stops", list(dl) == [] and dl.hit_deadline)

# checkpoint discovery and work dir
with tempfile.TemporaryDirectory() as d:
    root = Path(d) / "checkpoint"
    ok("no_checkpoint", T.latest_lora_checkpoint(root) is None)
    for s in ("3", "12", "7"):
        (root / s / "model_params").mkdir(parents=True)
    (root / "20").mkdir()
    ok("latest_checkpoint", T.latest_lora_checkpoint(root) == root / "12" / "model_params")
    ok("explicit_work_dir", T.pick_work_dir(str(Path(d) / "w")) == Path(d) / "w" and (Path(d) / "w").is_dir())

print(f"CHECKS: {CHECKS[0]}")
print(f"HARD FAILS: {len(FAILS)} -> {FAILS}")
raise SystemExit(1 if FAILS else 0)

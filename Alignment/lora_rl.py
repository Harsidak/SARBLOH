"""LoRA RL (GRPO) on chat / tool-call prompts with a verifiable reward. Plain torch + transformers + peft, no TRL; the
data, rendering, model loading and LoRA placement are lora_sft.py's, so both pipelines see identical prompts.

    python lora_rl.py --config rl.json [--set key=value ...]
    from lora_rl import run; card = run({...})

Pipeline: load + validate the same JSONL conversations as SFT -> one prompt per trained assistant turn (the template's
generation prompt; the turn itself is the reference) -> check the reward recognises every reference -> load the model
(``init_adapter``, e.g. an SFT adapter, is merged into the base first) -> attach a fresh LoRA -> greedy eval reward ->
GRPO -> greedy eval reward again -> save the adapter -> reload it into a fresh base and check it reproduces the
policy's log-probabilities -> ``result_card.{json,md}`` / ``reward.png`` with a PASS/FAIL verdict.

GRPO (DeepSeekMath, arXiv 2402.03300), one update per batch so it is exactly on-policy: per prompt, sample
``group_size`` completions; advantage = (r - mean) / (std + eps) inside the group (``scale_rewards: false`` drops the
std); groups whose rewards are all equal carry no signal and are skipped. Loss per completion token
``-A * exp(logp - logp.detach()) + beta * KL_k3``, summed and divided by the step's completion tokens. The reference
policy is the same model with the adapter disabled, i.e. the base (or the SFT-merged base), so no second copy of the
weights is held.

Reward: ``reward: "tool_match"`` (default) scores the action, not the wording, against the reference turn's tool
calls, in [0, 1]: stop 0.1 (ended at the end-of-turn token) + call 0.1 (a well-formed call of a known tool with its
required arguments) + name 0.4 (tool names match the reference) + args 0.4 (argument similarity). Any other reward is
``"module:function"`` or, from Python, a callable ``f(completion: str, item: dict, info: dict) -> float | dict`` (a dict
needs "total"); ``item`` has prompt / reference / tools / id, ``info`` has stopped / tokens.
"""

from __future__ import annotations

import argparse
import copy
import difflib
import gc
import importlib
import json
import math
import random
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import lora_sft as S

DEFAULTS: dict[str, Any] = {
    # model (as lora_sft)
    "model": "Qwen/Qwen3.5-2B",
    "model_class": "auto",
    "dtype": "auto",
    "load_in_4bit": False,
    "trust_remote_code": False,
    "allow_missing_weights": [r"(^|\.)mtp\.", r"lm_head"],
    "attn_implementation": None,
    "init_adapter": None,             # adapter dir merged into the base before RL (e.g. the SFT output); also the ref
    # data (as lora_sft)
    "data_path": "data/smoke3.jsonl",
    "eval_path": None,                # None = evaluate on the training prompts
    "max_samples": None,
    "filter_cleared": False,
    "images": "placeholder",
    "turns": "last",                  # all | last: which assistant turns become prompts
    "template_kwargs": {},
    "max_prompt_len": 3072,           # longer prompts are dropped (and counted), never cut: a cut prompt is a new task
    "mask_mode": "auto",              # unused by RL; kept so lora_sft's helpers see a complete config
    "max_seq_len": 4096,              # idem
    # reward
    "reward": "tool_match",           # tool_match | "module:function" | callable
    # LoRA (as lora_sft)
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.0,
    "lora_targets": S.DEFAULTS["lora_targets"],
    "lora_exclude": S.DEFAULTS["lora_exclude"],
    # GRPO
    "group_size": 4,                  # completions sampled per prompt
    "prompts_per_step": 3,            # prompts (groups) per optimiser step
    "max_steps": 15,
    "lr": 2e-4,
    "warmup_ratio": 0.0,
    "weight_decay": 0.0,
    "max_grad_norm": 1.0,
    "beta": 0.02,                     # KL(policy || reference) weight; 0 = no reference forward
    "scale_rewards": True,            # divide advantages by the group std
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 0,                       # 0 = off
    "max_new_tokens": 160,
    "gradient_checkpointing": True,
    "seed": 0,
    "log_every": 1,
    # evaluation and checks
    "eval_max_new_tokens": 160,
    "min_reward_gain": 0.1,           # PASS needs greedy eval reward after - before >= this (None = report only)
    "reload_check": True,
    "reload_tol": 0.02,               # relative tolerance on the reloaded mean log-prob
    # output
    "output_dir": "rl_out",
}


def make_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = copy.deepcopy({k: v for k, v in DEFAULTS.items()})
    unknown = sorted(set(overrides or {}) - set(cfg))
    if unknown:
        raise KeyError(f"unknown config keys: {unknown} (known: {sorted(cfg)})")
    cfg.update({k: (v if callable(v) else copy.deepcopy(v)) for k, v in (overrides or {}).items()})
    if cfg["turns"] not in ("all", "last"):
        raise ValueError(f"turns {cfg['turns']!r}")
    if cfg["group_size"] < 2:
        raise ValueError("group_size must be >= 2: GRPO's advantage is relative to the group")
    if cfg["init_adapter"] and cfg["load_in_4bit"]:
        raise ValueError("init_adapter is merged into the base, which a 4-bit base cannot take exactly")
    return cfg


# --- prompts ------------------------------------------------------------------------------------------------------
def build_items(tok, convs: list[dict], cfg: dict[str, Any], stats: S.Stats) -> list[dict[str, Any]]:
    """One item per trained assistant turn: the generation prompt, its ids, and the reference completion."""
    items = []
    for c in convs:
        turns = [i for i, m in enumerate(c["messages"]) if m["role"] == "assistant" and m.get("train", True)]
        if cfg["turns"] == "last":
            turns = turns[-1:]
        for i in turns:
            text, start, _ = S.target_span(tok, c["messages"], c["tools"], i, cfg, stats)
            ids = tok(text[:start], add_special_tokens=False)["input_ids"]
            if len(ids) > cfg["max_prompt_len"]:
                stats.dropped_too_long += 1
                continue
            items.append({"id": f"{c['id']}#t{i}", "prompt": text[:start], "prompt_ids": ids,
                          "reference": text[start:], "tools": c["tools"]})
    return items


# --- reward -------------------------------------------------------------------------------------------------------
def parse_tool_calls(text: str) -> list[tuple[str, dict[str, Any]]]:
    """(name, arguments) of the tool calls after the reasoning: Qwen3.5 XML (<function=..><parameter=..>) or Hermes
    JSON inside <tool_call>. Parameter values stay strings unless they parse as JSON."""
    import re
    body = text.split("</think>")[-1]
    calls = []
    for blob in re.findall(r"<tool_call>(.*?)(?:</tool_call>|$)", body, flags=re.S):
        fn = re.search(r"<function=([\w.\-]+)>(.*?)(?:</function>|$)", blob, flags=re.S)
        if fn:
            args = {}
            for k, v in re.findall(r"<parameter=([\w.\-]+)>\n?(.*?)\n?</parameter>", fn.group(2), flags=re.S):
                args[k] = _as_value(v)
            calls.append((fn.group(1), args))
            continue
        try:
            obj = json.loads(blob.strip())
            args = obj.get("arguments", {})
            calls.append((str(obj["name"]), json.loads(args) if isinstance(args, str) else dict(args)))
        except Exception:  # noqa: BLE001 - a malformed call is no call
            pass
    return calls


def _as_value(v: str) -> Any:
    try:
        return json.loads(v)
    except (json.JSONDecodeError, ValueError):
        return v.strip()


def _canon(v: Any) -> str:
    return v.strip() if isinstance(v, str) else json.dumps(v, sort_keys=True)


def value_similarity(a: Any, b: Any) -> float:
    ca, cb = _canon(a), _canon(b)
    return 1.0 if ca == cb else difflib.SequenceMatcher(None, ca, cb).ratio()


def tool_match(completion: str, item: dict[str, Any], info: dict[str, Any]) -> dict[str, float]:
    gen, ref = parse_tool_calls(completion), parse_tool_calls(item["reference"])
    schema = {}
    for t in item.get("tools") or []:
        fn = t.get("function", t)
        schema[fn["name"]] = list((fn.get("parameters") or {}).get("required") or [])
    stop = 0.1 if info.get("stopped") else 0.0
    if not ref:
        same = 1.0 if not gen else 0.0
        return {"stop": stop, "call": 0.1 * same, "name": 0.4 * same, "args": 0.4 * same,
                "total": stop + 0.9 * same}
    valid = any(n in schema and all(k in a for k in schema[n]) for n, a in gen) if schema else bool(gen)
    gn, rn = [n for n, _ in gen], [n for n, _ in ref]
    name = 1.0 if gn == rn else sum((Counter(gn) & Counter(rn)).values()) / max(len(gn), len(rn))
    arg_scores = []
    for j, (n, ra) in enumerate(ref):
        if j < len(gen) and gen[j][0] == n:
            ga = gen[j][1]
            keys = set(ra) | set(ga)
            arg_scores.append(sum(value_similarity(ga[k], ra[k]) if k in ga and k in ra else 0.0 for k in keys)
                              / max(len(keys), 1))
        else:
            arg_scores.append(0.0)
    parts = {"stop": stop, "call": 0.1 if valid else 0.0, "name": 0.4 * name,
             "args": 0.4 * sum(arg_scores) / len(arg_scores)}
    parts["total"] = sum(parts.values())
    return parts


BUILTIN_REWARDS = {"tool_match": tool_match}


def resolve_reward(spec):
    if callable(spec):
        return spec
    if spec in BUILTIN_REWARDS:
        return BUILTIN_REWARDS[spec]
    mod, sep, fn = str(spec).partition(":")
    if not sep:
        raise ValueError(f"reward {spec!r}: a builtin ({sorted(BUILTIN_REWARDS)}) or 'module:function'")
    return getattr(importlib.import_module(mod), fn)


def score(reward_fn, completion: str, item: dict[str, Any], info: dict[str, Any]) -> dict[str, float]:
    r = reward_fn(completion, item, info)
    parts = dict(r) if isinstance(r, dict) else {"total": float(r)}
    if not math.isfinite(float(parts["total"])):
        raise ValueError(f"reward returned a non-finite total for {item['id']}: {parts}")
    return {k: round(float(v), 4) for k, v in parts.items()}


# --- sampling and log-probs ---------------------------------------------------------------------------------------
def sample(model, tok, item: dict[str, Any], n: int, cfg: dict[str, Any], device, autocast, stops: list[int],
           greedy: bool, max_new_tokens: int) -> list[dict[str, Any]]:
    """n completions of one prompt: ids cut after the first stop token, whether it stopped, and the decoded text."""
    import torch
    ids = torch.tensor([item["prompt_ids"]], device=device)
    kw: dict[str, Any] = {"max_new_tokens": max_new_tokens, "pad_token_id": tok.pad_token_id, "eos_token_id": stops}
    if greedy:
        kw["do_sample"] = False
    else:
        kw.update(do_sample=True, num_return_sequences=n, temperature=cfg["temperature"], top_p=cfg["top_p"],
                  top_k=cfg["top_k"])   # explicit 0 / 1.0: None would fall back to the model's generation_config
    with S.generation_mode(model), torch.no_grad(), autocast():
        gen = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), **kw)
    out = []
    for row in gen[:, ids.shape[1]:].tolist():
        new, stopped = S.cut_at_stop(row, stops)
        if not stopped:   # generate pads finished rows; an unstopped row is max_new_tokens long, padding included
            new = new[:max_new_tokens]
        out.append({"ids": new, "stopped": stopped, "text": tok.decode(new, skip_special_tokens=False)})
    return out


class LogProbs:
    """Per-token log-probs of a completion given its prompt. Asks the model for logits at the completion positions
    only (``logits_to_keep`` as an index tensor) and falls back to full logits if the model ignores it."""

    def __init__(self) -> None:
        self.keep_ok: bool | None = None

    def __call__(self, model, prompt_ids: list[int], comp_ids: list[int], device, autocast, temperature: float):
        import torch
        import torch.nn.functional as F
        ids = torch.tensor([prompt_ids + comp_ids], device=device)
        p, n = len(prompt_ids), len(comp_ids)
        tgt = ids[0, p:]
        logits = None
        if self.keep_ok is not False:
            idx = torch.arange(p - 1, p + n - 1, device=device)
            try:
                with autocast():
                    out = model(input_ids=ids, logits_to_keep=idx)
                if out.logits.shape[1] == n:
                    self.keep_ok, logits = True, out.logits[0]
                else:
                    self.keep_ok = False
            except TypeError:
                self.keep_ok = False
        if logits is None:
            with autocast():
                logits = model(input_ids=ids).logits[0, p - 1: p + n - 1]
        return -F.cross_entropy(logits.float() / temperature, tgt, reduction="none")


def advantages(rewards: list[float], scale: bool) -> list[float]:
    m = sum(rewards) / len(rewards)
    sd = (sum((r - m) ** 2 for r in rewards) / len(rewards)) ** 0.5
    if sd < 1e-6:
        return [0.0] * len(rewards)
    return [(r - m) / (sd + 1e-4) if scale else r - m for r in rewards]


def policy_loss(model, logprobs: LogProbs, prompt_ids: list[int], comp_ids: list[int], adv: float, beta: float,
                n_tokens: int, device, autocast, temperature: float) -> tuple[Any, float]:
    """GRPO loss of one completion, already divided by the step's token count; returns (loss, mean KL)."""
    import torch
    lp = logprobs(model, prompt_ids, comp_ids, device, autocast, temperature)
    loss_t = -adv * torch.exp(lp - lp.detach())
    kl = 0.0
    if beta:
        with torch.no_grad(), model.disable_adapter():
            ref = logprobs(model, prompt_ids, comp_ids, device, autocast, temperature)
        d = ref - lp
        kl_t = torch.exp(d) - d - 1
        loss_t = loss_t + beta * kl_t
        kl = float(kl_t.detach().mean())
    return loss_t.sum() / max(n_tokens, 1), kl


# --- train / eval -------------------------------------------------------------------------------------------------
def evaluate(model, tok, items, reward_fn, cfg, device, autocast, stops) -> dict[str, Any]:
    rows = []
    for it in items:
        t0 = time.time()
        g = sample(model, tok, it, 1, cfg, device, autocast, stops, greedy=True,
                   max_new_tokens=cfg["eval_max_new_tokens"])[0]
        r = score(reward_fn, g["text"], it, {"stopped": g["stopped"], "tokens": len(g["ids"])})
        rows.append({"id": it["id"], "reward": r, "generated": g["text"], "ids": g["ids"], "stopped": g["stopped"],
                     "tool_calls": [n for n, _ in parse_tool_calls(g["text"])],
                     "tool_calls_reference": [n for n, _ in parse_tool_calls(it["reference"])],
                     "new_tokens": len(g["ids"]), "seconds": round(time.time() - t0, 2)})
    return {"reward": sum(r["reward"]["total"] for r in rows) / max(len(rows), 1), "samples": rows}


def mean_logprob(model, items, rows, logprobs: LogProbs, device, autocast) -> float:
    import torch
    tot, n = 0.0, 0
    model.eval()
    with torch.no_grad():
        for it, r in zip(items, rows):
            if r["ids"]:
                lp = logprobs(model, it["prompt_ids"], r["ids"], device, autocast, 1.0)
                tot, n = tot + float(lp.sum()), n + lp.numel()
    return tot / max(n, 1)


def train(model, tok, items, reward_fn, cfg, device, autocast, stops, logprobs: LogProbs, log) -> dict[str, Any]:
    import torch
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    total = cfg["max_steps"]
    warmup = int(total * cfg["warmup_ratio"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, S.lr_lambda(warmup, total))
    rng = random.Random(cfg["seed"])
    order: list[int] = []
    history, nonfinite, updated, t0, samples_seen = [], 0, 0, time.time(), 0
    for step in range(total):
        torch.manual_seed(cfg["seed"] * 100003 + step)
        groups = []
        for _ in range(min(cfg["prompts_per_step"], len(items))):
            if not order:
                order = list(range(len(items)))
                rng.shuffle(order)
            it = items[order.pop()]
            comps = sample(model, tok, it, cfg["group_size"], cfg, device, autocast, stops, greedy=False,
                           max_new_tokens=cfg["max_new_tokens"])
            rewards = [score(reward_fn, c["text"], it, {"stopped": c["stopped"], "tokens": len(c["ids"])})["total"]
                       for c in comps]
            groups.append((it, comps, rewards, advantages(rewards, cfg["scale_rewards"])))
            samples_seen += len(comps)
        live = [g for g in groups if any(a != 0 for a in g[3])]
        n_tokens = sum(len(c["ids"]) for _, comps, _, _ in live for c in comps)
        model.train()
        loss_sum, kls, ok = 0.0, [], True
        for it, comps, _, adv in live:
            for c, a in zip(comps, adv):
                if not c["ids"]:
                    continue
                loss, kl = policy_loss(model, logprobs, it["prompt_ids"], c["ids"], a, cfg["beta"], n_tokens,
                                       device, autocast, cfg["temperature"])
                if not torch.isfinite(loss):
                    ok = False
                    break
                loss.backward()
                loss_sum += float(loss.detach())
                kls.append(kl)
            if not ok:
                break
        gnorm = 0.0
        if not ok:
            nonfinite += 1
            opt.zero_grad(set_to_none=True)
            log(f"step {step + 1}: non-finite loss, update skipped ({nonfinite})")
            if nonfinite > 3:
                raise RuntimeError("more than 3 non-finite losses: aborting (try dtype fp32 or a lower lr)")
        elif live:
            gnorm = float(torch.nn.utils.clip_grad_norm_(params, cfg["max_grad_norm"]))
            opt.step()
            opt.zero_grad(set_to_none=True)
            updated += 1
        sched.step()
        all_r = [r for _, _, rs, _ in groups for r in rs]
        all_c = [c for _, cs, _, _ in groups for c in cs]
        row = {"step": step + 1, "reward_mean": sum(all_r) / len(all_r), "reward_max": max(all_r),
               "group_std": sum((sum((r - sum(rs) / len(rs)) ** 2 for r in rs) / len(rs)) ** 0.5
                                for _, _, rs, _ in groups) / len(groups),
               "live_groups": len(live), "groups": len(groups), "kl": sum(kls) / len(kls) if kls else 0.0,
               "loss": loss_sum, "grad_norm": gnorm, "lr": sched.get_last_lr()[0],
               "completion_len": sum(len(c["ids"]) for c in all_c) / len(all_c),
               "stopped_frac": sum(c["stopped"] for c in all_c) / len(all_c), "elapsed_s": round(time.time() - t0, 1)}
        history.append(row)
        if (step + 1) % cfg["log_every"] == 0 or step + 1 == total:
            log(f"step {row['step']}/{total} reward {row['reward_mean']:.3f} (max {row['reward_max']:.3f}, "
                f"group std {row['group_std']:.3f}) live {row['live_groups']}/{row['groups']} kl {row['kl']:.4f} "
                f"len {row['completion_len']:.0f} stop {row['stopped_frac']:.2f} gnorm {gnorm:.3f} {row['elapsed_s']}s")
    secs = time.time() - t0
    return {"steps": total, "updated_steps": updated, "nonfinite_steps": nonfinite, "history": history,
            "samples": samples_seen, "seconds": round(secs, 1)}


# --- result card --------------------------------------------------------------------------------------------------
def render_card(card: dict[str, Any]) -> str:
    c, e, d = card["config"], card.get("environment", {}), card.get("data", {})
    lines = [f"# LoRA RL (GRPO) result card — {card['verdict']}", "",
             f"run `{card['run_id']}` · model `{c['model']}`"
             f"{' + merged `' + str(c['init_adapter']) + '`' if c.get('init_adapter') else ''} · {e.get('gpu', 'cpu')}"
             f" · dtype `{card.get('dtype')}` · reward `{c['reward'] if isinstance(c['reward'], str) else 'callable'}`"
             f" · {card.get('seconds_total')}s total", ""]
    lines += ["## Checks", "", "| check | result | detail |", "|---|---|---|"]
    for k, v in card["checks"].items():
        lines.append(f"| {k} | {'PASS' if v['pass'] else 'FAIL'} | {v.get('detail', '')} |")
    if card.get("error"):
        lines += ["", "## Error", "", "```", card["error"][-3000:], "```"]
    ev0, ev1 = card.get("eval_before"), card.get("eval_after")
    if ev0 and ev1:
        lines += ["", "## Greedy reward on the eval prompts", "", "| | mean reward | per prompt |", "|---|---|---|",
                  f"| references | {card.get('reference_reward', float('nan')):.3f} | |",
                  f"| before | {ev0['reward']:.3f} | {[r['reward']['total'] for r in ev0['samples']]} |",
                  f"| after | {ev1['reward']:.3f} | {[r['reward']['total'] for r in ev1['samples']]} |"]
        if card.get("reload"):
            rl = card["reload"]
            lines.append(f"\nReload: mean log-prob of the after-training completions {rl['policy']:.4f} (policy) vs "
                         f"{rl['reloaded']:.4f} (reloaded adapter).")
    t = card.get("training")
    if t and t["history"]:
        h0, h1 = t["history"][0], t["history"][-1]
        lines += ["", "## Training", "",
                  f"{t['steps']} steps ({t['updated_steps']} with an update), {t['samples']} samples, {t['seconds']}s, "
                  f"non-finite {t['nonfinite_steps']}. Sampled reward {h0['reward_mean']:.3f} -> {h1['reward_mean']:.3f}"
                  f"; KL {h1['kl']:.4f}; completion length {h0['completion_len']:.0f} -> {h1['completion_len']:.0f}.",
                  "", "| step | reward | max | group std | live | kl | len | stopped | grad norm |",
                  "|---|---|---|---|---|---|---|---|---|"]
        lines += [f"| {h['step']} | {h['reward_mean']:.3f} | {h['reward_max']:.3f} | {h['group_std']:.3f} | "
                  f"{h['live_groups']}/{h['groups']} | {h['kl']:.4f} | {h['completion_len']:.0f} | "
                  f"{h['stopped_frac']:.2f} | {h['grad_norm']:.3f} |" for h in t["history"]]
        if card.get("plot"):
            lines += ["", f"![reward]({card['plot']})"]
    lines += ["", "## Data", "", "| stat | value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in d.items() if k != "errors"]
    if d.get("errors"):
        lines += ["", "Rejected conversations:", ""] + [f"- {x}" for x in d["errors"][:20]]
    if card.get("lora"):
        lo = card["lora"]
        lines += ["", "## LoRA", "", f"r={c['lora_r']} alpha={c['lora_alpha']}; {lo['target_layers']} layers; "
                  f"trainable {lo['trainable_params']:,} of {lo['total_params']:,}"]
    for tag in ("eval_before", "eval_after"):
        if card.get(tag):
            lines += ["", f"## Greedy generations {tag.split('_')[1]}", ""]
            for r in card[tag]["samples"]:
                lines += [f"- `{r['id']}` reward {r['reward']} · tools {r['tool_calls']} (ref "
                          f"{r['tool_calls_reference']}) · {r['new_tokens']} tokens"
                          f"{'' if r['stopped'] else ' (hit max tokens)'}", "",
                          "  ```", "  " + r["generated"][:600].replace("\n", "\n  "), "  ```"]
    lines += ["", "## Environment", "", "```json", json.dumps(e, indent=1), "```", "",
              "## Config", "", "```json", json.dumps(c, indent=1, default=str), "```", ""]
    return "\n".join(lines)


# --- driver -------------------------------------------------------------------------------------------------------
def load_policy_base(path: str, cfg: dict[str, Any], dtype):
    """The base model, with ``init_adapter`` merged in when set (so it is also the KL reference)."""
    model, load = S.load_model(path, cfg, dtype)
    if cfg["init_adapter"]:
        from peft import PeftModel
        S.guard_peft_torchao()
        model = PeftModel.from_pretrained(model, cfg["init_adapter"]).merge_and_unload()
        model.config.use_cache = False
        load["merged_adapter"] = str(cfg["init_adapter"])
    return model, load


def run(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    return S.run_with_card(make_config(overrides), _run, render_card)


def _run(cfg: dict[str, Any], card: dict[str, Any], out: Path, log) -> None:
    import torch
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    card["environment"] = env = S.environment()
    log(f"environment {json.dumps(env)}")
    if env["peft"] is None:
        raise RuntimeError("peft is not installed")
    dtype = S.pick_dtype(cfg["dtype"])
    card["dtype"] = str(dtype).replace("torch.", "")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reward_fn = resolve_reward(cfg["reward"])

    t = time.time()
    path = S.resolve_model(cfg["model"])
    card["model_path"] = path
    log(f"model files at {path} ({time.time() - t:.0f}s)")
    tok = S.load_tokenizer(path, cfg)

    stats = S.Stats()
    convs = S.load_conversations(cfg["data_path"], cfg, stats, cfg["max_samples"])
    items = build_items(tok, convs, cfg, stats)
    eval_items = (build_items(tok, S.load_conversations(cfg["eval_path"], cfg, S.Stats(), None), cfg, S.Stats())
                  if cfg["eval_path"] else items)
    card["data"] = {**asdict(stats), "prompts": len(items), "eval_prompts": len(eval_items),
                    "prompt_tokens_max": max((len(i["prompt_ids"]) for i in items), default=0),
                    "sources": [i["id"] for i in items][:50]}
    log(f"data: {len(items)} prompts from {stats.conversations} conversations, {stats.skipped_conversations} rejected, "
        f"{stats.dropped_too_long} too long")
    check(card, "data", bool(items) and bool(eval_items) and stats.skipped_conversations == 0,
          f"{len(items)} prompts from {stats.conversations} conversations, {stats.skipped_conversations} rejected, "
          f"{stats.dropped_too_long} over max_prompt_len")
    if not items:
        raise RuntimeError("no prompts")
    ref_rewards = [score(reward_fn, it["reference"], it, {"stopped": True, "tokens": 0}) for it in items]
    card["reference_reward"] = sum(r["total"] for r in ref_rewards) / len(ref_rewards)
    if cfg["reward"] in BUILTIN_REWARDS:
        worst = min(r["total"] for r in ref_rewards)
        check(card, "reward_sane", worst >= 0.99, f"every reference scores >= 0.99 under its own reward (min {worst})")

    t = time.time()
    model, load = load_policy_base(path, cfg, dtype)
    card["load"] = load
    log(f"loaded {load['class']} ({load['params']:,} params) in {time.time() - t:.0f}s"
        f"{'; merged ' + load['merged_adapter'] if load.get('merged_adapter') else ''}")
    check(card, "weights", not load["missing_not_allowed"],
          f"{len(load['missing_keys'])} missing ({len(load['missing_not_allowed'])} not allowed), "
          f"{len(load['unexpected_keys'])} unexpected")
    if load["missing_not_allowed"]:
        raise RuntimeError(f"weights missing from the checkpoint: {load['missing_not_allowed'][:10]}")
    model, lora = S.attach_lora(model, cfg)
    card["lora"] = lora
    check(card, "trainable", lora["trainable_params"] > 0 and not lora["trainable_non_lora"],
          f"{lora['trainable_params']:,} params; non-LoRA trainable {lora['trainable_non_lora']}")

    autocast = S.make_autocast(dtype)
    stops = S.stop_token_ids(model, tok)
    logprobs = LogProbs()
    card["eval_before"] = evaluate(model, tok, eval_items, reward_fn, cfg, device, autocast, stops)
    log(f"eval before: greedy reward {card['eval_before']['reward']:.3f}")

    card["training"] = tr = train(model, tok, items, reward_fn, cfg, device, autocast, stops, logprobs, log)
    card["logits_to_keep"] = logprobs.keep_ok
    if torch.cuda.is_available():
        card["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    card["eval_after"] = ev1 = evaluate(model, tok, eval_items, reward_fn, cfg, device, autocast, stops)
    log(f"eval after: greedy reward {ev1['reward']:.3f}")
    moved = S.lora_moved(model)
    check(card, "finite", tr["nonfinite_steps"] == 0 and all(math.isfinite(h["loss"]) for h in tr["history"]),
          f"{tr['nonfinite_steps']} non-finite steps")
    live = sum(h["live_groups"] for h in tr["history"])
    check(card, "learning_signal", tr["updated_steps"] > 0,
          f"{tr['updated_steps']}/{tr['steps']} steps updated; {live}/{sum(h['groups'] for h in tr['history'])} "
          f"groups had reward variance")
    check(card, "adapter_updated", moved > 0, f"max |lora_B| = {moved:.3g}")
    b, a = card["eval_before"]["reward"], ev1["reward"]
    if cfg["min_reward_gain"] is not None:
        check(card, "reward_improved", a - b >= cfg["min_reward_gain"],
              f"greedy reward {b:.3f} -> {a:.3f} (gain {a - b:+.3f}, need >= {cfg['min_reward_gain']})")

    policy_lp = mean_logprob(model, eval_items, ev1["samples"], logprobs, device, autocast)
    adapter_dir = out / "adapter"
    model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    (adapter_dir / "rl_config.json").write_text(json.dumps(cfg, indent=1, default=str), encoding="utf-8")
    files = sorted(p.name for p in adapter_dir.iterdir())
    size = sum(p.stat().st_size for p in adapter_dir.iterdir() if p.is_file())
    card["adapter"] = {"dir": str(adapter_dir), "files": files, "mb": round(size / 2**20, 2),
                       "base_merged_with": cfg["init_adapter"]}
    check(card, "adapter_saved", "adapter_config.json" in files and any(f.startswith("adapter_model") for f in files),
          f"{card['adapter']['mb']} MB")
    log(f"adapter saved: {files}")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if cfg["reload_check"]:
        from peft import PeftModel
        base, _ = load_policy_base(path, cfg, dtype)
        re_model = PeftModel.from_pretrained(base, adapter_dir)
        re_lp = mean_logprob(re_model, eval_items, ev1["samples"], LogProbs(), device, autocast)
        card["reload"] = {"policy": policy_lp, "reloaded": re_lp}
        log(f"reload: mean log-prob {policy_lp:.4f} (policy) vs {re_lp:.4f} (reloaded)")
        check(card, "reload_matches", abs(re_lp - policy_lp) <= cfg["reload_tol"] * max(abs(policy_lp), 1e-3) + 1e-4,
              f"mean log-prob {policy_lp:.4f} vs reloaded {re_lp:.4f}")
        del re_model, base
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    card["plot"] = "reward.png" if S.write_plot(tr["history"], out / "reward.png",
                                                (("reward_mean", "sampled reward"), ("kl", "KL to reference"))) else None


check = S.check


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="LoRA GRPO with a result card")
    ap.add_argument("--config", help="JSON file of overrides")
    ap.add_argument("--set", nargs="*", default=[], help="key=value overrides (value parsed as JSON when possible)")
    args = ap.parse_args(argv)
    overrides = json.loads(Path(args.config).read_text(encoding="utf-8")) if args.config else {}
    overrides.update(S._parse_set(args.set))
    card = run(overrides)
    print(render_card(card))
    return 0 if card["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())

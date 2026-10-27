"""Long-context training path for Qwen3.8-Flash-Next (transformers ``qwen4_exp``), used by lora_sft.py when
``long_context`` is on and the model is ``qwen4_exp``.

Why: transformers' reference code cannot train on 131k-token rows. Its Qwen Sparse Attention (QSA) indexer loops in
Python over every query position, and the model builds a dense L x L mask (131k x 131k = 17 GB per bool copy, several
copies). This module replaces both, for training only (no KV cache, no padding, causal), with the same maths:

- Indexer: the key of block j is the mean of the raw indexer keys of tokens 4j..4j+3, normed, roped at 4j. Query i
  sees the complete blocks j < (i+1)//4 and keeps the top ``indexer_budget / indexer_compress_ratio`` (512) of them by
  sum_h relu(q_ih . k_j) / sqrt(d), plus its own incomplete tail block. Computed in query chunks, without gradient (the
  selection is an index, so the reference gives the indexer no gradient either).
- Attention: each query attends to its selected tokens only (<= 2051), gathered per query chunk. Each chunk runs under
  its own activation checkpoint, so the gathered keys/values of only one chunk are ever alive.
- The MoE block (token-wise) runs in token chunks under checkpoint, so routed-token buffers stay small at 131k.

``check_equivalence`` compares this path with the reference on a small random model (tests/unit/test_alignment_long.py).
Everything else (Gated DeltaNet, gated residual, n-gram embedding) is the reference code; install
``flash-linear-attention`` and ``causal-conv1d`` for their fast kernels.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

DEFAULTS = {"query_chunk": 1024, "moe_chunk": 16384}
_STATE: dict[str, Any] = {"patched": False, **DEFAULTS}


def _mod():
    from transformers.models.qwen4_exp import modeling_qwen4_exp as m
    return m


def select_tokens(indexer, hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                  q_chunk: int) -> torch.Tensor:
    """(B, L, T) int64 token indices each query attends to, -1 = none. T = 4 * block_topk + compress_ratio - 1."""
    m = _mod()
    B, L, _ = hidden_states.shape
    r, d = indexer.compress_ratio, indexer.index_head_dim
    topk = indexer.block_topk
    T = topk * r + r - 1
    out = torch.full((B, L, T), -1, dtype=torch.long, device=hidden_states.device)
    with torch.no_grad():
        qk = indexer.index_qk_proj(hidden_states)
        q, raw = torch.split(qk, [indexer.index_n_heads * d, indexer.index_kv_heads * d], dim=-1)
        q = indexer.q_layernorm(q.reshape(B, L, -1, d))
        q = m.apply_rotary_pos_emb(q, cos=cos, sin=sin, unsqueeze_dim=2).float()       # (B, L, H, d)
        raw = raw.reshape(B, L, d)
        nb = L // r
        pos = torch.arange(L, device=hidden_states.device)
        if nb:
            pooled = raw[:, : nb * r].reshape(B, nb, r, d).float().mean(dim=2).to(raw.dtype)
            pooled = indexer.k_layernorm(pooled)
            starts = torch.arange(nb, device=raw.device) * r
            kb = m.apply_rotary_pos_emb(pooled.unsqueeze(2), cos=cos[:, starts], sin=sin[:, starts],
                                        unsqueeze_dim=2).squeeze(2).float()           # (B, nb, d)
        for s in range(0, L, q_chunk):
            e = min(L, s + q_chunk)
            qi = pos[s:e]
            n_complete = (qi + 1) // r                                                 # visible complete blocks
            nbv = int(n_complete.max())
            k_eff = min(topk, nbv)
            if k_eff:
                scores = torch.einsum("bchd,bnd->bchn", q[:, s:e], kb[:, :nbv]).relu_().sum(2) / math.sqrt(d)
                visible = torch.arange(nbv, device=raw.device)[None, :] < n_complete[:, None]
                scores = scores.masked_fill(~visible[None], -1.0)
                val, blk = scores.topk(k_eff, dim=-1)                                  # (B, C, k)
                tok = (blk[..., None] * r + torch.arange(r, device=raw.device)).flatten(-2)
                ok = (val >= 0)[..., None].expand(*val.shape, r).flatten(-2)
                out[:, s:e, : k_eff * r] = torch.where(ok, tok, -1)
            tail = n_complete[:, None] * r + torch.arange(r - 1, device=raw.device)        # (C, r-1)
            out[:, s:e, topk * r:] = torch.where(tail <= qi[:, None], tail, -1)[None].expand(B, -1, -1)
    return out


def _sparse_chunk(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, idx: torch.Tensor, scaling: float) -> torch.Tensor:
    """q (Hkv, G, C, D), k/v (Hkv, L, D), idx (C, T) -> (Hkv, G, C, D)."""
    valid = idx >= 0
    safe = idx.clamp_min(0)
    kg = k[:, safe]                                                                    # (Hkv, C, T, D)
    vg = v[:, safe]
    att = torch.einsum("hgcd,hctd->hgct", q.float(), kg.float()) * scaling
    att = att.masked_fill(~valid[None, None], float("-inf")).softmax(-1)
    return torch.einsum("hgct,hctd->hgcd", att.to(vg.dtype), vg)


def attention_forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None, **kwargs):
    """Qwen4ExpTextAttention.forward for training at long context (see the module docstring)."""
    if past_key_values is not None:
        raise RuntimeError("the long-context QSA path is for training only (no KV cache)")
    m = _mod()
    B, L, _ = hidden_states.shape
    cos, sin = (x[:, -L:, :] for x in position_embeddings)
    idx = select_tokens(self.indexer, hidden_states, cos, sin, _STATE["query_chunk"])
    hidden_shape = (B, L, -1, self.head_dim)
    query, gate = torch.chunk(self.q_proj(hidden_states).view(B, L, -1, self.head_dim * 2), 2, dim=-1)
    gate = gate.reshape(B, L, -1)
    query = self.q_norm(query.reshape(hidden_shape)).transpose(1, 2)
    key = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    query, key = m.apply_rotary_pos_emb(query, key, cos, sin)
    hkv = key.shape[1]
    g = query.shape[1] // hkv
    outs = []
    for b in range(B):
        qb = query[b].reshape(hkv, g, L, self.head_dim)
        parts = []
        for s in range(0, L, _STATE["query_chunk"]):
            e = min(L, s + _STATE["query_chunk"])
            args = (qb[:, :, s:e], key[b], value[b], idx[b, s:e], self.scaling)
            parts.append(checkpoint(_sparse_chunk, *args, use_reentrant=False) if torch.is_grad_enabled()
                         else _sparse_chunk(*args))
        outs.append(torch.cat(parts, dim=2).reshape(hkv * g, L, self.head_dim))
    attn = torch.stack(outs).transpose(1, 2).reshape(B, L, -1)
    attn = attn * torch.sigmoid(gate)
    return self.o_proj(attn), None


def moe_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """Qwen4ExpTextSparseMoeBlock.forward in token chunks (exact: the block is token-wise)."""
    n = _STATE["moe_chunk"]
    B, L, H = hidden_states.shape
    flat = hidden_states.reshape(-1, H)
    if flat.shape[0] <= n:
        return _ORIG["moe"](self, hidden_states)

    def part(x):
        return _ORIG["moe"](self, x[None])[0]

    outs = [checkpoint(part, flat[s:s + n], use_reentrant=False) if torch.is_grad_enabled() else part(flat[s:s + n])
            for s in range(0, flat.shape[0], n)]
    return torch.cat(outs).reshape(B, L, H)


_ORIG: dict[str, Any] = {}


def _text_forward(self, *args, attention_mask=None, past_key_values=None, **kwargs):
    """Skip the dense L x L mask: training rows are single unpadded sequences, and the patched attention is causal."""
    if past_key_values is None and not kwargs.get("use_cache") and (attention_mask is None or (
            isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 2 and bool(attention_mask.all()))):
        attention_mask = {"indexed_attention": None, "linear_attention": None}
        kwargs["use_cache"] = False
    return _ORIG["text"](self, *args, attention_mask=attention_mask, past_key_values=past_key_values, **kwargs)


def enable(query_chunk: int | None = None, moe_chunk: int | None = None) -> dict[str, Any]:
    """Patch the qwen4_exp classes in place (idempotent). Returns the active settings."""
    m = _mod()
    if query_chunk:
        _STATE["query_chunk"] = int(query_chunk)
    if moe_chunk:
        _STATE["moe_chunk"] = int(moe_chunk)
    if not _STATE["patched"]:
        _ORIG["attn"] = m.Qwen4ExpTextAttention.forward
        _ORIG["moe"] = m.Qwen4ExpTextSparseMoeBlock.forward
        _ORIG["text"] = m.Qwen4ExpTextModel.forward
        m.Qwen4ExpTextAttention.forward = attention_forward
        m.Qwen4ExpTextSparseMoeBlock.forward = moe_forward
        m.Qwen4ExpTextModel.forward = _text_forward
        _STATE["patched"] = True
    return {k: _STATE[k] for k in DEFAULTS}


def disable() -> None:
    if _STATE["patched"]:
        m = _mod()
        m.Qwen4ExpTextAttention.forward = _ORIG["attn"]
        m.Qwen4ExpTextSparseMoeBlock.forward = _ORIG["moe"]
        m.Qwen4ExpTextModel.forward = _ORIG["text"]
        _STATE["patched"] = False


def tiny_config(**over):
    """A small random qwen4_exp text config with every block type (DeltaNet, QSA, MoE, PLE, gated residual)."""
    from transformers import Qwen4ExpConfig
    text = dict(hidden_size=64, num_hidden_layers=4, layer_types=["linear_attention"] * 3 + ["full_attention"],
                num_attention_heads=4, num_key_value_heads=2, head_dim=32, linear_num_key_heads=2,
                linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16, num_experts=8,
                num_experts_per_tok=2, moe_intermediate_size=32, shared_expert_intermediate_size=32, hc_count=4,
                hc_lowrank=16, indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=16, indexer_budget=16, indexer_compress_ratio=4,
                ngram_vocab_size_base=1009, ple_embed_dim=64, ple_layer_ids=[2], heads_per_ngram=2,
                vocab_size=512, eos_token_id=511, bos_token_id=511, max_position_embeddings=4096, mtp_num_hidden_layers=0,
                rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
                                 "mrope_section": [2, 1, 1], "mrope_interleaved": True})
    text.update(over)
    return Qwen4ExpConfig(text_config=text, vision_config={"depth": 1, "hidden_size": 32, "num_heads": 2,
                                                           "intermediate_size": 64, "out_hidden_size": 64})


def check_equivalence(seq_len: int = 300, device: str = "cpu", seed: int = 0, query_chunk: int = 64,
                      moe_chunk: int = 128) -> dict[str, float]:
    """Max |logit| and |grad| difference between the reference and the patched model on one random sequence."""
    from transformers import Qwen4ExpForConditionalGeneration
    torch.manual_seed(seed)
    model = Qwen4ExpForConditionalGeneration(tiny_config()).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.dim() == 1 and "norm" in n:
                p.normal_(0, 0.1)    # zero-initialised norms would hide ordering bugs
    model.eval()
    ids = torch.randint(0, model.config.text_config.vocab_size, (1, seq_len), device=device)

    def run():
        model.zero_grad(set_to_none=True)
        out = model(input_ids=ids)
        out.logits.float().pow(2).mean().backward()
        grads = {n: p.grad.detach().clone() for n, p in model.named_parameters()
                 if p.grad is not None and ("self_attn" in n or "linear_attn" in n)}
        return out.logits.detach().float(), grads

    disable()
    ref, ref_g = run()
    enable(query_chunk=query_chunk, moe_chunk=moe_chunk)
    try:
        got, got_g = run()
    finally:
        disable()
    gdiff = max(float((ref_g[n] - got_g[n]).abs().max()) for n in ref_g)
    gscale = max(float(ref_g[n].abs().max()) for n in ref_g)
    return {"logit_max_diff": float((ref - got).abs().max()), "logit_scale": float(ref.abs().max()),
            "grad_max_diff": gdiff, "grad_scale": gscale, "grads_compared": len(ref_g),
            "missing_grads": sorted(set(ref_g) ^ set(got_g))}

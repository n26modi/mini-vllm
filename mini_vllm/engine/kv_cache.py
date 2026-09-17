from __future__ import annotations

import torch
from transformers import DynamicCache


# ---------------------------------------------------------------------------
# KV quantization helpers
# ---------------------------------------------------------------------------

def _quantize(x: torch.Tensor, bits: int):
    """Per-token, per-head min-max quantization. Returns (q, min, scale)."""
    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1
    x_min = x.amin(dim=-1, keepdim=True)
    x_max = x.amax(dim=-1, keepdim=True)
    scale = (x_max - x_min).clamp(min=1e-8) / (qmax - qmin)
    x_q = ((x - x_min) / scale).round().clamp(qmin, qmax).to(torch.int8)
    return x_q, x_min, scale


def _dequantize(x_q, x_min, scale, dtype):
    return (x_q.to(torch.float32) * scale + x_min).to(dtype)


def compress_cache(cache: DynamicCache, bits: int) -> DynamicCache:
    """Quantize-then-dequantize all KV tensors in place.

    Simulates the precision loss from a compressed cache. Actual memory savings
    require storing int8 and dequantizing at attention time (needs custom kernel).
    This measures the quality impact instead.
    """
    if not cache.key_cache:
        return cache
    dtype = cache.key_cache[0].dtype
    for i in range(len(cache.key_cache)):
        k_q, k_min, k_sc = _quantize(cache.key_cache[i].float(), bits)
        v_q, v_min, v_sc = _quantize(cache.value_cache[i].float(), bits)
        cache.key_cache[i] = _dequantize(k_q, k_min, k_sc, dtype)
        cache.value_cache[i] = _dequantize(v_q, v_min, v_sc, dtype)
    return cache


def cache_bytes(cache: DynamicCache, bits: int = 16) -> int:
    """Bytes the KV cache would occupy at a given bit width."""
    total = sum(k.numel() + v.numel() for k, v in zip(cache.key_cache, cache.value_cache))
    return total * (bits // 8)


# ---------------------------------------------------------------------------
# H2O eviction
# ---------------------------------------------------------------------------

def evict_h2o(
    cache: DynamicCache,
    attn_weights: tuple,
    budget: int,
    recent_window: int = 256,
) -> DynamicCache:
    """Keep heavy-hitter tokens (highest cumulative attention score) + recent window.

    attn_weights: tuple of [batch, heads, q_len, k_len] per layer,
                  from model(..., output_attentions=True).
    budget: max KV entries to keep.
    recent_window: always keep the last N tokens regardless of score.
    """
    if not cache.key_cache:
        return cache
    seq_len = cache.key_cache[0].shape[2]
    if seq_len <= budget:
        return cache

    device = cache.key_cache[0].device
    scores = torch.zeros(seq_len, device=device)
    for w in attn_weights:
        if w is None:
            continue
        scores += w[0].float().mean(dim=0).sum(dim=0)[:seq_len]

    # always protect recent window
    scores[max(0, seq_len - recent_window):] = float("inf")

    _, keep_idx = scores.topk(min(budget, seq_len))
    keep_idx = keep_idx.sort().values

    for i in range(len(cache.key_cache)):
        cache.key_cache[i] = cache.key_cache[i][:, :, keep_idx, :]
        cache.value_cache[i] = cache.value_cache[i][:, :, keep_idx, :]
    return cache


# ---------------------------------------------------------------------------
# SnapKV eviction
# ---------------------------------------------------------------------------

def evict_snapkv(
    cache: DynamicCache,
    attn_weights: tuple,
    budget: int,
    obs_window: int = 32,
) -> DynamicCache:
    """Query-aware eviction using attention from a recent observation window.

    Unlike H2O (global accumulation), SnapKV re-evaluates which positions
    the current queries care about — adapts to topic shifts mid-generation.
    """
    if not cache.key_cache:
        return cache
    seq_len = cache.key_cache[0].shape[2]
    if seq_len <= budget:
        return cache

    device = cache.key_cache[0].device
    scores = torch.zeros(seq_len, device=device)
    for w in attn_weights:
        if w is None:
            continue
        obs = w[0, :, -obs_window:, :seq_len].float()  # [heads, obs, seq]
        scores += obs.mean(dim=0).sum(dim=0)

    _, keep_idx = scores.topk(min(budget, seq_len))
    keep_idx = keep_idx.sort().values

    for i in range(len(cache.key_cache)):
        cache.key_cache[i] = cache.key_cache[i][:, :, keep_idx, :]
        cache.value_cache[i] = cache.value_cache[i][:, :, keep_idx, :]
    return cache


# ---------------------------------------------------------------------------
# MLA stub
# ---------------------------------------------------------------------------

class MLACache:
    """
    Multi-head Latent Attention cache stub.

    Instead of caching full K/V ([heads × head_dim] per token), MLA projects
    K/V down to a shared low-rank latent c_kv of shape [rank] and reconstructs
    full K/V on the fly via up-projection matrices W_K^up, W_V^up.

    Memory: 2 × rank per token  vs  2 × heads × head_dim.
    For Qwen3-0.6B (8 heads, 128 dim, rank=512): ~4x compression.

    Requires TransMLA weight conversion to add projection matrices post-hoc
    to an already-trained GQA model (no fine-tuning).
    See: https://arxiv.org/html/2502.07864v4 and https://github.com/bet0x/transmla-converter
    """
    def __init__(self):
        raise NotImplementedError(
            "MLACache requires TransMLA weight conversion of Qwen3-0.6B. "
            "See https://github.com/bet0x/transmla-converter"
        )

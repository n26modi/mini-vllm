from __future__ import annotations

import time
import uuid
from typing import Generator, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from mini_vllm.configs.flags import EngineConfig
from mini_vllm.engine.kv_cache import (
    cache_bytes, compress_cache, evict_h2o, evict_snapkv,
)
from mini_vllm.engine.prefix_cache import HashPrefixCache, RadixCache
from mini_vllm.engine.sampler import SampleParams, sample
from mini_vllm.instrumentation.metrics import RequestMetrics

MODEL_ID = "Qwen/Qwen3-0.6B"


# ---------------------------------------------------------------------------
# Shared model loader (singleton per process — load once, share)
# ---------------------------------------------------------------------------

_loaded: dict[str, tuple] = {}  # device -> (model, tokenizer)


def _load_model(device: str, eager: bool = False):
    key = f"{device}_{eager}"
    if key not in _loaded:
        print(f"Loading {MODEL_ID} on {device} ...")
        tok = AutoTokenizer.from_pretrained(MODEL_ID)
        attn_impl = "eager" if eager else "sdpa"
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, dtype=torch.bfloat16, attn_implementation=attn_impl
        ).to(device)
        model.eval()
        _loaded[key] = (model, tok)
        print("Ready.")
    return _loaded[key]


# ---------------------------------------------------------------------------
# NaiveEngine — Day 5: no KV cache, full recompute every step
# ---------------------------------------------------------------------------

class NaiveEngine:
    """Full forward pass over entire sequence on every decode step. O(n²)."""

    def __init__(self, config: EngineConfig):
        self.config = config
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model, self.tokenizer = _load_model(self.device, eager=False)

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        params: Optional[SampleParams] = None,
    ) -> RequestMetrics:
        if params is None:
            params = SampleParams()

        request_id = str(uuid.uuid4())
        prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)[0]
        seq = prompt_ids.tolist()
        generated: list[int] = []
        itl_ms: list[float] = []
        eos_id = self.tokenizer.eos_token_id
        t_start = time.perf_counter()

        for step in range(max_new_tokens):
            t_step = time.perf_counter()
            ids = torch.tensor([seq], dtype=torch.long, device=self.device)
            logits = self.model(ids).logits[0, -1]
            gen_t = torch.tensor(generated, dtype=torch.long, device=self.device) if generated else None
            next_token = sample(logits, params, gen_t)
            elapsed_ms = (time.perf_counter() - t_step) * 1000
            if step == 0:
                ttft_ms = (time.perf_counter() - t_start) * 1000
            else:
                itl_ms.append(elapsed_ms)
            seq.append(next_token)
            generated.append(next_token)
            if next_token == eos_id:
                break

        wall_time_s = time.perf_counter() - t_start
        peak_gb = torch.mps.current_allocated_memory() / 1e9 if self.device == "mps" else 0.0
        return RequestMetrics(
            request_id=request_id, prompt=prompt, max_new_tokens=max_new_tokens,
            ttft_ms=ttft_ms, itl_ms_per_token=itl_ms, total_tokens=len(generated),
            wall_time_s=wall_time_s, gpu_util_pct=0.0, peak_mem_gb=peak_gb,
        )

    @torch.inference_mode()
    def generate_ids(self, prompt: str, max_new_tokens: int) -> list[int]:
        """Greedy, returns token ids. Used for correctness checks."""
        prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)[0]
        seq = prompt_ids.tolist()
        eos_id = self.tokenizer.eos_token_id
        for _ in range(max_new_tokens):
            ids = torch.tensor([seq], dtype=torch.long, device=self.device)
            logits = self.model(ids).logits[0, -1]
            next_token = int(logits.argmax().item())
            seq.append(next_token)
            if next_token == eos_id:
                break
        return seq[len(prompt_ids):]


# ---------------------------------------------------------------------------
# CachedEngine — Day 6+: KV cache, quantization, eviction, prefix reuse
# ---------------------------------------------------------------------------

class CachedEngine:
    """
    Full engine with KV cache and all Phase 1-4 features.
    Behavior driven entirely by EngineConfig flags.
    """

    def __init__(self, config: EngineConfig):
        self.config = config
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"

        # eager attention needed to get attention weights for eviction
        needs_eager = config.eviction in ("h2o", "snapkv")
        self.model, self.tokenizer = _load_model(self.device, eager=needs_eager)
        self.needs_attn_weights = needs_eager
        self.eos_id = self.tokenizer.eos_token_id

        # prefix cache
        if config.prefix_cache == "hash":
            self._prefix_cache = HashPrefixCache(block_size=16)
        elif config.prefix_cache == "radix":
            self._prefix_cache = RadixCache(max_tokens=8192)
        else:
            self._prefix_cache = None

        # eviction budget (tokens to keep after eviction)
        self._eviction_budget = 512

    # ------------------------------------------------------------------
    # Low-level step interface (used by scheduler)
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def prefill(
        self,
        token_ids: list[int],
        params: Optional[SampleParams] = None,
        prefix_cache: Optional[DynamicCache] = None,
        prefix_len: int = 0,
    ) -> tuple[int, DynamicCache, float]:
        """Run prefill. Returns (first_token, kv_cache, ttft_ms)."""
        if params is None:
            params = SampleParams()
        t0 = time.perf_counter()

        if prefix_cache is not None and prefix_len > 0:
            # only run the suffix through the model
            suffix_ids = token_ids[prefix_len:]
            input_ids = torch.tensor([suffix_ids], dtype=torch.long, device=self.device)
            out = self.model(
                input_ids,
                past_key_values=prefix_cache,
                use_cache=True,
                output_attentions=self.needs_attn_weights,
            )
        else:
            input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
            out = self.model(
                input_ids,
                use_cache=True,
                output_attentions=self.needs_attn_weights,
            )

        past_kv: DynamicCache = out.past_key_values
        logits = out.logits[0, -1]
        attn = out.attentions if self.needs_attn_weights else None

        past_kv = self._apply_cache_ops(past_kv, attn)

        gen_t = None
        first_token = sample(logits, params, gen_t)
        ttft_ms = (time.perf_counter() - t0) * 1000
        return first_token, past_kv, ttft_ms

    @torch.inference_mode()
    def decode_step(
        self,
        last_token: int,
        past_kv: DynamicCache,
        generated: list[int],
        params: Optional[SampleParams] = None,
    ) -> tuple[int, DynamicCache, float]:
        """One decode step. Returns (next_token, updated_kv, step_ms)."""
        if params is None:
            params = SampleParams()
        t0 = time.perf_counter()

        input_ids = torch.tensor([[last_token]], dtype=torch.long, device=self.device)
        out = self.model(
            input_ids,
            past_key_values=past_kv,
            use_cache=True,
            output_attentions=self.needs_attn_weights,
        )
        past_kv = out.past_key_values
        logits = out.logits[0, -1]
        attn = out.attentions if self.needs_attn_weights else None

        past_kv = self._apply_cache_ops(past_kv, attn)

        gen_t = torch.tensor(generated, dtype=torch.long, device=self.device) if generated else None
        next_token = sample(logits, params, gen_t)
        step_ms = (time.perf_counter() - t0) * 1000
        return next_token, past_kv, step_ms

    def _apply_cache_ops(self, cache: DynamicCache, attn) -> DynamicCache:
        """Apply quantization and/or eviction to cache after a forward pass."""
        cfg = self.config
        # quantize first (before eviction so we evict already-compressed entries)
        if cfg.kv_quant_bits < 16:
            cache = compress_cache(cache, cfg.kv_quant_bits)
        # eviction
        if cfg.eviction == "h2o" and attn is not None:
            cache = evict_h2o(cache, attn, self._eviction_budget)
        elif cfg.eviction == "snapkv" and attn is not None:
            cache = evict_snapkv(cache, attn, self._eviction_budget)
        return cache

    # ------------------------------------------------------------------
    # High-level generate interface
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        params: Optional[SampleParams] = None,
    ) -> RequestMetrics:
        if params is None:
            params = SampleParams()

        request_id = str(uuid.uuid4())
        token_ids = self.tokenizer.encode(prompt)
        generated: list[int] = []
        itl_ms: list[float] = []

        # prefix cache lookup
        prefix_kv, prefix_len = None, 0
        if self._prefix_cache is not None:
            prefix_kv, prefix_len = self._prefix_cache.lookup(token_ids)

        t_start = time.perf_counter()
        token, past_kv, ttft_ms = self.prefill(
            token_ids, params, prefix_cache=prefix_kv, prefix_len=prefix_len
        )
        generated.append(token)

        for _ in range(max_new_tokens - 1):
            if token == self.eos_id:
                break
            token, past_kv, step_ms = self.decode_step(token, past_kv, generated, params)
            itl_ms.append(step_ms)
            generated.append(token)

        wall_time_s = time.perf_counter() - t_start

        # store prefix for future requests
        if self._prefix_cache is not None:
            self._prefix_cache.store(token_ids, past_kv)

        peak_gb = torch.mps.current_allocated_memory() / 1e9 if self.device == "mps" else 0.0
        kv_gb = cache_bytes(past_kv, bits=self.config.kv_quant_bits) / 1e9

        return RequestMetrics(
            request_id=request_id, prompt=prompt, max_new_tokens=max_new_tokens,
            ttft_ms=ttft_ms, itl_ms_per_token=itl_ms, total_tokens=len(generated),
            wall_time_s=wall_time_s, gpu_util_pct=0.0, peak_mem_gb=peak_gb + kv_gb,
        )

    def serve_request(
        self,
        prompt: str,
        max_new_tokens: int,
        params: Optional[SampleParams] = None,
    ) -> Generator[str, None, None]:
        """Streaming generation — yields one decoded token string at a time."""
        if params is None:
            params = SampleParams()

        token_ids = self.tokenizer.encode(prompt)
        generated: list[int] = []

        prefix_kv, prefix_len = None, 0
        if self._prefix_cache is not None:
            prefix_kv, prefix_len = self._prefix_cache.lookup(token_ids)

        token, past_kv, _ = self.prefill(
            token_ids, params, prefix_cache=prefix_kv, prefix_len=prefix_len
        )
        generated.append(token)
        yield self.tokenizer.decode([token], skip_special_tokens=True)

        for _ in range(max_new_tokens - 1):
            if token == self.eos_id:
                break
            token, past_kv, _ = self.decode_step(token, past_kv, generated, params)
            generated.append(token)
            yield self.tokenizer.decode([token], skip_special_tokens=True)

        if self._prefix_cache is not None:
            self._prefix_cache.store(token_ids, past_kv)


def make_engine(config: EngineConfig) -> "NaiveEngine | CachedEngine":
    """Factory — returns the right engine for the given config."""
    if config.cache_backend == "none":
        return NaiveEngine(config)
    return CachedEngine(config)

from __future__ import annotations

import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from transformers import DynamicCache

from mini_vllm.engine.sampler import SampleParams
from mini_vllm.instrumentation.metrics import RequestMetrics


# ---------------------------------------------------------------------------
# Request object
# ---------------------------------------------------------------------------

@dataclass
class Request:
    prompt: str
    max_new_tokens: int
    params: SampleParams = field(default_factory=SampleParams)
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # filled in by scheduler during execution
    generated: list[int] = field(default_factory=list)
    itl_ms: list[float] = field(default_factory=list)
    past_kv: Optional[DynamicCache] = None
    ttft_ms: float = 0.0
    t_start: float = field(default_factory=time.perf_counter)
    finished: bool = False


# ---------------------------------------------------------------------------
# Static batcher (Day 13)
# ---------------------------------------------------------------------------

class StaticBatcher:
    """Collect N requests, prefill all, decode until every sequence finishes.

    Failure mode: short requests waste their slot waiting for the longest one.
    This is the baseline that continuous batching fixes.
    """

    def __init__(self, engine, batch_size: int = 8):
        self.engine = engine
        self.batch_size = batch_size
        self._queue: deque[Request] = deque()

    def submit(self, req: Request):
        self._queue.append(req)

    def run_batch(self) -> list[RequestMetrics]:
        """Drain up to batch_size requests and run them to completion."""
        batch = [self._queue.popleft() for _ in range(min(self.batch_size, len(self._queue)))]
        if not batch:
            return []

        results = []
        for req in batch:
            # run each request independently (simulates fixed-slot batching)
            metrics = self.engine.generate(req.prompt, req.max_new_tokens, req.params)
            results.append(metrics)

        # track wasted slots: time all requests spent waiting for the last to finish
        total_tokens = [r.total_tokens for r in results]
        max_tokens = max(total_tokens)
        wasted_pct = 100 * (1 - sum(total_tokens) / (max_tokens * len(batch)))
        print(f"  [StaticBatcher] batch={len(batch)} max_tokens={max_tokens} wasted_slots={wasted_pct:.1f}%")
        return results


# ---------------------------------------------------------------------------
# Continuous scheduler (Day 14)
# ---------------------------------------------------------------------------

class ContinuousScheduler:
    """Iteration-level scheduling (Orca/vLLM style).

    Every step:
    1. Evict finished requests, free their KV slots.
    2. Admit waiting requests into free slots (run their prefill).
    3. Run one decode step for all active requests.
    4. Yield newly generated tokens per request.

    Key property: TTFT doesn't scale with batch size — a new request sees
    its first token within one prefill pass of being admitted, not after
    waiting for all current requests to finish.
    """

    def __init__(self, engine, max_active: int = 8):
        self.engine = engine
        self.max_active = max_active
        self._queue: deque[Request] = deque()
        self._active: dict[str, Request] = {}
        self.completed: list[RequestMetrics] = []

    def submit(self, req: Request):
        self._queue.append(req)

    def step(self) -> dict[str, str]:
        """Run one scheduler iteration. Returns {request_id: new_token_text}."""
        new_tokens: dict[str, str] = {}

        # 1. evict finished
        finished_ids = [rid for rid, r in self._active.items() if r.finished]
        for rid in finished_ids:
            req = self._active.pop(rid)
            wall = time.perf_counter() - req.t_start
            text = self.engine.tokenizer.decode(req.generated, skip_special_tokens=True)
            self.completed.append(RequestMetrics(
                request_id=req.request_id,
                prompt=req.prompt,
                max_new_tokens=req.max_new_tokens,
                ttft_ms=req.ttft_ms,
                itl_ms_per_token=req.itl_ms,
                total_tokens=len(req.generated),
                wall_time_s=wall,
                gpu_util_pct=0.0,
                peak_mem_gb=0.0,
            ))

        # 2. admit new requests (prefill them)
        while self._queue and len(self._active) < self.max_active:
            req = self._queue.popleft()
            req.t_start = time.perf_counter()
            token_ids = self.engine.tokenizer.encode(req.prompt)
            first_token, past_kv, ttft_ms = self.engine.prefill(
                token_ids, req.params
            )
            req.generated.append(first_token)
            req.past_kv = past_kv
            req.ttft_ms = ttft_ms
            self._active[req.request_id] = req
            tok_str = self.engine.tokenizer.decode([first_token], skip_special_tokens=True)
            new_tokens[req.request_id] = tok_str

        # 3. decode step for all active (that weren't just admitted)
        for rid, req in list(self._active.items()):
            if rid in new_tokens:
                continue  # just admitted, skip decode this step
            last_token = req.generated[-1]
            if last_token == self.engine.eos_id or len(req.generated) >= req.max_new_tokens:
                req.finished = True
                continue
            next_token, past_kv, step_ms = self.engine.decode_step(
                last_token, req.past_kv, req.generated, req.params
            )
            req.generated.append(next_token)
            req.past_kv = past_kv
            req.itl_ms.append(step_ms)
            tok_str = self.engine.tokenizer.decode([next_token], skip_special_tokens=True)
            new_tokens[rid] = tok_str
            if next_token == self.engine.eos_id or len(req.generated) >= req.max_new_tokens:
                req.finished = True

        return new_tokens

    def run_until_done(self) -> list[RequestMetrics]:
        """Keep stepping until all queued + active requests are finished."""
        while self._queue or self._active:
            self.step()
        return self.completed

    @property
    def queue_depth(self) -> int:
        return len(self._queue)

    @property
    def active_count(self) -> int:
        return len(self._active)

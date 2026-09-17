"""
Disaggregated prefill/decode (Day 30).

Prefill workers handle the compute-bound phase (full prompt forward pass).
Decode workers handle the memory-bandwidth-bound phase (one token at a time).
Communication: KV cache blob transferred via a queue between workers.

On a single machine: two threads sharing a queue. In production (DistServe):
separate GPU pools connected by high-bandwidth interconnect.

The key metric: TTFT *jitter* (p99 - p50), not mean TTFT.
Disaggregation reduces jitter because a new request's prefill doesn't block
ongoing decode steps on the decode worker.

Reference: https://arxiv.org/abs/2401.09670
"""
from __future__ import annotations

import queue
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Optional

from transformers import DynamicCache

from mini_vllm.engine.sampler import SampleParams


@dataclass
class PrefillResult:
    request_id: str
    token_ids: list[int]
    first_token: int
    past_kv: DynamicCache
    ttft_ms: float


class PrefillWorker(threading.Thread):
    """Runs prefill on incoming requests and sends KV blobs to decode queue."""

    def __init__(self, engine, decode_queue: queue.Queue):
        super().__init__(daemon=True)
        self.engine = engine
        self.decode_queue = decode_queue
        self._request_queue: queue.Queue = queue.Queue()
        self._running = True

    def submit(self, request_id: str, token_ids: list[int], params: Optional[SampleParams] = None):
        self._request_queue.put((request_id, token_ids, params or SampleParams()))

    def run(self):
        while self._running:
            try:
                request_id, token_ids, params = self._request_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            first_token, past_kv, ttft_ms = self.engine.prefill(token_ids, params)
            result = PrefillResult(
                request_id=request_id,
                token_ids=token_ids,
                first_token=first_token,
                past_kv=past_kv,
                ttft_ms=ttft_ms,
            )
            self.decode_queue.put(result)

    def stop(self):
        self._running = False


class DecodeWorker(threading.Thread):
    """Receives KV blobs from prefill worker, continues decode."""

    def __init__(self, engine, decode_queue: queue.Queue, results: list):
        super().__init__(daemon=True)
        self.engine = engine
        self.decode_queue = decode_queue
        self.results = results  # shared list that caller can read
        self._running = True
        self._active: dict[str, dict] = {}

    def run(self):
        while self._running:
            # admit new prefill results
            while True:
                try:
                    result: PrefillResult = self.decode_queue.get_nowait()
                    self._active[result.request_id] = {
                        "token_ids": result.token_ids,
                        "generated": [result.first_token],
                        "past_kv": result.past_kv,
                        "ttft_ms": result.ttft_ms,
                        "itl_ms": [],
                        "t_start": time.perf_counter(),
                        "params": SampleParams(temperature=0.0),
                    }
                except queue.Empty:
                    break

            if not self._active:
                time.sleep(0.001)
                continue

            # one decode step for each active request
            finished = []
            for rid, state in self._active.items():
                last_token = state["generated"][-1]
                eos = self.engine.eos_id
                if last_token == eos or len(state["generated"]) >= 256:
                    finished.append(rid)
                    continue
                next_token, past_kv, step_ms = self.engine.decode_step(
                    last_token, state["past_kv"], state["generated"], state["params"]
                )
                state["generated"].append(next_token)
                state["past_kv"] = past_kv
                state["itl_ms"].append(step_ms)

            for rid in finished:
                state = self._active.pop(rid)
                self.results.append({
                    "request_id": rid,
                    "ttft_ms": state["ttft_ms"],
                    "itl_ms": state["itl_ms"],
                    "total_tokens": len(state["generated"]),
                    "wall_time_s": time.perf_counter() - state["t_start"],
                })

    def stop(self):
        self._running = False


class DisaggregatedEngine:
    """Wires a PrefillWorker and DecodeWorker together on a single machine.

    For benchmarking jitter: compare p99 - p50 TTFT against ContinuousScheduler.
    """

    def __init__(self, engine):
        self.engine = engine
        self._kv_queue: queue.Queue = queue.Queue()
        self.results: list = []
        self._prefill_worker = PrefillWorker(engine, self._kv_queue)
        self._decode_worker = DecodeWorker(engine, self._kv_queue, self.results)

    def start(self):
        self._prefill_worker.start()
        self._decode_worker.start()

    def stop(self):
        self._prefill_worker.stop()
        self._decode_worker.stop()

    def submit(self, prompt: str, params: Optional[SampleParams] = None) -> str:
        request_id = str(uuid.uuid4())
        token_ids = self.engine.tokenizer.encode(prompt)
        self._prefill_worker.submit(request_id, token_ids, params)
        return request_id

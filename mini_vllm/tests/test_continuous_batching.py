"""Compare static batching vs continuous batching on a mixed-length workload."""
from __future__ import annotations

import time
from mini_vllm.configs.flags import EngineConfig
from mini_vllm.engine.model_runner import CachedEngine
from mini_vllm.engine.sampler import SampleParams
from mini_vllm.engine.scheduler import StaticBatcher, ContinuousScheduler, Request

WORKLOAD = [
    ("What is 2+2?",                                              16),
    ("What is the capital of France?",                            16),
    ("Name one programming language.",                            16),
    ("Explain how transformers work in natural language processing.", 128),
    ("Describe the water cycle in detail.",                       128),
    ("Write a Python function that computes Fibonacci numbers.",  128),
    ("What causes lightning?",                                     16),
    ("How does a neural network learn?",                          128),
]


def make_requests():
    params = SampleParams(temperature=0.0)
    return [Request(prompt=p, max_new_tokens=t, params=params) for p, t in WORKLOAD]


def run_static(engine, batch_size=8):
    batcher = StaticBatcher(engine, batch_size=batch_size)
    for req in make_requests():
        batcher.submit(req)
    results = []
    while batcher._queue:
        results.extend(batcher.run_batch())
    return results


def run_continuous(engine, max_active=4):
    scheduler = ContinuousScheduler(engine, max_active=max_active)
    for req in make_requests():
        scheduler.submit(req)
    return scheduler.run_until_done()


def summarize(label, results):
    n = len(results)
    avg_ttft = sum(r.ttft_ms for r in results) / n
    avg_tput = sum(r.throughput_tok_s for r in results) / n
    print(f"\n{label}")
    print(f"  requests:        {n}")
    print(f"  avg TTFT:        {avg_ttft:.1f} ms")
    print(f"  avg throughput:  {avg_tput:.1f} tok/s")


def main():
    engine = CachedEngine(EngineConfig())

    static = run_static(engine, batch_size=8)
    summarize("static batching (batch=8)", static)

    cont = run_continuous(engine, max_active=4)
    summarize("continuous batching (max_active=4)", cont)


if __name__ == "__main__":
    main()

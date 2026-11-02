"""Demonstrate static batching failure modes: wasted slots + TTFT scaling."""
from __future__ import annotations

from mini_vllm.configs.flags import EngineConfig
from mini_vllm.engine.model_runner import CachedEngine
from mini_vllm.engine.sampler import SampleParams
from mini_vllm.engine.scheduler import StaticBatcher, Request

# mixed-length workload: short + long requests together
WORKLOAD = [
    ("What is 2+2?",                                           16),
    ("What is the capital of France?",                         16),
    ("Name one programming language.",                         16),
    ("Explain how transformers work in natural language processing.", 128),
    ("Describe the water cycle in detail.",                   128),
    ("Write a Python function that computes Fibonacci numbers.", 128),
    ("What causes lightning?",                                  16),
    ("How does a neural network learn?",                       128),
]


def run_at_batch_size(engine, batch_size):
    batcher = StaticBatcher(engine, batch_size=batch_size)
    params = SampleParams(temperature=0.0)
    for prompt, max_tokens in WORKLOAD:
        batcher.submit(Request(prompt=prompt, max_new_tokens=max_tokens, params=params))

    all_results = []
    while batcher._queue:
        results = batcher.run_batch()
        all_results.extend(results)

    avg_ttft = sum(r.ttft_ms for r in all_results) / len(all_results)
    return avg_ttft


def main():
    engine = CachedEngine(EngineConfig())
    print(f"\n{'batch size':>12} {'avg TTFT (ms)':>15} {'wasted slots':>14}")
    print("-" * 44)
    for bs in [1, 2, 4, 8]:
        avg_ttft = run_at_batch_size(engine, bs)
        # wasted slot % is printed inside StaticBatcher.run_batch()
        print(f"{bs:>12} {avg_ttft:>15.1f}")


if __name__ == "__main__":
    main()

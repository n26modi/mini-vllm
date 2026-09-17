"""
Concurrent load test (Day 15).

Ramps from 1 → N concurrent requests, measures p50/p95/p99 TTFT and ITL,
finds the saturation point where throughput stops growing.

Run:
    python -m mini_vllm.bench.load_test --max-concurrency 8
"""
from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from dataclasses import dataclass

from mini_vllm.configs.flags import EngineConfig
from mini_vllm.engine.model_runner import CachedEngine
from mini_vllm.engine.sampler import SampleParams
from mini_vllm.engine.scheduler import ContinuousScheduler, Request

PROMPTS = [
    "Explain how transformers work.",
    "What is the capital of France?",
    "Write a Python function to sort a list.",
    "Describe the water cycle.",
    "What causes lightning?",
    "How does a neural network learn?",
    "Explain quantum entanglement simply.",
    "What is the difference between RAM and ROM?",
]


@dataclass
class LatencyStats:
    concurrency: int
    throughput_tok_s: float
    ttft_p50: float
    ttft_p95: float
    ttft_p99: float
    itl_p50: float
    itl_p95: float


def _percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    data = sorted(data)
    idx = int(len(data) * p / 100)
    return data[min(idx, len(data) - 1)]


def run_concurrency_level(
    engine: CachedEngine,
    concurrency: int,
    requests_per_level: int = 16,
    max_new_tokens: int = 64,
) -> LatencyStats:
    """Run requests_per_level requests at a given concurrency and collect stats."""
    scheduler = ContinuousScheduler(engine, max_active=concurrency)
    params = SampleParams(temperature=0.0)

    prompts = (PROMPTS * ((requests_per_level // len(PROMPTS)) + 1))[:requests_per_level]
    for p in prompts:
        scheduler.submit(Request(prompt=p, max_new_tokens=max_new_tokens, params=params))

    t_start = time.perf_counter()
    results = scheduler.run_until_done()
    wall = time.perf_counter() - t_start

    ttfts = [r.ttft_ms for r in results]
    itls = [ms for r in results for ms in r.itl_ms_per_token]
    total_tokens = sum(r.total_tokens for r in results)
    throughput = total_tokens / wall if wall > 0 else 0

    return LatencyStats(
        concurrency=concurrency,
        throughput_tok_s=throughput,
        ttft_p50=_percentile(ttfts, 50),
        ttft_p95=_percentile(ttfts, 95),
        ttft_p99=_percentile(ttfts, 99),
        itl_p50=_percentile(itls, 50),
        itl_p95=_percentile(itls, 95),
    )


def run_saturation_curve(
    engine: CachedEngine,
    max_concurrency: int = 8,
    requests_per_level: int = 16,
) -> list[LatencyStats]:
    levels = [1, 2, 4] + [c for c in range(4, max_concurrency + 1, 2) if c > 4]
    levels = sorted(set(levels))
    results = []
    print(f"\n{'concurrency':>12} {'tok/s':>10} {'ttft p50':>10} {'ttft p95':>10} {'ttft p99':>10} {'itl p50':>10}")
    print("-" * 70)
    for c in levels:
        stats = run_concurrency_level(engine, c, requests_per_level)
        results.append(stats)
        print(
            f"{c:>12} {stats.throughput_tok_s:>10.1f} "
            f"{stats.ttft_p50:>10.1f} {stats.ttft_p95:>10.1f} {stats.ttft_p99:>10.1f} "
            f"{stats.itl_p50:>10.1f}"
        )
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--requests-per-level", type=int, default=16)
    args = parser.parse_args()

    config = EngineConfig()
    engine = CachedEngine(config)
    run_saturation_curve(engine, args.max_concurrency, args.requests_per_level)


if __name__ == "__main__":
    main()

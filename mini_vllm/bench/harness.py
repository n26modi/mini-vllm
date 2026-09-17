"""
Benchmark harness — runs a workload against the engine and emits one ablation row.

Usage:
    python -m mini_vllm.bench.harness
    python -m mini_vllm.bench.harness --stub   # zero-metric stub (no model load)
    python -m mini_vllm.bench.harness --workload mini_vllm/bench/workloads/default.jsonl
"""
from __future__ import annotations

import argparse
import json
import uuid
from dataclasses import asdict
from pathlib import Path

from mini_vllm.configs.flags import EngineConfig
from mini_vllm.instrumentation.metrics import MetricsLogger, RequestMetrics


def _stub_run(config: EngineConfig, prompt: str, max_new_tokens: int) -> RequestMetrics:
    """Zero-metric placeholder. Useful for CI / harness smoke tests."""
    return RequestMetrics(
        request_id=str(uuid.uuid4()),
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        ttft_ms=0.0,
        itl_ms_per_token=[],
        total_tokens=0,
        wall_time_s=0.0,
        gpu_util_pct=0.0,
        peak_mem_gb=0.0,
    )


def run(
    config: EngineConfig,
    workload_path: "str | Path",
    output_path: "str | Path",
    stub: bool = False,
) -> None:
    workload = [json.loads(l) for l in Path(workload_path).read_text().splitlines() if l.strip()]

    if stub:
        engine = None
    else:
        from mini_vllm.engine.model_runner import CachedEngine
        engine = CachedEngine(config)

    results: list[RequestMetrics] = []
    with MetricsLogger(output_path) as logger:
        for item in workload:
            if stub or engine is None:
                metrics = _stub_run(config, item["prompt"], item["max_new_tokens"])
            else:
                metrics = engine.generate(item["prompt"], item["max_new_tokens"])
            logger.log(metrics)
            results.append(metrics)

    _print_summary(config, results)


def _print_summary(config: EngineConfig, results: list[RequestMetrics]) -> None:
    n = len(results)
    avg_ttft = sum(r.ttft_ms for r in results) / n if n else 0
    avg_itl = sum(r.mean_itl_ms for r in results) / n if n else 0
    avg_tput = sum(r.throughput_tok_s for r in results) / n if n else 0
    avg_mem = sum(r.peak_mem_gb for r in results) / n if n else 0

    print("\n" + "=" * 60)
    print("ablation row")
    print("=" * 60)
    for k, v in asdict(config).items():
        print(f"  {k}: {v}")
    print("-" * 60)
    print(f"  requests:        {n}")
    print(f"  avg ttft:        {avg_ttft:.1f} ms")
    print(f"  avg itl:         {avg_itl:.1f} ms/token")
    print(f"  avg throughput:  {avg_tput:.1f} tok/s")
    print(f"  avg peak mem:    {avg_mem:.3f} GB")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", default="mini_vllm/bench/workloads/default.jsonl")
    parser.add_argument("--output", default="results/metrics.jsonl")
    parser.add_argument("--stub", action="store_true", help="skip model load, emit zero metrics")
    args = parser.parse_args()

    config = EngineConfig()
    run(config, args.workload, args.output, stub=args.stub)


if __name__ == "__main__":
    main()

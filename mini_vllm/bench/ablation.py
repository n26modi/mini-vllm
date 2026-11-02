"""
Day 31 capstone: run the full ablation table.

Each row is one EngineConfig combination. Columns: TTFT, ITL, throughput, memory.

Usage:
    python -m mini_vllm.bench.ablation
    python -m mini_vllm.bench.ablation --workload mini_vllm/bench/workloads/ablation.jsonl
    python -m mini_vllm.bench.ablation --output results/ablation_table.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict
from pathlib import Path

from mini_vllm.configs.flags import EngineConfig
from mini_vllm.instrumentation.metrics import RequestMetrics


CONFIGS: list[tuple[str, EngineConfig]] = [
    ("baseline",             EngineConfig()),
    ("kv_quant_8bit",        EngineConfig(kv_quant_bits=8)),
    ("kv_quant_4bit",        EngineConfig(kv_quant_bits=4)),
    ("eviction_h2o",         EngineConfig(eviction="h2o")),
    ("eviction_snapkv",      EngineConfig(eviction="snapkv")),
    ("prefix_hash",          EngineConfig(prefix_cache="hash")),
    ("prefix_radix",         EngineConfig(prefix_cache="radix")),
    ("quant8_h2o",           EngineConfig(kv_quant_bits=8, eviction="h2o")),
    ("quant4_snapkv",        EngineConfig(kv_quant_bits=4, eviction="snapkv")),
    ("quant8_radix",         EngineConfig(kv_quant_bits=8, prefix_cache="radix")),
]


def _run_config(
    label: str,
    config: EngineConfig,
    workload: list[dict],
) -> dict:
    from mini_vllm.engine.model_runner import CachedEngine

    engine = CachedEngine(config)
    results: list[RequestMetrics] = []

    for item in workload:
        m = engine.generate(item["prompt"], item["max_new_tokens"])
        results.append(m)

    n = len(results)
    avg_ttft  = statistics.mean(r.ttft_ms for r in results)
    avg_itl   = statistics.mean(r.mean_itl_ms for r in results)
    avg_tput  = statistics.mean(r.throughput_tok_s for r in results)
    avg_mem   = statistics.mean(r.peak_mem_gb for r in results)
    total_tok = sum(r.total_tokens for r in results)

    return {
        "label":         label,
        "config":        asdict(config),
        "n_requests":    n,
        "total_tokens":  total_tok,
        "avg_ttft_ms":   round(avg_ttft, 1),
        "avg_itl_ms":    round(avg_itl, 1),
        "avg_tput_tok_s": round(avg_tput, 1),
        "avg_mem_gb":    round(avg_mem, 3),
    }


def _print_table(rows: list[dict]) -> None:
    header = f"{'config':<22} {'TTFT ms':>8} {'ITL ms':>8} {'tok/s':>8} {'mem GB':>8}"
    print()
    print("=" * len(header))
    print("Day 31 ablation table")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['label']:<22} "
            f"{r['avg_ttft_ms']:>8.1f} "
            f"{r['avg_itl_ms']:>8.1f} "
            f"{r['avg_tput_tok_s']:>8.1f} "
            f"{r['avg_mem_gb']:>8.3f}"
        )
    print("=" * len(header))
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workload",
        default="mini_vllm/bench/workloads/ablation.jsonl",
    )
    parser.add_argument(
        "--output",
        default="results/ablation_table.json",
    )
    parser.add_argument(
        "--configs",
        nargs="*",
        help="run only these config labels (space-separated)",
    )
    args = parser.parse_args()

    workload = [
        json.loads(l)
        for l in Path(args.workload).read_text().splitlines()
        if l.strip()
    ]

    configs = CONFIGS
    if args.configs:
        configs = [(l, c) for l, c in CONFIGS if l in args.configs]

    rows: list[dict] = []
    t_total = time.perf_counter()

    for i, (label, config) in enumerate(configs):
        print(f"[{i+1}/{len(configs)}] running {label} ...")
        t0 = time.perf_counter()
        row = _run_config(label, config, workload)
        elapsed = time.perf_counter() - t0
        rows.append(row)
        print(f"      done in {elapsed:.0f}s  —  ttft={row['avg_ttft_ms']:.0f}ms  tput={row['avg_tput_tok_s']:.1f} tok/s")

    total_elapsed = time.perf_counter() - t_total
    print(f"\nAll {len(rows)} configs done in {total_elapsed/60:.1f} min")

    _print_table(rows)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"Results saved to {out}")


if __name__ == "__main__":
    main()

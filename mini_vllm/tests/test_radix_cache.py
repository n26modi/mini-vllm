"""Compare no-cache vs hash vs radix prefix cache on a shared system prompt workload."""
from __future__ import annotations

from mini_vllm.configs.flags import EngineConfig
from mini_vllm.engine.model_runner import CachedEngine

SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Always answer concisely and accurately. "
    "If you are unsure, say so. "
    "Do not make up facts. "
    "Be direct and clear in your responses. " * 8
)

QUESTIONS = [
    " What is the capital of France?",
    " What is 2 + 2?",
    " Name three programming languages.",
]


def run_workload(label, config):
    engine = CachedEngine(config)
    prompts = [SYSTEM_PROMPT + q for q in QUESTIONS]

    print(f"\n{label}")
    print("-" * 44)
    results = []
    for i, p in enumerate(prompts):
        m = engine.generate(p, max_new_tokens=32)
        tag = "cold" if i == 0 else "warm"
        print(f"  request {i+1} ({tag}):  {m.ttft_ms:.1f} ms TTFT")
        results.append(m.ttft_ms)

    if len(results) > 1:
        avg_warm = sum(results[1:]) / len(results[1:])
        saved = (1 - avg_warm / results[0]) * 100
        print(f"  avg warm speedup:  {results[0]/avg_warm:.2f}x  ({saved:.0f}% TTFT saved)")


def main():
    run_workload("no prefix cache (baseline)", EngineConfig(prefix_cache="off"))
    run_workload("hash prefix cache", EngineConfig(prefix_cache="hash"))
    run_workload("radix prefix cache", EngineConfig(prefix_cache="radix"))


if __name__ == "__main__":
    main()

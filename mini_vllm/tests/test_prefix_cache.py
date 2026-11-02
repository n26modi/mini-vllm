"""Quick prefix cache hit/miss TTFT comparison."""
from __future__ import annotations

import time
from mini_vllm.configs.flags import EngineConfig
from mini_vllm.engine.model_runner import CachedEngine

SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Always answer concisely and accurately. "
    "If you are unsure, say so. "
    "Do not make up facts. "
    "Be direct and clear in your responses. " * 8  # ~200 tokens
)

QUESTION_A = " What is the capital of France?"
QUESTION_B = " What is 2 + 2?"


def ttft(engine, prompt, max_new_tokens=32):
    t0 = time.perf_counter()
    gen = engine.serve_request(prompt, max_new_tokens=max_new_tokens)
    next(gen)
    return (time.perf_counter() - t0) * 1000


def main():
    config = EngineConfig(prefix_cache="hash")
    engine = CachedEngine(config)

    prompt_a = SYSTEM_PROMPT + QUESTION_A
    prompt_b = SYSTEM_PROMPT + QUESTION_B

    print(f"System prompt length: ~{len(prompt_a.split())} tokens (approx)")
    print()

    # cold run — no cache
    t1 = ttft(engine, prompt_a)
    print(f"Request 1 (cold, no cache):     TTFT = {t1:.1f} ms")

    # warm run — shared system prompt should be cached now
    t2 = ttft(engine, prompt_b)
    print(f"Request 2 (warm, shared prefix): TTFT = {t2:.1f} ms")

    speedup = t1 / t2 if t2 > 0 else 0
    saved_pct = (1 - t2 / t1) * 100 if t1 > 0 else 0
    print(f"\nSpeedup: {speedup:.2f}x  ({saved_pct:.0f}% TTFT reduction)")


if __name__ == "__main__":
    main()

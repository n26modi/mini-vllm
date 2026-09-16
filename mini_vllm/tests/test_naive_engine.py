"""
Acceptance test: greedy output of NaiveEngine matches HF model.generate() token-for-token.
Also logs per-token time to confirm O(n²) degradation.

Run:
    python -m mini_vllm.tests.test_naive_engine
"""
from __future__ import annotations

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mini_vllm.configs.flags import EngineConfig
from mini_vllm.engine.model_runner import MODEL_ID, NaiveEngine
from mini_vllm.engine.sampler import SampleParams


def hf_greedy(tokenizer, model, prompt: str, max_new_tokens: int, device: str) -> list[int]:
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            use_cache=False,  # match naive engine: full recompute every step
        )
    return out[0, input_ids.shape[1]:].tolist()


def test_greedy_matches_hf():
    engine = NaiveEngine(EngineConfig())
    prompt = "The capital of France is"
    max_new_tokens = 10

    our_ids = engine.generate_ids(prompt, max_new_tokens)
    hf_ids = hf_greedy(engine.tokenizer, engine.model, prompt, max_new_tokens, engine.device)

    our_text = engine.tokenizer.decode(our_ids)
    hf_text = engine.tokenizer.decode(hf_ids)
    print(f"  ours: {our_ids}  →  '{our_text}'")
    print(f"  hf:   {hf_ids}  →  '{hf_text}'")

    assert our_ids == hf_ids, f"mismatch: {our_ids} vs {hf_ids}"
    print("  PASS: greedy output matches HF token-for-token\n")


def test_itl_degrades_without_cache():
    """Per-token time should visibly worsen as sequence grows — confirms O(n²) behavior."""
    engine = NaiveEngine(EngineConfig())
    prompt = "Explain the history of artificial intelligence in detail:"
    params = SampleParams(temperature=0.0)

    prompt_ids = engine.tokenizer.encode(prompt, return_tensors="pt").to(engine.device)[0]
    seq = prompt_ids.tolist()
    eos_id = engine.tokenizer.eos_token_id

    print("  step  |  seq_len  |  ms/step")
    print("  ------+-----------+---------")

    step_times = []
    with torch.inference_mode():
        for step in range(40):
            t = time.perf_counter()
            ids = torch.tensor([seq], dtype=torch.long, device=engine.device)
            logits = engine.model(ids).logits[0, -1].float()
            next_token = int(logits.argmax().item())
            elapsed_ms = (time.perf_counter() - t) * 1000
            step_times.append(elapsed_ms)
            seq.append(next_token)
            if step % 10 == 0:
                print(f"  {step:4d}  |  {len(seq):7d}  |  {elapsed_ms:.1f} ms")
            if next_token == eos_id:
                break

    # confirm the trend: second half slower than first half
    mid = len(step_times) // 2
    first_half_avg = sum(step_times[:mid]) / mid
    second_half_avg = sum(step_times[mid:]) / (len(step_times) - mid)
    ratio = second_half_avg / first_half_avg
    print(f"\n  first-half avg: {first_half_avg:.1f} ms  |  second-half avg: {second_half_avg:.1f} ms  |  ratio: {ratio:.2f}x")
    assert ratio > 1.1, f"expected second half to be slower (ratio={ratio:.2f})"
    print("  PASS: per-token time worsens with sequence length\n")


if __name__ == "__main__":
    print("=== test: greedy matches HF ===")
    test_greedy_matches_hf()
    print("=== test: ITL degrades without cache ===")
    test_itl_degrades_without_cache()
    print("All tests passed.")

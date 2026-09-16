from __future__ import annotations

import time
import uuid
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mini_vllm.configs.flags import EngineConfig
from mini_vllm.engine.sampler import SampleParams, sample
from mini_vllm.instrumentation.metrics import RequestMetrics

MODEL_ID = "Qwen/Qwen3-0.6B"


class NaiveEngine:
    """
    Naive decode loop — no KV cache. Full forward pass over the entire sequence
    on every step. O(n²) total compute. Baseline to beat on Day 5.
    """

    def __init__(self, config: EngineConfig):
        self.config = config
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Loading {MODEL_ID} on {self.device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16
        ).to(self.device)
        self.model.eval()
        print("Ready.")

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        params: Optional[SampleParams] = None,
    ) -> RequestMetrics:
        if params is None:
            params = SampleParams()

        request_id = str(uuid.uuid4())
        prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)[0]
        seq = prompt_ids.tolist()
        generated: list[int] = []
        itl_ms: list[float] = []
        eos_id = self.tokenizer.eos_token_id

        t_start = time.perf_counter()
        ttft_ms = 0.0

        for step in range(max_new_tokens):
            t_step = time.perf_counter()

            ids = torch.tensor([seq], dtype=torch.long, device=self.device)
            logits = self.model(ids).logits[0, -1].float()  # [vocab_size]

            gen_tensor = (
                torch.tensor(generated, dtype=torch.long, device=self.device)
                if generated else None
            )
            next_token = sample(logits, params, gen_tensor)

            elapsed_ms = (time.perf_counter() - t_step) * 1000

            if step == 0:
                ttft_ms = (time.perf_counter() - t_start) * 1000
            else:
                itl_ms.append(elapsed_ms)

            seq.append(next_token)
            generated.append(next_token)

            if next_token == eos_id:
                break

        wall_time_s = time.perf_counter() - t_start
        peak_mem_gb = (
            torch.mps.current_allocated_memory() / 1e9
            if self.device == "mps"
            else 0.0
        )

        return RequestMetrics(
            request_id=request_id,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            ttft_ms=ttft_ms,
            itl_ms_per_token=itl_ms,
            total_tokens=len(generated),
            wall_time_s=wall_time_s,
            gpu_util_pct=0.0,
            peak_mem_gb=peak_mem_gb,
        )

    @torch.inference_mode()
    def generate_ids(self, prompt: str, max_new_tokens: int) -> list[int]:
        """Greedy generation returning raw token ids. Used for correctness checks."""
        params = SampleParams(temperature=0.0)
        prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)[0]
        seq = prompt_ids.tolist()
        eos_id = self.tokenizer.eos_token_id

        for _ in range(max_new_tokens):
            ids = torch.tensor([seq], dtype=torch.long, device=self.device)
            logits = self.model(ids).logits[0, -1].float()
            next_token = int(logits.argmax().item())
            seq.append(next_token)
            if next_token == eos_id:
                break

        return seq[len(prompt_ids):]

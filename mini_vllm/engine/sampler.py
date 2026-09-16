from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class SampleParams:
    temperature: float = 1.0
    top_k: int = 0          # 0 = disabled
    top_p: float = 1.0      # 1.0 = disabled
    repetition_penalty: float = 1.0  # 1.0 = disabled


def sample(
    logits: torch.Tensor,
    params: SampleParams,
    generated_ids: Optional[torch.Tensor] = None,
) -> int:
    """Sample next token from logits [vocab_size]. Returns token id as int."""
    logits = logits.clone()

    # repetition penalty: reduce probability of already-generated tokens
    if params.repetition_penalty != 1.0 and generated_ids is not None and generated_ids.numel() > 0:
        scores = logits[generated_ids]
        scores = torch.where(scores < 0, scores * params.repetition_penalty, scores / params.repetition_penalty)
        logits[generated_ids] = scores

    # greedy shortcut
    if params.temperature == 0.0:
        return int(logits.argmax().item())

    # temperature scaling
    if params.temperature != 1.0:
        logits = logits / params.temperature

    # top-k: zero out all but the k highest logits
    if params.top_k > 0:
        k = min(params.top_k, logits.size(-1))
        threshold = logits.topk(k).values[-1]
        logits = logits.masked_fill(logits < threshold, float("-inf"))

    # top-p (nucleus): keep smallest set of tokens whose cumulative prob >= p
    if params.top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumprobs = torch.cumsum(probs, dim=-1)
        # shift right: always keep the token that first crosses the threshold
        remove = (cumprobs - probs) > params.top_p
        sorted_logits[remove] = float("-inf")
        logits = torch.empty_like(logits).scatter_(0, sorted_idx, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())

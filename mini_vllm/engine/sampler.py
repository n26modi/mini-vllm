from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class SampleParams:
    temperature: float = 1.0
    top_k: int = 0               # 0 = disabled
    top_p: float = 1.0           # 1.0 = disabled
    repetition_penalty: float = 1.0  # 1.0 = disabled


def sample(
    logits: torch.Tensor,
    params: SampleParams,
    generated_ids: Optional[torch.Tensor] = None,
) -> int:
    """Sample next token from logits [vocab_size]. Returns token id as int."""
    logits = logits.clone()

    if params.repetition_penalty != 1.0 and generated_ids is not None and generated_ids.numel() > 0:
        scores = logits[generated_ids]
        scores = torch.where(scores < 0, scores * params.repetition_penalty, scores / params.repetition_penalty)
        logits[generated_ids] = scores

    if params.temperature == 0.0:
        return int(logits.argmax().item())

    if params.temperature != 1.0:
        logits = logits / params.temperature

    if params.top_k > 0:
        k = min(params.top_k, logits.size(-1))
        threshold = logits.topk(k).values[-1]
        logits = logits.masked_fill(logits < threshold, float("-inf"))

    if params.top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumprobs = torch.cumsum(probs, dim=-1)
        remove = (cumprobs - probs) > params.top_p
        sorted_logits[remove] = float("-inf")
        logits = torch.empty_like(logits).scatter_(0, sorted_idx, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


# ---------------------------------------------------------------------------
# Speculative decoding — Day 21: draft + verify (greedy)
# ---------------------------------------------------------------------------

class DraftVerifyDecoder:
    """Standard speculative decoding: draft model proposes K tokens,
    target verifies all K in one parallel forward pass.

    Cost intuition: verifying K tokens in parallel costs barely more than
    verifying 1 (memory-bound regime), so accepted draft tokens are nearly free.

    Requires a compatible draft model — ideally a smaller model from the same
    family (e.g. a pruned Qwen3-0.6B) for high acceptance rates.
    """

    def __init__(self, draft_engine, target_engine, k: int = 4):
        self.draft = draft_engine
        self.target = target_engine
        self.k = k
        self._total_proposed = 0
        self._total_accepted = 0

    @property
    def acceptance_rate(self) -> float:
        if self._total_proposed == 0:
            return 0.0
        return self._total_accepted / self._total_proposed

    @torch.inference_mode()
    def step(
        self,
        token_ids: list[int],
        draft_kv,
        target_kv,
        params: Optional[SampleParams] = None,
    ) -> tuple[list[int], object, object]:
        """One speculative step. Returns (accepted_tokens, new_draft_kv, new_target_kv)."""
        from transformers import DynamicCache

        if params is None:
            params = SampleParams(temperature=0.0)

        device = self.target.device

        # --- draft: propose K tokens ---
        draft_tokens = []
        current_kv = draft_kv
        for _ in range(self.k):
            last = draft_tokens[-1] if draft_tokens else token_ids[-1]
            inp = torch.tensor([[last]], dtype=torch.long, device=device)
            out = self.draft.model(inp, past_key_values=current_kv, use_cache=True)
            current_kv = out.past_key_values
            tok = int(out.logits[0, -1].argmax().item())
            draft_tokens.append(tok)
        new_draft_kv = current_kv

        # --- target: verify all K draft tokens in one parallel pass ---
        # input is the last real token + all K draft tokens
        verify_ids = [token_ids[-1]] + draft_tokens
        inp = torch.tensor([verify_ids], dtype=torch.long, device=device)
        out = self.target.model(inp, past_key_values=target_kv, use_cache=True)
        new_target_kv = out.past_key_values
        # logits[i] is the prediction at position i (for position i+1)
        target_logits = out.logits[0]  # [K+1, vocab]

        # --- accept/reject left to right ---
        accepted = []
        self._total_proposed += self.k
        for i, dt in enumerate(draft_tokens):
            target_tok = int(target_logits[i].argmax().item())
            if target_tok == dt:
                accepted.append(dt)
                self._total_accepted += 1
            else:
                # take target's correction token and stop
                accepted.append(target_tok)
                break
        else:
            # all K accepted — take the bonus target token too
            accepted.append(int(target_logits[self.k].argmax().item()))
            self._total_accepted += 1

        return accepted, new_draft_kv, new_target_kv


# ---------------------------------------------------------------------------
# Lookahead decoding — Day 23: no auxiliary model required
# ---------------------------------------------------------------------------

class LookaheadDecoder:
    """Jacobi-iteration n-gram speculation.

    No auxiliary model. Builds a pool of n-gram candidates observed during
    generation. At each step: propose continuations from the pool,
    verify against the target model, accept the longest matching prefix.

    The longer the generation and more repetitive the content, the better
    the acceptance rate — because the n-gram pool grows with seen sequences.
    """

    def __init__(self, target_engine, n: int = 4, window: int = 5):
        self.target = target_engine
        self.n = n            # n-gram length
        self.window = window  # how many candidates to try per step
        self._ngram_pool: dict[tuple, list[int]] = {}  # prefix -> next tokens seen

    def _update_pool(self, token_ids: list[int]):
        """Add all n-grams from a sequence to the pool."""
        for i in range(len(token_ids) - self.n):
            prefix = tuple(token_ids[i : i + self.n - 1])
            next_tok = token_ids[i + self.n - 1]
            if prefix not in self._ngram_pool:
                self._ngram_pool[prefix] = []
            if next_tok not in self._ngram_pool[prefix]:
                self._ngram_pool[prefix].append(next_tok)

    @torch.inference_mode()
    def step(
        self,
        token_ids: list[int],
        past_kv,
        params: Optional[SampleParams] = None,
    ) -> tuple[list[int], object]:
        """Returns (new_tokens, updated_kv). Falls back to single token if no candidates."""
        if params is None:
            params = SampleParams(temperature=0.0)

        device = self.target.device

        # look up n-gram candidates for current suffix
        suffix = tuple(token_ids[-(self.n - 1):])
        candidates = self._ngram_pool.get(suffix, [])[:self.window]

        if not candidates:
            # no candidates — standard decode step
            inp = torch.tensor([[token_ids[-1]]], dtype=torch.long, device=device)
            out = self.target.model(inp, past_key_values=past_kv, use_cache=True)
            tok = int(out.logits[0, -1].argmax().item())
            self._update_pool(token_ids + [tok])
            return [tok], out.past_key_values

        # try first candidate: verify the n-gram in one pass
        candidate_ids = [token_ids[-1]] + candidates
        inp = torch.tensor([candidate_ids], dtype=torch.long, device=device)
        out = self.target.model(inp, past_key_values=past_kv, use_cache=True)
        target_logits = out.logits[0]  # [len, vocab]

        accepted = []
        for i, cand_tok in enumerate(candidates):
            target_tok = int(target_logits[i].argmax().item())
            if target_tok == cand_tok:
                accepted.append(cand_tok)
            else:
                accepted.append(target_tok)
                break
        else:
            accepted.append(int(target_logits[len(candidates)].argmax().item()))

        self._update_pool(token_ids + accepted)
        return accepted, out.past_key_values


# ---------------------------------------------------------------------------
# Medusa stub — Day 22
# ---------------------------------------------------------------------------

class MedusaDecoder:
    """
    Medusa: k extra linear decoding heads on the target model's last hidden state.
    Each head i predicts token at position t+i+1.
    Verification uses a tree of candidate continuations.

    No separate draft model — heads share target model parameters.
    Requires trained Medusa heads (not available publicly for Qwen3-0.6B).

    Reference: https://arxiv.org/abs/2401.10774
    """
    def __init__(self, target_engine, heads=None):
        if heads is None:
            raise NotImplementedError(
                "MedusaDecoder requires pre-trained Medusa heads for Qwen3-0.6B. "
                "Train heads on a text corpus or find a compatible checkpoint."
            )
        self.target = target_engine
        self.heads = heads  # list of nn.Linear([hidden_dim, vocab_size])


# ---------------------------------------------------------------------------
# EAGLE stub — Day 24
# ---------------------------------------------------------------------------

class EAGLEDecoder:
    """
    EAGLE: feature-level speculation using the target model's second-to-last
    layer features. The EAGLE head predicts future feature vectors,
    not tokens — higher acceptance rates than draft-model approaches.

    Requires a trained EAGLE head specific to Qwen3-0.6B.
    Reference: https://arxiv.org/abs/2401.15077
    """
    def __init__(self, target_engine, eagle_head=None):
        if eagle_head is None:
            raise NotImplementedError(
                "EAGLEDecoder requires a trained EAGLE head for Qwen3-0.6B."
            )
        self.target = target_engine
        self.head = eagle_head

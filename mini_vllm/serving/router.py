"""
Difficulty-based router (Day 27).

Routes each request to a tier (adapter ID) based on estimated difficulty.
Easy requests go to a cheaper/faster adapter; hard requests get the full model.
Saves $/1M tokens when workload is mixed difficulty.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class RoutingDecision:
    tier: str
    reason: str


class DifficultyRouter:
    """Heuristic router: classifies requests by prompt length and keywords.

    Production routers use a small classifier model. This version uses
    simple rules to demonstrate the routing concept and $/token tradeoff.

    Tiers map to adapter IDs in a LoRAPool (or to separate model configs).
    """

    HARD_KEYWORDS = frozenset([
        "analyze", "explain", "compare", "evaluate", "describe in detail",
        "write a", "implement", "prove", "derive", "reason", "multi-step",
    ])
    EASY_KEYWORDS = frozenset([
        "what is", "define", "list", "name", "translate", "summarize briefly",
        "yes or no",
    ])

    def __init__(
        self,
        easy_tier: str = "base",
        hard_tier: str = "instruct",
        length_threshold: int = 100,
    ):
        self.easy_tier = easy_tier
        self.hard_tier = hard_tier
        self.length_threshold = length_threshold
        self._decisions: list[RoutingDecision] = []

    def route(self, prompt: str) -> RoutingDecision:
        prompt_lower = prompt.lower()
        n_tokens_approx = len(prompt.split())

        if any(kw in prompt_lower for kw in self.HARD_KEYWORDS):
            d = RoutingDecision(tier=self.hard_tier, reason="hard keyword")
        elif any(kw in prompt_lower for kw in self.EASY_KEYWORDS):
            d = RoutingDecision(tier=self.easy_tier, reason="easy keyword")
        elif n_tokens_approx > self.length_threshold:
            d = RoutingDecision(tier=self.hard_tier, reason="long prompt")
        else:
            d = RoutingDecision(tier=self.easy_tier, reason="default easy")

        self._decisions.append(d)
        return d

    def routing_stats(self) -> dict:
        if not self._decisions:
            return {}
        easy = sum(1 for d in self._decisions if d.tier == self.easy_tier)
        hard = len(self._decisions) - easy
        return {
            "total": len(self._decisions),
            "easy_pct": 100 * easy / len(self._decisions),
            "hard_pct": 100 * hard / len(self._decisions),
        }

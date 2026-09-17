from __future__ import annotations

import hashlib
import time
from typing import Optional

import torch
from transformers import DynamicCache


def _hash_chunk(token_ids: list[int]) -> str:
    return hashlib.md5(bytes(token_ids)).hexdigest()


def _slice_cache(cache: DynamicCache, end: int) -> DynamicCache:
    """Return a new DynamicCache containing only positions [:end]."""
    sliced = DynamicCache()
    for k, v in zip(cache.key_cache, cache.value_cache):
        sliced.key_cache.append(k[:, :, :end, :].clone())
        sliced.value_cache.append(v[:, :, :end, :].clone())
    return sliced


# ---------------------------------------------------------------------------
# Hash-based prefix cache (Day 11)
# ---------------------------------------------------------------------------

class HashPrefixCache:
    """Hash each chunk of block_size tokens; reuse any matching prefix blocks.

    On new request: walk the token sequence in blocks, hash each block,
    look up in store. The longest contiguous prefix match is returned as a
    pre-filled DynamicCache so the engine only needs to compute the suffix.
    """

    def __init__(self, block_size: int = 16):
        self.block_size = block_size
        self._store: dict[str, DynamicCache] = {}  # block_hash -> KV slice up to this block

    def lookup(self, token_ids: list[int]) -> tuple[Optional[DynamicCache], int]:
        """Return (cached_kv_prefix, n_matched_tokens).

        cached_kv_prefix is the KV cache for token_ids[:n_matched_tokens].
        n_matched_tokens is always a multiple of block_size.
        """
        matched = 0
        last_cache = None

        n_blocks = len(token_ids) // self.block_size
        for b in range(n_blocks):
            chunk = token_ids[b * self.block_size : (b + 1) * self.block_size]
            key = _hash_chunk(chunk)
            if key in self._store:
                last_cache = self._store[key]
                matched = (b + 1) * self.block_size
            else:
                break

        return last_cache, matched

    def store(self, token_ids: list[int], full_cache: DynamicCache):
        """Store KV blocks for all complete blocks in this sequence."""
        n_blocks = len(token_ids) // self.block_size
        for b in range(n_blocks):
            chunk = token_ids[b * self.block_size : (b + 1) * self.block_size]
            key = _hash_chunk(chunk)
            if key not in self._store:
                end = (b + 1) * self.block_size
                self._store[key] = _slice_cache(full_cache, end)

    def size(self) -> int:
        return len(self._store)


# ---------------------------------------------------------------------------
# Radix trie prefix cache (Day 12)
# ---------------------------------------------------------------------------

class RadixNode:
    __slots__ = ("children", "kv_ref", "last_used", "n_tokens")

    def __init__(self):
        self.children: dict[int, RadixNode] = {}
        self.kv_ref: Optional[DynamicCache] = None  # KV for this node's prefix
        self.last_used: float = 0.0
        self.n_tokens: int = 0  # number of tokens represented by this node's prefix


class RadixCache:
    """Radix trie keyed on token sequences with LRU eviction on leaves.

    Shared prefixes map to shared trie paths. Evicting a leaf never
    disturbs a prefix shared by another active branch — unlike a flat
    hash map where evicting a hash entry can break a shared prefix.
    """

    def __init__(self, max_tokens: int = 8192):
        self.root = RadixNode()
        self.max_tokens = max_tokens
        self._total_tokens = 0

    def lookup(self, token_ids: list[int]) -> tuple[Optional[DynamicCache], int]:
        """Walk trie matching token_ids. Return best prefix cache + match length."""
        node = self.root
        matched = 0
        best_cache = None
        best_matched = 0

        for tok in token_ids:
            if tok not in node.children:
                break
            node = node.children[tok]
            node.last_used = time.time()
            matched += 1
            if node.kv_ref is not None:
                best_cache = node.kv_ref
                best_matched = matched

        return best_cache, best_matched

    def insert(self, token_ids: list[int], full_cache: DynamicCache):
        """Insert the full token sequence into the trie, storing KV slices at each node."""
        self._evict_if_needed(len(token_ids))

        node = self.root
        for i, tok in enumerate(token_ids):
            if tok not in node.children:
                child = RadixNode()
                child.n_tokens = i + 1
                node.children[tok] = child
            node = node.children[tok]
            node.last_used = time.time()
            if node.kv_ref is None:
                node.kv_ref = _slice_cache(full_cache, i + 1)
                self._total_tokens += 1

    def _evict_if_needed(self, incoming: int):
        """LRU eviction on leaf nodes until we have room."""
        while self._total_tokens + incoming > self.max_tokens:
            leaf = self._find_lru_leaf(self.root)
            if leaf is None:
                break
            leaf.kv_ref = None
            self._total_tokens -= 1

    def _find_lru_leaf(self, node: RadixNode) -> Optional[RadixNode]:
        if not node.children:
            return node if node.kv_ref is not None else None
        candidates = [self._find_lru_leaf(c) for c in node.children.values()]
        candidates = [c for c in candidates if c is not None]
        if not candidates:
            return None
        return min(candidates, key=lambda n: n.last_used)

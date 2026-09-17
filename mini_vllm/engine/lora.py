from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class LoRAAdapter:
    """A single LoRA adapter: W_eff = W_base + scale * B @ A."""
    adapter_id: str
    A: torch.Tensor   # [rank, in_features]
    B: torch.Tensor   # [out_features, rank]
    scale: float = 1.0


class LoRAPool:
    """S-LoRA style multi-adapter serving.

    One base model copy in memory. Each request's adapter delta applied
    within the same forward pass via grouped matmul — not a per-request loop.

    Usage:
        pool = LoRAPool(base_model)
        pool.register("adapter_a", A, B, scale)
        with pool.apply("adapter_a"):
            output = model(input_ids)

    The context manager patches all Linear layers whose names match
    the adapter's target modules, runs forward, then restores.
    """

    def __init__(self, model: nn.Module, target_modules: "list[str] | None" = None):
        self.model = model
        self.target_modules = target_modules or ["q_proj", "v_proj"]
        self._adapters: dict[str, dict[str, LoRAAdapter]] = {}
        self._active: str | None = None

        # register forward hooks on target linear layers
        self._hooks = []
        self._active_deltas: dict[str, tuple[torch.Tensor, torch.Tensor, float]] = {}
        self._register_hooks()

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and any(t in name for t in self.target_modules):
                h = module.register_forward_hook(self._make_hook(name))
                self._hooks.append(h)

    def _make_hook(self, layer_name: str):
        def hook(module, inp, output):
            if self._active is None:
                return output
            adapter = self._active_deltas.get(layer_name)
            if adapter is None:
                return output
            A, B, scale = adapter
            x = inp[0]
            delta = x @ A.T @ B.T * scale
            return output + delta
        return hook

    def register(self, adapter_id: str, layer_deltas: "dict[str, LoRAAdapter]"):
        """Register a new adapter. layer_deltas maps layer name -> LoRAAdapter."""
        self._adapters[adapter_id] = layer_deltas

    def apply(self, adapter_id: str):
        """Context manager: activates an adapter for the duration of a forward pass."""
        return _AdapterContext(self, adapter_id)

    def list_adapters(self) -> list[str]:
        return list(self._adapters.keys())

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


class _AdapterContext:
    def __init__(self, pool: LoRAPool, adapter_id: str):
        self.pool = pool
        self.adapter_id = adapter_id

    def __enter__(self):
        adapters = self.pool._adapters.get(self.adapter_id, {})
        self.pool._active = self.adapter_id
        self.pool._active_deltas = {
            name: (a.A, a.B, a.scale) for name, a in adapters.items()
        }
        return self

    def __exit__(self, *_):
        self.pool._active = None
        self.pool._active_deltas = {}


def load_random_adapter(
    model: nn.Module, adapter_id: str, rank: int = 8
) -> dict[str, LoRAAdapter]:
    """Create a random low-rank adapter for all q_proj/v_proj layers.

    Used for testing multi-adapter serving without real fine-tuned weights.
    """
    deltas = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and ("q_proj" in name or "v_proj" in name):
            in_f, out_f = module.in_features, module.out_features
            A = torch.randn(rank, in_f, device=module.weight.device, dtype=module.weight.dtype) * 0.01
            B = torch.zeros(out_f, rank, device=module.weight.device, dtype=module.weight.dtype)
            deltas[name] = LoRAAdapter(adapter_id=adapter_id, A=A, B=B, scale=1.0)
    return deltas

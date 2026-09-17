"""
torch.compile wrapper + weight quantization (Days 28 + 17 weight quant).

torch.compile on MPS has limited support — primarily reduces Python overhead.
Full kernel fusion (Inductor) targets CUDA. On MPS: compile with mode="default"
and check graph breaks with torch._dynamo.explain().

Weight quantization: per-output-channel int8 min-max, dequantize at runtime.
Reduces model memory footprint; won't be faster on MPS without fused kernels.
"""
from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# torch.compile wrapper (Day 28)
# ---------------------------------------------------------------------------

def compile_model(model: nn.Module, mode: str = "default") -> nn.Module:
    """Wrap model with torch.compile.

    mode options: "default", "reduce-overhead", "max-autotune"
    On MPS: "default" is safest. "reduce-overhead" may trigger graph breaks
    from dynamic sequence lengths in continuous batching.
    """
    try:
        compiled = torch.compile(model, mode=mode)
        print(f"torch.compile applied (mode={mode})")
        return compiled
    except Exception as e:
        print(f"torch.compile failed ({e}), returning eager model")
        return model


def explain_graph_breaks(model: nn.Module, sample_input: torch.Tensor):
    """Print graph break diagnostics for a sample input."""
    try:
        explanation = torch._dynamo.explain(model)(sample_input)
        print(explanation)
    except Exception as e:
        print(f"explain failed: {e}")


# ---------------------------------------------------------------------------
# Weight quantization (Day 17 — weight_quant_bits flag)
# ---------------------------------------------------------------------------

def quantize_weights_int8(model: nn.Module) -> nn.Module:
    """Per-output-channel int8 quantization of all Linear layers.

    Quantizes weights to int8 and stores scale factors as buffers.
    Dequantizes to bfloat16 at runtime before each matmul.

    Memory reduction: ~2x (bfloat16 → int8).
    Speed: no gain on MPS without fused dequant+matmul kernel.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            _quantize_linear_int8(module)
    return model


def _quantize_linear_int8(module: nn.Linear):
    """Replace module.weight with int8 + scale, add dequant in forward."""
    W = module.weight.data.float()
    # per-output-channel: each row of W gets its own scale
    w_max = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    scale = w_max / 127.0
    W_q = (W / scale).round().clamp(-128, 127).to(torch.int8)

    module.register_buffer("weight_q", W_q)
    module.register_buffer("weight_scale", scale.to(module.weight.dtype))
    del module.weight
    module.weight = None  # type: ignore

    original_forward = module.forward

    def _dequant_forward(x):
        W_fp = module.weight_q.to(x.dtype) * module.weight_scale
        return nn.functional.linear(x, W_fp, module.bias)

    module.forward = _dequant_forward  # type: ignore


def memory_footprint_mb(model: nn.Module) -> float:
    """Return model parameter memory in MB."""
    total = sum(
        p.numel() * p.element_size()
        for p in model.parameters()
    )
    # also count registered buffers (for quantized weights)
    for name, buf in model.named_buffers():
        if buf is not None:
            total += buf.numel() * buf.element_size()
    return total / 1e6

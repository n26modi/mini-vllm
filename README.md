# mini-vllm

A from-scratch LLM inference engine built one optimization at a time. Every technique is a flag on a shared `EngineConfig` dataclass, so a controlled ablation table is the natural output.

**Model:** Qwen3-0.6B  
**Hardware:** M4 Mac Air (MPS / CPU)  
**Reference targets:** [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm), [tachyon](https://github.com/JINO-ROHIT/tachyon)

---

## What's built

| Phase | Technique | Flag | Status |
|---|---|---|---|
| 1 | Naive decode (no cache, O(n²)) | `cache_backend="none"` | done |
| 1 | KV cache | `cache_backend="naive"` | done |
| 1 | Streaming HTTP server | - | done |
| 2 | Hash-based prefix cache | `prefix_cache="hash"` | done |
| 2 | Radix trie prefix cache + LRU | `prefix_cache="radix"` | done |
| 3 | Static batching | `batching="static"` | done |
| 3 | Continuous batching (Orca-style) | `batching="continuous"` | done |
| 3 | Concurrent load test | - | done |
| 4 | KV quantization (int8 / int4) | `kv_quant_bits=8\|4` | done |
| 4 | H2O eviction | `eviction="h2o"` | done |
| 4 | SnapKV eviction | `eviction="snapkv"` | done |
| 5 | Lookahead decoding (no aux model) | `spec_method="lookahead"` | done |
| 5 | Draft + verify speculative decoding | `spec_method="draft_verify"` | done |
| 6 | Multi-LoRA serving (S-LoRA style) | `use_lora=True` | done |
| 6 | Difficulty-based request router | - | done |
| 7 | torch.compile wrapper | `compiled=True` | done |
| 8 | Disaggregated prefill/decode | `disaggregated=True` | done |

**Not implemented** (requires external artifacts not available for Qwen3-0.6B):

- `cache_backend="mla"` — requires running the [TransMLA converter](https://github.com/bet0x/transmla-converter) on the model weights to add low-rank projection matrices post-hoc
- `spec_method="medusa"` — requires pre-trained Medusa heads (none publicly available for Qwen3-0.6B)
- `spec_method="eagle"` — requires a trained EAGLE head (none publicly available for Qwen3-0.6B)
- `weight_quant_bits=8` — weight quantization is implemented in `kernels/matmul_triton.py:quantize_weights_int8` but intentionally not wired into the ablation because it mutates the shared model singleton; run it in isolation

---

## Quick start

```bash
pip install torch transformers fastapi uvicorn
```

**Run one config:**
```bash
python -m mini_vllm.bench.harness
```

**Run the full ablation table:**
```bash
python -m mini_vllm.bench.ablation
```

**Run with a specific subset:**
```bash
python -m mini_vllm.bench.ablation --configs baseline kv_quant_8bit prefix_radix
```

**Concurrent load test:**
```bash
python -m mini_vllm.bench.load_test --max-concurrency 8
```

**Start the HTTP server:**
```bash
python -m mini_vllm.serving.server
# POST /generate  {"prompt": "...", "max_new_tokens": 64}
# GET  /generate/stream  (SSE)
```

---

## EngineConfig flags

```python
from mini_vllm.configs.flags import EngineConfig

config = EngineConfig(
    cache_backend="naive",    # "naive" (mla: needs TransMLA converter)
    kv_quant_bits=16,         # 16 | 8 | 4
    weight_quant_bits=16,     # 16 | 8 (run in isolation, see kernels/matmul_triton.py)
    eviction="none",          # "none" | "h2o" | "snapkv"
    prefix_cache="off",       # "off" | "hash" | "radix"
    batching="static",        # "static" | "continuous"
    decoding="standard",      # "standard" | "speculative"
    spec_method=None,         # "lookahead" | "draft_verify" (medusa/eagle need trained heads)
    use_lora=False,           # use engine/lora.py:LoRAPool directly
    compiled=False,           # wraps model with torch.compile
    disaggregated=False,      # use serving/disagg.py:DisaggregatedEngine directly
)
```

---

## Ablation results

Qwen3-0.6B, M4 Mac Air (MPS), 5 requests, up to 64 new tokens each.

```
==========================================================
Day 31 ablation table
==========================================================
config                   TTFT ms   ITL ms    tok/s   mem GB
----------------------------------------------------------
baseline                   338.3     21.6     35.1    1.211
kv_quant_8bit               39.1     33.1     30.3    1.207
kv_quant_4bit               45.0     34.2     29.1    1.204
eviction_h2o               256.7     40.8     23.7    2.411
eviction_snapkv             37.8     30.7     32.4    2.411
prefix_hash                 65.1     25.1     38.3    2.410
prefix_radix                33.0     25.7     39.2    2.410
quant8_h2o                  80.7     40.6     24.1    2.407
quant4_snapkv               43.3     39.4     25.3    2.403
quant8_radix                65.9     31.0     31.3    2.407
==========================================================
```

Speculative decoding × KV compression cross-phase results (8 requests, results in `results/day25.json`):

```
config                   TTFT ms   ITL ms    tok/s   acceptance
----------------------------------------------------------------
standard_16bit             224.5     21.2     42.6        0.00
standard_8bit               39.6     33.7     29.8        0.00
standard_4bit               35.7     30.6     32.6        0.00
lookahead_16bit             29.4     23.2     43.2        0.61
lookahead_8bit              36.7     33.2     30.1        0.00
lookahead_4bit              36.3     33.5     29.8        0.00
```

Key finding: lookahead decoding achieves 61% n-gram acceptance and +1.4% throughput at 16-bit precision, but KV quantization drops acceptance to zero — the precision noise breaks n-gram matching between requests.

---

## Repo layout

```
mini_vllm/
  configs/flags.py              EngineConfig dataclass
  engine/
    model_runner.py             NaiveEngine + CachedEngine (all Phase 1-5 flags)
    kv_cache.py                 KV quantization, H2O + SnapKV eviction, MLA stub
    scheduler.py                StaticBatcher + ContinuousScheduler
    sampler.py                  sampling strategies + speculative decoders
    prefix_cache.py             HashPrefixCache + RadixCache
    lora.py                     LoRAPool (S-LoRA style multi-adapter)
  serving/
    server.py                   FastAPI server with streaming
    router.py                   DifficultyRouter (heuristic)
    disagg.py                   PrefillWorker + DecodeWorker threads
  kernels/
    matmul_triton.py            torch.compile wrapper + weight quantization
  bench/
    harness.py                  single-config benchmark runner
    ablation.py                 full ablation table (Day 31)
    load_test.py                concurrent load test, saturation curve
    workloads/                  JSONL prompt sets
  instrumentation/
    metrics.py                  RequestMetrics + MetricsLogger (JSONL/CSV)
  tests/
    test_naive_engine.py        greedy correctness vs HF generate()
    test_continuous_batching.py static vs continuous batching comparison
    test_prefix_cache.py        hash cache hit/miss
    test_radix_cache.py         radix trie insertion + LRU eviction
    test_static_batcher.py      slot waste measurement
results/
  ablation_table.json           full ablation results
  day25.json                    speculative × KV compression cross-phase
  metrics.jsonl                 per-request metric log
```

---

## Running tests

```bash
python -m pytest mini_vllm/tests/ -v
```

Tests cover: greedy output matching HF `model.generate()`, static vs continuous batching throughput comparison, prefix cache hit rates, radix trie LRU eviction.

---

## Multi-LoRA serving

The `LoRAPool` in `engine/lora.py` applies multiple adapter deltas via forward hooks — one base model copy in memory, adapter applied per-request within the same forward pass.

```python
from mini_vllm.engine.lora import LoRAPool, load_random_adapter
from mini_vllm.engine.model_runner import CachedEngine

engine = CachedEngine(EngineConfig())
pool = LoRAPool(engine.model)

deltas = load_random_adapter(engine.model, adapter_id="adapter_a", rank=8)
pool.register("adapter_a", deltas)

with pool.apply("adapter_a"):
    output = engine.model(input_ids)
```

## Disaggregated prefill/decode

`serving/disagg.py` wires a `PrefillWorker` and `DecodeWorker` as two threads connected by a KV queue — simulating the DistServe architecture on a single machine. The metric to watch is TTFT jitter (p99 - p50), not mean TTFT.

```python
from mini_vllm.serving.disagg import DisaggregatedEngine
from mini_vllm.engine.model_runner import CachedEngine

engine = CachedEngine(EngineConfig())
disagg = DisaggregatedEngine(engine)
disagg.start()
disagg.submit("Explain transformers.")
```

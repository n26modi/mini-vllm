## mini-vllm

a from-scratch LLM inference engine, built one technique at a time.

- KV cache, prefix caching (hash + radix trie), continuous batching
- KV quantization (int8/int4), H2O and SnapKV eviction
- lookahead speculative decoding
- multi-LoRA serving, difficulty-based router
- disaggregated prefill/decode
- torch.compile support
- streaming HTTP server

**model:** Qwen3-0.6B &nbsp;|&nbsp; **hardware:** M4 Mac Air (MPS)

## results

```
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
```

speculative decoding × KV compression cross-phase (`results/day25.json`):

```
config                   TTFT ms   ITL ms    tok/s   acceptance
----------------------------------------------------------------
standard_16bit             224.5     21.2     42.6        0.00
lookahead_16bit             29.4     23.2     43.2        0.61
lookahead_8bit              36.7     33.2     30.1        0.00
lookahead_4bit              36.3     33.5     29.8        0.00
```

lookahead gets 61% n-gram acceptance at fp16, but KV quantization drops it to zero — precision noise breaks n-gram matching between requests.

## usage

```bash
pip install torch transformers fastapi uvicorn
```

run a single benchmark:

```bash
python -m mini_vllm.bench.harness
```

run the full ablation table:

```bash
python -m mini_vllm.bench.ablation
```

start the server:

```bash
python -m mini_vllm.serving.server
# POST /generate  {"prompt": "...", "max_new_tokens": 64}
```

concurrent load test:

```bash
python -m mini_vllm.bench.load_test --max-concurrency 8
```

## config flags

every technique is a flag on `EngineConfig`:

```python
from mini_vllm.configs.flags import EngineConfig
from mini_vllm.engine.model_runner import CachedEngine

engine = CachedEngine(EngineConfig(
    kv_quant_bits=8,        # 16 | 8 | 4
    eviction="snapkv",      # "none" | "h2o" | "snapkv"
    prefix_cache="radix",   # "off" | "hash" | "radix"
    batching="continuous",  # "static" | "continuous"
    decoding="speculative",
    spec_method="lookahead",
    compiled=True,
))

metrics = engine.generate("Explain transformers.", max_new_tokens=64)
print(f"{metrics.throughput_tok_s:.1f} tok/s, {metrics.ttft_ms:.0f}ms TTFT")
```

## tests

```bash
python -m pytest mini_vllm/tests/ -v
```

## layout

```
mini_vllm/
  configs/flags.py        EngineConfig
  engine/
    model_runner.py       NaiveEngine + CachedEngine
    kv_cache.py           quantization + eviction
    scheduler.py          static + continuous batching
    sampler.py            sampling + speculative decoders
    prefix_cache.py       hash + radix prefix cache
    lora.py               multi-LoRA serving
  serving/
    server.py             FastAPI + SSE streaming
    router.py             difficulty router
    disagg.py             disaggregated prefill/decode
  kernels/
    matmul_triton.py      torch.compile + weight quant
  bench/
    harness.py            single-config runner
    ablation.py           full ablation table
    load_test.py          saturation curve
  instrumentation/
    metrics.py            TTFT / ITL / tok/s logging
results/
  ablation_table.json
  day25.json
```

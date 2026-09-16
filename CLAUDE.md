# mini-vllm

From-scratch LLM inference engine. Every technique is a flag on `EngineConfig` — the ablation table is the deliverable.

**Model:** Qwen3-0.6B. Avoid Qwen3.5/3.6 — they use Gated DeltaNet (no conventional KV cache; breaks Phases 2-4).
**Hardware:** Mac CPU/MPS locally. Cloud GPU (GCP) for Day 7 stretch (custom matmul) if wanted.
**Blog:** each build session → post in ~/vscode/personal-portfolio writing/mini-vllm/ folder.

## Repo layout

```
mini_vllm/
  engine/
    model_runner.py       # wraps HF model, uses SDPA for attention
    kv_cache.py            # cache backends: naive, mla, quantized, evicted
    scheduler.py            # request queue + batching policy
    sampler.py              # decoding strategies incl. speculative
  kernels/
    matmul_triton.py      # stretch: custom matmul (Day 7)
  serving/
    server.py             # request entrypoint (single + batched, streaming)
    router.py             # multi-LoRA / difficulty router
  bench/
    harness.py            # benchmark runner + ablation table generator
    load_test.py          # concurrent request generator
  instrumentation/
    metrics.py            # TTFT / ITL / tok/s / GPU-util logging
  configs/
    flags.py              # EngineConfig / feature flags
  tests/
```

## EngineConfig (single source of truth)

```python
@dataclass
class EngineConfig:
    cache_backend: str = "naive"    # "naive" | "mla"               (Day 16)
    kv_quant_bits: int = 16         # 16 | 8 | 4                    (Day 17)
    weight_quant_bits: int = 16     # 16 | 8                        (Day 17)
    eviction: str = "none"          # "none" | "h2o" | "snapkv"     (Days 18-19)
    prefix_cache: str = "off"       # "off" | "hash" | "radix"      (Days 11-12)
    batching: str = "static"        # "static" | "continuous"       (Days 13-14)
    decoding: str = "standard"      # "standard" | "speculative"    (Days 21-24)
    spec_method: str | None = None  # "draft_verify" | "medusa" | ...
    use_lora: bool = False          #                               (Day 26)
    compiled: bool = False          #                               (Day 28)
    disaggregated: bool = False     #                               (Day 30)
```

Every new technique branches on these fields — never a new script.

## Benchmark methodology

- Fixed: Qwen3-0.6B + fixed GPU + fixed prompt/output-length distribution
- Per-request metrics: TTFT (ms), ITL (ms/token), throughput (tok/s), peak KV memory (GB)
- Derived: $/1M tokens = (GPU $/hr ÷ 3600 ÷ throughput) × 1,000,000
- Ablation table: rows = flag combination, columns = metrics above

---

## Phase 0 — Foundations (Days 2-4)

No code. Output is your own notes/explainer before any implementation.

### Day 2 — What LLM inference actually is

**Teach:**
- Training optimizes for throughput only. Inference optimizes throughput AND latency simultaneously — that tension is the whole field.
- Prefill (whole prompt, one parallel pass) vs. decode (one token at a time, sequentially dependent) — two different computational regimes with opposite bottlenecks.
  - Prefill is compute-bound: lots of matrix multiplications, GPU works hard.
  - Decode is memory-bandwidth-bound: very few FLOPs per step, but you're reading the entire KV cache and all model weights from HBM on every step. No amount of batching fully fixes this.
- The KV cache: without it, decode recomputes K,V for every past token every step → O(n²) total. With it, K,V computed once per token, stored, never recomputed → O(n) total. The trade is memory for compute.
- TTFT (time to first token) is dominated by prefill time. ITL/TPOT (inter-token latency) is dominated by decode step time — specifically by KV cache read bandwidth.
- Inference framework (the forward pass: kernels, KV cache management) ≠ inference server (the API layer: queuing, batching users, streaming, autoscaling). vLLM ships both, which is why they get conflated. They're separable concerns.

**Notes to produce (your own words):** why is decode O(n²) without cache / O(n) with it? What's the difference between an inference framework and an inference server?

**Reference:** https://handbook.modular.com/llm-inference-basics/what-is-llm-inference/

---

### Day 3 — The roofline model

**Teach:**
- Arithmetic intensity = FLOPs / bytes read from HBM. A matmul is compute-bound only above a hardware-specific critical batch size B_crit (roughly 240 tokens on a TPU v5e; similar order on modern GPUs, varies by precision).
- Attention during decode is always memory-bandwidth-bound regardless of batch size — you load a large KV cache to do very little math per step.
- KV cache size per token = `2 × bytes_per_element × num_kv_heads × head_dim × num_layers`. At long context and large batch, this dominates over model weight size.
- Theoretical minimum step time ≈ `(batch_size × KV_per_token + param_size) / memory_bandwidth` in the memory-bound regime.

**Exercise (do by hand before code):** compute B_crit and per-token KV cache size for Qwen3-0.6B on your hardware. This is the number every later benchmark gets compared against.

**Reference:** https://jax-ml.github.io/scaling-book/inference/ ("The Basics of Transformer Inference" and "What about memory?" sections)

---

### Day 4 — Landscape survey + benchmark methodology

**Teach:**
- vLLM: PagedAttention (non-contiguous paged KV cache) + continuous batching.
- SGLang: RadixAttention (prefix caching via radix trie) + structured output.
- TensorRT-LLM: ahead-of-time compiled graphs + custom kernel plugin system.
- Each targets a different point on the latency/throughput/flexibility tradeoff.
- **Know this concept — tensor parallelism (multi-GPU, not built here):** split the model's weight matrices across N GPUs so each device does 1/N of the compute per layer. For attention: each GPU owns a subset of the heads — Q/K/V projections are column-parallel, the output projection is row-parallel and requires an all-reduce at the end. For MLPs: same idea, column-parallel on the up-projection, row-parallel on the down-projection, all-reduce after. The KV cache shards naturally alongside the attention heads each GPU owns (covered more in Day 16). The tradeoff: reduces per-GPU memory and per-GPU compute, but every layer now requires an all-reduce — adds latency proportional to inter-GPU bandwidth. Works well when the all-reduce is fast (NVLink) and the model is large enough that the memory/compute savings dwarf the communication cost.
- **Know this concept — pipeline parallelism (multi-GPU, not built here):** split the model's *layers* across N GPUs (GPU 0 runs layers 0-7, GPU 1 runs layers 8-15, etc.). Instead of each GPU running the full model at reduced width, each GPU runs a slice of the full model depth. A request flows through GPU 0, its activations are sent to GPU 1, and so on — like a factory pipeline. The cost is pipeline bubbles: GPUs sit idle while waiting for the previous stage to finish. Micro-batching (filling the pipeline with many small chunks of a batch simultaneously) reduces bubbles. Pipeline parallelism is complementary to tensor parallelism — production systems like Megatron-LM use both together. Relevant here because it's a third way to split work across workers alongside tensor parallelism and disaggregation (Day 30).

**Spec:**
- `configs/flags.py`: `EngineConfig` dataclass, all flags defaulted to simplest option
- `instrumentation/metrics.py`: `RequestMetrics` dataclass (`request_id`, `ttft_ms`, `itl_ms_per_token: list[float]`, `total_tokens`, `wall_time_s`, `gpu_util_pct`, `peak_mem_gb`) + logger that appends rows to CSV/JSONL
- `bench/harness.py`: takes `EngineConfig` + workload (prompts + max_new_tokens), runs against whatever engine exists, emits one ablation-table row. Run against a stub/naive backend today.

**Acceptance:** `harness.py` produces one row of output (placeholder/zero metrics fine) by end of day.

---

## Phase 1 — Core Runtime Loop (Days 5-8)

### Day 5 — Naive decode loop (no cache)

**Teach:** the naive loop re-runs the full forward pass over the entire sequence at every step. O(n) tokens × O(n) work per token = O(n²) total. This is the baseline to beat.

Also define the sampler properly here — every later day calls it, and the benchmark results depend on it being realistic:
- **Greedy:** argmax of logits. Deterministic. Used for correctness checks (bit-identical comparison against HF).
- **Temperature:** divide logits by T before softmax. T < 1 sharpens the distribution (more confident), T > 1 flattens it (more random). T = 1 is unchanged.
- **Top-k:** zero out all logits except the k highest before softmax. Hard cutoff.
- **Top-p (nucleus):** sort tokens by probability descending, keep the smallest set whose cumulative probability exceeds p, zero out the rest. Adaptive — uses fewer tokens when the model is confident.
- **Repetition penalty:** divide logits for already-generated tokens by a penalty factor (> 1 makes them less likely). Prevents loops.

All five are pure tensor operations on a `[vocab_size]` logit vector — fully implementable on CPU/MPS, no CUDA needed.

**Spec:**
```python
NaiveEngine.generate(prompt_ids, max_new_tokens):
    seq = prompt_ids
    for _ in range(max_new_tokens):
        logits = model.forward(seq)       # full recompute every step
        next_token = sample(logits[-1])   # sample() implements the above strategies
        seq = seq + [next_token]
        if next_token == eos: break
    return seq
```

**Acceptance:** greedy output matches HF `model.generate()` token-for-token. Top-p/top-k outputs are non-deterministic but statistically sane (no infinite loops, distribution looks right). Log wall-clock time vs. sequence length as `baseline_no_cache` — per-token time should visibly worsen as sequence grows.

---

### Day 6 — Naive KV cache

**Teach:**
- Caching K,V projections means a new token only computes its own Q/K/V and attends over cache + itself. Per-token cost becomes constant.
- **Know this concept — PagedAttention:** the naive cache preallocates a contiguous tensor `[batch, num_heads, max_seq_len, head_dim]`. If max_seq_len is 4096 and most requests finish at 200 tokens, you've allocated 20x what's needed — memory utilization ~60-70%. PagedAttention (vLLM's core innovation) applies virtual memory ideas: a pool of fixed-size pages allocated on demand, a page table mapping logical positions to physical pages. Utilization rises above 90%, enabling far more concurrent requests. We build the naive version here; understanding why it wastes memory makes the later compression phases more meaningful.

**Spec:**
```
KVCache:
  tensors: [num_layers] of (K, V) each [batch, num_heads, max_seq_len, head_dim], preallocated
  .append(layer_idx, k_new, v_new)   # write at current position, advance pointer
  .get(layer_idx) -> (K[:cur_len], V[:cur_len])
```
Modify forward pass to accept `past_key_values`; only run new token through Q/K/V projection + attention over `cache ++ new`.

**Acceptance:** output matches Day 5 within fp tolerance. Per-token time is now flat vs. sequence length. Log as `baseline_naive_cache`. Flag: `cache_backend = "naive"`.

---

### Day 7 — Single-request serving + baseline tok/s + instrumentation

**Teach:** prefill and decode wired into one serving path — the first day the "engine" framing is actually true.

**Spec:** `Engine.serve_request(prompt) -> Generator[token]` (streaming). Wire `metrics.py`: TTFT at first yielded token, ITL at each subsequent token, device utilization polled via `pynvml` (GPU) or `psutil` (CPU/MPS).

**Acceptance / milestone:** first full `baseline` row in ablation table (SDPA attention + naive KV cache, single request, no batching). This is the number every later phase is compared against.

---

### Day 8 (stretch, optional) — Custom matmul kernel vs. cuBLAS

**Teach:** matmul is ~90% of transformer FLOPs. Beating (or characterizing yourself against) a vendor GEMM is the highest-signal kernel-writing exercise — saves raw compute, not memory traffic.

**Spec:** Triton or CUTLASS matmul; benchmark vs. `torch.matmul`/cuBLAS across FFN and attention projection shapes.

**Acceptance:** report % of cuBLAS performance achieved and where it wins/loses.

---

## Phase 2 — Prefix Caching (Days 11-12)

### Day 11 — Hash-based prefix cache reuse

**Teach:** identical prompt prefixes produce identical KV cache entries. Recompute only the divergent suffix.

**Spec:** hash each prefix chunk (per `block_size` tokens); `PrefixCacheStore: dict[hash] -> KVCache slice`. On a new request, walk block-by-block matching hashes to find longest cached prefix, splice in cached KV, compute only the remainder.

**Acceptance:** second request sharing a prefix has prefill time drop proportional to shared fraction.

---

### Day 12 — RadixAttention-style trie + LRU eviction

**Teach:** a flat hash map can't manage memory well — a radix trie keyed on token sequences represents shared prefixes as shared trie paths, and LRU eviction on leaves reclaims memory from stale branches without disturbing shared ancestors.

**Spec:**
```
RadixNode: { children: dict[token_id -> RadixNode], kv_ref, last_used_ts }
insert on request completion; evict LRU leaves under memory pressure
```

**Acceptance:** cache hit rate + throughput improvement vs. Day 11 on a workload with repeated system prompts/few-shot prefixes. Flag: `prefix_cache = "off" | "hash" | "radix"`.

---

## Phase 3 — Batching & Scheduling (Days 13-15)

### Day 13 — Static batching baseline

**Teach:** naive batching blocks the whole batch until the longest sequence finishes. Early finishers waste their slot. TTFT is tied to the entire batch's prefill time.

**Spec:** `StaticBatcher` — collect N requests, prefill all, decode-step together until all hit EOS/max_len (pad short sequences).

**Acceptance:** quantify the failure modes — log wasted slot percentage and TTFT at a few batch sizes. This day is about demonstrating the problem, not solving it.

---

### Day 14 — Continuous batching scheduler

**Teach:**
- The fix (Orca/vLLM): iteration-level scheduling. Every step: evict finished requests, admit new ones (prefilling them), run one decode step for all active requests.
- **Know this concept — chunked prefill:** continuous batching still has a problem — a very long prefill (8k token prompt) occupies the GPU for many milliseconds while active decode requests wait, causing ITL spikes. Chunked prefill breaks the prefill into fixed-size chunks (e.g., 512 tokens) interleaved with decode steps. Every 512 tokens of prefill, all active requests get a decode step. Tradeoff: longer total prefill time (chunk overhead), but much smoother ITL for concurrent requests. This is a natural extension to the scheduler built today — worth noting in the blog post.

**Spec:**
```python
Scheduler.step():
    1. evict finished requests
    2. admit queued requests into free KV slots (prefill them)
    3. run one decode step for all active requests
    4. yield newly generated tokens per request
```

**Acceptance:** TTFT stops scaling with batch size; throughput clearly beats Day 13 at same concurrency. Flag: `batching = "static" | "continuous"`.

---

### Day 15 — Concurrent load test

**Teach:** small-batch testing doesn't reveal scheduler bugs — slot-allocation races, queueing fairness, memory fragmentation only show up under real concurrent load.

**Spec:** `bench/load_test.py` (asyncio or Locust) ramping 1 → N concurrent requests; measure p50/p95/p99 TTFT and ITL; find saturation point.

**Acceptance / deliverable:** saturation curve (concurrency vs. throughput, vs. p95 latency).

---

## Phase 4 — KV Cache Compression (Days 16-20)

### Day 16 — GQA → MLA conversion

**Teach:**
- GQA shares K/V heads across multiple Q heads to shrink the cache.
- MLA (DeepSeek-style) goes further — project K/V into a shared low-rank latent space, reconstruct per-head K/V from that latent on the fly. Cache the latent instead of full K/V.
- TransMLA converts an already-trained GQA model into this scheme post-hoc, no retraining needed.
- **Know this concept — KV cache sharding (multi-GPU, not built here):** a direct consequence of tensor parallelism (introduced Day 4). Each GPU owns a subset of attention heads and therefore stores only that subset's K,V tensors. Qwen3-0.6B has 8 KV heads — fits cleanly on 1, 2, 4, or 8 GPUs. No extra communication needed during the attention computation itself; the all-reduce only happens on the attention output. With GQA you must ensure KV head count divides evenly across the tensor-parallel degree — Qwen3-0.6B satisfies this.
- **Know this concept — sequence parallelism (multi-GPU, not built here):** tensor parallelism shards by head; sequence parallelism shards by token range. Each GPU owns a contiguous slice of the sequence and its corresponding KV entries. Attention across slice boundaries requires a communication step — in ring attention, each GPU passes its KV slice to the next in a ring while computing attention against the locally-held slice, then against the received slice, and so on for all N GPUs. The communication and compute overlap so the bandwidth cost is partially hidden. Sequence parallelism becomes necessary when a single request's KV cache at very long context (e.g., 128k+ tokens) doesn't fit on one GPU — distinct from tensor parallelism which is primarily about model weight memory and compute throughput.

**Spec:** `MLACacheBackend` — store low-rank latent `c_kv` per token instead of full K/V; fold up-projection into attention math directly.

**Acceptance:** bytes/token before vs. after; quality sanity check (perplexity or a few generations). Flag: `cache_backend = "naive" | "mla"`.

**Reference:** https://arxiv.org/html/2502.07864v4

---

### Day 17 — KV cache quantization

**Teach:**
- KV cache tolerates lower precision better than weights. TurboQuant approach: PolarQuant (polar-coordinate quantization) + QJL (quantized Johnson-Lindenstrauss projection), sensitivity-based mixed precision.
- **Weight quantization vs. KV quantization — distinct techniques:** KV quantization (what this day builds) reduces the memory cost of the cache written during inference. Weight quantization reduces the memory cost of the model itself at load time. They're independent and stack. Production GPTQ/AWQ/bitsandbytes weight quantization requires CUDA for calibration kernels. But naive post-training int8 weight quantization — round each weight matrix to int8 with per-channel scales, dequantize to fp32/bf16 before each matmul — is pure PyTorch tensor math and is fully implementable on Mac/MPS. Add a `quantize_weights(model, bits=8)` function: per-output-channel min-max scaling → round to int8 → store scale factors. Dequantize at runtime before each linear layer. This demonstrates the concept and the memory reduction; it won't be as fast as CUDA-optimized kernels (dequantize + matmul fused), but the memory footprint reduction is real and measurable. Add `weight_quant_bits: int = 16` to `EngineConfig`.

**Spec:**
- `quantize_kv(k, v, bits) -> (k_q, v_q, scale_params)` / `dequantize_kv(...)`, per-channel or per-block scaling
- `quantize_weights(model, bits=8)` — per-output-channel int8 quantization of all linear layers, with stored scale factors and runtime dequantization

**Acceptance:** KV memory reduction tracks roughly `bits/16` vs. bf16. Weight quantization measurably reduces model memory footprint. Sweep int8/int4 for KV, plot quality vs. compression. Note separately where weight quantization hurts quality vs. KV quantization. Flags: `kv_quant_bits = 16 | 8 | 4`, `weight_quant_bits = 16 | 8`.

**Reference:** https://arxiv.org/abs/2504.19874

---

### Day 18 — H2O eviction

**Teach:** "heavy-hitter" tokens — a small fraction by cumulative attention score — account for most attention mass. Evict everything else when cache exceeds budget, keeping heavy-hitters plus a protected recent window.

**Spec:** `H2OEvictor` tracks running attention-score sum per cached token; on overflow, evicts lowest-scoring tokens outside the recent window.

**Acceptance:** quality vs. a fixed memory budget vs. no eviction (which would OOM at that budget). Flag: `eviction = "none"`.

**Reference:** https://arxiv.org/abs/2306.14048

---

### Day 19 — SnapKV eviction

**Teach:** query-aware eviction — score using attention pattern from a small recent observation window of queries, re-evaluated periodically. Adapts better than H2O when topic/task shifts mid-generation.

**Spec:** `SnapKVEvictor` — at each eviction point, score using only recent queries, retain top-k KV entries.

**Acceptance:** direct comparison vs. H2O, especially on a workload with a topic shift partway through. Flag: `eviction = "none" | "h2o" | "snapkv"`.

**Reference:** https://arxiv.org/abs/2404.14469

---

### Day 20 — Phase 4 ablation day

**Spec:** run full compatible matrix — `{naive, mla} × {16, 8, 4-bit} × {none, h2o, snapkv}` — log tok/s, memory/token, quality, $/1M tokens.

**Deliverable:** Phase 4 ablation table.

---

## Phase 5 — Speculative Decoding (Days 21-25)

### Day 21 — Draft + verify (greedy)

**Teach:** a small draft model proposes K tokens; the target model verifies all K in one parallel forward pass. Because decode is memory-bound, scoring K tokens costs barely more than scoring 1 — accepted tokens are close to free.

**Spec:**
```python
step(draft_model, target_model, k):
    draft_tokens = draft_model.sample_k(k)
    target_logits = target_model.forward(draft_tokens)   # one parallel pass, k+1 positions
    accept tokens left-to-right while draft.argmax == target.argmax
    on first mismatch: take target's token, stop the round
```

**Acceptance:** acceptance rate (avg tokens accepted per round) + tok/s vs. Day 7. Flags: `decoding = "standard" | "speculative"`, `spec_method = "draft_verify"`.

---

### Day 22 — Medusa

**Teach:** k extra linear decoding heads on the target model's final hidden state, each predicting t+1...t+k. No separate model — heads share the target's parameters, so predictions track its distribution more closely. Verification against a candidate tree (several continuations per round), not a single linear sequence.

**Spec:** `MedusaHeads` — k small linear heads; tree-based candidate verification.

**Acceptance:** acceptance rate + tok/s vs. Day 21; note tradeoff (needs head training vs. draft+verify needing a compatible small model).

---

### Day 23 — Lookahead decoding

**Teach:** candidate n-grams generated in parallel via Jacobi-iteration-style fixed-point trick — no auxiliary model, zero extra memory footprint. Verify against a pool of previously-seen n-grams.

**Spec:** `LookaheadDecoder` — maintain n-gram candidate pool, verify + accept against target model.

**Acceptance:** compare vs. Days 21-22; the key differentiator is no auxiliary model required.

---

### Day 24 — EAGLE-2/3

**Teach:** feature-level speculation — speculate on the target model's hidden-state features one layer early, rather than tokens from a separate model. Higher acceptance rates than draft-model approaches because the speculator sees inside the target. Hardest of the four to implement correctly.

**Spec:** lightweight EAGLE head consuming target model's second-to-last-layer features + embeddings, predicting future feature vectors, verifying.

**Acceptance:** acceptance rate + tok/s vs. Days 21-23 — should show best acceptance-rate/overhead tradeoff.

---

### Day 25 — Acceptance rate vs. cache variant

**Teach:** does a compressed (MLA/quantized) KV cache change speculative acceptance rates? Compression adds noise to K/V — worth measuring rather than assuming it doesn't matter.

**Spec:** run best speculative method across Phase 4 cache variants; log acceptance rate + tok/s per combination.

**Deliverable:** cross-phase ablation table.

---

## Phase 6 — Multi-LoRA Serving + Routing (Days 26-27)

### Day 26 — S-LoRA-style multi-adapter serving

**Teach:** LoRA adapters are small low-rank deltas on frozen base weights. Serving N adapters concurrently: one base-model copy in memory, each request's adapter delta applied within the same batched forward pass via grouped matmul — not a per-request loop.

**Spec:** `LoRAPool` loads N adapters as `(A, B)` pairs; batched forward applies `W_eff = W_base + B_i @ A_i × scale_i` per request, gathered by adapter id.

**Acceptance:** 2-3 adapters served concurrently; output matches single-adapter runs; throughput vs. one-at-a-time.

---

### Day 27 — Difficulty-based router

**Teach:** not every request needs the biggest tier. A lightweight router sends easy requests to a cheaper adapter/model, saving cost without hurting quality where it matters.

**Spec:** `Router.route(request) -> tier_id`, initially heuristic (prompt length/keywords), feeding into Day 26 adapter pool.

**Acceptance:** $/1M tokens with routing vs. always using biggest tier, on mixed-difficulty synthetic workload.

---

## Phase 7 — Compiler Stack (Days 28-29)

### Day 28 — torch.compile

**Teach:**
- torch.compile captures the model as a graph rather than executing op-by-op in eager Python, enabling kernel fusion and reduced Python overhead. Watch for graph breaks triggered by dynamic shapes from continuous batching — a real gotcha, not hypothetical.
- **Know this concept — CUDA graphs (GPU-only, not on Mac):** where torch.compile reduces Python overhead by compiling the computation graph, CUDA graphs go further — they capture the entire sequence of GPU kernel launches as a static graph that can be replayed in a single GPU API call, eliminating per-step kernel launch overhead (which matters a lot for decode, where you do the same operations thousands of times). vLLM uses CUDA graphs aggressively for the decode phase. The limitation: CUDA graphs require static shapes (fixed batch size, fixed sequence length) — incompatible with dynamic continuous batching unless you maintain a set of pre-captured graphs for each batch size you expect to serve. If you run Day 28 on a cloud GPU (alongside the Triton kernel days), `torch.cuda.CUDAGraph` is worth implementing as a comparison against torch.compile. On Mac/MPS there is no equivalent.

**Spec:** `torch.compile(model, mode=...)`; benchmark eager vs. compiled at a few batch sizes. Diagnose graph breaks with `torch._dynamo.explain`. On GPU: optionally add a CUDA-graph-based decode path for fixed batch size and compare all three (eager / compiled / CUDA graph).

**Acceptance:** report speedup (or lack of one) and graph-break diagnostics.

---

### Day 29 — Trace the compiler stack

**Teach:** TorchDynamo (Python bytecode → FX graph) → AOTAutograd (ahead-of-time tracing, forward-only for inference) → TorchInductor (generates Triton/C++ kernels from the graph).

**Spec:** instrument with `TORCH_COMPILE_DEBUG=1` or `torch._dynamo.explain(model)(inputs)`, capture intermediate graphs + generated kernels.

**Deliverable:** "here's literally what torch.compile generated for my attention kernel."

**Reference:** https://jino-rohit.github.io/blogs/

---

## Phase 8 — Disaggregated Prefill/Decode (Day 30, stretch)

### Day 30 — Disaggregation

**Teach:**
- Prefill and decode have fundamentally different hardware preferences. Co-locating them forces a compromise.
- Prefill is compute-bound: large prompt batch, high arithmetic intensity, wants maximum FLOP/s.
- Decode is memory-bandwidth-bound: many requests active simultaneously, wants fast KV reads, low latency per step.
- The interference: a long prefill (10k token prompt) blocks all decode steps for its duration, causing ITL spikes for active users and TTFT jitter for the new request. Day 14's continuous batching makes this more granular but doesn't eliminate it.
- Disaggregation (DistServe): separate prefill workers (optimized for compute) and decode workers (optimized for memory bandwidth). After prefill, the KV cache blob is serialized and transferred to a decode worker. Transfer bandwidth is the cost; reduced TTFT jitter is the payoff.
- Simulated on one machine: two processes or CUDA streams connected by a queue carrying the KV blob.
- **The metric to watch is jitter (variance of TTFT), not just mean TTFT** — that's what disaggregation actually fixes.
- **Relationship to pipeline parallelism (Day 4):** pipeline parallelism splits work by *layer* — GPU 0 runs layers 0-7, GPU 1 runs layers 8-15, activations flow forward. Disaggregation splits work by *phase* — prefill workers do the full model forward pass for new prompts, decode workers do the full model forward pass for ongoing generation. Both are "split the computation across workers" but at different granularities. In a real production cluster you'd often see all three: tensor parallelism within a worker (to fit the model), pipeline parallelism across worker groups (depth), and prefill/decode disaggregation across worker pools (phase). Understanding how they compose is what separates knowing the techniques from knowing how they're actually deployed.

**Spec:** `PrefillWorker` (runs prefill, sends KV blob to queue) + `DecodeWorker` (receives blob, inserts into batch, continues generation).

**Acceptance:** TTFT jitter (variance or p99-p50 gap) drops vs. Day 14 continuous batching, even if raw throughput is similar or slightly worse due to transfer overhead.

**Reference:** https://arxiv.org/abs/2401.09670

---

## Capstone (Day 31)

Run complete ablation table across all mutually-compatible flag combinations (or curated interesting subset). Final report: tok/s, TTFT, ITL, $/1M tokens per combination + retrospective + comparison against nano-vllm and tachyon reference numbers.

---

## Paper (Day 32)

**Title (working):** *Characterizing LLM Inference Optimization Interactions: A Controlled Ablation Study*

**Novel contribution:** the cross-phase interaction between KV cache compression and speculative decoding acceptance rates. Existing work on MLA, quantization, and speculative decoding treat these as independent. Nobody has measured what happens when you combine them in a controlled setting — specifically whether compression-induced noise in K/V degrades draft/target agreement in EAGLE-style speculation. The Day 25 experiment is the core result.

**Structure:**
1. Introduction — motivation for controlled ablation; why separate papers on individual techniques aren't enough
2. System design — the EngineConfig flag architecture and why it enables fair comparison
3. Individual technique ablations — results from Phases 2-7 in isolation
4. Cross-phase interactions — the central finding: speculative decoding × KV compression
5. Cost analysis — $/1M tokens across all configurations
6. Comparison to reference tiers — nano-vllm, tachyon
7. Discussion + limitations

**Format:** 6-8 page workshop paper. Target venues: MLSys, EfficientML workshop at NeurIPS, or arXiv preprint first.

**What you need before writing:** Day 31 ablation table complete + Day 25 cross-phase numbers. Everything else is already done by the time you get there.

**Acceptance:** submitted to arXiv or a workshop venue.

---

## Resources

- Modular LLM Inference Handbook: https://handbook.modular.com/llm-inference-basics/what-is-llm-inference/ (Day 2)
- jax-ml scaling book inference chapter: https://jax-ml.github.io/scaling-book/inference/ (Day 3)
- Inference Engineering (Baseten Books): https://www.baseten.co/library/inference-engineering/
- nano-vllm: https://github.com/GeeeekExplorer/nano-vllm
- tachyon: https://github.com/JINO-ROHIT/tachyon
- elizabetht/100-days-of-inference: https://github.com/elizabetht/100-days-of-inference
- TransMLA paper: https://arxiv.org/html/2502.07864v4 (Day 16)
- TransMLA code: https://github.com/bet0x/transmla-converter (Day 16)
- TurboQuant paper: https://arxiv.org/abs/2504.19874 (Day 17)
- H2O paper: https://arxiv.org/abs/2306.14048 (Day 18)
- SnapKV paper: https://arxiv.org/abs/2404.14469 (Day 19)
- DistServe paper: https://arxiv.org/abs/2401.09670 (Day 30)
- vLLM Triton backend deep dive: https://vllm.ai/blog/2026-03-04-vllm-triton-backend-deep-dive (background reading)
- torch.compile internals series: https://jino-rohit.github.io/blogs/ (Days 28-29)

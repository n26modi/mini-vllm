# X posts — mini-vLLM build series

One post per day. Replace [brackets] with real numbers when you have them.
Thread format = multiple tweets separated by ---

---

## Day 2 — What LLM inference actually is

without a KV cache, every decode step recomputes keys and values for every past token.

step 100 does 100× the work of step 1.
step 1000 does 1000× the work of step 1.

O(n²) total. the cache makes it O(n). that one insight is the foundation of everything else in mini-vLLM

---

## Day 3 — The roofline model

did the Day 3 exercise for Qwen3-0.6B: computing KV cache size per token by hand before writing any engine code

2 × 2 bytes × 8 KV heads × 128 head_dim × 28 layers = [X] bytes/token

at 4k context, batch 8: [X] MB just for the cache. the model weights are [X] MB

decode is memory-bandwidth-bound. no matter how fast your compute, you're always waiting on the memory bus

---

## Day 4 — Landscape survey + harness

the architectural decision for mini-vLLM:

every technique is a flag, not a script

```python
EngineConfig(cache_backend="mla", kv_quant_bits=8, eviction="snapkv", batching="continuous", decoding="speculative")
```

the benchmark harness runs different configs against the same prompts. the delta between rows IS the ablation table

first ablation row printed today (all zeros, stub backend). the scaffold is ready

---

## Day 5 — Naive decode loop

baseline is in

the naive loop re-runs the full forward pass over the entire sequence at every step

[X] ms/token at 100 tokens
[X] ms/token at 500 tokens
[X] ms/token at 1000 tokens

per-token cost grows linearly with sequence length. this is what O(n²) looks like in practice

---

## Day 6 — Naive KV cache

added a KV cache. store keys and values the first time, never recompute them.

[X] ms/token at 100 tokens
[X] ms/token at 500 tokens
[X] ms/token at 1000 tokens

flat. same cost at 1k tokens as at 100. O(n) instead of O(n²).

output is bit-identical to the naive loop — same math, just not repeated

---

## Day 7 — Single-request serving + baseline

milestone: first real ablation row

`Engine.serve_request(prompt) -> Generator[token]` — streaming, instrumented end-to-end

TTFT: [X]ms
ITL: [X]ms/token
throughput: [X] tok/s
peak KV memory: [X] GB

attention: torch SDPA. KV cache: naive contiguous. no batching yet.

this is the number everything else gets compared against. every later phase either moves one of these or explains why it doesn't

---

## Day 8 (stretch) — Custom matmul vs cuBLAS

matmul is ~90% of transformer FLOPs. wrote a Triton GEMM kernel and benchmarked it

| shape              | torch.matmul | mine  | % of cuBLAS |
|--------------------|--------------|-------|-------------|
| [FFN up-proj]      | [X]ms        | [X]ms | [X]%        |
| [attn q-proj]      | [X]ms        | [X]ms | [X]%        |

[X]% of cuBLAS at the shapes that matter. the wins and losses tell you exactly where vendor libraries are hard to beat and why

---

## Day 12 — Hash-based prefix cache

"I like dogs" and "I like cats" share a prefix. the KV cache for "I like" is identical for both.

hash each block of tokens. on a new request, walk the hashes to find the longest cached match, splice it in, only compute the suffix.

second request with 80% shared prefix: prefill time dropped from [X]ms to [X]ms. [X]% reduction, proportional to the shared fraction

---

## Day 13 — RadixAttention trie + LRU eviction

flat hash maps don't handle memory pressure well — you can't selectively evict "the stale part" of a cached prefix without a structure that represents sharing explicitly

radix trie: shared prefixes live on shared trie paths. LRU eviction removes leaves (stale branches) without touching ancestors (shared prefixes still in use)

workload: 100 requests, 20-token shared system prompt + variable suffix

cache hit rate: [X]%
throughput vs. no prefix cache: +[X]%

---

## Day 14 — Static batching (demonstrating the problem)

static batching: collect N requests, prefill all, decode until everyone finishes

the failure modes are real and measurable

TTFT at batch=1: [X]ms
TTFT at batch=8: [X]ms  ← tied to the longest prefill in the batch
TTFT at batch=32: [X]ms

wasted slot %: [X]% — sequences that finished early but couldn't be replaced

this is the problem Day 15 fixes

---

## Day 15 — Continuous batching

iteration-level scheduling: every step, evict finished requests, admit new ones (prefilling them), decode step everyone active

TTFT at batch=1: [X]ms
TTFT at batch=8: [X]ms ← flat now, doesn't scale with batch
TTFT at batch=32: [X]ms

throughput vs. static batching at same concurrency: +[X]%

also worth knowing: chunked prefill exists as an extension — break long prefills into 512-token chunks interleaved with decode steps to prevent ITL spikes on active requests

---

## Day 16 — Load test

concurrent requests 1 → N on the continuous batching scheduler

| concurrency | throughput | p50 TTFT | p95 TTFT |
|-------------|-----------|----------|----------|
| 1           | [X] tok/s | [X]ms    | [X]ms    |
| 8           | [X] tok/s | [X]ms    | [X]ms    |
| 32          | [X] tok/s | [X]ms    | [X]ms    |
| [sat point] | plateau   | blows up |          |

saturation at [X] concurrent requests. past that, throughput flatlines and p95 TTFT doubles

---

## Day 17 — GQA → MLA conversion

GQA: K/V heads shared across Q heads. shrinks cache size.

MLA goes further: project K/V into a shared low-rank latent. cache the latent, not the full K/V. reconstruct per-head K/V at attention time.

Qwen3-0.6B before: [X] bytes/token
after MLA: [X] bytes/token — [X]% reduction

output quality delta: [perplexity change or qualitative note]

TransMLA paper does this post-hoc on an already-trained GQA model, no fine-tuning required

---

## Day 18 — KV cache quantization + weight quantization

two distinct things, often conflated:

**KV quantization** — reduce precision of the cache written during inference
int8: [X]× memory reduction, [X] quality delta
int4: [X]× memory reduction, [X] quality delta

**weight quantization** — reduce precision of the model weights at load time
int8 per-channel: [X]× model size reduction, [X] quality delta

they're independent and stack. both implemented in pure PyTorch — no CUDA-specific kernels needed

---

## Day 19 — H2O eviction

"heavy hitter" hypothesis: a small fraction of tokens account for most attention mass

H2O tracks cumulative attention scores per token. when the cache hits budget, evict the lowest-scoring tokens outside a protected recent window.

at [X]k token budget (would OOM without eviction):
full cache baseline: [perplexity/quality baseline]
H2O: [quality at budget]

the tradeoff curve is the deliverable — not "does it work" but "how much quality do you trade for how much memory"

---

## Day 20 — SnapKV eviction

H2O uses one global score. SnapKV uses the attention pattern from recent queries to decide what matters right now, re-evaluated periodically.

when the topic/task shifts mid-generation, H2O holds onto tokens that were important at the start. SnapKV adapts.

benchmark: [X]-token prompt with a topic shift at the midpoint

| method  | quality (first half) | quality (second half) |
|---------|---------------------|----------------------|
| H2O     | [X]                 | [X]                  |
| SnapKV  | [X]                 | [X]                  |

SnapKV is better on the second half by [X] points

---

## Day 21 — Phase 4 ablation

ran the full matrix: {naive, mla} × {16, 8, 4-bit KV} × {none, h2o, snapkv}

| config                        | tok/s  | mem/token | $/1M tokens |
|-------------------------------|--------|-----------|-------------|
| naive + fp16 + no eviction    | [X]    | [X]       | $[X]        |
| naive + int8 KV + h2o         | [X]    | [X]       | $[X]        |
| mla + int4 KV + snapkv        | [X]    | [X]       | $[X]        |
| ... (best combo)              | [X]    | [X]       | $[X]        |

best combo vs. baseline: [X]% cheaper per 1M tokens at comparable quality

---

## Day 22 — Draft + verify speculative decoding

draft model proposes k tokens in sequence. target model verifies all k in one parallel pass.

decode is memory-bound — scoring k tokens costs barely more than scoring 1. accepted tokens are close to free.

k=5, acceptance rate: [X]% (avg [X] tokens accepted per round)
throughput vs. Day 10 baseline: +[X]%

if the acceptance rate is low, speculative decoding makes things worse (wasted draft work). the baseline stays the baseline

---

## Day 23 — Medusa

instead of a separate draft model: k small linear heads on the target's final hidden state, each predicting t+1...t+k

no separate model. predictions come from inside the target, so they track its distribution better.

verification against a candidate tree, not a linear sequence — more accepted tokens per round

vs. draft+verify at same k:
acceptance rate: [X]% vs [X]%
throughput: [X] tok/s vs [X] tok/s

tradeoff: Medusa needs the heads trained/calibrated. draft+verify just needs a small compatible model

---

## Day 24 — Lookahead decoding

Jacobi iteration trick: generate candidate n-grams in parallel using fixed-point iteration, verify them against the target model

no auxiliary model. no extra memory. just redundant compute traded for parallelism.

acceptance rate: [X]%
throughput vs. Day 10: +[X]%

comparison across all three so far:
draft+verify: needs small model, [X] tok/s
Medusa: needs trained heads, [X] tok/s
lookahead: needs nothing extra, [X] tok/s

---

## Day 25 — EAGLE-2/3

feature-level speculation: predict the target model's hidden-state features one layer early, not tokens from a separate model

the speculator sees inside the target. it knows what the target was "thinking," not just what it output last

highest acceptance rates of the four methods:
acceptance rate: [X]% (vs [X]% draft+verify, [X]% Medusa, [X]% lookahead)
throughput: [X] tok/s

hardest to get right. budget extra time for this one

---

## Day 26 — Acceptance rate vs. cache variant

does a compressed KV cache hurt speculative acceptance rates?

MLA and int4 quantization add noise to K/V. the question is whether that noise causes draft/target disagreement.

ran best speculative method (EAGLE) across Phase 4 cache variants:

| cache config       | acceptance rate | tok/s  |
|--------------------|-----------------|--------|
| naive + fp16       | [X]%            | [X]    |
| naive + int8       | [X]%            | [X]    |
| mla + int8         | [X]%            | [X]    |
| mla + int4         | [X]%            | [X]    |

[conclusion based on actual numbers]

most builds don't test this interaction. the result is either reassuring or a genuine warning

---

## Day 27 — Multi-LoRA serving

one base model in GPU memory. N adapters, each a pair of low-rank matrices (A, B).

serving them concurrently means: one batched forward pass, each request's adapter delta applied via grouped matmul

W_eff = W_base + B_i @ A_i × scale_i

3 adapters served simultaneously, verified output matches single-adapter runs

throughput vs. one-at-a-time: [X]× — the batch amortizes the weight loading

---

## Day 28 — Difficulty-based router

not every request needs the most capable adapter. a lightweight router sends easy requests to a cheaper tier.

heuristic: prompt length + keyword signals → tier_id

mixed-difficulty workload (50% easy / 50% hard):
always-max-tier: $[X]/1M tokens
with routing: $[X]/1M tokens — [X]% cheaper

quality on hard requests: unchanged (they still go to the capable tier)

---

## Day 29 — torch.compile

`torch.compile` captures the model as a graph, enabling kernel fusion and reduced Python overhead

the real issue: continuous batching produces dynamic shapes. dynamic shapes cause graph recompilations. recompilations eat the speedup.

eager: [X] tok/s
compiled (max-autotune): [X] tok/s
graph breaks diagnosed: [X] — [where they happen]

the speedup exists but graph breaks require fixing before it shows up cleanly

---

## Day 30 — Tracing the compiler stack

Dynamo → AOTAutograd → Inductor → Triton

set TORCH_COMPILE_DEBUG=1 and traced exactly what happened to the attention kernel

Dynamo captured: [describe what it found / what caused graph breaks]
Inductor generated kernel: [describe the fused kernel it emitted]
vs. my hand-written Triton kernel: [comparison]

the gap between "torch.compile generates this" and "hand-optimized Triton" is [small/large] at my model's shapes

---

## Day 31 — Disaggregated prefill/decode

prefill wants compute. decode wants memory bandwidth. running both in the same process forces a compromise.

PrefillWorker → KV blob via queue → DecodeWorker

mean TTFT: [X]ms vs [X]ms (continuous batching)
TTFT p99-p50 gap: [X]ms vs [X]ms ← this is the metric

disaggregation doesn't necessarily improve mean TTFT. it flattens the variance — long prefills no longer spike the latency for active decode requests

---

## Day 32 — Capstone

32 days. every major inference optimization technique. one codebase, one benchmark harness.

final ablation table across all flag combinations:

| config                                    | tok/s | TTFT  | ITL   | $/1M  |
|-------------------------------------------|-------|-------|-------|-------|
| baseline (naive + no opts)                | [X]   | [X]ms | [X]ms | $[X]  |
| + KV cache                                | [X]   | [X]ms | [X]ms | $[X]  |
| + prefix caching                          | [X]   | [X]ms | [X]ms | $[X]  |
| + continuous batching                     | [X]   | [X]ms | [X]ms | $[X]  |
| + MLA + int8 KV                           | [X]   | [X]ms | [X]ms | $[X]  |
| + EAGLE speculative                       | [X]   | [X]ms | [X]ms | $[X]  |
| full stack                                | [X]   | [X]ms | [X]ms | $[X]  |

vs. nano-vllm: [X] tok/s
vs. tachyon: [X] tok/s

what moved the needle most: [X]
what didn't: [X]
what surprised me: [X]

full writeup at [link]

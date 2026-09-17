# Blog post samples — mini-vLLM series

Full drafts for three key posts. Structure matches your existing posts exactly.
Replace [brackets] with real numbers. HTML template to copy is in your existing posts.

---

## Post 1: Foundations (Days 2-3)
**Title:** Why LLM inference is harder than it looks
**Subtitle:** Prefill, decode, the KV cache, and the roofline model
**Read time:** 7 min

### Key finding

LLM inference optimizes two things simultaneously: throughput (tokens per second, total) and latency (how fast this specific request gets a response). They pull against each other. Every technique in this series is a different way of navigating that tradeoff.

Before writing a single line of engine code, I worked through the roofline model by hand for Qwen3-0.6B. The numbers tell you what's worth optimizing before you profile anything.

---

### Prefill vs. decode

When you send a prompt, the model processes it all at once in a single forward pass. Every token's attention is computed simultaneously. This is **prefill**. It scales with prompt length and is compute-bound — the GPU is doing real work.

Then the model generates output one token at a time, each token depending on every token before it. This is **decode**. You can't parallelize across tokens — position 4 doesn't exist until position 3 is done. Each step is a separate forward pass over one new token.

Prefill is compute-bound. Decode is memory-bandwidth-bound. You're doing very few FLOPs per decode step, but on every step you're reading the entire KV cache and all model weights from HBM. The GPU's compute units sit mostly idle waiting for memory.

No amount of batching fully fixes this. Batching more decode requests together amortizes the memory reads, which helps, but you're always bounded by how fast you can move bytes off the chip.

---

### The KV cache

Attention needs every past token's key and value vectors to compute attention scores for the current token.

Without caching, you recompute K and V for every past token on every decode step:

- Step 1: compute K,V for position 1
- Step 2: recompute K,V for positions 1-2
- Step 100: recompute K,V for positions 1-100

Step n does n times the work of step 1. Total work across all steps is 1 + 2 + ... + n = O(n²).

With a KV cache, you compute K,V for each position exactly once and store it:

- Step 1: compute K,V for position 1. Store it.
- Step n: compute K,V for position n only. Attend over cached 1..n-1 plus new n.

Work per step is constant. Total work is O(n). The cache trades memory for compute. That tradeoff is the foundation of everything else in this project.

---

### TTFT and ITL

**TTFT** (time to first token) is dominated by prefill. The user sees nothing until prefill finishes.

**ITL** (inter-token latency) is dominated by decode step time, which is dominated by how fast you can read the KV cache from memory.

A fast prefill helps TTFT but doesn't affect streaming speed. A slow decode step makes tokens come out slowly even once they start. The two are separate knobs.

---

### The roofline calculation

Before writing any engine code, I computed the key numbers for Qwen3-0.6B.

KV cache size per token:

```
2 × bytes_per_element × num_kv_heads × head_dim × num_layers
= 2 × 2 bytes (bf16) × 8 × 128 × 28
= [X] bytes/token
```

At a 4k context window with batch size 8, that's [X] MB of KV cache alone. The model weights are [X] MB. Past [X] tokens, the KV cache exceeds model weight size — this is why KV compression phases (Days 17-21) end up mattering more than I expected.

Theoretical minimum decode step time at batch size 1 (memory-bound regime):

```
(model_size + KV_size) / memory_bandwidth
= ([X] MB + [X] MB) / [X] GB/s
= [X] ms/step
```

This is the floor. Everything above it is overhead. A well-optimized engine at batch=1 should approach this number.

---

### Inference framework vs. inference server

These get conflated constantly. They're different.

An **inference framework** is the forward pass: kernels, KV cache management, batching logic. It answers "how do I run the model efficiently?"

An **inference server** adds the API layer: HTTP endpoint, request queue, streaming, autoscaling. It answers "how do I serve this to multiple users?"

vLLM does both, which is why people treat it as one thing. mini-vLLM also does both. But they're built separately — `engine/` vs. `serving/` — because debugging them together is miserable.

---

### What's next

Day 4: build the benchmark harness that will produce every number in this series. The ablation table lives or dies on this infrastructure being correct.

---

### Stack

Python, PyTorch, Qwen3-0.6B, Mac MPS

---

---

## Post 2: The KV cache (Days 5-6)
**Title:** From O(n²) to O(n): building the KV cache
**Subtitle:** A naive decode loop, then the fix, then the numbers
**Read time:** 6 min

### Key finding

Per-token decode time with no cache: [X]ms at 100 tokens, [X]ms at 1000 tokens. It grows linearly with sequence length.

Per-token decode time with a KV cache: [X]ms flat, regardless of sequence length. Output is bit-identical to the naive loop.

The fix is one data structure.

---

### The naive loop

The first version of mini-vLLM's generate loop looks like this:

```python
def generate(prompt_ids, max_new_tokens):
    seq = prompt_ids
    for _ in range(max_new_tokens):
        logits = model.forward(seq)     # full recompute every step
        next_token = sample(logits[-1])
        seq = seq + [next_token]
        if next_token == eos:
            break
    return seq
```

Every iteration passes the entire sequence through the model. At step 100, that's 100 tokens going through every transformer layer. At step 500, it's 500. The forward pass cost grows with sequence length because attention computes scores between every token pair.

The graph of per-token time vs. position tells the story: it's a straight line up. This is what O(n²) looks like empirically — total work is the area under that line.

---

### What the cache stores

Every transformer layer computes Q, K, V projections for each token. The attention output for the current token depends on Q for the current token plus K and V for every previous token.

K and V for past tokens don't change. They're a deterministic function of those tokens' embeddings and the weight matrices. Computing them again on the next step produces the same values.

The cache structure is straightforward:

```python
class KVCache:
    # [num_layers, batch, num_heads, max_seq_len, head_dim]
    # preallocated, filled incrementally
    
    def append(self, layer_idx, k_new, v_new):
        # write at current position, advance pointer
    
    def get(self, layer_idx):
        # return K[:cur_len], V[:cur_len]
```

Modify the forward pass to accept `past_key_values`. Each layer's attention runs Q/K/V projection only for the new token, then computes attention over `[cached KV] ++ [new KV]`. Past tokens contribute their cached vectors; no recomputation.

---

### The results

With a KV cache:

| position | no cache (ms/token) | with cache (ms/token) |
|----------|---------------------|----------------------|
| 50       | [X]                 | [X]                  |
| 200      | [X]                 | [X]                  |
| 500      | [X]                 | [X]                  |
| 1000     | [X]                 | [X]                  |

The cached version is flat. The no-cache version's cost at position 1000 is [X]× higher than at position 50.

Both produce identical output for greedy decoding, verified token-for-token against HF `model.generate()`.

---

### What the cache costs

The KV cache for a single sequence at 4k tokens is [X] MB. At a 32k context window, [X] MB. This is the memory cost of O(n) decode: you trade compute for storage.

The version built here is the simplest possible implementation: one contiguous preallocated tensor per layer, filled left to right. It wastes memory when sequences finish early — a 4k allocation used for a 200-token sequence leaves 3800 slots empty. PagedAttention (vLLM's core contribution) fixes this with a virtual memory scheme, getting utilization above 90%. For now, the naive version is the baseline.

---

### Caveats

1. **Preallocated size.** The cache is allocated at `max_seq_len` upfront. Running out of space means a hard stop. A production implementation would grow dynamically or use pages.
2. **Batch dimension.** This cache is per-sequence. Batching multiple sequences together requires padding or packing — Day 14-15 handles that.

---

### What's next

Days 7-9: a Triton fused attention kernel. The KV cache fixes the O(n²) problem. The kernel fixes the memory bandwidth bottleneck inside each decode step.

---

### Stack

Python, PyTorch, Qwen3-0.6B, Mac MPS

---

---

## Post 3: Continuous batching (Days 14-15)
**Title:** Continuous batching: why static batching wastes half your GPU
**Subtitle:** Demonstrating the failure, then building the fix
**Read time:** 7 min

### Key finding

Static batching at batch=32: TTFT is [X]ms — [X]× slower than batch=1. Wasted slot rate: [X]%. GPU is idle waiting for the longest sequence to finish.

Continuous batching at batch=32: TTFT is [X]ms — flat, same as batch=1. Throughput: +[X]% vs. static at the same concurrency.

The improvement comes from a scheduler change, not a kernel change.

---

### The static batching problem

The simplest batching strategy: collect N requests, run prefill for all of them together, then decode until every sequence hits EOS or max length.

The problem is that sequences finish at different times. A 50-token request finishes in [X]ms. A 2000-token request needs [X]ms. Until the longest sequence finishes, every other slot in the batch is sitting idle — the computation still runs for the full batch size, but the finished requests aren't producing anything useful.

I measured this directly on Day 14 before building the fix:

| batch size | TTFT    | wasted slot % |
|------------|---------|---------------|
| 1          | [X]ms   | 0%            |
| 4          | [X]ms   | [X]%          |
| 8          | [X]ms   | [X]%          |
| 32         | [X]ms   | [X]%          |

TTFT grows with batch size because every request has to wait for the full prefill of all N sequences before its first token is generated. At batch=32, a short request waits for the longest prompt in the batch before seeing its first output token.

---

### The fix: iteration-level scheduling

The Orca paper (2022) identified that the problem is scheduling granularity. Instead of committing to a batch for its entire lifetime, schedule at the iteration level — every single decode step.

The scheduler loop:

```python
def step():
    # 1. remove finished requests, free their KV slots
    # 2. admit waiting requests into freed slots (run their prefill)
    # 3. run one decode step for all currently active requests
    # 4. yield newly generated tokens per request
```

When a request finishes, its slot is freed immediately and a waiting request takes it — prefilling on the very next step. No idle slots. The batch composition changes every iteration.

---

### What changed in the numbers

| metric         | static (batch=8) | continuous (batch=8) |
|----------------|-----------------|----------------------|
| TTFT           | [X]ms           | [X]ms                |
| ITL            | [X]ms/token     | [X]ms/token          |
| throughput     | [X] tok/s       | [X] tok/s            |
| wasted slots   | [X]%            | ~0%                  |

TTFT is now independent of batch size — a new request starts seeing tokens within one prefill pass of joining the active set, not after waiting for all N current requests to finish.

---

### The detail that matters: prefill-decode mixing

Continuous batching mixes prefill steps (for newly admitted requests) with decode steps (for existing ones) in the same scheduler loop. These have different computational profiles — prefill processes many tokens at once (compute-bound), decode processes one token (memory-bound).

In practice this means a new request's prefill step is slower if many decode requests are also active in the same step. The scheduler doesn't currently separate them; they share compute budget.

The extension that fixes this is **chunked prefill**: break a new request's prefill into fixed-size chunks (512 tokens) and interleave them with decode steps, so the decode requests don't get blocked for the full prefill duration of a long prompt. It's a natural extension to this scheduler — the loop structure doesn't change, the chunk budget does.

---

### Caveats

1. **Memory accounting.** The scheduler here admits requests as long as there are free KV slots. It doesn't account for the varying KV memory cost of different sequence lengths mid-generation. A production scheduler (vLLM's) tracks physical pages. This one tracks slots.
2. **Prefill-decode interference.** Mixing prefill and decode in the same step causes measurable ITL spikes when a large prefill lands. Chunked prefill (mentioned above) mitigates this.

---

### What's next

Day 16: a concurrent load test ramping 1 → N requests to find the saturation point. The scheduler works correctly at small concurrency. At high concurrency, slot-allocation bugs and queueing fairness issues show up that unit tests don't catch.

---

### Stack

Python, PyTorch, Qwen3-0.6B, asyncio, Mac MPS

---

---

## Templates for remaining days

The remaining posts follow the same structure. Notes on what makes each one interesting:

**Day 8 stretch (custom matmul)**
Lead with % of cuBLAS achieved. Show the shape-vs-latency table for FFN and attention projections. The interesting angle: where does a hand-written GEMM lose to cuBLAS and why? Arithmetic intensity analysis explains the gap.

**Day 13 (Radix trie)**
Lead with cache hit rate numbers on a realistic workload. The interesting angle is why a flat hash map can't do LRU eviction correctly on a shared prefix — evicting one hash entry might invalidate a prefix that another active request depends on. The trie's structure makes sharing explicit.

**Day 17 (MLA)**
Lead with bytes/token before and after. The interesting angle is that this is done post-hoc on an already-trained model — no fine-tuning. Compare TransMLA's approach to just using a smaller model: MLA preserves the base model's quality while shrinking the cache; a smaller model degrades quality.

**Day 21 (Phase 4 ablation)**
The table IS the post. Lead with the best and worst combination. Discuss which technique contributed the most (likely: MLA or int8 KV quantization). Note which combinations are additive vs. which interfere.

**Day 25 (EAGLE)**
Lead with acceptance rate comparison across all four methods. The interesting angle: feature-level speculation has higher acceptance because the speculator sees the target's own internal state. But it requires access to intermediate activations, which complicates the implementation.

**Day 26 (acceptance vs. cache variant)**
This is the cross-phase post. Lead with whether or not MLA/quantization actually hurts acceptance rates — the answer is either reassuring (compounding optimizations stack cleanly) or a warning (compression interferes with speculation).

**Day 30 (compiler trace)**
Lead with a literal code snippet of what Inductor generated for the attention block. Compare it to the hand-written Triton kernel from Day 7. The interesting question: does the compiler-generated kernel look like what an experienced Triton author would write?

**Day 32 (capstone)**
The full ablation table. Then the retrospective: what moved the needle most, what didn't, what surprised you. Compare to nano-vllm and tachyon numbers. End with what you'd build next.

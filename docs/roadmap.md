# Future improvements

What could make the server faster or more useful next, roughly in order of expected value.
Each item says what was measured that motivates it, and what would show it worked. Results of
the work done so far, including what didn't help, are in [devlog.md](devlog.md).

All numbers are from an M5 Max (36 GB), stock Qwen3.8-27B 4-bit, and the benchmark harness in
`bench/runtimes` unless noted.

## Speed

### 1. A different long-context attention kernel
- **Why:** at 32K, decode is 56 tok/s against Splash's 62. The verify pass's attention costs
  10.6 ms for 8 queries and 18.7 ms for 16 over 16 layers, while reading the K/V cache once
  takes ~5.4 ms. The time grows with the number of 16-row simdgroups per KV head, each of which
  streams the same K/V tiles itself (`bench/attn/ablate.py`).
- **Tried without gain:** threadgroup-staged tiles, launch shapes, persistent accumulators,
  32-row simdgroups (devlog, 2026-10-30).
- **Next:** a new kernel rather than edits to MTPLX's. Matmuls cooperative across simdgroups
  (`execution_simdgroups<N>`) on threadgroup K/V tiles, loaded asynchronously and double
  buffered, so one load serves every query row without a barrier between load and compute.
- **Done when:** `bench/attn/flash_bench.py` shows n=8-16 at 32K within 1.5x of the K/V read
  time, and S1 decode at 16K-32K rises.

### 2. A faster int8 path for the 8-bit KV cache
- **Why:** 8-bit KV is the default, but its flash kernel is 15-30% slower than bf16 from 4
  queries (24.3 vs 18.7 ms at n=16, 32K), which costs up to ~5% decode at 16K.
- **Next:** convert int8 tiles once per threadgroup instead of in every simdgroup, or run the
  QK matmul on int8 directly if the matrix units take int8 inputs. This comes with item 1.
- **Done when:** kv8 attention is no slower than bf16 at n=8-16.

### 3. Faster GPU prefill
- **Why:** with its Neural Engine off, Splash still reaches the first token ~8% sooner (44.7 vs
  48.5 s at 32K, 849 vs 794 tok/s at 4K). For agents with long prompts, the first token matters
  more than decode.
- **Profiled** (devlog, 2026-10-30): matmuls are 65-85% of a chunk but already run at the rate
  of MLX's dense GEMM (~43 TFLOP/s), and no tile shape helps. Attention grows from 2% to 23% of
  a chunk between 0 and 24K context, at ~26 TFLOP/s. The GatedDeltaNet recurrence is ~9%.
- **Next:** a prefill flash-attention kernel for head dim 256 that beats MLX's NAX one (~26
  TFLOP/s), e.g. GQA-aware (one K/V tile for all 6 query heads of a group). A GEMM faster than
  MLX's design would matter more, but would be a new kernel, not a retune.
- **Done when:** TTFT at 16K-32K drops by 5% or more, with the prefill result unchanged.

### 4. A copy branch in the draft tree
- **Why:** Splash's trained drafter accepts more tokens per pass on code and JSON (5.2 vs 4.7
  on code, 6.4 vs 5.6 on JSON). Without training a drafter, repeated spans (identifiers, JSON
  keys, quoted text from the prompt) can be drafted by copying.
- **Next:** first measure, from `--trace-log` files, the share of output tokens that continue
  an n-gram seen earlier in the prompt or the output. If that share is large, add a branch that
  drafts the longest such match. Acceptance stays target-exact, so the output doesn't change.
- **Done when:** tokens per step rise on code, JSON and edit without dropping elsewhere.

### 5. Grammar decoding, version 2
- **Why:** version 1 is exact and costs nothing measurable, but drafts the grammar forbids still
  take verify rows, and spans the grammar forces (closing brackets, fixed keys) are drafted and
  verified like any other token.
- **Next:** check drafted tokens against the grammar before the pass. That needs them on the CPU
  first, i.e. one more sync per step, so measure that cost first. Draft forced spans with
  `compute_ff_tokens` instead of the MTP head. When the grammar overrides the next root, rerun
  the MTP head on the chosen token.
- **Done when:** tokens per step under a grammar rise, with the step cost up less than 5%.

### 6. A trained block drafter (not planned)
- DFlash2-style drafters accept more per pass on code and math, but need training and a 3.8 GB
  second model. Revisit only if items 1-4 stall.

## Serving

### 7. Requests from agent harnesses
- **Sessions and cache points:** explicit snapshot positions per request (end of system prompt,
  end of a repo map), pins that keep shared prefixes alive while any session uses them, and a
  byte budget for the in-memory bank instead of `BANK = 4`.
- **Priorities:** admit the request the harness is waiting on first, and give it more tree
  nodes when batched.
- **Reasoning budget:** `max_reasoning_tokens` forces `</think>` when reached.
- **Capacity:** `GET /v1/capacity` (free slots, memory, throughput), so a harness sizes its
  fan-out to what the server can take.
- **Timings:** queue, prefill and decode seconds in `usage`.

### 8. Batching against other engines
- **Why:** the Splash comparison is one request at a time. With 4 requests this server does
  128 tok/s in total, but no other engine was measured the same way.
- **Next:** add a batched scenario to `bench/runtimes` (2 and 4 concurrent requests, the same
  prompts) and run it for this server, Splash and MTPLX.

### 9. Disk snapshots, version 2
- **Why:** each new conversation leaves two snapshots a few tokens apart (user-turn start and
  shared prefix), both ~1.2 GB at 16K. A load right after a reboot (from the SSD, not the page
  cache) was not timed.
- **Next:** skip a snapshot within a few hundred tokens of one already on disk, and time cold
  loads at 16K and 32K.

## Ruled out

- **Neural Engine FFN split for prefill** (devlog, 2026-10-30). Through public Core ML it gains
  ~12% prefill, but needs a resident int8 copy of ~3.5 GB. Its error per layer is 1.6-5x the
  GPU's, and the ANE's own arithmetic stays coarser even with exact weights.
- **MLX's `scaled_dot_product_attention` for verify passes:** 1.5-4x slower than tree_flash
  from 2 queries up at 32K.

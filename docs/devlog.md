# Development log

Measurements from building the decoder, newest first. All on one M5 Max (36 GB) with Swift 1.5 4-bit (a Qwen3.8-27B fine-tune), greedy, unless noted. Raw JSON for the scenario sweeps is in [results/](../results); the agentic coding eval mentioned below is a separate private suite (10 coding-agent tasks, 3 trials each, pass/fail graded) and is not part of this repo.

# Grammar-constrained decoding (2026-10-30)

`grammar.py`, llguidance 1.9. A request's `response_format` (`json_schema` / `json_object`) or `grammar` (`json_schema`, `lark`, `regex`) constrains its reply, after `</think>` when thinking is on.
- Target-exact under the grammar: at every position the emitted token is the target's choice among the allowed tokens. In the accept walk (`tree_verify.accept_path`), a node's unconstrained choice that the grammar allows is also its constrained choice (the argmax over a set that contains the overall argmax), so only one llguidance check per walked node is needed. Only a disallowed choice computes the mask (0.06 ms for 248K tokens) and redoes the argmax on that node's logits, which `batch_pass` now also returns. For sampling, a disallowed draw is redrawn from the masked distribution, which gives exactly the masked distribution. The prefill's first token goes through the same check.
- The matcher follows each candidate path with consume/rollback and commits the accepted one. A finished grammar stops the matcher, and from then only the end-of-turn token is allowed.
- Server check: JSON schema with an enum and `maxItems`, the same schema with thinking on, a Lark `"yes" | "no"`, a regex date, `json_object`, an invalid grammar (400), and a constrained request batched with a free one. Every constrained reply parsed and matched its grammar. Greedy replies repeat exactly, and the free request in the batch gave the same text as alone.
- Speed: the schema reply ran at 89.4 tok/s, 5.0 tokens/step, with 4 re-picks in 59 tokens; the free reply to the same prompt (fenced, more tags) ran at 81.2 tok/s, 4.94 tokens/step.
- Not done: drafts the grammar forbids are still verified, then rejected, instead of being pruned before the pass, and forced spans aren't fast-forwarded (both need the drafted tokens on the CPU before the pass). When the grammar overrides the target's choice for the next root, that step's MTP hidden still pairs with the overridden token, which only costs acceptance on the next step.

# Long-context attention kernel: five variants, none faster (2026-10-30)

The verify pass's attention at 16-32K (tree_flash, MTPLX's NAX flash kernel with an ancestor mask) is the largest cost that grows with context. Benchmarks are in `bench/attn/` and need no model: 16 layers, 24 query heads, 4 KV heads, head dim 256, chains of n queries.

| ms, 16 layers | n=1 | 2 | 4 | 8 | 12 | 16 |
|---|---:|---:|---:|---:|---:|---:|
| tree_flash, bf16, 32K | 6.8 | 6.9 | 7.9 | 10.6 | 15.9 | 18.7 |
| tree_flash, kv8, 32K | 5.2 | 5.5 | 8.7 | 12.7 | 20.3 | 24.3 |
| MLX `scaled_dot_product_attention` + mask, bf16, 32K | 6.0 | 9.7 | 16.9 | 41.6 | 53.8 | 28.7 |

- Ceiling: reading 2.15 GB with `mx.sum` reaches ~400 GB/s, so 32K K+V takes at least ~5.4 ms. n=1 is near that (316 GB/s); from n=4 the time follows the number of 16-row simdgroups per KV head (6n rows: 1, 1, 2, 3, 5, 6 simdgroups).
- Ablation at n=16, 32K (`ablate.py`), ms saved by removing: K and V loads 8.0, the PV matmul 4.9, the QK matmul 3.1. Each simdgroup streams the same K/V tiles from device memory on its own, and its loads are not coalesced (8 bytes per lane from rows 512 bytes apart).
- Tried, all exact (bit-identical output), none faster:
  - K and V tiles staged once per threadgroup in threadgroup memory with coalesced 16-byte loads (`staged.py`): 0.6-0.97x. The barriers serialise loads and matmuls that used to overlap across simdgroups.
  - Launch shape, 32-256 key blocks x 1-4 key-split simdgroups (`sweep.py`, `blocks.py`): 64 blocks with no split is within 2-3% of the best everywhere.
  - PV accumulators kept in matmul destination tensors across tiles, P converted once per tile instead of once per output group (`persist.py`): 0.97-1.0x. The compiler already removed those copies.
  - 32 rows per simdgroup and half of D for P x V (`wide.py`): ~2.7x less K/V traffic per row, but 0.65-0.89x, presumably from register pressure.
- What's left is a different kernel design rather than edits to this one, e.g. matmuls cooperative across simdgroups (`execution_simdgroups<N>`) on threadgroup tiles with asynchronous loads, so one K/V load serves all rows without barriers between load and compute.
- The kv8 kernel is 15-30% slower than bf16 from 4 queries, which matches the -5% decode at 16K seen end to end with `--kv8`. A faster int8 path is worth more now that kv8 is the default.

# Startup curve cache, prefix snapshots on disk, and a Neural Engine prefill test (2026-10-30)

## Saved cost curves
The server saves its startup cost curves to `~/.cache/mlx-tree-spec/cost-curves.json` and reuses them. The key covers the checkpoint's config and weight sizes, this package's source, the MLX, mlx-lm and MTPLX versions, the GPU, macOS, `--kv8` and `--curve-context`.
- Cold start 43 s, warm start 7 s. A warm start still runs one short tree pass so that nothing compiles on the first request: the same 256-token reply took 1.97 s after a warm start and 2.02 s after a cold one.
- Two cold measurements agree within 0-4% at most node counts, and within 12% at a few (2 and 20 nodes). A saved curve is as good as a fresh one.
- `--remeasure-curves` replaces the saved curves; `--no-curve-cache` measures at every start and saves nothing.

## Prefix snapshots on disk (snapshot_store.py)
Every prefix snapshot the server takes (user-turn start, longest prefix shared with a recent prompt) is queued and written to `~/.cache/mlx-tree-spec/snapshots/<setup key>/` by the engine thread while no request is active, as one safetensors file with the tokens, every target and MTP cache layer exactly as prefill left it (bf16) and the last hidden state. A prompt resumes from the longest of the in-memory and on-disk prefixes; a disk hit is loaded into the in-memory bank. Files beyond `--snapshot-disk-gb` (default 20) are deleted, least recently used first, across all setup directories.
- 15.4K-token system prompt + question: prefill 21.0 s. After a restart, the same system prompt with another question resumed 15,445 tokens from disk: prefill 0.20 s (0.16 s from memory). The greedy reply matched the one before the restart exactly.
- One snapshot at 15.4K tokens is 1.17 GB and blocked the idle engine for 0.14 s (the write lands in the page cache). A load after a reboot reads from the SSD and was not timed.
- Each new conversation leaves two snapshots a few tokens apart (the user-turn start and the shared prefix), so the disk holds about 8 conversations of 16K tokens at the default budget.

## Neural Engine FFN split for prefill (prototype: no-go through public Core ML)
Splash prefills about 13% faster with its ANE split than without it (39.4 vs 44.7 s TTFT at 32K). Splash goes through the private AppleNeuralEngine framework, with W8A8 Hadamard-rotated programs and IOSurface handoffs. This tests the public route: a Core ML ML Program (1x1 convs on (1, C, 1, T), fp16 activations) runs the trailing intermediate channels of one FFN layer, while MLX runs the leading channels in 4-bit. Scripts and setup are in `bench/ane/`.

Speed: one layer at the real shapes (5120 -> 17408 -> 5120). Times include copying the input out of MLX and the partial sum back.

| rows | ANE share | weights | GPU alone | GPU lead | ANE tail | both | gain |
|---:|---:|---|---:|---:|---:|---:|---:|
| 2048 | 14.7% | int8 | 29.1 | 24.6 | 16.5 | 25.1 | 1.16x |
| 2048 | 20.6% | int8 | 29.1 | 22.2 | 20.5 | 23.7 | **1.23x** |
| 2048 | 26.5% | int8 | 29.1 | 20.4 | 25.3 | 26.1 | 1.12x |
| 2048 | 20.6% | fp16 | 28.7 | 22.8 | 34.5 | 35.6 | 0.81x |
| 2048 | 20.6% | uint4/64 | 29.1 | 22.2 | 152.7 | 156.8 | 0.19x |
| 1024 | 20.6% | int8 | 14.0 | 11.4 | 10.3 | 11.8 | 1.19x |

- The ANE really runs it: the same model on CPU only takes 108 ms vs 24 ms at 14.7%. 4-bit per-block weights (the only Core ML format that could hold MLX's affine 4-bit weights exactly) leave the ANE.
- Estimated end to end: 5.4 ms saved per layer x 64 = 0.35 s per 2048-row chunk, against ~2.6 s per chunk at 8K. That is ~12% faster prefill, about the gain Splash gets from its split, and it would close roughly half of Splash's 20% TTFT lead. The GPU side here is MLX's stock `quantized_matmul`, not this repo's packed prefill kernel; the two are within a few percent.
- Weights have to be baked in. With fp16 weights passed as inputs, the work leaves the ANE: 1465 ms (397 ms on the GPU through Core ML). So every layer's ANE share needs a resident int8 copy: 55 MB per layer at 20.6%, **3.5 GB for 64 layers** (2.5 GB at 14.7%). That is more than the 8-bit KV default saves.
- Accuracy, real activations from 2048 tokens of code and real weights (layers 3 / 31 / 62), relative L2 error against an fp32 reference of the 4-bit FFN:

  | | layer 3 | layer 31 | layer 62 |
  |---|---:|---:|---:|
  | GPU alone (bf16) | 3.8e-3 | 4.0e-3 | 3.6e-3 |
  | split, 14.7% on the ANE | 1.6e-2 | 1.0e-2 | 5.7e-3 |
  | split, 20.6% on the ANE | 1.9e-2 | 1.2e-2 | 6.4e-3 |

  The ANE part alone is off by 1.4-6.7% (int8 per-channel weights over 4-bit group-64 values, plus fp16 math), worst in the early layers where activations are small. Nothing overflowed. The split is 1.6-5x the GPU's own rounding error per layer, and that error would land in every prompt's KV cache and GatedDeltaNet state.
- Exact weights don't help. MLX's format (`w = s*q + b`, group 64) maps exactly onto Core ML's blockwise `s*(q - offset)` with `offset = -b/s` in fp16 (`bench/ane/exact_split.py`), but the ANE doesn't run it. With uint4 or int8 data, the 14.7% share takes 95-104 ms, the same as on CPU only, with the CPU's error: it falls back. On the ANE itself, even exact dense fp16 weights give 1.0e-2 relative error, against 5.0e-3 for MLX's bf16 GPU path and 4.8e-4 for Core ML on the GPU with the same fp16 model (`exact_control.py`). The ANE's own arithmetic is coarser, so no weight format makes the split match the GPU. The closest is fp16 weights: the 14.7% share takes 22.7 ms against 24.6 ms for the GPU's lead, about 1.13x on the FFN, with an fp16 copy of +5 GB.
- Verdict: not worth building on this route. It would gain ~12% prefill at a cost of +3.5 GB resident, 1.6-5x per-layer error, and 64 Core ML models to compile and keep loaded (load time not measured). Worth revisiting only with weights the ANE can read without a full copy (Splash requantizes one layer ahead into double-buffered surfaces, which public Core ML can't do), and with an end-to-end acceptance and quality check first.

# Prefill kernel on the packed layout (2026-10-27)

Prompts of 64+ rows per chunk now go to `qmm_packed.prefill_kernel` instead of unpacking each weight back to MLX's layout per call. It is MLX's own prefill GEMM (`affine_qmm_t_nax`: 64x64 output tile, 2x2 simdgroups, NAX tiles) with the weight loader rewritten for the packed layout. Each thread dequantizes one row's 16-wide K block (one 8-byte read; a simdgroup reads 256 contiguous bytes) into threadgroup memory. MLX's NAX tile helpers (`steel/gemm/nax.h`) are inlined from the installed package at runtime, because custom kernels compile without MLX's include path.

- Output is bit-identical to `mx.quantized_matmul` wherever MLX uses its NAX GEMM: from 64 rows on the large shapes, from ~320 on k/v (5120->1024). Where MLX picks another kernel, ours is closer to a float32 reference than MLX's own.
- Per matmul (`experiments/qmm_prefill.py`, 8-call dependent chains, time vs MLX's kernel): BK 64 runs 0.92-0.99x on most shapes at 320-2048 rows. The down projection (K 17408) at 2048 rows was 1.06-1.09x with BK 64 and 0.95-1.0x with BK 128, so that shape uses BK 128. The unpack path was 1.02-1.21x.
- From 64 rows it matches or beats the 32-row NAX path (`nax_multi_kernel`) on the large shapes, so `PREFILL_M` went from 320 to 64. Below 64 rows the 32-row path stays; it is faster at 24 rows.
- Real model, in-process, `tree_spec.prefill` (target + MTP, 2048-token chunks), medians of 3, swap flat (`experiments/prefill_e2e.py`, `results/prefill_kernel.json`):

| tokens | stock MLX | unpack (before) | packed prefill kernel |
|---|---|---|---|
| 200 | 357 ms | +1.3% | -3.1% |
| 1,024 | 1,329 ms | +7.5% | +1.1% |
| 4,096 | 5,427 ms | +4.6% | +1.9% |
| 8,192 | 11,164 ms | +4.0% | +0.9% |

  From 1K tokens up the prefill result (first token and the MTP hidden after the prompt) is bit-identical to stock. Stock has to run first, because installing the packed layout drops MLX's arrays. Its repetitions got slower as the run went on (4K: 5,232 / 5,427 / 5,485 ms), while the kernel's stayed within 0.3%. Part of the remaining 1-2% is therefore run order.
- Through the server (bench/runtimes T1/T2 rerun the same day, swap flat; MTPLX rows still from 2026-10-25), time to first token went from 0.97-0.98x of MTPLX's MTP from 4K up to 1.02-1.05x (>1 = faster): T2 vs Q5 1.21 / 1.10 / 1.03 / 1.03 / 1.02 / 1.02x at 128 / 1K / 4K / 8K / 16K / 32K; T1 vs C2 1.18 / 1.10 / 1.04 / 1.03 / 1.03 / 1.02x. At 32K the stock model takes 48.0 s vs 50.0 s before. 128 and 1K gained too (0.37 -> 0.34 s, 1.53 -> 1.40 s): those prompts now take the new kernel instead of the 32-row NAX path. Decode cells moved within run-to-run noise (T2 S1 -4% to +7%, S2 -1% to +1%); `results-tree.jsonl` and `summary-tree.md` now hold this run.

# Prefill: idle residency and weight unpacking (2026-10-27)

The first S1 run (below) showed the tree server 0.1-1.4 s slower to the first token than MTPLX. There were two causes.

**Metal residency lost after idle (fixed).** macOS drops an idle process's Metal residency ~2.5 s after its last command buffer. The next submit pays ~10 ms per GiB to restore it. `bench.py` sleeps 3 s between runs, so every run except the first after warm-up paid it. MTPLX's sources describe the same effect (`model_scheduler.arm_idle_keepalive`) and avoid it with a 64x64 matmul every 1 s while idle. In-process, with nothing between prefills except the gap:
- 200 tokens: 337-351 ms after 0-2 s of idle, 481-539 ms after 3 s.
- 970 tokens: 1,359-1,386 ms after 0-2 s, 1,539-1,571 ms after 3 s.

The server now does the same: a 64x64 matmul each second while idle, for `--keepalive-window` (default 600 s) after the last request. Through the server with 3-6 s gaps, 200-token prefill went from 457-552 to 324-373 ms and 970-token prefill from ~1,580 to ~1,380 ms.

Ruled out:
- the MTP-head pass over the prompt (under 1%);
- Python GC (0 ms inside prefill);
- snapshot copies (none taken);
- streaming;
- lingering handler threads (all idle).

**Unpacking the packed weights (fixed above).** Prompts of 320+ rows per chunk rebuild each 4-bit weight in MLX's layout per forward pass. Same model, one process, stock MLX kernels vs packed:
- 200 tokens: 350 vs 336 ms (the NAX multi-tile path is 4% faster);
- 1,024 tokens: 1,268 vs 1,390 ms (+9.6%);
- 4,096 tokens: 5,284 vs 5,566 ms (+5.3%);
- 8,192 tokens: 11,001 vs 11,361 ms (+3.3%).

# Like-for-like runs against MTPLX, and the stock model (2026-10-27)

**bench/runtimes S1/S2** (now possible because the server streams). New labels in `run_all.sh`: T1 is the tree server with shipped defaults on Swift 1.5 4-bit text (the same build as C2), and T2 is the same on stock Qwen3.8 bare-speed (the same build as Q5). Raw rows are in `bench/runtimes/results-tree.jsonl`; all tables are in `bench/runtimes/summary-tree.md`, written by `compare_tree.py`. Settings: greedy, thinking off, 512 tokens, nonce-led prompts (cached_tokens was 0 on every request), one request at a time. The MTPLX rows date from 2026-10-25.

The tree rows were measured twice:
- first without the idle keepalive (swap flat at 759.94M);
- then again with it (swap flat at 743.94M). The numbers below and in the summary are from this second run.

Decode speeds matched between the two runs within noise; only TTFT changed.

Decode tok/s, tree / MTPLX MTP depth 3:
- S1, stock model, 128 to 32K context: 75.4 / 75.8 / 73.9 / 76.2 / 61.9 / 56.5 vs Q5 51.8 / 54.4 / 54.3 / 54.1 / 47.0 / 40.0, i.e. 1.32-1.45x.
- S1, Swift: 1.28-1.47x vs C2 and 1.30-1.46x vs C2n.
- S2 vs Q5 (stock model): code 1.41x, prose 1.31x, math 1.62x, json 1.56x, edit 1.82x, creative 1.29x.
- S2, Swift: T1 vs C2 is 1.29-1.62x, except edit at 0.80x. C2's context copy drafts spans from the prompt, which reaches 158 tok/s on a task that rewrites the prompt's file. Against C2n (no copy) the edit task is 1.82x.
- Tokens per target pass (Swift, S2): tree 4.65 / 3.30 / 5.46 / 6.25 / 7.95 / 2.57 vs C2n 3.31 / 2.64 / 3.59 / 3.73 / 3.98 / 2.26.

Costs:
- TTFT relative to MTPLX (>1 is faster) with the keepalive: 1.10-1.12x at 128, 1.00-1.01x at 1K, 0.97-0.98x from 4K up, i.e. 0.1-1.2 s slower. The remaining gap is the unpacking above.
- Without the keepalive TTFT was 0.78-0.98x.
- Load takes 40 s (34 s of it is cost curves) vs MTPLX's 10 s.
- Single tree runs vary more than MTPLX's (e.g. T2 at 8K in the first run: 77.6 / 64.3 / 77.8). Cells are medians of 3 runs (2 at 32K).

**Scenario sweep on the stock model** (`results/scenarios/scenarios_base_v1.json`; the same command and configs as the v3 sweep, with `--model ~/models/qwen38-base/mtplx-bare-speed --flash-attn --curve-context --cache-limit-gb 4`). Plain / adaptive chain / fork chain+siblings / fork full, tok/s:

| scenario | stock Qwen3.8 | Swift 1.5 (v3) |
|---|---|---|
| summarize codebase (27.9K) | 21.2 / 54.8 / 55.0 / **56.9** | 21.5 / 58.2 / **60.9** / 56.3 |
| summarize manual (27.9K) | 21.2 / 61.5 / 63.1 / **64.1** | 21.2 / 62.5 / **63.0** / 62.7 |
| write module (0.2K -> 8.4K) | 23.7 / 96.2 / 98.3 / **99.0** | 23.9 / 97.2 / 96.3 / **102.1** |
| rewrite file (11K, 137 tok) | 23.2 / **134.9** / 124.2 / 125.0 | 23.0 / **134.0** / 129.7 / 128.1 |
| write tests (15.1K -> 19.2K) | 22.6 / 88.4 / **92.3** / 87.0 | 22.7 / 111.7 / 111.8 / **111.9** |

- Write tests is the one real gap. The stock model accepts ~5.9 tokens/step there against Swift's ~7.5. Every other scenario is within ~±10%, mostly within noise.
- Outputs diverge from plain decoding earlier on the stock model: tokens 19-768. Rewrite file matches plain decoding exactly.
- Summarize manual stops at EOS at 1992-2045 tokens, depending on the mode.
- The first attempt was discarded because swap grew from 458M to 760M during load. The rerun ran with swap flat.

# Comparison with MTPLX's own MTP decoding (2026-10-25)

Superseded by the bench/runtimes runs above. At the time, only one matched comparison existed: the agentic eval below (same 10 tasks x 3), where the tree server
did +24% output tokens per second of model time sequentially (48.9 vs 39.4) and finished 2.1x sooner with 4
sessions in parallel (29.2 vs 62.4 min). The MTPLX run used the AX MXFP4 build and the tree server the 4-bit
affine build; MTPLX itself is 10-15% faster on the 4-bit build in `bench/runtimes`, so part of the +24% is the
build. The tree server has not been run through `bench/runtimes` (S1 context sweep, S2 task mix), so there is
no like-for-like single-request tok/s against MTPLX's MTP depth 3 yet. Different prompts suggest ~1.5x at
28-32K context (61-63 vs 39.5 tok/s) and 93-102 vs 55-60 tok/s on short code, but text type moves these
numbers a lot.

# Batching server (2026-10-25)

`mlx-tree-spec-server --max-batch 4`: continuous batching. tree_verify is batch-aware (batch_forward / batch_gdn / batch_attention / batch_mtp_entries / batch_pass / batch_draft; ForkState + batch_step); one request is bit-identical to the old single-request code (passes, drafts, decode checked against a saved reference). Requests share matmuls, norms, MLPs, the MTP layer and the draft head; GatedDeltaNet recurrence and attention run per request. Node cap per request: 16 total for 2, 32 for 3-4, 48 for 5+. A request whose client disconnects is dropped at the next step.

- Direct (decode only, greedy): alone 59-85 tok/s; 2 together 92, 3: 108, 4: 128 tok/s total.
- Concurrency replay of eval requests (prefill included, contexts up to 8.7K; `results/bench_concurrency_tree.json`): 47.6 / 66.7 / 81.5 out tok/s at 1 / 2 / 4 concurrent requests; projected 4-way eval 50.6 -> ~30 min.
- Agentic eval, 4 sessions in parallel: 34 min 40 s elapsed vs 50.6 min sequential (-31%), no swap, min 4 GB free. 26/30 passed: 2 runs of a long-answer task hit the eval client's fixed 120 s timeout (long replies at ~25 tok/s per request), and 2 runs of a rubric-graded explanation task missed the same 3 rubric points (3/3 sequential, 2/3 with the MTPLX baseline).
- With a configurable client timeout, same 10 tasks x 3:

  | arm | elapsed | pass |
  |---|---|---|
  | MTPLX baseline, sequential (sum of runs) | 62.4 min | 29/30 |
  | tree, sequential | 50.6 min | 30/30 |
  | batch, 2 parallel, 120 s timeout | 37.4 min (-26%) | 29/30 |
  | batch, 4 parallel, 600 s timeout | **29.2 min (-42%)** | 29/30 |

  The only failures left are one run of the rubric-graded task per arm (as with the baseline); no timeouts, no swap, min 4.5 GB free.
- Greedy decoding at ~9K context is not exactly repeatable run to run even without the long-context kernels (near-tie flips, e.g. at token 14); prefill and each attention kernel are deterministic in isolation. Source not found.

# Agentic coding eval on the server (2026-10-24)

The server (OpenAI-compatible, decode_fork 15,2,4,4, temperature sampled exactly, prompt-prefix snapshot bank, thinking split at </think>) driving the agentic coding eval, 3 trials of 10 tasks, against a baseline run on the MTPLX server (AX MXFP4 build of the same model):

| | baseline (MTPLX, MXFP4) | tree decoder (4-bit affine) |
|---|---|---|
| pass | 29/30 | 30/30 |
| total wall | 62.4 min | 50.6 min (-19%) |
| mean / median wall per run | 124.8 / 93.4 s | 101.1 / 93.5 s |
| output tokens per LLM second | 39.4 | 48.9 (+24%) |
| prompt tokens reused | 43% | 61% |

Server side: decode 53 tok/s token-weighted (64 median for replies >= 50 tokens), 4.0 tokens per step, prefill 630 tok/s on the uncached part, prompts up to 6.7K tokens, no truncations. Caveats: different model build and day, and the eval code changed slightly between the two runs.

# Round 3: flash block tuning + read-only long drafts (2026-10-24)

`results/scenarios/scenarios_bf16_v3.json` (+ `scenario3_v3b.json`: write module rerun after the below-8K fix). tok/s, previous round (tree drafting, "tree" column below) -> now:

| scenario | adaptive chain | fork chain+siblings | fork full | best vs first sweep |
|---|---|---|---|---|
| summarize codebase (27.9K) | 53.8 -> 58.2 | 57.9 -> **60.9** | 57.0 -> 56.3 | 55.4 -> 60.9 (+10%) |
| summarize manual (27.9K) | 60.8 -> 62.5 | 60.4 -> **63.0** | 61.0 -> 62.7 | 60.3 -> 63.0 (+4%) |
| write module (0.2K -> 8.4K) | 94.2 -> 97.2 | 96.7 -> 96.3 | 97.5 -> **102.1** | 95.7 -> 102.1 (+7%) |
| write tests (15.1K -> 19.2K) | 108.2 -> 111.7 | 108.3 -> 111.8 | 105.8 -> **111.9** | 106.9 -> 111.9 (+5%) |

Plain decoding unchanged (21-24 tok/s). Swap 0 throughout.

What changed:
- Flash kernels use 64 key blocks up to 48K (tree_flash.flash_blocks; MTPLX's chain kernels via their env vars): 16 layers, 8-query chain 6.9 -> 5.7 ms at 16K, 11.0 -> 10.5 at 32K; 16-node tree at 16K 12.2 -> 9.5.
- Drafting over a history >= 8K no longer appends to the MTP cache: inside one lazy graph each append copied the whole 128 MB buffer (queued reads keep it alive, so no in-place update). Draft entries now sit beside the cache; the history is read by the flash kernel (block stats) and merged with the entries' softmax (tree_verify._draft_attention). Draft step at 28K 2.0 -> 1.5-1.66 ms; also used by the adaptive chain (Drafter.chain_lazy, only >= 8K: below it MTPLX's unmasked path is faster, a first version without that limit cost 7% on write module).

Tried, no gain (reverted or kept off):
- Prefill matmuls already run at 43-46 TFLOPS (bf16 matmul 47-50); the packed NAX kernel is slower at 2048 rows.
- Layer-major prefill (expand packed weights once per layer instead of per chunk): bit-identical but 42.8-43.2 s vs 41.7 s; the per-chunk expansion overlaps compute. Reverted.
- Staging Q in threadgroup memory (QTG=1), key-splitting (KS 2/4): slower.

# 32K-window scenario sweeps (2026-10-24)

All runs: full-power charger, ~29 GB free before each run, swap 0 before and after, no other model processes, MLX buffer cache capped at 4 GB, memory wired.

tok/s. "old" = first clean sweep (copy-based branch drafting, `results/scenarios/scenarios_results.json`); "tree" = branches drafted as a tree in the MTP cache (`scenarios_bf16_v2.json`); "kv8" = tree drafting + 8-bit KV cache from 8K context (`scenarios_kv8_v2.json`).

| scenario | plain old/tree/kv8 | adaptive chain old/tree/kv8 | fork chain+siblings old/tree/kv8 | fork full old/tree/kv8 |
|---|---|---|---|---|
| summarize codebase (27.9K) | 21.5 / 21.6 / 21.8 | 55.4 / 53.8 / 54.5 | 54.6 / **57.9** / 56.0 | 53.2 / 57.0 / 54.2 |
| summarize manual (27.9K) | 21.6 / 21.5 / 21.8 | 60.3 / 60.8 / 60.0 | 59.0 / 60.4 / 61.6 | 59.9 / 61.0 / **61.7** |
| write module (0.2K -> 8.4K) | 24.1 / 23.6 / 23.9 | 95.7 / 94.2 / 94.7 | 94.3 / 96.7 / 94.3 | 90.8 / 97.5 / **99.0** |
| rewrite file (11K, stops at 137 tokens) | 23.1 / 23.0 / 23.1 | 124.0 / 128.9 / 126.3 | 113.0 / 124.2 / 123.8 | 121.4 / 124.3 / 123.6 |
| write tests (15.1K -> 19.2K) | 22.5 / 22.5 / 22.6 | 106.9 / 108.2 / 106.9 | 102.7 / **108.3** / 105.0 | 99.9 / 105.8 / 104.3 |

Run-to-run noise is about +-3% (outputs differ between runs because the packed kernels are not bit-exact, so the texts and acceptance differ).

## Tree drafting (tree_verify.draft_tree)
- Each level drafts the next token of every growing branch as one batch of entries appended to the shared MTP cache, with an ancestor bitmask (tree_flash now takes a tail longer than the queries and 64-bit masks); the entries are trimmed before the verify pass. No per-branch copy of the MTP history.
- Same drafts as the copy-based version (96/96 tokens at 200 and 27.9K context).
- Draft time at 27.9K: main 15 + two 4-branches 54.7 -> 29.9 ms; 4 x 4 branches 13.8 -> 6.6 ms; extra chain per call +1.29 -> +0.12 ms.
- The fork engine gained 2-7% overall (fork full 53-100 -> 57-106 tok/s), also without branches (the old path copied the MTP cache even for one chain). Fork configs now match or beat the adaptive chain in 4 of 5 scenarios.

## 8-bit KV (kv8.py, `--kv8`)
- int8 rows + one fp32 scale per row (MTPLX's format), 0.51x the bytes; quantized once the context reaches 8,192 tokens; per-token appends quantized by one fused kernel.
- Memory: -1.06 GB of KV at 32K (2.15 -> 1.09 GB); measured -0.9 GB active at 28K.
- Kernels (16 layers, 32K, ms bf16 -> int8): 1 query 5.9 -> 5.0 (MTPLX int8 vector kernel), 2 queries 6.8 -> 5.5, 4: 7.7 -> 9.3, 8: 11.0 -> 13.2, 16: 20.4 -> 26.1. From 4 queries the flash kernel is compute-bound, not KV-bandwidth-bound; fp16 matmuls with integer conversion and key-splitting did not help.
- End to end: plain decoding +1%; speculative configs within noise (-3% to +2%). Worth it for memory, not speed. Outputs first differ from bf16 KV after ~110-1450 tokens.
- 2026-10-30: the server default (`--no-kv8` turns it off; `tree_spec`'s CLI still needs `--kv8`). Runtime harness, same night, T2k vs T2 (`bench/runtimes/summary-splash.md`): decode within noise at every context and task (-5% to +1%), TTFT identical, tokens/step the same, peak footprint 23 vs 24 GB.

## Open
- Verify passes of 4-16 queries are compute-bound in the flash kernel at long context (8 queries at 32K: 11 ms for 2.15 GB, ~195 GB/s): a faster multi-query attention kernel would help bf16 and int8 alike.
- Drafting one chain at 27.9K still costs ~2 ms per MTP call (vs 1.2 at short context).

## Constrained decoding for agent harnesses (plan; version 1 built 2026-10-30, see "Grammar-constrained decoding" above)
Version 1 takes a JSON schema, Lark grammar or regex per request, so per-request value lists can be written into the grammar. llguidance's mask costs 0.06 ms, so it is applied only where the target's choice is disallowed (the fallback in the last bullet), not for every drafted node. Draft pruning and forced spans are still open ([roadmap](roadmap.md), item 5).
An agent harness that validates each structured reply rejects about 3% of them and asks again, costing a whole extra call each time. Most rejects are a value that isn't valid at that step (e.g. citing a file the agent never read) or a reply in the wrong shape. A grammar applied while decoding makes these impossible to generate.
- Request field: `grammar` (a JSON schema, or a TypeScript-subset grammar if the harness moves to writing programs) applied to the reply after `</think>`; thinking stays free.
- Per-request value lists: the harness sends what is valid for this step, e.g. a fact's `from` must be one of the files read so far plus `search` | `command` | `note`, and plan ids must exist. The grammar is built for each request, so it can't be a fixed compiled schema.
- Library: compare llguidance and xgrammar for MLX on the per-token cost of the mask over the 248K vocabulary, and the time to compile a per-request grammar.
- Tree speculation needs: a mask for every drafted node (16-32 per step), not one per step. Drafted tokens the grammar forbids are pruned before verify, which frees verify rows. Spans the grammar forces (e.g. `"status": {`) are appended without sampling.
- Measure first, on recorded harness requests: mask cost per step against a 15-40 ms step, tokens per step with and without pruning, and the share of reply tokens that are forced.
- Go/no-go: rejects go to 0 on the harness's eval with tokens per step not lower and step time up less than 5%. If the mask costs more than that, apply it only to the drafted path the target accepts, at verify.

## Drafting experiments (not started)
Acceptance ideas ranked by expected gain per effort. Code-heavy scenarios run at 97-112 tok/s and 28K summaries at 58-63, but much of that gap comes from verify cost at long context and from prose being less predictable to the target, which no drafter can beat.
- Measure first: log acceptance at each tree depth per scenario, so the options below can be compared against the same baseline.
- Tree shape by task: fork full wins on code and chain+siblings on summaries in the tables above. Track acceptance per depth during a request and resize the tree online. No training needed.
- N-gram (prompt-lookup) branches: add a branch copied from the context where the last few tokens match an earlier span. Summaries, code edits and tests quote the context a lot, which is where the MTP head is weakest. It needs no extra model passes; the branch rides in the same verify pass.
- Align the head with the served model: check whether the MTP head was trained against base Qwen3.8-27B or the Swift fine-tune. If against the base, distil it on the fine-tune's hidden states and outputs. This helps every task, not only one domain.
- Domain-specific MTP heads (e.g. code vs prose), routed per request: last on the list. The stock head is already trained on the full mix, so expect a few points of acceptance in-domain. Each head needs its own MTP cache (rebuild on switch, snapshot bank keyed by head), and requests with different heads stop sharing the draft matmuls under `--max-batch`. Mixed agentic turns (prose + code + tool JSON) make per-request routing wrong for much of the output. Go/no-go test: fine-tune a code-only copy (LoRA is enough) on a few thousand target outputs and compare on code. Under ~5 points of acceptance gain, drop it.

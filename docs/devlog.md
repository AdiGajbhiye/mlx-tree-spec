# Development log

Measurements from building the decoder, newest first. All on one M5 Max (36 GB) with Swift 1.5 4-bit (a Qwen3.8-27B fine-tune), greedy unless noted. Raw JSON for the scenario sweeps is in [results/](../results); the agentic coding eval mentioned below is a separate private suite (10 coding-agent tasks, 3 trials each, pass/fail graded) and is not part of this repo.

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

## Open
- Verify passes of 4-16 queries are compute-bound in the flash kernel at long context (8 queries at 32K: 11 ms for 2.15 GB, ~195 GB/s): a faster multi-query attention kernel would help bf16 and int8 alike.
- Drafting one chain at 27.9K still costs ~2 ms per MTP call (vs 1.2 at short context).

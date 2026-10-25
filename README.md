# mlx-tree-spec

Tree-speculative decoding with continuous batching for **Qwen3.8-27B on Apple Silicon**, as an
OpenAI-compatible server. It builds on [MTPLX](https://github.com/youssofal/mtplx)'s runtime and
kernels: the model's own multi-token-prediction (MTP) head drafts a small **tree** of candidate
continuations instead of one chain, and the target verifies all branches in one pass.

On an M5 Max this decodes a 27B model at **60–112 tok/s for one request, even at 28K context**
(plain decoding: 21–24 tok/s), and **128 tok/s total with 4 requests at once**.

> Status: research code that has run a few hundred agentic-coding sessions. Measured on one
> machine (M5 Max, 36 GB). Expect rough edges.

## Results

M5 Max (36 GB), Swift 1.5 4-bit (a Qwen3.8-27B fine-tune), greedy, one request, tok/s:

| scenario | plain decoding | MTP chain | MTP tree (this repo) |
|---|---:|---:|---:|
| summarize a codebase (27.9K-token prompt) | 21.6 | 58.2 | **60.9** |
| summarize a manual (27.9K-token prompt) | 21.5 | 62.5 | **63.0** |
| write a module (0.2K → 8.4K tokens) | 23.6 | 97.2 | **102.1** |
| write tests (15.1K → 19.2K tokens) | 22.5 | 111.7 | **111.9** |

"MTP chain" is this repo's adaptive single-chain drafter with the same kernels, so the gap to
it is what the tree itself adds; most of the gain over plain decoding is shared by both. Runs
vary about ±3%. Raw results: [results/scenarios](results/scenarios).

**Batching** (decode only, greedy): 1 request 59–85 tok/s; 2 together 92, 3: 108, 4: 128 tok/s
total. Requests share every matmul, norm and MLP and the draft head; only the linear-attention
recurrence and attention run per request.

**End to end**, a private agentic coding eval (10 tasks × 3 trials, prompts up to 8.7K tokens,
prefill included; not part of this repo):

| setup | wall time | passed |
|---|---:|---:|
| MTPLX server (MTP chain), sequential | 62.4 min | 29/30 |
| this server, sequential | 50.6 min | 30/30 |
| this server, 4 sessions in parallel | **29.2 min** | 29/30 |

## How it works

- **Tree drafting.** At every depth the MTP head proposes its top tokens; confident stretches stay
  a chain, and the tree widens only where the head is unsure. Branches are drafted as one batch
  in the shared MTP cache with an ancestor mask, with no per-branch copy of the history.
- **Verifying a tree on a hybrid model.** Qwen3.8 has three Gated-DeltaNet (linear attention)
  layers per full-attention layer. Their recurrent state can't take a tree attention mask, so
  each verify pass records the recurrence inputs per token. After the accepted path is chosen,
  only that recurrence is replayed for the accepted tokens, and the KV caches are trimmed. The
  weights are not run again.
- **Cost-aware tree shape.** At startup the server measures how much a verify pass costs per node,
  at short and long context. The tree is sized to maximise expected accepted tokens per
  millisecond, not per pass.
- **Kernels.** Packed 4-bit matmuls tuned for the 8–16-row verify passes. A tree-mask variant of
  MTPLX's flash-decoding kernel for long context. Fused GatedDeltaNet and norm ops.
- **Continuous batching.** Up to `--max-batch` requests decode together. Each step drafts all of
  them at once and verifies every tree in one target pass.
- **Prompt-prefix reuse.** Linear-attention state can't be trimmed, so the server keeps snapshots
  at the user-turn boundary and at prefixes shared with recent prompts, and resumes from the
  longest match.
- **Target-exact acceptance.** A drafted token is kept only if it is the token the target itself
  picks at that position (its argmax, or its sample at `temperature > 0`). The draft head changes
  speed, not what gets generated.

The development log with every measurement, including what didn't help, is in
[docs/devlog.md](docs/devlog.md).

## Requirements

- **Apple M5-family GPU** (the packed matmuls and flash kernels use the M5's GPU neural
  accelerators) and macOS 26.2 or later. Tested only on an M5 Max with 36 GB.
- About 20 GB of free unified memory for the 4-bit 27B model plus KV cache.
- Python 3.11+.

## Quickstart

```bash
git clone https://github.com/AdiGajbhiye/mlx-tree-spec && cd mlx-tree-spec
uv venv && uv pip install -e .
.venv/bin/mlx-tree-spec-server --port 8090
```

On first run it downloads [`Youssofal/Qwen3.8-27B-MTPLX-Bare-Speed`](https://huggingface.co/Youssofal/Qwen3.8-27B-MTPLX-Bare-Speed)
(Qwen3.8-27B in 4-bit with its MTP head, Apache-2.0). Startup measures cost curves for about
40 seconds. `--model` takes any local MTPLX-compatible checkpoint or Hugging Face repo id.

```bash
curl -s localhost:8090/v1/chat/completions -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"Write a haiku about caches."}],
       "max_tokens":200, "chat_template_kwargs":{"enable_thinking":false}}'
```

The server answers `POST /v1/chat/completions`, `GET /v1/models` and `GET /health`. Thinking is on
by default (the model's chat template). The reasoning comes back in `message.reasoning_content`,
and `--default-effort` sets `reasoning_effort` for requests that send none. `--help` lists the
options.

Run the tests (the lossless check loads the model):

```bash
uv pip install -e ".[dev]"
MLX_TREE_SPEC_MODEL=Youssofal/Qwen3.8-27B-MTPLX-Bare-Speed .venv/bin/pytest
```

## Limitations

- **No streaming yet**: responses come back whole. `"stream": true` is rejected.
- Only the Qwen3.8-27B architecture (hybrid GatedDeltaNet with an MTP head) loaded through MTPLX.
- **Greedy output is not token-for-token identical to plain decoding.** Verify passes run 8–16
  rows through different kernels than one-token decoding, so bf16 rounding flips near-ties and
  the texts drift apart. In the saved benchmark runs about half the outputs diverge, some within
  the first 20 tokens. `tests/test_lossless.py` re-scores the output in one teacher-forced pass and
  checks that every emitted token is the target's own top choice, up to such near-ties. Task
  quality held in the eval above (30/30 vs the MTPLX baseline's 29/30).
- Greedy decoding at around 9K context isn't exactly repeatable from run to run, even with
  plain decoding; the cause isn't found yet.

## Repository layout

```
src/mlx_tree_spec/   server, decoder (tree_spec, tree_verify) and Metal kernels
tests/               unit tests; lossless check against plain greedy decoding
bench/prompts/       long-context benchmark scenarios
bench/runtimes/      the llama.cpp / MTPLX / mlx-lm / DSpark comparison harness and its results
experiments/         one-off probes kept for reference (rank probe, draft vocab builder, ...)
results/             raw JSON behind the numbers above
docs/devlog.md       development log
```

Benchmark the decoding modes against plain greedy decoding on your own machine:

```bash
.venv/bin/python -m mlx_tree_spec.tree_spec --packed --draft-vocab general \
  --config fork:15,2,4,4 --prompt-file "bench/prompts/scenarios/3_write_module@8192.txt"
```

## Credits

Powered by [MTPLX](https://github.com/youssofal/mtplx) by Youssof Altoukhi. Built on
[MLX](https://github.com/ml-explore/mlx) and [mlx-lm](https://github.com/ml-explore/mlx-lm), with a
matmul kernel adapted from [dflash-mlx](https://github.com/bstnxbt/dflash-mlx). See
[NOTICE](NOTICE).

## License

Apache-2.0

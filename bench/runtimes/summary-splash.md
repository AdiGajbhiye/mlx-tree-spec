# mlx-tree-spec vs Splash: runtime harness S1/S2

Medians per cell (S1: 3 runs, 2 at 32768; S2: 3 runs). Greedy, thinking off, 512 max tokens, unique nonce per request, one request at a time, M5 Max 36 GB. SP and T2 were measured the same day (results-splash.jsonl); T2 ran after a 10-minute cooldown, because a run started 60 s after Splash finished came out 25–38% slower from 8K (results-splash-t2-after-splash.jsonl, discarded). B2, Q1, Q2 and Q5 are earlier runs, for context. **T2 vs SP** is tree / Splash (TTFT: Splash / tree); >1 means the tree server is faster.

## S1 decode tok/s by context

| config | 128 | 1024 | 4096 | 8192 | 16384 | 32768 |
|---|---:|---:|---:|---:|---:|---:|
| T2 mlx-tree-spec · Qwen3.8 bare-speed MLX 4bit · MTP tree (defaults) | 76.2 | 75.5 | 71.3 | 73.5 | 65.3 | 55.3 |
| SP Splash 1.5.0 · Qwen3.8 UD-Q4_K_M GGUF · DFlash2 (defaults) | 74.0 | 81.6 | 71.3 | 61.7 | 54.5 | 55.2 |
| B2 mlx-dspark · Qwen3.8 MLX 4bit · DFlash2 | 53.1 | 54.5 | 49.9 | 45.7 | 40.0 | 35.9 |
| Q5 MTPLX · Qwen3.8 bare-speed · MTP d3 only | 51.8 | 54.4 | 54.3 | 54.1 | 47.0 | 40.0 |
| Q2 llama.cpp · Qwen3.8 UD-Q4_K_M GGUF · MTP draft | 37.8 | 37.3 | 37.1 | 35.9 | 36.2 | 32.3 |
| Q1 llama.cpp · Qwen3.8 UD-Q4_K_M GGUF · no draft | 23.3 | 23.2 | 22.8 | 22.4 | 21.5 | 19.8 |
| **T2 vs SP** | 1.03x | 0.93x | 1.00x | 1.19x | 1.20x | 1.00x |

## S1 time to first token (s) by context

| config | 128 | 1024 | 4096 | 8192 | 16384 | 32768 |
|---|---:|---:|---:|---:|---:|---:|
| T2 mlx-tree-spec · Qwen3.8 bare-speed MLX 4bit · MTP tree (defaults) | 0.35 | 1.40 | 5.21 | 10.60 | 22.14 | 48.16 |
| SP Splash 1.5.0 · Qwen3.8 UD-Q4_K_M GGUF · DFlash2 (defaults) | 0.35 | 1.35 | 4.65 | 9.17 | 20.52 | 46.10 |
| B2 mlx-dspark · Qwen3.8 MLX 4bit · DFlash2 | 0.39 | 1.40 | 5.16 | 10.31 | 21.46 | 46.45 |
| Q5 MTPLX · Qwen3.8 bare-speed · MTP d3 only | 0.42 | 1.54 | 5.39 | 10.85 | 22.53 | 48.84 |
| Q2 llama.cpp · Qwen3.8 UD-Q4_K_M GGUF · MTP draft | 0.54 | 2.15 | 7.62 | 15.49 | 33.06 | 71.51 |
| Q1 llama.cpp · Qwen3.8 UD-Q4_K_M GGUF · no draft | 0.53 | 2.08 | 7.40 | 14.88 | 31.11 | 68.32 |
| **T2 vs SP** | 1.00x | 0.96x | 0.89x | 0.87x | 0.93x | 0.96x |

## S1 prefill tok/s by context

| config | 128 | 1024 | 4096 | 8192 | 16384 | 32768 |
|---|---:|---:|---:|---:|---:|---:|
| T2 mlx-tree-spec · Qwen3.8 bare-speed MLX 4bit · MTP tree (defaults) | 586 | 782 | 800 | 780 | 743 | 682 |
| SP Splash 1.5.0 · Qwen3.8 UD-Q4_K_M GGUF · DFlash2 (defaults) | 588 | 813 | 896 | 901 | 802 | 713 |
| B2 mlx-dspark · Qwen3.8 MLX 4bit · DFlash2 | 511 | 787 | 809 | 801 | 767 | 707 |
| Q5 MTPLX · Qwen3.8 bare-speed · MTP d3 only | 488 | 711 | 774 | 762 | 730 | 672 |
| Q2 llama.cpp · Qwen3.8 UD-Q4_K_M GGUF · MTP draft | 371 | 512 | 548 | 534 | 498 | 459 |
| Q1 llama.cpp · Qwen3.8 UD-Q4_K_M GGUF · no draft | 386 | 528 | 563 | 556 | 529 | 481 |
| **T2 vs SP** | 1.00x | 0.96x | 0.89x | 0.87x | 0.93x | 0.96x |

## S2 decode tok/s by task

| config | code | prose | math | json | edit | creative |
|---|---:|---:|---:|---:|---:|---:|
| T2 mlx-tree-spec · Qwen3.8 bare-speed MLX 4bit · MTP tree (defaults) | 84.3 | 63.2 | 99.4 | 113.7 | 126.8 | 51.6 |
| SP Splash 1.5.0 · Qwen3.8 UD-Q4_K_M GGUF · DFlash2 (defaults) | 94.4 | 57.0 | 111.7 | 118.5 | 81.6 | 44.4 |
| B2 mlx-dspark · Qwen3.8 MLX 4bit · DFlash2 | 59.7 | 47.5 | 67.4 | 66.1 | 64.7 | 40.2 |
| Q5 MTPLX · Qwen3.8 bare-speed · MTP d3 only | 61.0 | 48.5 | 62.2 | 66.2 | 69.7 | 39.8 |
| Q2 llama.cpp · Qwen3.8 UD-Q4_K_M GGUF · MTP draft | 40.2 | 32.2 | 42.6 | 45.7 | 46.8 | 28.1 |
| Q1 llama.cpp · Qwen3.8 UD-Q4_K_M GGUF · no draft | 23.3 | 23.3 | 23.3 | 23.3 | 23.0 | 23.3 |
| **T2 vs SP** | 0.89x | 1.11x | 0.89x | 0.96x | 1.55x | 1.16x |

## S2 tokens per target pass (tree: tokens_per_step; Splash: 1 + accepted/cycles)

| config | code | prose | math | json | edit | creative |
|---|---:|---:|---:|---:|---:|---:|
| T2 mlx-tree-spec · Qwen3.8 bare-speed MLX 4bit · MTP tree (defaults) | 4.58 | 3.26 | 5.39 | 6.47 | 7.83 | 2.59 |
| SP Splash 1.5.0 · Qwen3.8 UD-Q4_K_M GGUF · DFlash2 (defaults) | 5.22 | 3.28 | 6.10 | 6.48 | 4.83 | 2.59 |

## Load time (s, to first successful 1-token reply)

- T2: 41
- SP: 110
- B2: 10
- Q5: 10
- Q2: 11
- Q1: 11

## Swap used (MB) before / after each same-day run

- SP: 443.88 / 443.88
- T2: 443.88 / 443.88

# mlx-tree-spec vs Splash: runtime harness S1/S2

Medians per cell (S1: 3 runs, 2 at 32768; S2: 3 runs). Greedy, thinking off, 512 max tokens, unique nonce per request, one request at a time, M5 Max 36 GB. T2, T2k, SP, SPa and SPb ran the night of 2026-10-29/30, one config at a time with a 10-minute idle gap before each, and swap `used` did not grow during any of them. SP@1 is Splash's first run, on 2026-10-29: slower from 8K than the three Splash runs that night, which agree with each other. B2, Q1, Q2 and Q5 are earlier runs, for context. **T2 vs SP** is tree / Splash (TTFT: Splash / tree); >1 means the tree server is faster.

## S1 decode tok/s by context

| config | 128 | 1024 | 4096 | 8192 | 16384 | 32768 |
|---|---:|---:|---:|---:|---:|---:|
| T2 mlx-tree-spec · Qwen3.8 bare-speed MLX 4bit · MTP tree (defaults) | 82.2 | 77.0 | 72.1 | 74.9 | 64.3 | 55.6 |
| T2k mlx-tree-spec · same, `--kv8` | 83.3 | 75.8 | 73.2 | 71.8 | 61.1 | 54.8 |
| SP Splash 1.5.0 · Qwen3.8 UD-Q4_K_M GGUF · DFlash2 · `--max-context 36864` | 70.4 | 79.2 | 70.7 | 69.9 | 64.4 | 61.5 |
| SPa Splash 1.5.0 · README quick start (automatic context) | 78.6 | 81.7 | 68.1 | 68.4 | 65.3 | 62.9 |
| SPb Splash 1.5.0 · its llama.cpp-comparison settings (`--kv-format bf16 --disable-ane`) | 77.8 | 81.8 | 71.4 | 69.1 | 64.7 | 60.1 |
| SP@1 Splash 1.5.0 · as SP, first run on 2026-10-29 | 74.0 | 81.6 | 71.3 | 61.7 | 54.5 | 55.2 |
| B2 mlx-dspark · Qwen3.8 MLX 4bit · DFlash2 | 53.1 | 54.5 | 49.9 | 45.7 | 40.0 | 35.9 |
| Q5 MTPLX · Qwen3.8 bare-speed · MTP d3 only | 51.8 | 54.4 | 54.3 | 54.1 | 47.0 | 40.0 |
| Q2 llama.cpp · Qwen3.8 UD-Q4_K_M GGUF · MTP draft | 37.8 | 37.3 | 37.1 | 35.9 | 36.2 | 32.3 |
| Q1 llama.cpp · Qwen3.8 UD-Q4_K_M GGUF · no draft | 23.3 | 23.2 | 22.8 | 22.4 | 21.5 | 19.8 |
| **T2 vs SP** | 1.17x | 0.97x | 1.02x | 1.07x | 1.00x | 0.90x |

## S1 time to first token (s) by context

| config | 128 | 1024 | 4096 | 8192 | 16384 | 32768 |
|---|---:|---:|---:|---:|---:|---:|
| T2 mlx-tree-spec · Qwen3.8 bare-speed MLX 4bit · MTP tree (defaults) | 0.34 | 1.41 | 5.26 | 10.66 | 22.35 | 48.52 |
| T2k mlx-tree-spec · same, `--kv8` | 0.34 | 1.41 | 5.25 | 10.68 | 22.35 | 48.54 |
| SP Splash 1.5.0 · Qwen3.8 UD-Q4_K_M GGUF · DFlash2 · `--max-context 36864` | 0.34 | 1.27 | 4.32 | 8.59 | 17.88 | 39.39 |
| SPa Splash 1.5.0 · README quick start (automatic context) | 0.34 | 1.24 | 4.37 | 8.71 | 17.94 | 39.31 |
| SPb Splash 1.5.0 · its llama.cpp-comparison settings (`--kv-format bf16 --disable-ane`) | 0.34 | 1.35 | 4.91 | 9.82 | 20.51 | 44.73 |
| SP@1 Splash 1.5.0 · as SP, first run on 2026-10-29 | 0.35 | 1.35 | 4.65 | 9.17 | 20.52 | 46.10 |
| B2 mlx-dspark · Qwen3.8 MLX 4bit · DFlash2 | 0.39 | 1.40 | 5.16 | 10.31 | 21.46 | 46.45 |
| Q5 MTPLX · Qwen3.8 bare-speed · MTP d3 only | 0.42 | 1.54 | 5.39 | 10.85 | 22.53 | 48.84 |
| Q2 llama.cpp · Qwen3.8 UD-Q4_K_M GGUF · MTP draft | 0.54 | 2.15 | 7.62 | 15.49 | 33.06 | 71.51 |
| Q1 llama.cpp · Qwen3.8 UD-Q4_K_M GGUF · no draft | 0.53 | 2.08 | 7.40 | 14.88 | 31.11 | 68.32 |
| **T2 vs SP** | 1.01x | 0.90x | 0.82x | 0.81x | 0.80x | 0.81x |

## S1 prefill tok/s by context

| config | 128 | 1024 | 4096 | 8192 | 16384 | 32768 |
|---|---:|---:|---:|---:|---:|---:|
| T2 mlx-tree-spec · Qwen3.8 bare-speed MLX 4bit · MTP tree (defaults) | 600 | 778 | 794 | 775 | 736 | 677 |
| T2k mlx-tree-spec · same, `--kv8` | 597 | 778 | 794 | 774 | 736 | 676 |
| SP Splash 1.5.0 · Qwen3.8 UD-Q4_K_M GGUF · DFlash2 · `--max-context 36864` | 588 | 866 | 965 | 963 | 920 | 834 |
| SPa Splash 1.5.0 · README quick start (automatic context) | 602 | 887 | 955 | 949 | 917 | 835 |
| SPb Splash 1.5.0 · its llama.cpp-comparison settings (`--kv-format bf16 --disable-ane`) | 600 | 812 | 849 | 842 | 802 | 734 |
| SP@1 Splash 1.5.0 · as SP, first run on 2026-10-29 | 588 | 813 | 896 | 901 | 802 | 713 |
| B2 mlx-dspark · Qwen3.8 MLX 4bit · DFlash2 | 511 | 787 | 809 | 801 | 767 | 707 |
| Q5 MTPLX · Qwen3.8 bare-speed · MTP d3 only | 488 | 711 | 774 | 762 | 730 | 672 |
| Q2 llama.cpp · Qwen3.8 UD-Q4_K_M GGUF · MTP draft | 371 | 512 | 548 | 534 | 498 | 459 |
| Q1 llama.cpp · Qwen3.8 UD-Q4_K_M GGUF · no draft | 386 | 528 | 563 | 556 | 529 | 481 |
| **T2 vs SP** | 1.02x | 0.90x | 0.82x | 0.81x | 0.80x | 0.81x |

## S2 decode tok/s by task

| config | code | prose | math | json | edit | creative |
|---|---:|---:|---:|---:|---:|---:|
| T2 mlx-tree-spec · Qwen3.8 bare-speed MLX 4bit · MTP tree (defaults) | 85.3 | 62.7 | 103.0 | 100.0 | 127.5 | 50.3 |
| T2k mlx-tree-spec · same, `--kv8` | 86.7 | 62.1 | 101.4 | 101.1 | 126.9 | 51.2 |
| SP Splash 1.5.0 · Qwen3.8 UD-Q4_K_M GGUF · DFlash2 · `--max-context 36864` | 97.1 | 62.2 | 110.1 | 120.2 | 89.1 | 48.2 |
| SPa Splash 1.5.0 · README quick start (automatic context) | 98.6 | 54.4 | 100.9 | 111.6 | 77.8 | 43.1 |
| SPb Splash 1.5.0 · its llama.cpp-comparison settings (`--kv-format bf16 --disable-ane`) | 97.7 | 62.2 | 109.4 | 120.3 | 89.8 | 46.8 |
| SP@1 Splash 1.5.0 · as SP, first run on 2026-10-29 | 94.4 | 57.0 | 111.7 | 118.5 | 81.6 | 44.4 |
| B2 mlx-dspark · Qwen3.8 MLX 4bit · DFlash2 | 59.7 | 47.5 | 67.4 | 66.1 | 64.7 | 40.2 |
| Q5 MTPLX · Qwen3.8 bare-speed · MTP d3 only | 61.0 | 48.5 | 62.2 | 66.2 | 69.7 | 39.8 |
| Q2 llama.cpp · Qwen3.8 UD-Q4_K_M GGUF · MTP draft | 40.2 | 32.2 | 42.6 | 45.7 | 46.8 | 28.1 |
| Q1 llama.cpp · Qwen3.8 UD-Q4_K_M GGUF · no draft | 23.3 | 23.3 | 23.3 | 23.3 | 23.0 | 23.3 |
| **T2 vs SP** | 0.88x | 1.01x | 0.94x | 0.83x | 1.43x | 1.04x |

## S2 tokens per target pass (tree: tokens_per_step; Splash: 1 + accepted/cycles)

| config | code | prose | math | json | edit | creative |
|---|---:|---:|---:|---:|---:|---:|
| T2 mlx-tree-spec · Qwen3.8 bare-speed MLX 4bit · MTP tree (defaults) | 4.67 | 3.29 | 5.81 | 5.61 | 7.95 | 2.54 |
| T2k mlx-tree-spec · same, `--kv8` | 4.69 | 3.20 | 5.55 | 5.61 | 7.80 | 2.58 |
| SP Splash 1.5.0 · Qwen3.8 UD-Q4_K_M GGUF · DFlash2 · `--max-context 36864` | 5.17 | 3.30 | 5.89 | 6.40 | 4.83 | 2.56 |
| SPa Splash 1.5.0 · README quick start (automatic context) | 5.28 | 3.22 | 5.89 | 6.48 | 4.83 | 2.55 |
| SPb Splash 1.5.0 · its llama.cpp-comparison settings (`--kv-format bf16 --disable-ane`) | 5.22 | 3.30 | 5.82 | 6.40 | 4.88 | 2.48 |

## Load time (s, to first successful 1-token reply)

- T2: 40
- T2k: 40
- SP: 11
- SPa: 82
- SPb: 12
- SP@1: 110
- B2: 10
- Q5: 10
- Q2: 11
- Q1: 11

## Peak memory (sum of the server processes' phys_footprint_peak; footprint(1) rounds to 1 GB)

- T2: 24.0 GB
- T2k: 23.0 GB
- SP: 25.7 GB
- SPa: 25.7 GB
- SPb: 25.6 GB

## Swap used (MB) before / after each run

- T2: 1235.88 / 1235.88
- T2k: 1235.88 / 1227.88
- SPa: 3977.94 / 2487.12
- SPb: 1275.88 / 1235.88
- SP: 1235.88 / 1235.88
- SP@1: 443.88 / 443.88

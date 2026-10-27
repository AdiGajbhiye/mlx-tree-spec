# mlx-tree-spec vs MTPLX: runtime harness S1/S2

Medians per cell (S1: 3 runs, 2 at 32768; S2: 3 runs). Greedy, thinking off, 512 max tokens, unique nonce per request, one request at a time. C1/C2/C2n from results.jsonl, Q4/Q5 from results-base.jsonl, T1/T2 from results-tree.jsonl. Speedup rows are tree / MTPLX decode tok/s (TTFT: MTPLX / tree, >1 means the tree server is faster).

## Swift 1.5 4-bit text (Swift-1.5-4bit-MLX-text)

### S1 decode tok/s by context

| config | 128 | 1024 | 4096 | 8192 | 16384 | 32768 |
|---|---:|---:|---:|---:|---:|---:|
| T1 mlx-tree-spec · Swift 4bit · tree (defaults) | 75.0 | 73.6 | 67.8 | 70.9 | 65.1 | 53.6 |
| C2 MTPLX · Swift 4bit · MTP d3 + context copy | 55.1 | 55.4 | 52.9 | 51.4 | 44.4 | 39.5 |
| C2n MTPLX · Swift 4bit · MTP d3 only | 53.5 | 54.4 | 52.1 | 51.0 | 44.7 | 39.2 |
| C1 MTPLX · Swift 4bit · no MTP | 24.8 | 24.6 | 24.0 | 23.6 | 22.8 | 21.4 |
| **T1 vs C2** | 1.36x | 1.33x | 1.28x | 1.38x | 1.47x | 1.36x |
| **T1 vs C2n** | 1.40x | 1.35x | 1.30x | 1.39x | 1.46x | 1.37x |
| **T1 vs C1** | 3.03x | 3.00x | 2.82x | 3.00x | 2.86x | 2.51x |

### S1 time to first token (s) by context

| config | 128 | 1024 | 4096 | 8192 | 16384 | 32768 |
|---|---:|---:|---:|---:|---:|---:|
| T1 mlx-tree-spec · Swift 4bit · tree (defaults) | 0.37 | 1.53 | 5.52 | 11.08 | 23.10 | 49.98 |
| C2 MTPLX · Swift 4bit · MTP d3 + context copy | 0.41 | 1.53 | 5.39 | 10.89 | 22.77 | 49.12 |
| C2n MTPLX · Swift 4bit · MTP d3 only | 0.41 | 1.53 | 5.42 | 10.89 | 22.71 | 50.58 |
| C1 MTPLX · Swift 4bit · no MTP | 0.40 | 1.51 | 7.85 | 13.23 | 24.91 | 50.68 |
| **T1 vs C2** | 1.10x | 1.00x | 0.98x | 0.98x | 0.99x | 0.98x |
| **T1 vs C2n** | 1.09x | 1.00x | 0.98x | 0.98x | 0.98x | 1.01x |
| **T1 vs C1** | 1.08x | 0.99x | 1.42x | 1.19x | 1.08x | 1.01x |

### S1 prefill tok/s by context

| config | 128 | 1024 | 4096 | 8192 | 16384 | 32768 |
|---|---:|---:|---:|---:|---:|---:|
| T1 mlx-tree-spec · Swift 4bit · tree (defaults) | 535 | 720 | 756 | 746 | 712 | 657 |
| C2 MTPLX · Swift 4bit · MTP d3 + context copy | 498 | 715 | 774 | 759 | 723 | 669 |
| C2n MTPLX · Swift 4bit · MTP d3 only | 496 | 715 | 769 | 760 | 725 | 650 |
| C1 MTPLX · Swift 4bit · no MTP | 504 | 726 | 531 | 624 | 661 | 648 |
| **T1 vs C2** | 1.08x | 1.01x | 0.98x | 0.98x | 0.99x | 0.98x |
| **T1 vs C2n** | 1.08x | 1.01x | 0.98x | 0.98x | 0.98x | 1.01x |
| **T1 vs C1** | 1.06x | 0.99x | 1.42x | 1.19x | 1.08x | 1.01x |

### S2 decode tok/s by task

| config | code | prose | math | json | edit | creative |
|---|---:|---:|---:|---:|---:|---:|
| T1 mlx-tree-spec · Swift 4bit · tree (defaults) | 84.6 | 63.6 | 99.6 | 110.3 | 126.6 | 51.0 |
| C2 MTPLX · Swift 4bit · MTP d3 + context copy | 60.3 | 47.3 | 63.2 | 68.0 | 158.0 | 39.5 |
| C2n MTPLX · Swift 4bit · MTP d3 only | 60.0 | 47.9 | 65.5 | 68.0 | 69.4 | 41.0 |
| C1 MTPLX · Swift 4bit · no MTP | 24.6 | 24.7 | 24.8 | 24.8 | 24.3 | 24.7 |
| **T1 vs C2** | 1.40x | 1.34x | 1.58x | 1.62x | 0.80x | 1.29x |
| **T1 vs C2n** | 1.41x | 1.33x | 1.52x | 1.62x | 1.82x | 1.24x |
| **T1 vs C1** | 3.44x | 2.57x | 4.02x | 4.44x | 5.21x | 2.06x |

### S2 tokens per target pass (tree: tokens_per_step; MTPLX: 1 + accepted/verify)

| config | code | prose | math | json | edit | creative |
|---|---:|---:|---:|---:|---:|---:|
| T1 mlx-tree-spec · Swift 4bit · tree (defaults) | 4.65 | 3.30 | 5.46 | 6.25 | 7.95 | 2.57 |
| C2 MTPLX · Swift 4bit · MTP d3 + context copy | 3.31 | 2.61 | 3.47 | 3.73 | 2.69 | 2.19 |
| C2n MTPLX · Swift 4bit · MTP d3 only | 3.31 | 2.64 | 3.59 | 3.73 | 3.98 | 2.26 |
| C1 MTPLX · Swift 4bit · no MTP | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

## Stock Qwen3.8-27B (mtplx-bare-speed)

### S1 decode tok/s by context

| config | 128 | 1024 | 4096 | 8192 | 16384 | 32768 |
|---|---:|---:|---:|---:|---:|---:|
| T2 mlx-tree-spec · Qwen3.8 bare-speed · tree (defaults) | 75.4 | 75.8 | 73.9 | 76.2 | 61.9 | 56.5 |
| Q5 MTPLX · Qwen3.8 bare-speed · MTP d3 only | 51.8 | 54.4 | 54.3 | 54.1 | 47.0 | 40.0 |
| Q4 MTPLX · Qwen3.8 bare-speed · no MTP | 24.6 | 24.5 | 24.0 | 23.5 | 22.7 | 21.6 |
| **T2 vs Q5** | 1.45x | 1.39x | 1.36x | 1.41x | 1.32x | 1.41x |
| **T2 vs Q4** | 3.06x | 3.09x | 3.08x | 3.24x | 2.72x | 2.62x |

### S1 time to first token (s) by context

| config | 128 | 1024 | 4096 | 8192 | 16384 | 32768 |
|---|---:|---:|---:|---:|---:|---:|
| T2 mlx-tree-spec · Qwen3.8 bare-speed · tree (defaults) | 0.37 | 1.53 | 5.51 | 11.09 | 23.11 | 50.03 |
| Q5 MTPLX · Qwen3.8 bare-speed · MTP d3 only | 0.42 | 1.54 | 5.39 | 10.85 | 22.53 | 48.84 |
| Q4 MTPLX · Qwen3.8 bare-speed · no MTP | 0.41 | 1.51 | 5.30 | 10.59 | 22.13 | 47.62 |
| **T2 vs Q5** | 1.12x | 1.01x | 0.98x | 0.98x | 0.97x | 0.98x |
| **T2 vs Q4** | 1.09x | 0.99x | 0.96x | 0.95x | 0.96x | 0.95x |

### S1 prefill tok/s by context

| config | 128 | 1024 | 4096 | 8192 | 16384 | 32768 |
|---|---:|---:|---:|---:|---:|---:|
| T2 mlx-tree-spec · Qwen3.8 bare-speed · tree (defaults) | 542 | 719 | 756 | 745 | 712 | 657 |
| Q5 MTPLX · Qwen3.8 bare-speed · MTP d3 only | 488 | 711 | 774 | 762 | 730 | 672 |
| Q4 MTPLX · Qwen3.8 bare-speed · no MTP | 495 | 725 | 787 | 781 | 744 | 690 |
| **T2 vs Q5** | 1.11x | 1.01x | 0.98x | 0.98x | 0.98x | 0.98x |
| **T2 vs Q4** | 1.09x | 0.99x | 0.96x | 0.95x | 0.96x | 0.95x |

### S2 decode tok/s by task

| config | code | prose | math | json | edit | creative |
|---|---:|---:|---:|---:|---:|---:|
| T2 mlx-tree-spec · Qwen3.8 bare-speed · tree (defaults) | 86.0 | 63.4 | 100.6 | 103.4 | 127.1 | 51.4 |
| Q5 MTPLX · Qwen3.8 bare-speed · MTP d3 only | 61.0 | 48.5 | 62.2 | 66.2 | 69.7 | 39.8 |
| Q4 MTPLX · Qwen3.8 bare-speed · no MTP | 24.5 | 24.9 | 24.9 | 24.9 | 24.5 | 24.9 |
| **T2 vs Q5** | 1.41x | 1.31x | 1.62x | 1.56x | 1.82x | 1.29x |
| **T2 vs Q4** | 3.50x | 2.55x | 4.04x | 4.15x | 5.18x | 2.06x |

### S2 tokens per target pass (tree: tokens_per_step; MTPLX: 1 + accepted/verify)

| config | code | prose | math | json | edit | creative |
|---|---:|---:|---:|---:|---:|---:|
| T2 mlx-tree-spec · Qwen3.8 bare-speed · tree (defaults) | 4.88 | 3.30 | 5.51 | 5.67 | 8.05 | 2.62 |
| Q5 MTPLX · Qwen3.8 bare-speed · MTP d3 only | 3.35 | 2.68 | 3.45 | 3.62 | 3.98 | 2.21 |
| Q4 MTPLX · Qwen3.8 bare-speed · no MTP | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

## Load time (s, to first successful 1-token reply)

- T1: 40
- C2: 11
- C2n: 11
- C1: 10
- T2: 41
- Q5: 10
- Q4: 11

## Rows per config

- T1: 35
- C2: 35
- C2n: 35
- C1: 35
- T2: 35
- Q5: 35
- Q4: 35

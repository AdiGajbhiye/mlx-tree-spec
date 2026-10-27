# mlx-tree-spec vs MTPLX: runtime harness S1/S2

Medians per cell (S1: 3 runs, 2 at 32768; S2: 3 runs). Greedy, thinking off, 512 max tokens, unique nonce per request, one request at a time. C1/C2/C2n from results.jsonl, Q4/Q5 from results-base.jsonl, T1/T2 from results-tree.jsonl. Speedup rows are tree / MTPLX decode tok/s (TTFT: MTPLX / tree, >1 means the tree server is faster).

## Swift 1.5 4-bit text (Swift-1.5-4bit-MLX-text)

### S1 decode tok/s by context

| config | 128 | 1024 | 4096 | 8192 | 16384 | 32768 |
|---|---:|---:|---:|---:|---:|---:|
| T1 mlx-tree-spec · Swift 4bit · tree (defaults) | 76.7 | 76.4 | 69.2 | 71.5 | 62.6 | 55.6 |
| C2 MTPLX · Swift 4bit · MTP d3 + context copy | 55.1 | 55.4 | 52.9 | 51.4 | 44.4 | 39.5 |
| C2n MTPLX · Swift 4bit · MTP d3 only | 53.5 | 54.4 | 52.1 | 51.0 | 44.7 | 39.2 |
| C1 MTPLX · Swift 4bit · no MTP | 24.8 | 24.6 | 24.0 | 23.6 | 22.8 | 21.4 |
| **T1 vs C2** | 1.39x | 1.38x | 1.31x | 1.39x | 1.41x | 1.41x |
| **T1 vs C2n** | 1.43x | 1.40x | 1.33x | 1.40x | 1.40x | 1.42x |
| **T1 vs C1** | 3.10x | 3.11x | 2.88x | 3.03x | 2.75x | 2.60x |

### S1 time to first token (s) by context

| config | 128 | 1024 | 4096 | 8192 | 16384 | 32768 |
|---|---:|---:|---:|---:|---:|---:|
| T1 mlx-tree-spec · Swift 4bit · tree (defaults) | 0.35 | 1.40 | 5.21 | 10.55 | 22.10 | 48.01 |
| C2 MTPLX · Swift 4bit · MTP d3 + context copy | 0.41 | 1.53 | 5.39 | 10.89 | 22.77 | 49.12 |
| C2n MTPLX · Swift 4bit · MTP d3 only | 0.41 | 1.53 | 5.42 | 10.89 | 22.71 | 50.58 |
| C1 MTPLX · Swift 4bit · no MTP | 0.40 | 1.51 | 7.85 | 13.23 | 24.91 | 50.68 |
| **T1 vs C2** | 1.18x | 1.10x | 1.04x | 1.03x | 1.03x | 1.02x |
| **T1 vs C2n** | 1.17x | 1.10x | 1.04x | 1.03x | 1.03x | 1.05x |
| **T1 vs C1** | 1.16x | 1.08x | 1.51x | 1.25x | 1.13x | 1.06x |

### S1 prefill tok/s by context

| config | 128 | 1024 | 4096 | 8192 | 16384 | 32768 |
|---|---:|---:|---:|---:|---:|---:|
| T1 mlx-tree-spec · Swift 4bit · tree (defaults) | 581 | 787 | 801 | 783 | 745 | 684 |
| C2 MTPLX · Swift 4bit · MTP d3 + context copy | 498 | 715 | 774 | 759 | 723 | 669 |
| C2n MTPLX · Swift 4bit · MTP d3 only | 496 | 715 | 769 | 760 | 725 | 650 |
| C1 MTPLX · Swift 4bit · no MTP | 504 | 726 | 531 | 624 | 661 | 648 |
| **T1 vs C2** | 1.17x | 1.10x | 1.04x | 1.03x | 1.03x | 1.02x |
| **T1 vs C2n** | 1.17x | 1.10x | 1.04x | 1.03x | 1.03x | 1.05x |
| **T1 vs C1** | 1.15x | 1.08x | 1.51x | 1.25x | 1.13x | 1.06x |

### S2 decode tok/s by task

| config | code | prose | math | json | edit | creative |
|---|---:|---:|---:|---:|---:|---:|
| T1 mlx-tree-spec · Swift 4bit · tree (defaults) | 86.4 | 62.9 | 99.7 | 112.3 | 129.3 | 51.4 |
| C2 MTPLX · Swift 4bit · MTP d3 + context copy | 60.3 | 47.3 | 63.2 | 68.0 | 158.0 | 39.5 |
| C2n MTPLX · Swift 4bit · MTP d3 only | 60.0 | 47.9 | 65.5 | 68.0 | 69.4 | 41.0 |
| C1 MTPLX · Swift 4bit · no MTP | 24.6 | 24.7 | 24.8 | 24.8 | 24.3 | 24.7 |
| **T1 vs C2** | 1.43x | 1.33x | 1.58x | 1.65x | 0.82x | 1.30x |
| **T1 vs C2n** | 1.44x | 1.31x | 1.52x | 1.65x | 1.86x | 1.25x |
| **T1 vs C1** | 3.52x | 2.54x | 4.02x | 4.52x | 5.32x | 2.08x |

### S2 tokens per target pass (tree: tokens_per_step; MTPLX: 1 + accepted/verify)

| config | code | prose | math | json | edit | creative |
|---|---:|---:|---:|---:|---:|---:|
| T1 mlx-tree-spec · Swift 4bit · tree (defaults) | 4.70 | 3.27 | 5.40 | 6.22 | 7.95 | 2.57 |
| C2 MTPLX · Swift 4bit · MTP d3 + context copy | 3.31 | 2.61 | 3.47 | 3.73 | 2.69 | 2.19 |
| C2n MTPLX · Swift 4bit · MTP d3 only | 3.31 | 2.64 | 3.59 | 3.73 | 3.98 | 2.26 |
| C1 MTPLX · Swift 4bit · no MTP | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

## Stock Qwen3.8-27B (mtplx-bare-speed)

### S1 decode tok/s by context

| config | 128 | 1024 | 4096 | 8192 | 16384 | 32768 |
|---|---:|---:|---:|---:|---:|---:|
| T2 mlx-tree-spec · Qwen3.8 bare-speed · tree (defaults) | 76.6 | 76.2 | 71.0 | 76.3 | 66.1 | 56.0 |
| Q5 MTPLX · Qwen3.8 bare-speed · MTP d3 only | 51.8 | 54.4 | 54.3 | 54.1 | 47.0 | 40.0 |
| Q4 MTPLX · Qwen3.8 bare-speed · no MTP | 24.6 | 24.5 | 24.0 | 23.5 | 22.7 | 21.6 |
| **T2 vs Q5** | 1.48x | 1.40x | 1.31x | 1.41x | 1.41x | 1.40x |
| **T2 vs Q4** | 3.11x | 3.11x | 2.96x | 3.25x | 2.90x | 2.60x |

### S1 time to first token (s) by context

| config | 128 | 1024 | 4096 | 8192 | 16384 | 32768 |
|---|---:|---:|---:|---:|---:|---:|
| T2 mlx-tree-spec · Qwen3.8 bare-speed · tree (defaults) | 0.34 | 1.40 | 5.21 | 10.57 | 22.15 | 48.03 |
| Q5 MTPLX · Qwen3.8 bare-speed · MTP d3 only | 0.42 | 1.54 | 5.39 | 10.85 | 22.53 | 48.84 |
| Q4 MTPLX · Qwen3.8 bare-speed · no MTP | 0.41 | 1.51 | 5.30 | 10.59 | 22.13 | 47.62 |
| **T2 vs Q5** | 1.21x | 1.10x | 1.03x | 1.03x | 1.02x | 1.02x |
| **T2 vs Q4** | 1.18x | 1.08x | 1.02x | 1.00x | 1.00x | 0.99x |

### S1 prefill tok/s by context

| config | 128 | 1024 | 4096 | 8192 | 16384 | 32768 |
|---|---:|---:|---:|---:|---:|---:|
| T2 mlx-tree-spec · Qwen3.8 bare-speed · tree (defaults) | 593 | 784 | 800 | 782 | 743 | 684 |
| Q5 MTPLX · Qwen3.8 bare-speed · MTP d3 only | 488 | 711 | 774 | 762 | 730 | 672 |
| Q4 MTPLX · Qwen3.8 bare-speed · no MTP | 495 | 725 | 787 | 781 | 744 | 690 |
| **T2 vs Q5** | 1.22x | 1.10x | 1.03x | 1.03x | 1.02x | 1.02x |
| **T2 vs Q4** | 1.20x | 1.08x | 1.02x | 1.00x | 1.00x | 0.99x |

### S2 decode tok/s by task

| config | code | prose | math | json | edit | creative |
|---|---:|---:|---:|---:|---:|---:|
| T2 mlx-tree-spec · Qwen3.8 bare-speed · tree (defaults) | 87.2 | 62.8 | 100.8 | 102.2 | 126.5 | 51.3 |
| Q5 MTPLX · Qwen3.8 bare-speed · MTP d3 only | 61.0 | 48.5 | 62.2 | 66.2 | 69.7 | 39.8 |
| Q4 MTPLX · Qwen3.8 bare-speed · no MTP | 24.5 | 24.9 | 24.9 | 24.9 | 24.5 | 24.9 |
| **T2 vs Q5** | 1.43x | 1.29x | 1.62x | 1.54x | 1.81x | 1.29x |
| **T2 vs Q4** | 3.55x | 2.52x | 4.05x | 4.10x | 5.16x | 2.06x |

### S2 tokens per target pass (tree: tokens_per_step; MTPLX: 1 + accepted/verify)

| config | code | prose | math | json | edit | creative |
|---|---:|---:|---:|---:|---:|---:|
| T2 mlx-tree-spec · Qwen3.8 bare-speed · tree (defaults) | 4.69 | 3.32 | 5.51 | 5.61 | 8.12 | 2.58 |
| Q5 MTPLX · Qwen3.8 bare-speed · MTP d3 only | 3.35 | 2.68 | 3.45 | 3.62 | 3.98 | 2.21 |
| Q4 MTPLX · Qwen3.8 bare-speed · no MTP | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

## Load time (s, to first successful 1-token reply)

- T1: 40
- C2: 11
- C2n: 11
- C1: 10
- T2: 40
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

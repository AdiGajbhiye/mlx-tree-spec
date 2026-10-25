## S1 prefill tok/s

| config | 128 | 1024 | 4096 | 8192 | 16384 | 32768 |
|---|---:|---:|---:|---:|---:|---:|
| A1 Lithos · Qwen3.8 NVFP4 · no draft | 3.03 | 3.68 | – | – | – | – |
| A2 Lithos · Qwen3.8 NVFP4 · DSpark | 2.81 | 3.42 | – | – | – | – |
| B1 mlx-dspark · Qwen3.8 4bit · no draft | 473.09 | 757.78 | 808.54 | 804.67 | 771.0 | 709.24 |
| B2 mlx-dspark · Qwen3.8 4bit · DFlash2 | 511.14 | 786.87 | 808.59 | 801.18 | 766.8 | 707.01 |
| B3 mlx-dspark · Qwen3.8 4bit · DSpark | 537.77 | 787.74 | 802.65 | 794.51 | 758.58 | 696.44 |
| C1 MTPLX · Swift 4bit · no MTP | 504.02 | 726.11 | 530.95 | 624.41 | 660.65 | 647.98 |
| C2 MTPLX · Swift 4bit · MTP d3 + context copy | 498.03 | 714.67 | 773.85 | 758.91 | 722.58 | 668.64 |
| C2n MTPLX · Swift 4bit · MTP d3 only | 496.49 | 715.09 | 768.74 | 759.53 | 724.79 | 649.97 |
| C3 mlx-lm (ukisai) · Swift 4bit · no MTP | 312.61 | 579.02 | 683.86 | 689.55 | 663.67 | 606.55 |
| D1 llama.cpp · Swift IQ3_S · no MTP | 368.71 | 530.67 | 569.73 | 563.73 | 538.99 | 489.65 |
| D2 llama.cpp · Swift IQ3_S · MTP | 363.22 | 513.39 | 551.8 | 546.1 | 520.8 | 469.58 |
| E1 MTPLX · Swift AX MXFP4 · no MTP | 512.79 | 723.31 | 783.6 | 624.73 | 662.34 | 652.1 |
| E2 MTPLX · Swift AX MXFP4 · MTP d3 + context copy | 503.85 | 712.57 | 774.96 | 762.27 | 726.87 | 675.93 |
| E2n MTPLX · Swift AX MXFP4 · MTP d3 only | 506.44 | 713.82 | 772.4 | 764.91 | 729.66 | 653.22 |

## S1 decode tok/s

| config | 128 | 1024 | 4096 | 8192 | 16384 | 32768 |
|---|---:|---:|---:|---:|---:|---:|
| A1 Lithos · Qwen3.8 NVFP4 · no draft | 17.46 | 17.39 | – | – | – | – |
| A2 Lithos · Qwen3.8 NVFP4 · DSpark | 8.3 | 9.02 | – | – | – | – |
| B1 mlx-dspark · Qwen3.8 4bit · no draft | 25.95 | 25.79 | 25.37 | 24.87 | 24.05 | 22.85 |
| B2 mlx-dspark · Qwen3.8 4bit · DFlash2 | 53.09 | 54.55 | 49.89 | 45.66 | 40.02 | 35.93 |
| B3 mlx-dspark · Qwen3.8 4bit · DSpark | 59.64 | 57.5 | 58.01 | 50.06 | 42.86 | 37.21 |
| C1 MTPLX · Swift 4bit · no MTP | 24.79 | 24.56 | 24.01 | 23.62 | 22.77 | 21.41 |
| C2 MTPLX · Swift 4bit · MTP d3 + context copy | 55.14 | 55.42 | 52.95 | 51.4 | 44.41 | 39.53 |
| C2n MTPLX · Swift 4bit · MTP d3 only | 53.53 | 54.4 | 52.13 | 51.05 | 44.71 | 39.16 |
| C3 mlx-lm (ukisai) · Swift 4bit · no MTP | 22.85 | 22.78 | 22.49 | 22.0 | 21.33 | 20.02 |
| D1 llama.cpp · Swift IQ3_S · no MTP | 27.64 | 27.44 | 26.83 | 26.26 | 25.01 | 22.82 |
| D2 llama.cpp · Swift IQ3_S · MTP | 27.77 | 30.95 | 30.31 | 29.83 | 28.39 | 26.8 |
| E1 MTPLX · Swift AX MXFP4 · no MTP | 25.14 | 24.96 | 24.45 | 23.92 | 23.09 | 21.67 |
| E2 MTPLX · Swift AX MXFP4 · MTP d3 + context copy | 46.56 | 49.53 | 48.04 | 46.14 | 42.63 | 38.01 |
| E2n MTPLX · Swift AX MXFP4 · MTP d3 only | 48.86 | 53.26 | 48.19 | 46.81 | 44.2 | 38.09 |

## S1 time to first token (s)

| config | 128 | 1024 | 4096 | 8192 | 16384 | 32768 |
|---|---:|---:|---:|---:|---:|---:|
| A1 Lithos · Qwen3.8 NVFP4 · no draft | 67.01 | 298.77 | – | – | – | – |
| A2 Lithos · Qwen3.8 NVFP4 · DSpark | 71.65 | 321.03 | – | – | – | – |
| B1 mlx-dspark · Qwen3.8 4bit · no draft | 0.42 | 1.44 | 5.15 | 10.27 | 21.35 | 46.3 |
| B2 mlx-dspark · Qwen3.8 4bit · DFlash2 | 0.39 | 1.4 | 5.16 | 10.31 | 21.46 | 46.45 |
| B3 mlx-dspark · Qwen3.8 4bit · DSpark | 0.38 | 1.4 | 5.2 | 10.41 | 21.69 | 47.16 |
| C1 MTPLX · Swift 4bit · no MTP | 0.4 | 1.51 | 7.85 | 13.23 | 24.91 | 50.68 |
| C2 MTPLX · Swift 4bit · MTP d3 + context copy | 0.41 | 1.53 | 5.39 | 10.89 | 22.77 | 49.12 |
| C2n MTPLX · Swift 4bit · MTP d3 only | 0.41 | 1.53 | 5.42 | 10.89 | 22.71 | 50.58 |
| C3 mlx-lm (ukisai) · Swift 4bit · no MTP | 0.66 | 1.89 | 6.09 | 11.99 | 24.8 | 54.15 |
| D1 llama.cpp · Swift IQ3_S · no MTP | 0.54 | 2.07 | 7.32 | 14.66 | 30.53 | 67.07 |
| D2 llama.cpp · Swift IQ3_S · MTP | 0.56 | 2.13 | 7.56 | 15.13 | 31.59 | 69.94 |
| E1 MTPLX · Swift AX MXFP4 · no MTP | 0.39 | 1.52 | 5.32 | 13.23 | 24.84 | 50.36 |
| E2 MTPLX · Swift AX MXFP4 · MTP d3 + context copy | 0.4 | 1.54 | 5.38 | 10.84 | 22.64 | 48.59 |
| E2n MTPLX · Swift AX MXFP4 · MTP d3 only | 0.4 | 1.54 | 5.4 | 10.81 | 22.56 | 50.31 |

## S2 decode tok/s by task

| config | code | prose | math | json | edit | creative |
|---|---:|---:|---:|---:|---:|---:|
| A1 Lithos · Qwen3.8 NVFP4 · no draft | 17.49 | 17.49 | 17.46 | 17.47 | 17.26 | 17.48 |
| A2 Lithos · Qwen3.8 NVFP4 · DSpark | 11.52 | 6.75 | 14.23 | 14.91 | 17.64 | 5.48 |
| B1 mlx-dspark · Qwen3.8 4bit · no draft | 25.88 | 25.9 | 25.95 | 25.94 | 25.6 | 25.95 |
| B2 mlx-dspark · Qwen3.8 4bit · DFlash2 | 59.73 | 47.47 | 67.36 | 66.11 | 64.73 | 40.22 |
| B3 mlx-dspark · Qwen3.8 4bit · DSpark | 65.79 | 49.2 | 66.73 | 67.85 | 68.91 | 44.13 |
| C1 MTPLX · Swift 4bit · no MTP | 24.57 | 24.75 | 24.78 | 24.82 | 24.3 | 24.72 |
| C2 MTPLX · Swift 4bit · MTP d3 + context copy | 60.34 | 47.35 | 63.15 | 68.02 | 158.0 | 39.51 |
| C2n MTPLX · Swift 4bit · MTP d3 only | 60.01 | 47.95 | 65.5 | 68.05 | 69.44 | 40.98 |
| C3 mlx-lm (ukisai) · Swift 4bit · no MTP | 22.84 | 22.89 | 22.89 | 22.88 | 22.6 | 22.88 |
| D1 llama.cpp · Swift IQ3_S · no MTP | 27.58 | 27.59 | 27.57 | 27.56 | 27.13 | 27.59 |
| D2 llama.cpp · Swift IQ3_S · MTP | 32.01 | 25.92 | 34.08 | 35.44 | 37.05 | 22.63 |
| E1 MTPLX · Swift AX MXFP4 · no MTP | 25.03 | 25.12 | 25.13 | 25.1 | 24.67 | 25.1 |
| E2 MTPLX · Swift AX MXFP4 · MTP d3 + context copy | 51.01 | 42.66 | 55.23 | 58.25 | 150.71 | 35.86 |
| E2n MTPLX · Swift AX MXFP4 · MTP d3 only | 51.33 | 42.27 | 55.77 | 58.33 | 60.35 | 36.71 |

## S2 draft acceptance rate

| config | code | prose | math | json | edit | creative |
|---|---:|---:|---:|---:|---:|---:|
| A1 Lithos · Qwen3.8 NVFP4 · no draft | None | None | None | None | None | None |
| A2 Lithos · Qwen3.8 NVFP4 · DSpark | None | None | None | None | None | None |
| B1 mlx-dspark · Qwen3.8 4bit · no draft | None | None | None | None | None | None |
| B2 mlx-dspark · Qwen3.8 4bit · DFlash2 | None | None | None | None | None | None |
| B3 mlx-dspark · Qwen3.8 4bit · DSpark | None | None | None | None | None | None |
| C1 MTPLX · Swift 4bit · no MTP | None | None | None | None | None | None |
| C2 MTPLX · Swift 4bit · MTP d3 + context copy | 0.77 | 0.54 | 0.82 | 0.91 | 1.0 | 0.4 |
| C2n MTPLX · Swift 4bit · MTP d3 only | 0.77 | 0.55 | 0.86 | 0.91 | 0.99 | 0.42 |
| C3 mlx-lm (ukisai) · Swift 4bit · no MTP | None | None | None | None | None | None |
| D1 llama.cpp · Swift IQ3_S · no MTP | None | None | None | None | None | None |
| D2 llama.cpp · Swift IQ3_S · MTP | 0.81 | 0.58 | 0.86 | 0.91 | 0.98 | 0.46 |
| E1 MTPLX · Swift AX MXFP4 · no MTP | None | None | None | None | None | None |
| E2 MTPLX · Swift AX MXFP4 · MTP d3 + context copy | 0.76 | 0.58 | 0.85 | 0.92 | 1.0 | 0.44 |
| E2n MTPLX · Swift AX MXFP4 · MTP d3 only | 0.76 | 0.59 | 0.86 | 0.92 | 1.0 | 0.45 |

## Load time (s, to first successful 1-token reply)

- A1: 113
- A2: 22
- B1: 11
- B2: 10
- B3: 10
- C1: 10
- C2: 11
- C2n: 11
- C3: 10
- D1: 11
- D2: 10
- E1: 10
- E2: 11
- E2n: 11

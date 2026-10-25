#!/bin/bash
# Benchmark stock Qwen3.8-27B 4-bit: Q1-Q5 (llama.cpp and MLX, each with and without MTP).
cd "$(dirname "$0")"
ls -la ~/models/qwen38-base/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf ~/models/qwen38-base/gguf/MTP/mtp-Qwen3.8-27B-Q4_0.gguf || { echo "GGUF download missing"; exit 1; }
OUT=results-base.jsonl ./run_all.sh Q1 Q2 Q3
ls ~/models/qwen38-base/mtplx-bare-speed/model*.safetensors >/dev/null || { echo "MTPLX download missing"; exit 1; }
OUT=results-base.jsonl ./run_all.sh Q4 Q5
echo "=== base benchmark done $(date +%T)"

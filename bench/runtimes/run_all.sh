#!/bin/bash
# Runs S1 (context sweep) + S2 (task mix) for each config, one server at a time.
# Usage: ./run_all.sh [LABEL ...]   (default: all configs, in table order)
set -u
cd "$(dirname "$0")"
mkdir -p logs

PORT=8000
PY=$HOME/.local/share/mlx-dspark/.venv/bin/python
MTPLX=$HOME/.local/share/mtplx/.venv/bin/mtplx
DSPARK=$HOME/.local/share/mlx-dspark/.venv/bin/mlx-dspark
TREE=$HOME/projects/mlx-tree-spec/.venv/bin/mlx-tree-spec-server  # this repo's server
SPLASH=/opt/homebrew/bin/splash  # brew install incoai/tap/splash
DFLASH2=$(ls -d $HOME/.cache/huggingface/hub/models--incoai--Qwen3.8-27B-DFlash2/snapshots/*/ 2>/dev/null | head -1)
SWIFT_MLX=$HOME/models/swift15-mlx/Swift-1.5-4bit-MLX
SWIFT_MLX_TEXT=$HOME/models/swift15-mlx/Swift-1.5-4bit-MLX-text  # vision tensors stripped; MTPLX rejects them
SWIFT_AX=$HOME/models/swift15-mlx/AX-Swift-1.5-MXFP4-MTP
SWIFT_GGUF=$HOME/models/swift15
SWIFT_TOK=$SWIFT_MLX/tokenizer.json
BASE=$HOME/models/qwen38-base  # stock Qwen3.8-27B: unsloth UD-Q4_K_M GGUF + its MTP head, and MTPLX "Bare Speed" (flat 4-bit + MTP head)
QWEN_TOK=$(ls -d $HOME/.cache/huggingface/hub/models--nvidia--Qwen3.8-27B-NVFP4/snapshots/*/ | head -1)tokenizer.json
CTX=36864
SCENARIOS=${SCENARIOS:-s1,s2}
EXTRA_BENCH_ARGS=${EXTRA_BENCH_ARGS:-}
OUT=${OUT:-results.jsonl}

server_cmd() {
  case "$1" in
    A1) echo "lithos-metal serve --model nvidia/Qwen3.8-27B-NVFP4 --no-draft --max-context 4096 --port $PORT" ;;
    A2) echo "lithos-metal serve --model nvidia/Qwen3.8-27B-NVFP4 --max-context 4096 --port $PORT" ;;
    B1) echo "$DSPARK serve --model mlx-community/Qwen3.8-27B-4bit --mode baseline --no-lookup-drafts --no-prefix-cache --no-thinking --context-window $CTX --port $PORT" ;;
    B2) echo "$DSPARK serve --model mlx-community/Qwen3.8-27B-4bit --mode dflash --drafter incoai/Qwen3.8-27B-DFlash2 --no-lookup-drafts --no-prefix-cache --no-thinking --context-window $CTX --port $PORT" ;;
    B3) echo "$DSPARK serve --model mlx-community/Qwen3.8-27B-4bit --mode dspark --drafter RedHatAI/Qwen3.8-27B-speculator.dspark --no-lookup-drafts --no-prefix-cache --no-thinking --context-window $CTX --port $PORT" ;;
    C1) echo "$MTPLX serve --model $SWIFT_MLX_TEXT --generation-mode ar --no-mtp --profile sustained --ssd-session-cache off --no-auth --yes --port $PORT" ;;
    C2) echo "$MTPLX serve --model $SWIFT_MLX_TEXT --generation-mode mtp --mtp --depth 3 --profile sustained --ssd-session-cache off --no-auth --yes --port $PORT" ;;
    C3) echo "$HOME/.local/share/swift15-mlx/4bit/serve --port $PORT" ;;
    D1) echo "llama-server -m $SWIFT_GGUF/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf -c $CTX -np 1 --port $PORT" ;;
    D2) echo "llama-server -m $SWIFT_GGUF/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf --spec-type draft-mtp -c $CTX -np 1 --port $PORT" ;;
    # MTP only: MTPLX's default "context copy" also drafts by copying spans from the prompt
    C2n) echo "env MTPLX_CONTEXT_COPY=0 $MTPLX serve --model $SWIFT_MLX_TEXT --generation-mode mtp --mtp --depth 3 --profile sustained --ssd-session-cache off --no-auth --yes --port $PORT" ;;
    E2n) echo "env MTPLX_CONTEXT_COPY=0 $MTPLX serve --model $SWIFT_AX --generation-mode mtp --mtp --depth 3 --profile sustained --ssd-session-cache off --no-auth --yes --port $PORT" ;;
    E1) echo "$MTPLX serve --model $SWIFT_AX --generation-mode ar --no-mtp --profile sustained --ssd-session-cache off --no-auth --yes --port $PORT" ;;
    E2) echo "$MTPLX serve --model $SWIFT_AX --generation-mode mtp --mtp --depth 3 --profile sustained --ssd-session-cache off --no-auth --yes --port $PORT" ;;
    # Stock Qwen3.8-27B 4-bit, no fine-tune
    Q1) echo "llama-server -m $BASE/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf -c $CTX -np 1 --port $PORT" ;;
    Q2) echo "llama-server -m $BASE/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf -md $BASE/gguf/MTP/mtp-Qwen3.8-27B-Q4_0.gguf --spec-type draft-mtp -c $CTX -np 1 --port $PORT" ;;
    Q3) echo "$HOME/.local/share/mlx-dspark/.venv/bin/python -m mlx_lm server --model mlx-community/Qwen3.8-27B-4bit --port $PORT" ;;
    # mlx-lm keeps every past request's KV in its prompt cache by default; at 32k that ran the GPU out of memory
    Q3b) echo "$HOME/.local/share/mlx-dspark/.venv/bin/python -m mlx_lm server --model mlx-community/Qwen3.8-27B-4bit --prompt-cache-size 1 --port $PORT" ;;
    Q4) echo "$MTPLX serve --model $BASE/mtplx-bare-speed --generation-mode ar --no-mtp --profile sustained --ssd-session-cache off --no-auth --yes --port $PORT" ;;
    Q5) echo "env MTPLX_CONTEXT_COPY=0 $MTPLX serve --model $BASE/mtplx-bare-speed --generation-mode mtp --mtp --depth 3 --profile sustained --ssd-session-cache off --no-auth --yes --port $PORT" ;;
    # mlx-tree-spec (this repo), shipped defaults: tree-speculative decoding with the MTP head drafting.
    # Measures cost curves for ~40 s before it listens. No prefix reuse here: the nonce leads every prompt.
    T1) echo "$TREE --model $SWIFT_MLX_TEXT --no-kv8 --port $PORT" ;;  # vs C2 / C2n
    T2) echo "$TREE --model $BASE/mtplx-bare-speed --no-kv8 --port $PORT" ;;  # vs Q5 (Q4 for plain decoding); bf16 KV, as measured before kv8 became the default
    T2k) echo "$TREE --model $BASE/mtplx-bare-speed --kv8 --port $PORT" ;;  # T2 with 8-bit KV from 8K (the server default since 2026-10-30)
    # Splash (incoai), shipped defaults (INT8 KV, ANE prefill, DFlash2 draft) on Q1's GGUF.
    # Context capped like the llama.cpp rows; left on auto it plans for the native 256K window.
    SP) echo "$SPLASH serve --model local/qwen38-27b --model-dir $BASE/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf --draft-model $DFLASH2 --default-reasoning-effort none --max-context $CTX --port $PORT" ;;
    # README's quick start as is (automatic context and memory), thinking off as for every config
    SPa) echo "$SPLASH serve --model local/qwen38-27b --model-dir $BASE/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf --draft-model $DFLASH2 --default-reasoning-effort none --port $PORT" ;;
    # the settings of Splash's own accuracy comparison with llama.cpp: BF16 KV, GPU-only prefill
    SPb) echo "$SPLASH serve --model local/qwen38-27b --model-dir $BASE/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf --draft-model $DFLASH2 --default-reasoning-effort none --max-context $CTX --kv-format bf16 --disable-ane --port $PORT" ;;
  esac
}

model_for() {  # some servers (mlx-lm, lithos) reject unknown model names
  case "$1" in A*) echo nvidia/Qwen3.8-27B-NVFP4 ;; B*|Q3|Q3b) echo mlx-community/Qwen3.8-27B-4bit ;; C3) echo "$SWIFT_MLX" ;; T*) echo local ;; SP*) echo local/qwen38-27b ;; *) echo local ;; esac
}

bench_args_for() {
  # Lithos on the 32-core M5 Max: ~4 tok/s prefill (unmeasured fallback backend), so only short contexts
  case "$1" in A*) echo "--strict-openai --contexts 128,1024" ;; esac
}

tokenizer_for() { case "$1" in A*|B*|Q*|T2|T2k|SP*) echo "$QWEN_TOK" ;; *) echo "$SWIFT_TOK" ;; esac; }

free_port() {
  lsof -ti tcp:$PORT -sTCP:LISTEN | xargs kill 2>/dev/null
  for _ in $(seq 30); do lsof -ti tcp:$PORT -sTCP:LISTEN >/dev/null || return 0; sleep 2; done
  lsof -ti tcp:$PORT -sTCP:LISTEN | xargs kill -9 2>/dev/null
}

wait_ready() {  # a real 1-token completion, so "listening but still loading" doesn't count
  local deadline=$(( $(date +%s) + 2700 ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    kill -0 "$1" 2>/dev/null || return 1
    code=$(curl -s -o /dev/null -w '%{http_code}' -m 120 http://127.0.0.1:$PORT/v1/chat/completions \
      -H 'Content-Type: application/json' -H 'Authorization: Bearer local' \
      -d "{\"model\":\"$2\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1}")
    [ "$code" = "200" ] && return 0
    sleep 10
  done
  return 1
}

# Wrapped in main so bash parses the whole loop up front (safe to edit this file mid-run).
main() {
LABELS=("$@")
[ ${#LABELS[@]} -eq 0 ] && LABELS=(A1 A2 B1 B2 B3 C1 C2 C3 D1 D2 E1 E2)

for L in "${LABELS[@]}"; do
  free_port
  cmd=$(server_cmd "$L")
  echo "=== $L $(date +%T): $cmd" | tee -a logs/run_all.log
  t0=$(date +%s)
  bash -c "exec $cmd" > "logs/$L.server.log" 2>&1 &
  spid=$!
  if wait_ready $spid "$(model_for $L)"; then
    echo "$L ready after $(( $(date +%s) - t0 ))s" | tee -a logs/run_all.log
    echo "{\"label\":\"$L\",\"scenario\":\"load\",\"load_s\":$(( $(date +%s) - t0 ))}" >> "$OUT"
    $PY -I bench.py --label "$L" --model "$(model_for $L)" --url http://127.0.0.1:$PORT/v1 --tokenizer "$(tokenizer_for $L)" \
      --scenarios "$SCENARIOS" --out "$OUT" $(bench_args_for $L) $EXTRA_BENCH_ARGS 2>&1 | grep --line-buffered -v PyTorch | tee -a logs/run_all.log
  else
    echo "$L FAILED to become ready (see logs/$L.server.log)" | tee -a logs/run_all.log
  fi
  kill $spid 2>/dev/null; pkill -P $spid 2>/dev/null
  case "$L" in SP*) pkill -f "splash.*serve" 2>/dev/null ;; esac  # its engine runs as a child process
  case "$L" in C*|E*|Q4|Q5) $MTPLX stop --port $PORT >/dev/null 2>&1 ;; esac  # MTPLX only, not T*
  free_port
  sleep 60  # cool down between configs
done
echo "=== ALL DONE $(date +%T)" | tee -a logs/run_all.log
}
main "$@"; exit

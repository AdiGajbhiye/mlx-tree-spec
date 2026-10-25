#!/bin/bash
# Restart the C2 server per KV mode (no session caches, so every prompt is computed fresh) and record outputs.
# Usage: ./kvq_run.sh off:off_a off:off_b q8:q8_a q8:q8_b
cd "$(dirname "$0")"
mkdir -p kvq
PY=$HOME/.local/share/mlx-dspark/.venv/bin/python

restart() {  # $1 = KV mode, $2 = log name
  lsof -ti tcp:8080 -sTCP:LISTEN | xargs kill 2>/dev/null
  for _ in $(seq 30); do lsof -ti tcp:8080 -sTCP:LISTEN >/dev/null || break; sleep 2; done
  MTPLX_SESSION_BANK_MAX_BYTES=64M nohup "$HOME/.local/share/mtplx/swift-c2-server" \
    --ssd-session-cache off --paged-kv-quantization "$1" > "kvq/server-$2.log" 2>&1 &
  for _ in $(seq 90); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' -m 60 http://127.0.0.1:8080/v1/chat/completions \
      -H 'Content-Type: application/json' -H 'Authorization: Bearer mtplx-local' \
      -d '{"model":"swift-1.5-4bit-mtp","messages":[{"role":"user","content":"hi"}],"max_tokens":1}')" = 200 ] && return 0
    sleep 5
  done
  return 1
}

for spec in "$@"; do
  mode=${spec%%:*}; label=${spec#*:}
  restart "$mode" "$label" || { echo "$label: server failed to start"; continue; }
  grep -aoE "session-bank budget: [^ ]+|paged_kv[^,]*" "kvq/server-$label.log" | head -2
  $PY -I kvq_compare.py run "$label" 2>&1 | grep -v PyTorch
done

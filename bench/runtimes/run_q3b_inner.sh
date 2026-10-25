#!/bin/bash
# Q3-Q5 of the stock-Qwen benchmark, aborted if the Mac starts swapping (swap used > 1 GB).
cd "$(dirname "$0")"
OUT=results-base.jsonl ./run_all.sh Q3b &
RUN=$!
while kill -0 $RUN 2>/dev/null; do
  used=$(sysctl -n vm.swapusage | awk '{print $6}' | tr -d M)
  if [ "${used%.*}" -gt 1024 ]; then
    echo "=== SWAP ABORT $(date +%T): swap used ${used}M"
    top -l 1 -o mem -n 4 -stats pid,command,mem | tail -4
    pkill -f run_all.sh; pkill -f "bench.py --label Q"
    lsof -ti tcp:8000 -sTCP:LISTEN | xargs kill 2>/dev/null
    $HOME/.local/share/mtplx/.venv/bin/mtplx stop --port 8000 >/dev/null 2>&1
    exit 1
  fi
  sleep 10
done
echo "=== Q3b done $(date +%T)"

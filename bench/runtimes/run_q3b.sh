#!/bin/bash
# After Q3-Q5 finish: rerun plain mlx-lm with a one-entry prompt cache, same swap watchdog.
cd "$(dirname "$0")"
while pgrep -f run_rest.sh >/dev/null; do sleep 20; done
grep -q "SWAP ABORT" logs/run_rest.log && { echo "skipping Q3b: Q3-Q5 run aborted on swap"; exit 1; }
sed 's/run_all.sh Q3 Q4 Q5/run_all.sh Q3b/; s/Q3-Q5 done/Q3b done/' run_rest.sh > run_q3b_inner.sh && chmod +x run_q3b_inner.sh
./run_q3b_inner.sh

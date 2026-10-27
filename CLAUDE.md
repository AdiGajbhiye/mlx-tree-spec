# Notes for coding agents

## Running the server, benchmarks or anything that loads the model

The 27B model takes ~16 GB plus KV cache on a 36 GB Mac. Swapping both slows the machine and invalidates
any timing, so start a model process only when all of these hold:

1. **No other model process is running**: `pgrep -fl "mlx-tree-spec|mtplx|llama-server|mlx_lm"` prints nothing.
   Port 8090 may be serving other local experiments. Never stop or replace a server you didn't start;
   use another port, and only if the checks below still pass.
2. **At least 24 GB free**: `memory_pressure | tail -1` reports free percentage × 36 GB ≥ 24 GB,
   i.e. 67% or more.
3. **Not swapping**: `sysctl vm.swapusage` shows the same `used` value in two readings 30 s apart.
   Swap left over from earlier is fine as long as it isn't growing. macOS frees it only when those
   pages are read back in.

If any check fails, wait and re-check. Don't start anyway. If it still fails after ~30 min, stop and
tell the user what is holding the memory.

For benchmarks, record `vm.swapusage` before and after each run. If `used` grew during a run, the
numbers are invalid: discard them and rerun. Stop every server you started when you're done.

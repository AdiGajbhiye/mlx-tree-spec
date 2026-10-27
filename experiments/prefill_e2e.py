"""Prompt prefill on the real model: MLX's stock kernels vs the packed layout's
previous prefill path (unpack per call) vs qmm_packed.prefill_kernel.

One process, back to back (no idle gaps, so Metal residency stays warm). Stock
runs first because packed_linear.install drops MLX's weight layout; the two
packed paths are then interleaved per repetition. Each run is tree_spec.prefill
(target + MTP history, 2048-token chunks) of the first N tokens of the bash
manual. Reports median seconds and whether the result (first generated token and
the MTP hidden after the prompt, which depends on every layer) is bit-identical
to stock. Follows CLAUDE.md: refuses to start unless no other model process
runs, >= 67% memory is free and swap is not growing; records swap before and
after every run.

  python experiments/prefill_e2e.py --model ~/models/swift15-mlx/Swift-1.5-4bit-MLX-text
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import time
from pathlib import Path

import mlx.core as mx

from mlx_tree_spec import packed_linear
from mlx_tree_spec import tree_spec as ts

ROOT = Path(__file__).resolve().parents[1]


def swap_used_mb() -> float:
    out = subprocess.run(["sysctl", "vm.swapusage"], capture_output=True, text=True, check=False).stdout
    return float(re.search(r"used = ([\d.]+)M", out).group(1))


def preflight() -> None:
    busy = subprocess.run(["pgrep", "-fl", "mlx-tree-spec|mtplx|llama-server|mlx_lm"], capture_output=True, text=True, check=False).stdout
    busy = "\n".join(line for line in busy.splitlines() if "prefill_e2e" not in line)
    if busy.strip():
        raise SystemExit(f"another model process is running:\n{busy}")
    free = int(re.search(r"(\d+)%", subprocess.run(["memory_pressure"], capture_output=True, text=True, check=False).stdout.splitlines()[-1]).group(1))
    a = swap_used_mb()
    time.sleep(30)
    b = swap_used_mb()
    print(f"preflight: {free}% free, swap used {a:.0f} -> {b:.0f} MB over 30 s", flush=True)
    if free < 67 or b != a:
        raise SystemExit("not enough free memory or swapping; try later")


def run(rt, inner, prompt):
    ts._PREFILLED.clear()
    s0 = swap_used_mb()
    _, _, bonus, root_g, secs = ts.prefill(rt, inner, prompt)
    s1 = swap_used_mb()
    return secs, bonus, root_g, s1 - s0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokens", default="200,1024,4096,8192")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--json", default=str(ROOT / "results" / "prefill_kernel.json"))
    args = ap.parse_args()
    sizes = [int(t) for t in args.tokens.split(",")]

    preflight()
    mx.set_cache_limit(int(2e9))
    mx.set_wired_limit(int(mx.device_info()["max_recommended_working_set_size"]))
    rt = ts.load(ts.resolve_model(args.model))
    inner = getattr(rt.model, "language_model", rt.model)
    ids = rt.tokenizer.encode((ROOT / "bench" / "prompts" / "bash_manual_32k.txt").read_text())
    prompts = {n: ids[:n] for n in sizes}

    times: dict[str, dict[int, list[float]]] = {m: {n: [] for n in sizes} for m in ("stock", "unpack", "kernel")}
    outs: dict[str, dict[int, tuple]] = {m: {} for m in times}
    swap_growth = []

    def measure(mode):
        for n in sizes:
            secs, bonus, g, grew = run(rt, inner, prompts[n])
            times[mode][n].append(secs)
            outs[mode][n] = (bonus, g)
            swap_growth.append(grew)
            print(f"  {mode:<6} {n:>5} tokens {secs * 1e3:8.1f} ms  swap {grew:+.0f} MB", flush=True)

    run(rt, inner, prompts[sizes[0]])  # warm-up
    for _ in range(args.reps):
        measure("stock")
    packed_linear.install(rt.model, verbose=False)
    for mode in ("unpack", "kernel"):  # warm-up both (kernel compiles)
        packed_linear.PREFILL_UNPACK, packed_linear.PREFILL_M = mode == "unpack", 320 if mode == "unpack" else 64
        for n in sizes:
            run(rt, inner, prompts[n])
    for _ in range(args.reps):
        for mode in ("unpack", "kernel"):
            packed_linear.PREFILL_UNPACK, packed_linear.PREFILL_M = mode == "unpack", 320 if mode == "unpack" else 64
            measure(mode)

    rows = []
    print(f"\n{'tokens':>6} {'stock ms':>9} {'unpack':>14} {'kernel':>14}  kernel == stock")
    for n in sizes:
        med = {m: statistics.median(times[m][n]) for m in times}
        same = {
            m: outs[m][n][0] == outs["stock"][n][0] and mx.array_equal(outs[m][n][1], outs["stock"][n][1]).item()
            for m in ("unpack", "kernel")
        }
        rows.append(dict(tokens=n, ms={m: [t * 1e3 for t in times[m][n]] for m in times}, identical_to_stock=same))
        print(
            f"{n:>6} {med['stock'] * 1e3:9.1f} {med['unpack'] * 1e3:8.1f} {med['unpack'] / med['stock'] - 1:+5.1%}"
            f" {med['kernel'] * 1e3:8.1f} {med['kernel'] / med['stock'] - 1:+5.1%}  {same['kernel']}"
        )
    grew = max(swap_growth)
    print(f"max swap growth during a run: {grew:+.0f} MB" + ("  (INVALID: rerun)" if grew > 0 else ""))
    Path(args.json).write_text(json.dumps(dict(model=args.model, reps=args.reps, swap_growth_mb=max(swap_growth), rows=rows), indent=1))


if __name__ == "__main__":
    main()

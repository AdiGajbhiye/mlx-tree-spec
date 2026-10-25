"""Fastest 4-bit matmul per (layer shape, row count) for verify passes.

A verify forward multiplies M = rows x tokens activations (2..16) by every 4-bit
weight matrix of the target. MLX's kernels are near-flat up to ~5 rows, then cost
grows to 2.5-5x a 1-row pass by 8-16 rows (profile_verify.py). MTPLX ships
verify kernels for Apple's M5 matrix accelerators (mtplx/nax_verify.py, ported
from dflash-mlx): a 16-row NAX tile that costs ~1.15-1.6x a 1-row pass for ANY
M in 2..16, and K-split kernels for exactly 4 and 5-6 rows. None wins
everywhere, so install() times every candidate on the model's own layers for
each (K, N, M) at startup and routes each call to the fastest. M = 1 and M > 16
stay on MLX.

Not bit-exact: the NAX kernels accumulate in a different order (~6e-3 relative
drift, like any other kernel change), which can flip greedy argmax at near-ties.
MTPLX keeps them off for greedy verification for that reason; here they are an
explicit opt-in (--fast-verify).

My own multi-row kernels (one weight read for all M rows, plain SIMD and 8x8
simdgroup-matrix variants) were 1.5-2.5x slower than MLX at small M on this GPU
and were dropped in favour of these.
"""

from __future__ import annotations

import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from mtplx.nax_detect import nax_available
from mtplx.nax_verify import nax_qmm_m4, nax_qmm_m6, nax_qmm_m16

M_MAX = 16
_ORIG_CALL = nn.QuantizedLinear.__call__
_TABLE: dict[tuple[int, int, int], str] = {}
STATS: dict[str, int] = defaultdict(int)


def _candidates(k: int, n: int, m: int, gs: int, dtype) -> dict:
    """Kernels that can run this shape: name -> fn(x2, module) -> (m, n)."""
    out = {"stock": lambda x2, mod: _ORIG_CALL(mod, x2)}
    if dtype not in (mx.bfloat16, mx.float16) or gs not in (32, 64, 128):
        return out
    if nax_available() and k % 256 == 0 and n % 32 == 0:
        out["nax16"] = lambda x2, mod: nax_qmm_m16(x2, mod.weight, mod.scales, mod.biases, group_size=mod.group_size)
    if m == 4 and k % 32 == 0 and n % 4 == 0:
        out["nax4"] = lambda x2, mod: nax_qmm_m4(x2, mod.weight, mod.scales, mod.biases, group_size=mod.group_size)
    if m in (5, 6) and k % 32 == 0 and n % 4 == 0:
        out["nax6"] = lambda x2, mod: nax_qmm_m6(x2, mod.weight, mod.scales, mod.biases, group_size=mod.group_size)
    return out


def _eligible(mod) -> bool:
    return (
        isinstance(mod, nn.QuantizedLinear)
        and mod.bits == 4
        and getattr(mod, "mode", "affine") == "affine"
        and mod.get("biases") is not None
        and "bias" not in mod
    )


def _time(fn, reps: int = 6) -> float:
    for _ in range(2):
        mx.eval(fn())
    ts = []
    for _ in range(reps):
        mx.synchronize()
        t = time.perf_counter()
        mx.eval(fn())
        mx.synchronize()
        ts.append(time.perf_counter() - t)
    return statistics.median(ts)


def tune(model, cache_file: str | None = None, verbose: bool = True) -> dict:
    """Pick the fastest kernel for every (K, N, M) among the model's 4-bit layers."""
    if cache_file and Path(cache_file).exists():
        raw = json.loads(Path(cache_file).read_text())
        return {tuple(int(v) for v in key.split(",")): name for key, name in raw.items()}
    groups: dict[tuple[int, int], list] = defaultdict(list)
    for _, mod in model.named_modules():
        if _eligible(mod):
            groups[(mod.weight.shape[1] * 8, mod.weight.shape[0])].append(mod)
    table = {}
    for (k, n), mods in groups.items():
        mods = mods[:8]  # enough distinct copies that the weights don't sit in cache
        dtype = mods[0].scales.dtype
        wins = []
        for m in range(2, M_MAX + 1):
            x = (mx.random.normal((m, k)) * 0.5).astype(dtype)
            times = {
                name: _time(lambda f=f: [f(x, mod) for mod in mods])
                for name, f in _candidates(k, n, m, mods[0].group_size, dtype).items()
            }
            table[(k, n, m)] = best = min(times, key=times.get)
            wins.append(f"{m}:{best}({times['stock'] / times[best]:.2f}x)")
        if verbose:
            print(f"  {k:>6}->{n:<6} x{len(groups[(k, n)]):<3} " + " ".join(wins))
    if cache_file:
        Path(cache_file).write_text(json.dumps({f"{k},{n},{m}": v for (k, n, m), v in table.items()}, indent=0))
    return table


def _call(self, x):
    rows = 1
    for d in x.shape[:-1]:
        rows *= d
    if 2 <= rows <= M_MAX and _eligible(self):
        k, n = x.shape[-1], self.weight.shape[0]
        name = _TABLE.get((k, n, rows), "stock")
        STATS[name] += 1
        if name != "stock":
            f = _candidates(k, n, rows, self.group_size, x.dtype)[name]
            y = f(mx.contiguous(x.reshape(rows, k)), self)
            return y.reshape(*x.shape[:-1], n)
    return _ORIG_CALL(self, x)


def install(model, cache_file: str | None = None, verbose: bool = True) -> dict:
    """Tune (or load the tuned table) and route verify-shaped calls through it."""
    _TABLE.clear()
    _TABLE.update(tune(model, cache_file, verbose))
    nn.QuantizedLinear.__call__ = _call
    return dict(_TABLE)


def uninstall() -> None:
    nn.QuantizedLinear.__call__ = _ORIG_CALL

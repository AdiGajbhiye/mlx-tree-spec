"""Repack the target's 4-bit linears in place for the packed NAX kernels.

qmm_packed's NAX tile costs about the same for any row count up to 16 (verify
of 8-16 tokens ~1.05-1.25x a 1-token matmul, vs 2-4.5x with MLX's kernels), but
it needs its own weight layout, and keeping both layouts would double the
model's ~14 GB of matmul weights (this machine has 36 GB). So install() rewrites
every eligible QuantizedLinear (4-bit affine, N % 32 == 0, K % 128 == 0) into
the packed layout and drops the original arrays; every call goes to a kernel on
that layout:
  M <= 16 rows (decode, verify): the fastest of the 8-/16-row NAX tiles and,
      for M <= 4, the packed GEMV (qmv_packed), timed per shape at install as a
      dependent chain. The GEMV is what keeps 1-token decode at MLX's speed: an
      8-row tile padded around 1 row cost ~12% per token.
  16 < M < PREFILL_M (short prompts): one NAX dispatch over 32-row chunks.
  M >= PREFILL_M (prompts): qmm_packed.prefill_kernel, MLX's own prefill GEMM
      with a loader for the packed layout; same speed and bit-identical output.
      (It replaced unpacking the weight to MLX's layout per call, which cost
      ~100-120 ms per 2048-token forward.)
For M <= 16 each module has its launches prebuilt (shape-specialized kernels
with the weights bound): ~400 calls per forward through the generic path took
as much CPU as MLX's whole stock forward graph.
Linears that don't qualify (e.g. GDN's 48-wide a/b projections) stay on MLX.
Not bit-exact vs MLX's own kernels (~5e-3 relative per matmul).
"""

from __future__ import annotations

import gc
import math
import statistics
import time
from collections import defaultdict

import mlx.core as mx
import mlx.nn as nn

from .qmm_packed import _qmv_kernel, nax_kernel, nax_multi_kernel, pack_nax, prefill_kernel

_ORIG_CALL = nn.QuantizedLinear.__call__
# (K, N, M) -> ("nax", bm) | ("qmv", split, rows per lane)
_PLAN: dict[tuple[int, int, int], tuple] = {}
QMV_CONFIGS = ((8, 4), (8, 8), (16, 4))  # (split, rows per lane)
MULTI_TILE = (32, 2, 32)  # (rows, simdgroups, K step) for 17 .. PREFILL_M - 1 rows
PREFILL_M = 64
PREFILL_UNPACK = False  # the previous M >= PREFILL_M path, for comparison
STATS: dict[str, int] = defaultdict(int)


def eligible(mod) -> bool:
    return (
        isinstance(mod, nn.QuantizedLinear)
        and mod.bits == 4
        and getattr(mod, "mode", "affine") == "affine"
        and mod.get("biases") is not None
        and "bias" not in mod
        and mod.weight.shape[0] % 32 == 0
        and (mod.weight.shape[1] * 8) % 128 == 0
    )


def _launcher(mod, m: int, plan: tuple, dtype):
    """mod's packed matmul for m rows under plan, kernel and weights bound."""
    k = mod._wq2.size * 8 // mod._n
    if plan[0] == "qmv":
        kern = _qmv_kernel(k, mod._n, mod.group_size, m, plan[1], plan[2], dtype)
    else:
        kern = nax_kernel(k, mod._n, mod.group_size, m, plan[1], dtype)
    wq2, sp2, bp2 = mod._wq2, mod._sp2, mod._bp2
    return lambda x: kern(x, wq2, sp2, bp2)


def _unpacked(mod):
    """The weight in MLX's own layout again (inverse of pack_nax), built per call."""
    n = mod._n
    kw = mod._wq2.size // n
    g = mod._sp2.size // n
    w = mod._wq2.reshape(n // 32, kw // 2, 32, 2).transpose(0, 2, 1, 3).reshape(n, kw)
    s = mod._sp2.reshape(n // 32, g, 32).transpose(0, 2, 1).reshape(n, g)
    b = mod._bp2.reshape(n // 32, g, 32).transpose(0, 2, 1).reshape(n, g)
    return w, s, b


def _call(self, x):
    fast = self.__dict__.get("_fast")
    if fast is None:
        return _ORIG_CALL(self, x)
    sh = x.shape
    rows = sh[0] * sh[1] if len(sh) == 3 else math.prod(sh[:-1])
    run = fast.get(rows)
    if run is not None and x.dtype == self._dt:
        return run(x)
    if rows >= PREFILL_M:
        k = sh[-1]
        if PREFILL_UNPACK or self._n % 64:
            STATS["unpacked"] += 1
            w, s, b = _unpacked(self)
            return mx.quantized_matmul(x, w, s, b, transpose=True, group_size=self.group_size, bits=4)
        STATS["prefill"] += 1
        # BK 128 for the down projection (K 17408): 0.95-1.0x of MLX's time vs up to 1.09x with 64.
        tile = (64, 64, 128 if k >= 16384 else 64, 2, 2)
        return prefill_kernel(k, self._n, self.group_size, x.dtype, tile)(x, self._wq2, self._sp2, self._bp2)
    STATS["multi"] += 1
    bm, nsg, bk = MULTI_TILE
    return nax_multi_kernel(sh[-1], self._n, self.group_size, bm, nsg, x.dtype, bk)(x, self._wq2, self._sp2, self._bp2)


def _time(fn, reps: int = 5) -> float:
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


def _time_chain(mods, x, plan) -> float:
    """Time the mods as a dependent chain (each input from the previous output),
    as in a forward; independent calls overlap on the GPU and flatter kernels
    that are slow to ramp up."""
    k = x.shape[1]

    def f():
        h = x
        for d in mods:
            y = _launcher(d, h.shape[0], plan, h.dtype)(h)
            h = mx.concatenate([y] * -(-k // y.shape[1]), axis=1)[:, :k] * 0.01
        return h

    return _time(f)


def install(model, verbose: bool = True) -> dict:
    """Repack eligible linears in place, pick kernels per shape, patch calls."""
    before = mx.get_active_memory()
    groups: dict[tuple[int, int], list] = defaultdict(list)
    for _, mod in model.named_modules():
        if eligible(mod) and "_wq2" not in mod.__dict__:
            k, n = mod.weight.shape[1] * 8, mod.weight.shape[0]
            wq2, sp2, bp2 = pack_nax(mod.weight, mod.scales, mod.biases)
            mx.eval(wq2, sp2, bp2)
            # Plain attributes (not module parameters), so nothing else sees them.
            object.__setattr__(mod, "_wq2", wq2)
            object.__setattr__(mod, "_sp2", sp2)
            object.__setattr__(mod, "_bp2", bp2)
            object.__setattr__(mod, "_n", n)
            object.__setattr__(mod, "_dt", sp2.dtype)
            object.__setattr__(mod, "_fast", {})
            # Drop the original layout (a 1-element stand-in keeps the module valid).
            mod.weight = mx.zeros((1, 1), dtype=mx.uint32)
            mod.scales = mx.zeros((1, 1), dtype=sp2.dtype)
            mod.biases = mx.zeros((1, 1), dtype=bp2.dtype)
            groups[(k, n)].append(mod)
    gc.collect()
    mx.clear_cache()
    nn.QuantizedLinear.__call__ = _call
    for (k, n), mods in groups.items():
        picks = []
        for m in range(1, 17):
            x = (mx.random.normal((m, k)) * 0.5).astype(mods[0]._dt)
            plans = [("nax", bm) for bm in (8, 16) if m <= bm]
            if m <= 4:
                plans += [("qmv", sp, rpl) for sp, rpl in QMV_CONFIGS]
            times = {p: _time_chain(mods[:8], x, p) for p in plans}
            _PLAN[(k, n, m)] = best = min(times, key=times.get)
            picks.append(f"{m}:{best[0]}{best[1]}")
            for d in mods:
                d._fast[m] = _launcher(d, m, best, d._dt)
        if verbose:
            print(f"  packed {k:>6}->{n:<6} x{len(mods):<3} " + " ".join(picks))
    after = mx.get_active_memory()
    return dict(
        packed=sum(len(v) for v in groups.values()),
        memory_before_gb=before / 1e9,
        memory_after_gb=after / 1e9,
    )

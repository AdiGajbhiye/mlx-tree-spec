"""Prefill matmul on the packed layout vs MLX's quantized_matmul and the unpack path.

For each linear shape of the 27B model (random 4-bit weights, group 64, bf16),
times y = x @ W.T at prefill row counts three ways: MLX's own kernel on MLX's
layout (the speed to match), packed_linear's current path for M >= PREFILL_M
(unpack per call + quantized_matmul) and qmm_packed.prefill_kernel. Times are
medians of a dependent chain of 8 calls, as in a forward. Also reports the max
relative error of the packed kernel against quantized_matmul.

  python experiments/qmm_prefill.py [--tiles BM,BN,BK,WM,WN ...] [--rows 512,2048]
"""

from __future__ import annotations

import argparse
import statistics
import time

import mlx.core as mx

from mlx_tree_spec.packed_linear import _unpacked
from mlx_tree_spec.qmm_packed import PREFILL_TILE, pack_nax, prefill_kernel

# (K, N) of the eligible linears: MLP gate/up and down, GDN in_proj_qkv / in_proj_z /
# out_proj, full attention q / k,v / o.
SHAPES = ((5120, 17408), (17408, 5120), (5120, 10240), (5120, 6144), (6144, 5120), (5120, 12288), (5120, 1024))
GS = 64
DT = mx.bfloat16


class _Mod:
    def __init__(self, wq2, sp2, bp2, n):
        self._wq2, self._sp2, self._bp2, self._n = wq2, sp2, bp2, n


def _time(fn, reps=7):
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


def chain(call, x, k, depth=8):
    def f():
        h = x
        for _ in range(depth):
            y = call(h)
            h = mx.concatenate([y] * -(-k // y.shape[-1]), axis=-1)[..., :k] * 0.01
        return h

    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", nargs="*", default=[",".join(map(str, PREFILL_TILE))])
    ap.add_argument("--rows", default="320,512,2048")
    ap.add_argument("--shapes", default="all")
    args = ap.parse_args()
    tiles = [tuple(int(v) for v in t.split(",")) for t in args.tiles]
    rows = [int(r) for r in args.rows.split(",")]
    shapes = SHAPES if args.shapes == "all" else [SHAPES[int(i)] for i in args.shapes.split(",")]

    for k, n in shapes:
        w = mx.random.normal((n, k)) * 0.02
        wq, s, b = mx.quantize(w.astype(DT), group_size=GS, bits=4)
        wq2, sp2, bp2 = pack_nax(wq, s, b)
        mx.eval(wq, s, b, wq2, sp2, bp2)
        mod = _Mod(wq2, sp2, bp2, n)
        del w

        def mlx_qmm(h, wq=wq, s=s, b=b):
            return mx.quantized_matmul(h, wq, s, b, transpose=True, group_size=GS, bits=4)

        def unpack_qmm(h, mod=mod):
            w2, s2, b2 = _unpacked(mod)
            return mx.quantized_matmul(h, w2, s2, b2, transpose=True, group_size=GS, bits=4)

        for m in rows:
            x = (mx.random.normal((m, k)) * 0.5).astype(DT)
            ref = mlx_qmm(x)
            t_mlx = _time(chain(mlx_qmm, x, k))
            t_unp = _time(chain(unpack_qmm, x, k))
            line = f"{k:>6}->{n:<6} M={m:<5} mlx {t_mlx * 1e3:7.2f} ms  unpack {t_unp / t_mlx:5.3f}x"
            for tile in tiles:
                kern = prefill_kernel(k, n, GS, DT, tile)

                def packed(h, kern=kern, wq2=wq2, sp2=sp2, bp2=bp2):
                    return kern(h, wq2, sp2, bp2)

                y = packed(x)
                err = (mx.abs(y.astype(mx.float32) - ref.astype(mx.float32)).max() / mx.abs(ref.astype(mx.float32)).max()).item()
                t = _time(chain(packed, x, k))
                line += f"  {'x'.join(map(str, tile))}: {t / t_mlx:5.3f}x err {err:.1e}"
            print(line, flush=True)


if __name__ == "__main__":
    main()

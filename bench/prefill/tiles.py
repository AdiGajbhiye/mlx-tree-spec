"""The packed prefill GEMM's tile shape (qmm_packed.prefill_kernel) on the model's matmul shapes
at 2048 rows: ms per call and TFLOP/s for each (BM, BN, BK, WM, WN), and the max difference from
the default tile. No model needed. Usage: python bench/prefill/tiles.py"""
import statistics
import time

import mlx.core as mx

from mlx_tree_spec.qmm_packed import PREFILL_TILE, pack_nax, prefill_kernel

T = 2048
SHAPES = {"FFN gate/up 5120->17408": (5120, 17408), "FFN down 17408->5120": (17408, 5120),
          "GDN in_proj 5120->12288": (5120, 12288), "attn q 5120->12288": (5120, 12288)}
TILES = [PREFILL_TILE, (64, 128, 32, 2, 2), (128, 64, 32, 2, 2), (128, 128, 32, 2, 2), (128, 128, 32, 4, 4),
         (64, 64, 64, 2, 2), (128, 64, 32, 4, 2), (64, 128, 32, 2, 4), (32, 64, 32, 1, 2), (64, 32, 32, 2, 1)]


def med(f, n=9, w=3):
    ts = []
    for i in range(n + w):
        mx.synchronize(); t = time.perf_counter(); mx.eval(f()); mx.synchronize()
        if i >= w:
            ts.append(time.perf_counter() - t)
    return statistics.median(ts)


for name, (k, n) in SHAPES.items():
    x = mx.random.normal((1, T, k)).astype(mx.bfloat16)
    w = (mx.random.normal((n, k)) * 0.02).astype(mx.bfloat16)
    wq, s, b = mx.quantize(w, group_size=64, bits=4)
    packed = pack_nax(wq, s, b)
    mx.eval(x, *packed)
    ref = prefill_kernel(k, n, 64, mx.bfloat16)(x, *packed)
    print(name)
    for tile in TILES:
        try:
            f = prefill_kernel(k, n, 64, mx.bfloat16, tile)
            y = f(x, *packed)
            d = mx.abs(y.astype(mx.float32) - ref.astype(mx.float32)).max().item()
            t = med(lambda: f(x, *packed))
            print(f"  {tile!s:22s} {t * 1e3:6.2f} ms  {2 * T * k * n / t / 1e12:5.1f} TFLOP/s  max diff {d:.2g}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  {tile!s:22s} failed: {str(e).splitlines()[0][:100]}", flush=True)

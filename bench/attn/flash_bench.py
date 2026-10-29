"""Multi-query attention at long context, as one verify pass sees it: 16 attention layers
(24 query heads, 4 KV heads, head dim 256) over a `ctx`-token bf16 (or kv8) cache, for chains of
n queries. Times tree_flash and MLX's own scaled_dot_product_attention (with an additive mask)
and checks they agree. No model needed. Usage: python bench/attn/flash_bench.py [CTX...]"""
import statistics
import sys
import time

import mlx.core as mx
from mlx_lm.models.cache import KVCache

from mlx_tree_spec import kv8, tree_flash

HQ, HK, D, LAYERS = 24, 4, 256, 16


def med(f, n=9, warm=3):
    ts = []
    for i in range(n + warm):
        mx.synchronize()
        t = time.perf_counter()
        mx.eval(f())
        mx.synchronize()
        if i >= warm:
            ts.append((time.perf_counter() - t) * 1e3)
    return statistics.median(ts)


def caches(ctx, q8):
    out = []
    for i in range(LAYERS):
        c = KVCache()
        k = mx.random.normal((1, HK, ctx, D), key=mx.random.key(2 * i)).astype(mx.bfloat16)
        v = mx.random.normal((1, HK, ctx, D), key=mx.random.key(2 * i + 1)).astype(mx.bfloat16)
        c.update_and_fetch(k, v)
        if q8:
            n = kv8.Q8KVCache()
            n.offset = c.offset
            (n.keys, n.k_scale), (n.values, n.v_scale) = kv8.quantize_rows(c.keys), kv8.quantize_rows(c.values)
            c = n
        mx.eval(c.state)
        out.append(c)
    return out


def main():
    ctxs = [int(a) for a in sys.argv[1:]] or [16384, 32768]
    scale = D ** -0.5
    print(f"{'ctx':>6} {'kv':>4} {'n':>3} | {'tree_flash':>10} {'mlx sdpa':>9} | {'max err':>8}")
    for ctx in ctxs:
        for q8 in (False, True):
            cs = caches(ctx, q8)
            for n in (1, 2, 4, 8, 12, 16):
                q = mx.random.normal((1, HQ, n, D), key=mx.random.key(99)).astype(mx.bfloat16)
                bits = mx.array(tree_flash.ancestor_bits([[j <= i for j in range(n)] for i in range(n)]), dtype=mx.uint64)
                t_tree = med(lambda: [tree_flash.tree_flash(q, c, scale, bits) for c in cs])
                t_sdpa = err = None
                if not q8:
                    mask = mx.concatenate([mx.zeros((n, ctx - n)), mx.where(mx.tri(n) > 0, 0.0, -mx.inf)], axis=1).astype(mx.bfloat16)
                    kvs = [(c.keys[..., : c.offset, :], c.values[..., : c.offset, :]) for c in cs]
                    t_sdpa = med(lambda: [mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask) for k, v in kvs])
                    a = tree_flash.tree_flash(q, cs[0], scale, bits)
                    b = mx.fast.scaled_dot_product_attention(q, *kvs[0], scale=scale, mask=mask)
                    err = mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max().item()
                sd = f"{t_sdpa:9.2f}" if t_sdpa else " " * 9
                print(f"{ctx:6d} {'q8' if q8 else 'bf16':>4} {n:3d} | {t_tree:10.2f} {sd} | {err if err is not None else '':>8}", flush=True)
            del cs


if __name__ == "__main__":
    main()

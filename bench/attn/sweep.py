"""tree_flash's launch shape at long context: key blocks per KV head x key-split simdgroups per
threadgroup (KS), ms over 16 layers including the reduce. Usage: python bench/attn/sweep.py [CTX]"""
import sys
import mlx.core as mx
from mlx_lm.models.cache import KVCache
from mlx_tree_spec import tree_flash as tf
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from staged import HQ, HK, D, L, kernel, med  # noqa: E402

CTX = int(sys.argv[1]) if len(sys.argv) > 1 else 32768


def attend(k, q, c, bits, n, blocks, ks):
    from mtplx.kernels.sdpa_gqa_packed import _paged_reduce_kernel
    nthreads = 32 * ((6 * n + 15) // 16) * ks
    p, s, m = k(inputs=[q, c.keys, c.values, mx.array([c.offset], dtype=mx.int32), c.keys.shape[2], D ** -0.5,
                        mx.array([blocks], dtype=mx.int32), bits, n],
                template=[("InT", mx.bfloat16), ("PartT", mx.float32), ("D", D), ("QL", n), ("GQA_F", 6), ("KS", ks), ("QTG", 0)],
                grid=(HK * nthreads, 1, blocks), threadgroup=(nthreads, 1, 1),
                output_shapes=[(1, HQ, n, blocks, D), (1, HQ, n, blocks), (1, HQ, n, blocks)],
                output_dtypes=[mx.float32, mx.float32, mx.float32])
    return _paged_reduce_kernel()(inputs=[p, s, m, blocks], template=[("InT", mx.bfloat16), ("V", D)],
                                  grid=(HQ * 1024, n, 1), threadgroup=(1024, 1, 1),
                                  output_shapes=[(1, HQ, n, D)], output_dtypes=[mx.bfloat16])[0]


cs = []
for i in range(L):
    c = KVCache()
    c.update_and_fetch(*(mx.random.normal((1, HK, CTX - 100, D), key=mx.random.key(2 * i + j)).astype(mx.bfloat16) for j in (0, 1)))
    mx.eval(c.state); cs.append(c)
k = kernel([])
shapes = [(b, s) for s in (1, 2, 4) for b in (32, 64, 128, 256)]
print(f"ctx {CTX - 100}; columns blocks/KS")
print("n   " + " ".join(f"{b:>4}/{s}" for b, s in shapes))
for n in (1, 2, 4, 8, 12, 16):
    q = mx.random.normal((1, HQ, n, D)).astype(mx.bfloat16)
    bits = mx.array(tf.ancestor_bits([[j <= i for j in range(n)] for i in range(n)]), dtype=mx.uint64)
    ref = attend(k, q, cs[0], bits, n, 64, 1)
    row = []
    for b, s in shapes:
        if 32 * ((6 * n + 15) // 16) * s > 1024:
            row.append("     -"); continue
        out = attend(k, q, cs[0], bits, n, b, s)
        bad = mx.abs(out.astype(mx.float32) - ref.astype(mx.float32)).max().item() > 1e-2
        row.append(f"{med(lambda: [attend(k, q, c, bits, n, b, s) for c in cs]):6.2f}" + ("!" if bad else ""))
    print(f"{n:<3} " + " ".join(row), flush=True)

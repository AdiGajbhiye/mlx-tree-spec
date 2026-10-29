"""Where does tree_flash's time go at long context? Variants of the kernel with parts removed
(results are wrong on purpose; only the timing matters). Usage: python bench/attn/ablate.py"""
import statistics, time
import mlx.core as mx
from mlx_lm.models.cache import KVCache
from mlx_tree_spec import tree_flash as tf

HQ, HK, D, L, CTX = 24, 4, 256, 16, 32768
K_LOAD = "kk4 = reinterpret_cast<const device vec<InT, 4>*>( k_head + (size_t)kr * D + kk * 16 + sc.x)[0];"
V_LOAD = "v4 = reinterpret_cast<const device vec<InT, 4>*>( v_head + (size_t)kr * D + g * 32 + hh * 16 + sc.x)[0];"
VARIANTS = {
    "baseline": [],
    "no K loads": [(K_LOAD, "kk4 = vec<InT, 4>(InT(kr & 7));")],
    "no V loads": [(V_LOAD, "v4 = vec<InT, 4>(InT(kr & 7));")],
    "no K/V loads": [(K_LOAD, "kk4 = vec<InT, 4>(InT(kr & 7));"), (V_LOAD, "v4 = vec<InT, 4>(InT(kr & 7));")],
    "no QK matmul": [("mm_qk.run(ct_qa, ct_kb, ct_s);", "ct_s[0] += float(ct_kb[0]) * float(ct_qa[0]);")],
    "no PV matmul": [("mm_pv.run(ct_pa, ct_vb, ct_o);", "ct_o[0] += float(ct_vb[0]) * float(ct_pa[0]);")],
}


def kernel(edits):
    from mtplx.kernels.sdpa_nax_flash import _HEADER, _SOURCE
    src = tf._edit(tf._edit(_SOURCE, tf._OLD_VIS, tf._NEW_VIS), tf._OLD_TAIL, tf._NEW_TAIL)
    for old, new in edits:
        src = tf._edit(src, old, new)
    return mx.fast.metal_kernel(name="abl" + str(abs(hash(str(edits)))), header=_HEADER, source=src,
                                input_names=["queries", "keys", "values", "offset", "kcap", "scale", "blocks", "anc", "tail_n"],
                                output_names=["partials", "sums", "maxs"])


def run(k, q, c, bits, n):
    blocks = tf.flash_blocks(c.keys.shape[2])
    nthreads = 32 * ((6 * n + 15) // 16)
    return k(inputs=[q, c.keys, c.values, mx.array([c.offset], dtype=mx.int32), c.keys.shape[2], D ** -0.5,
                     mx.array([blocks], dtype=mx.int32), bits, n],
             template=[("InT", mx.bfloat16), ("PartT", mx.float32), ("D", D), ("QL", n), ("GQA_F", 6), ("KS", 1), ("QTG", 0)],
             grid=(HK * nthreads, 1, blocks), threadgroup=(nthreads, 1, 1),
             output_shapes=[(1, HQ, n, blocks, D), (1, HQ, n, blocks), (1, HQ, n, blocks)],
             output_dtypes=[mx.float32, mx.float32, mx.float32])


def med(f, n=9, warm=3):
    ts = []
    for i in range(n + warm):
        mx.synchronize(); t = time.perf_counter(); mx.eval(f()); mx.synchronize()
        if i >= warm: ts.append((time.perf_counter() - t) * 1e3)
    return statistics.median(ts)


cs = []
for i in range(L):
    c = KVCache()
    c.update_and_fetch(*(mx.random.normal((1, HK, CTX, D), key=mx.random.key(2 * i + j)).astype(mx.bfloat16) for j in (0, 1)))
    mx.eval(c.state); cs.append(c)
print(f"{'variant':>14} | " + " ".join(f"n={n:<5}" for n in (1, 4, 8, 16)) + "  (ms, 16 layers, partials only)")
for name, edits in VARIANTS.items():
    k = kernel(edits)
    row = []
    for n in (1, 4, 8, 16):
        q = mx.random.normal((1, HQ, n, D)).astype(mx.bfloat16)
        bits = mx.array(tf.ancestor_bits([[j <= i for j in range(n)] for i in range(n)]), dtype=mx.uint64)
        row.append(med(lambda: [run(k, q, c, bits, n) for c in cs]))
    print(f"{name:>14} | " + " ".join(f"{t:7.2f}" for t in row), flush=True)

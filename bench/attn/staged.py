"""tree_flash with each 32-key K and V tile staged once per threadgroup in threadgroup memory
(coalesced 16-byte loads, rows padded against bank conflicts), shared by all its simdgroups,
against the current kernel: time over 16 layers and max difference of the full attention.
Usage: python bench/attn/staged.py [CTX]"""
import statistics, sys, time
import mlx.core as mx
from mlx_lm.models.cache import KVCache
from mlx_tree_spec import tree_flash as tf

HQ, HK, D, L = 24, 4, 256, 16
CTX = int(sys.argv[1]) if len(sys.argv) > 1 else 32768

DECL = ("threadgroup float tg_o[NSGM * MROWS * (PHG * 32)];",
        "threadgroup float tg_o[KS > 1 ? NSGM * MROWS * (PHG * 32) : 1];\n"
        "    constexpr int DP = D + 8;  // padded row stride (elements)\n"
        "    threadgroup InT tg_kv[TK * DP];")


def stage(head):
    return ("      threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "      for (int idx = int(lid); idx < TK * (D / 8); idx += NTHREADS) {\n"
            "        const int r = idx / (D / 8), c8 = idx % (D / 8);\n"
            "        uint4 w = uint4(0u);\n"
            f"        if (t0 + r < kv_end) w = reinterpret_cast<const device uint4*>({head} + (size_t)(t0 + r) * D + c8 * 8)[0];\n"
            "        reinterpret_cast<threadgroup uint4*>(tg_kv + r * DP + c8 * 8)[0] = w;\n"
            "      }\n"
            "      threadgroup_barrier(mem_flags::mem_threadgroup);\n")


K_EDITS = [
    ("// ── QK: S[16,32] = Q[16,256] x Ktile[32,256]^T ──", stage("k_head") + "      // QK"),
    ("kk4 = reinterpret_cast<const device vec<InT, 4>*>( k_head + (size_t)kr * D + kk * 16 + sc.x)[0];",
     "kk4 = reinterpret_cast<const threadgroup vec<InT, 4>*>(tg_kv + (kr - t0) * DP + kk * 16 + sc.x)[0];"),
]
V_EDITS = [
    ("// ── PV: O[16, 32g..] += P[16,32] x V[32 keys, 32 dims] (NN, V straight from device) ──", stage("v_head") + "      // PV"),
    ("v4 = reinterpret_cast<const device vec<InT, 4>*>( v_head + (size_t)kr * D + g * 32 + hh * 16 + sc.x)[0];",
     "v4 = reinterpret_cast<const threadgroup vec<InT, 4>*>(tg_kv + (kr - t0) * DP + g * 32 + hh * 16 + sc.x)[0];"),
]


def kernel(edits):
    from mtplx.kernels.sdpa_nax_flash import _HEADER, _SOURCE
    src = tf._edit(tf._edit(_SOURCE, tf._OLD_VIS, tf._NEW_VIS), tf._OLD_TAIL, tf._NEW_TAIL)
    for old, new in edits:
        src = tf._edit(src, old, new)
    return mx.fast.metal_kernel(name="stg" + str(abs(hash(str(edits)))), header=_HEADER, source=src,
                                input_names=["queries", "keys", "values", "offset", "kcap", "scale", "blocks", "anc", "tail_n"],
                                output_names=["partials", "sums", "maxs"])


def attend(k, q, c, bits, n):
    from mtplx.kernels.sdpa_gqa_packed import _paged_reduce_kernel
    blocks = tf.flash_blocks(c.keys.shape[2])
    nthreads = 32 * ((6 * n + 15) // 16)
    p, s, m = k(inputs=[q, c.keys, c.values, mx.array([c.offset], dtype=mx.int32), c.keys.shape[2], D ** -0.5,
                        mx.array([blocks], dtype=mx.int32), bits, n],
                template=[("InT", mx.bfloat16), ("PartT", mx.float32), ("D", D), ("QL", n), ("GQA_F", 6), ("KS", 1), ("QTG", 0)],
                grid=(HK * nthreads, 1, blocks), threadgroup=(nthreads, 1, 1),
                output_shapes=[(1, HQ, n, blocks, D), (1, HQ, n, blocks), (1, HQ, n, blocks)],
                output_dtypes=[mx.float32, mx.float32, mx.float32])
    return _paged_reduce_kernel()(inputs=[p, s, m, blocks], template=[("InT", mx.bfloat16), ("V", D)],
                                  grid=(HQ * 1024, n, 1), threadgroup=(1024, 1, 1),
                                  output_shapes=[(1, HQ, n, D)], output_dtypes=[mx.bfloat16])[0]


def med(f, n=9, warm=3):
    ts = []
    for i in range(n + warm):
        mx.synchronize(); t = time.perf_counter(); mx.eval(f()); mx.synchronize()
        if i >= warm: ts.append((time.perf_counter() - t) * 1e3)
    return statistics.median(ts)


if __name__ == "__main__":
    cs = []
    for i in range(L):
        c = KVCache()
        c.update_and_fetch(*(mx.random.normal((1, HK, CTX - 100, D), key=mx.random.key(2 * i + j)).astype(mx.bfloat16) for j in (0, 1)))
        mx.eval(c.state); cs.append(c)  # CTX - 100: a partly filled last tile, like a real cache
    variants = {"current": [], "K staged": [DECL] + K_EDITS, "V staged": [DECL] + V_EDITS, "K+V staged": [DECL] + K_EDITS + V_EDITS}
    ks = {name: kernel(e) for name, e in variants.items()}
    print(f"ctx {CTX - 100}: ms over {L} layers (and max |diff| vs current)")
    for n in (1, 2, 4, 8, 12, 16):
        q = mx.random.normal((1, HQ, n, D)).astype(mx.bfloat16)
        bits = mx.array(tf.ancestor_bits([[j <= i for j in range(n)] for i in range(n)]), dtype=mx.uint64)
        ref = attend(ks["current"], q, cs[0], bits, n)
        cells = []
        for name, k in ks.items():
            t = med(lambda: [attend(k, q, c, bits, n) for c in cs])
            d = mx.abs(attend(k, q, cs[0], bits, n).astype(mx.float32) - ref.astype(mx.float32)).max().item()
            cells.append(f"{name} {t:6.2f} ({d:.0e})")
        print(f"n={n:2d}: " + " | ".join(cells), flush=True)

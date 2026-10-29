"""tree_flash with the PV accumulators kept in matmul destination tensors across key tiles (no
per-tile copy of all 256 output columns in and out of plain registers) and the P tile converted
once per tile instead of once per 32-column output group. Time over 16 layers and max difference
vs the current kernel. Usage: python bench/attn/persist.py [CTX]"""
import sys
import mlx.core as mx
from mlx_lm.models.cache import KVCache
from mlx_tree_spec import tree_flash as tf
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from staged import HQ, HK, D, L, attend, kernel, med  # noqa: E402

CTX = int(sys.argv[1]) if len(sys.argv) > 1 else 32768

G = range(256 // 32)  # NGROUPS for D = 256: one named accumulator each (cooperative tensors can't form arrays)
DST = "mm_pv.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(ct_pa)>, metal::remove_addrspace_t<decltype(ct_vb)>, float>()"
LEFT = "mm_pv.get_left_input_cooperative_tensor<InT, InT, float>()"
V_LOAD = """for (short hh = 0; hh < 2; hh++) {
            for (short i = 0; i < 2; i++) {
              const int kr = t0 + kk * 16 + sc.y + i * kElemRowsJump;
              vec<InT, 4> v4 = vec<InT, 4>(InT(0));
              if (tile_full || kr < kv_end) {
                v4 = reinterpret_cast<const device vec<InT, 4>*>(v_head + (size_t)kr * D + GG * 32 + hh * 16 + sc.x)[0];
              }
              for (short j = 0; j < kElemCols; j++) ct_vb[hh * kElemsPerFrag + i * kElemCols + j] = v4[j];
            }
          }"""
OLD_PV_START = """for (int g = 0; g < NGROUPS; g++) {
        for (short i = 0; i < kElemsPerFrag; i++) {
          ct_o[i] = o_frag[g][0][i];
          ct_o[kElemsPerFrag + i] = o_frag[g][1][i];
        }"""

PERSIST = [
    ("""float o_frag[NGROUPS][2][kElemsPerFrag];
    for (int g = 0; g < NGROUPS; g++)
      for (int h = 0; h < 2; h++)
        for (int i = 0; i < kElemsPerFrag; i++) o_frag[g][h][i] = 0.0f;""",
     "float o_frag[NGROUPS][2][kElemsPerFrag];\n"
     + "".join(f"    auto acc{g} = {DST};\n    for (short e = 0; e < 2 * kElemsPerFrag; e++) acc{g}[e] = 0.0f;\n" for g in G)
     + f"    auto pa0 = {LEFT};\n    auto pa1 = {LEFT};"),
    ("""for (short i = 0; i < 2; i++) {
        const float f = tile_f[i];
        for (int g = 0; g < NGROUPS; g++)
          for (short hh = 0; hh < 2; hh++)
            for (short j = 0; j < kElemCols; j++)
              o_frag[g][hh][i * kElemCols + j] *= f;
      }""",
     "for (short e = 0; e < 2 * kElemsPerFrag; e++) {\n        const float f = tile_f[(e % kElemsPerFrag) / kElemCols];\n"
     + "".join(f"        acc{g}[e] *= f;\n" for g in G) + "      }"),
]


def pv_block():
    out = ["for (short i = 0; i < 2; i++)\n        for (short j = 0; j < kElemCols; j++) {",
           "          pa0[i * kElemCols + j] = InT(s_p[0][i * kElemCols + j]);",
           "          pa1[i * kElemCols + j] = InT(s_p[1][i * kElemCols + j]);\n        }"]
    for g in G:
        for kk in (0, 1):
            out.append("      {\n        const int kk = %d;\n          " % kk + V_LOAD.replace("GG", str(g))
                       + f"\n        mm_pv.run(pa{kk}, ct_vb, acc{g});\n      }}")
    return "\n".join(out)


def kernel_persist():
    """The persistent-accumulator kernel: the PV loop of MTPLX's source replaced as a whole."""
    from mtplx.kernels.sdpa_nax_flash import _HEADER, _SOURCE
    src = tf._edit(tf._edit(_SOURCE, tf._OLD_VIS, tf._NEW_VIS), tf._OLD_TAIL, tf._NEW_TAIL)
    for old, new in PERSIST:
        src = tf._edit(src, old, new)
    a = src.index("// ── PV:")
    a = src.index("\n", a) + 1
    b = src.index("// ── merge the KS key-split")
    # the PV loop ends with the main key loop's closing brace: keep that brace
    body_end = src.rindex("}", a, b)
    src = src[:a] + "      " + pv_block() + "\n    " + src[body_end:b] \
        + "".join(f"    for (short i = 0; i < kElemsPerFrag; i++) {{ o_frag[{g}][0][i] = acc{g}[i]; o_frag[{g}][1][i] = acc{g}[kElemsPerFrag + i]; }}\n" for g in G) \
        + "    " + src[b:]
    return mx.fast.metal_kernel(name="persist", header=_HEADER, source=src,
                                input_names=["queries", "keys", "values", "offset", "kcap", "scale", "blocks", "anc", "tail_n"],
                                output_names=["partials", "sums", "maxs"]), src

if __name__ == "__main__":
    cs = []
    for i in range(L):
        c = KVCache()
        c.update_and_fetch(*(mx.random.normal((1, HK, CTX - 100, D), key=mx.random.key(2 * i + j)).astype(mx.bfloat16) for j in (0, 1)))
        mx.eval(c.state); cs.append(c)
    ks = {"current": kernel([]), "persistent acc": kernel_persist()[0]}
    print(f"ctx {CTX - 100}: ms over {L} layers (max |diff| vs current)")
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

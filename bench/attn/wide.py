"""A tree_flash variant with a different work split: each simdgroup takes 32 query rows (two
16-row M fragments) and half of the head dimension for P x V. A K fragment is loaded once for 32
rows instead of 16, and a V fragment once for 32 rows over half of D, so K/V traffic per row is
~2.7x lower. The cost is that the two D-halves of a row pair both compute its QK. Same inputs,
outputs and mask as tree_flash (bf16 only). Time over 16 layers and max difference vs the current
kernel. Usage: python bench/attn/wide.py [CTX]"""
import sys

import mlx.core as mx
from mlx_lm.models.cache import KVCache

from mlx_tree_spec import tree_flash as tf

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from staged import HQ, HK, D, L, attend, kernel, med  # noqa: E402

CTX = int(sys.argv[1]) if len(sys.argv) > 1 else 32768

# One block per (frag f, column group g of this simdgroup's D half): accumulators can't form arrays.
F, GH = (0, 1), range(4)  # 2 M fragments; D/2/32 = 4 column groups per half


def _src():
    acc_decl = "".join(f"    auto acc{f}{g} = mm_pv.get_destination_cooperative_tensor<"
                       "metal::remove_addrspace_t<decltype(pa00)>, metal::remove_addrspace_t<decltype(ct_vb)>, float>();\n"
                       f"    for (short e = 0; e < 2 * kElemsPerFrag; e++) acc{f}{g}[e] = 0.0f;\n" for f in F for g in GH)
    rescale = "".join(f"        acc{f}{g}[e] *= tile_f[{f}][(e % kElemsPerFrag) / kElemCols];\n" for f in F for g in GH)
    pv = []
    for g in GH:
        for kk in (0, 1):
            pv.append(f"""      {{
        const int col = dh * (D / 2) + {g} * 32;
        for (short hh = 0; hh < 2; hh++)
          for (short i = 0; i < 2; i++) {{
            const int kr = t0 + {kk} * 16 + sc.y + i * kElemRowsJump;
            vec<InT, 4> v4 = vec<InT, 4>(InT(0));
            if (tile_full || kr < kv_end) v4 = reinterpret_cast<const device vec<InT, 4>*>(v_head + (size_t)kr * D + col + hh * 16 + sc.x)[0];
            for (short j = 0; j < kElemCols; j++) ct_vb[hh * kElemsPerFrag + i * kElemCols + j] = v4[j];
          }}
        mm_pv.run(pa0{kk}, ct_vb, acc0{g});
        mm_pv.run(pa1{kk}, ct_vb, acc1{g});
      }}""")
    write = "".join(f"""        if (f == {f}) for (short hh = 0; hh < 2; hh++) for (short j = 0; j < kElemCols; j++) {{
          prow[dh * (D / 2) + {g} * 32 + hh * 16 + sc.x + j] = PartT(acc{f}{g}[hh * kElemsPerFrag + i * kElemCols + j]);
        }}
""" for f in F for g in GH)
    return f"""
    constexpr int TK = 32;
    constexpr int MROWS = 16;
    constexpr int LIVE = GQA_F * QL;
    constexpr int DFRAGS = D / 16;

    const uint lid = thread_position_in_threadgroup.x;
    const uint sg = lid >> 5;
    const ushort lane = ushort(lid & 31);
    const int mg = int(sg) >> 1;                 // 32-row group
    const int dh = int(sg) & 1;                  // which half of D this simdgroup outputs
    const int kv_head = int(threadgroup_position_in_grid.x);
    const int block_idx = int(threadgroup_position_in_grid.z);
    const int n_blocks = int(blocks[0]);
    const int n_kv = static_cast<int>(offset[0]);
    const int tail_lo = n_kv - tail_n;
    const int chunk = ((n_kv + n_blocks - 1) / n_blocks + TK - 1) / TK * TK;
    const int kv_start = block_idx * chunk;
    const int kv_end = metal::min(kv_start + chunk, n_kv);
    const short2 sc = nax_get_coord(lane);
    const int m_base = mg * 2 * MROWS;

    const device InT* k_head = keys + (size_t)kv_head * kcap * D;
    const device InT* v_head = values + (size_t)kv_head * kcap * D;
    const device InT* q_head = queries + (size_t)kv_head * LIVE * D;

    constexpr auto desc_nt = mpp::tensor_ops::matmul2d_descriptor(16, 32, 16, false, true, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    constexpr auto desc_nn = mpp::tensor_ops::matmul2d_descriptor(16, 32, 16, false, false, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc_nt, metal::execution_simdgroup> mm_qk;
    mpp::tensor_ops::matmul2d<desc_nn, metal::execution_simdgroup> mm_pv;
    auto qa0 = mm_qk.get_left_input_cooperative_tensor<InT, InT, float>();
    auto qa1 = mm_qk.get_left_input_cooperative_tensor<InT, InT, float>();
    auto ct_kb = mm_qk.get_right_input_cooperative_tensor<InT, InT, float>();
    auto s0 = mm_qk.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(qa0)>, metal::remove_addrspace_t<decltype(ct_kb)>, float>();
    auto s1 = mm_qk.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(qa0)>, metal::remove_addrspace_t<decltype(ct_kb)>, float>();
    auto pa00 = mm_pv.get_left_input_cooperative_tensor<InT, InT, float>();
    auto pa01 = mm_pv.get_left_input_cooperative_tensor<InT, InT, float>();
    auto pa10 = mm_pv.get_left_input_cooperative_tensor<InT, InT, float>();
    auto pa11 = mm_pv.get_left_input_cooperative_tensor<InT, InT, float>();
    auto ct_vb = mm_pv.get_right_input_cooperative_tensor<InT, InT, float>();
{acc_decl}
    float row_m[2][2] = {{{{-1e38f, -1e38f}}, {{-1e38f, -1e38f}}}};
    float row_l[2][2] = {{{{0.0f, 0.0f}}, {{0.0f, 0.0f}}}};

    for (int t0 = kv_start; t0 < kv_end; t0 += TK) {{
      const bool tile_full = (t0 + TK) <= kv_end;
      for (short i = 0; i < 2 * kElemsPerFrag; i++) {{ s0[i] = 0.0f; s1[i] = 0.0f; }}
      for (int kk = 0; kk < DFRAGS; kk++) {{
        for (short i = 0; i < 2; i++) {{
          const int r0 = m_base + sc.y + i * kElemRowsJump, r1 = r0 + MROWS;
          vec<InT, 4> q0 = vec<InT, 4>(InT(0)), q1 = vec<InT, 4>(InT(0));
          if (r0 < LIVE) q0 = reinterpret_cast<const device vec<InT, 4>*>(q_head + (size_t)r0 * D + kk * 16 + sc.x)[0];
          if (r1 < LIVE) q1 = reinterpret_cast<const device vec<InT, 4>*>(q_head + (size_t)r1 * D + kk * 16 + sc.x)[0];
          for (short j = 0; j < kElemCols; j++) {{ qa0[i * kElemCols + j] = q0[j]; qa1[i * kElemCols + j] = q1[j]; }}
        }}
        for (short hh = 0; hh < 2; hh++)
          for (short i = 0; i < 2; i++) {{
            const int kr = t0 + hh * 16 + sc.y + i * kElemRowsJump;
            vec<InT, 4> k4 = vec<InT, 4>(InT(0));
            if (tile_full || kr < kv_end) k4 = reinterpret_cast<const device vec<InT, 4>*>(k_head + (size_t)kr * D + kk * 16 + sc.x)[0];
            for (short j = 0; j < kElemCols; j++) ct_kb[hh * kElemsPerFrag + i * kElemCols + j] = k4[j];
          }}
        mm_qk.run(qa0, ct_kb, s0);
        mm_qk.run(qa1, ct_kb, s1);
      }}

      float tile_f[2][2];
      float sp[2][2][kElemsPerFrag];
      for (short f = 0; f < 2; f++) {{
        float tmax[2] = {{-1e38f, -1e38f}};
        for (short i = 0; i < 2; i++) {{
          const int m_local = m_base + f * MROWS + sc.y + i * kElemRowsJump;
          const bool live = m_local < LIVE;
          for (short hh = 0; hh < 2; hh++)
            for (short j = 0; j < kElemCols; j++) {{
              const int gp = t0 + hh * 16 + sc.x + j;
              const float raw = f == 0 ? s0[hh * kElemsPerFrag + i * kElemCols + j] : s1[hh * kElemsPerFrag + i * kElemCols + j];
              const int tp = gp - tail_lo;
              const bool vis = live && gp < kv_end && (tp < 0 || ((anc[m_local % QL] >> ulong(tp)) & 1ul) != 0ul);
              const float s = vis ? raw * scale : -1e38f;
              sp[f][hh][i * kElemCols + j] = s;
              tmax[i] = metal::max(tmax[i], s);
            }}
        }}
        for (short i = 0; i < 2; i++) {{
          tmax[i] = metal::max(tmax[i], simd_shuffle_xor(tmax[i], ushort(1)));
          tmax[i] = metal::max(tmax[i], simd_shuffle_xor(tmax[i], ushort(8)));
          const float new_m = metal::max(row_m[f][i], tmax[i]);
          tile_f[f][i] = metal::exp(row_m[f][i] - new_m);
          float tsum = 0.0f;
          for (short hh = 0; hh < 2; hh++)
            for (short j = 0; j < kElemCols; j++) {{
              const float s = sp[f][hh][i * kElemCols + j];
              const float p = (s > -1e37f) ? metal::exp(s - new_m) : 0.0f;
              sp[f][hh][i * kElemCols + j] = p;
              tsum += p;
            }}
          tsum += simd_shuffle_xor(tsum, ushort(1));
          tsum += simd_shuffle_xor(tsum, ushort(8));
          row_m[f][i] = new_m;
          row_l[f][i] = row_l[f][i] * tile_f[f][i] + tsum;
        }}
      }}
      for (short e = 0; e < 2 * kElemsPerFrag; e++) {{
{rescale}      }}
      for (short i = 0; i < 2; i++)
        for (short j = 0; j < kElemCols; j++) {{
          pa00[i * kElemCols + j] = InT(sp[0][0][i * kElemCols + j]);
          pa01[i * kElemCols + j] = InT(sp[0][1][i * kElemCols + j]);
          pa10[i * kElemCols + j] = InT(sp[1][0][i * kElemCols + j]);
          pa11[i * kElemCols + j] = InT(sp[1][1][i * kElemCols + j]);
        }}
{chr(10).join(pv)}
    }}

    for (short f = 0; f < 2; f++)
      for (short i = 0; i < 2; i++) {{
        const int m_local = m_base + f * MROWS + sc.y + i * kElemRowsJump;
        if (m_local >= LIVE) continue;
        const int hq_row = kv_head * LIVE + m_local;
        device PartT* prow = partials + ((size_t)hq_row * n_blocks + block_idx) * D;
{write}        if (dh == 0 && (lane & 0x9) == 0) {{
          sums[hq_row * n_blocks + block_idx] = row_l[f][i];
          maxs[hq_row * n_blocks + block_idx] = (row_l[f][i] > 0.0f) ? row_m[f][i] : -1e38f;
        }}
      }}
"""


def kernel_wide():
    from mtplx.kernels.sdpa_nax_flash import _HEADER
    return mx.fast.metal_kernel(name="tree_flash_wide", header=_HEADER, source=_src(),
                                input_names=["queries", "keys", "values", "offset", "kcap", "scale", "blocks", "anc", "tail_n"],
                                output_names=["partials", "sums", "maxs"])


def attend_wide(k, q, c, bits, n, blocks=None):
    from mtplx.kernels.sdpa_gqa_packed import _paged_reduce_kernel
    blocks = blocks or tf.flash_blocks(c.keys.shape[2])
    nthreads = 64 * ((6 * n + 31) // 32)
    p, s, m = k(inputs=[q, c.keys, c.values, mx.array([c.offset], dtype=mx.int32), c.keys.shape[2], D ** -0.5,
                        mx.array([blocks], dtype=mx.int32), bits, n],
                template=[("InT", mx.bfloat16), ("PartT", mx.float32), ("D", D), ("QL", n), ("GQA_F", 6)],
                grid=(HK * nthreads, 1, blocks), threadgroup=(nthreads, 1, 1),
                output_shapes=[(1, HQ, n, blocks, D), (1, HQ, n, blocks), (1, HQ, n, blocks)],
                output_dtypes=[mx.float32, mx.float32, mx.float32])
    return _paged_reduce_kernel()(inputs=[p, s, m, blocks], template=[("InT", mx.bfloat16), ("V", D)],
                                  grid=(HQ * 1024, n, 1), threadgroup=(1024, 1, 1),
                                  output_shapes=[(1, HQ, n, D)], output_dtypes=[mx.bfloat16])[0]


if __name__ == "__main__":
    cs = []
    for i in range(L):
        c = KVCache()
        c.update_and_fetch(*(mx.random.normal((1, HK, CTX - 100, D), key=mx.random.key(2 * i + j)).astype(mx.bfloat16) for j in (0, 1)))
        mx.eval(c.state); cs.append(c)
    cur, wide = kernel([]), kernel_wide()
    print(f"ctx {CTX - 100}: ms over {L} layers (max |diff| vs current); tree = a random 3-branch tree of n nodes")
    import random
    for n in (2, 4, 6, 8, 12, 16):
        q = mx.random.normal((1, HQ, n, D)).astype(mx.bfloat16)
        for shape in ("chain", "tree"):
            parents = [-1] + [i - 1 if shape == "chain" else random.Random(n).randrange(max(0, i - 3), i) for i in range(1, n)]
            anc = [[False] * n for _ in range(n)]
            for i in range(n):
                j = i
                while j >= 0:
                    anc[i][j] = True
                    j = parents[j]
            bits = mx.array(tf.ancestor_bits(anc), dtype=mx.uint64)
            ref = attend(cur, q, cs[0], bits, n)
            d = mx.abs(attend_wide(wide, q, cs[0], bits, n).astype(mx.float32) - ref.astype(mx.float32)).max().item()
            t0 = med(lambda: [attend(cur, q, c, bits, n) for c in cs])
            t1 = med(lambda: [attend_wide(wide, q, c, bits, n) for c in cs])
            print(f"n={n:2d} {shape:5s}: current {t0:6.2f}  wide {t1:6.2f}  ({t0 / t1:.2f}x, max diff {d:.1e})", flush=True)

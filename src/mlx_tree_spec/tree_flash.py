"""Tree-mask flash-decoding attention: MTPLX's sdpa_nax_flash with an ancestor mask.

MTPLX's TensorOps flash kernel (mtplx/kernels/sdpa_nax_flash.py) reads each KV
block once for every query of a head group, but its mask is tail-causal: query
q sees the committed context plus tail positions 0..q, i.e. a chain. A tree pass
appends n nodes in topological order and node q must see the context plus its
own ancestors. Here the kernel source is MTPLX's with two lines changed: the
tail is the last `tail` cache rows (at least the queries; branch drafting also
keeps earlier draft entries there), and tail visibility comes from a per-query
bitmask (bit j = tail row j is an ancestor or the row itself), passed as a
uint64 per query. Everything else, including the split-KV partials and MTPLX's
reduce kernel, is reused as is.

For kv8.Q8KVCache (int8 rows with one fp32 scale each) a second variant loads
the int8 K/V values (exact in fp16, in which its matmuls run) and applies the
scales outside the matmuls: each key's scale multiplies its scores, each
value's scale the softmax weight it is multiplied by. The running sums use the
unscaled weights, so the result equals attention over the dequantized cache up
to fp16 rounding of Q and P.

MTPLX's wrapper caps the window at 10 queries (64 head-group rows); the kernel's
threadgroup buffers fit 16 queries x 6 heads (~24 KB), so this allows 16.
"""

from __future__ import annotations

from collections import Counter
import re
from functools import lru_cache

import mlx.core as mx

STATS: Counter = Counter()
MAX_QUERIES = 16
MAX_TAIL = 64  # bits per mask

_OLD_TAIL = "const int tail_lo = n_kv - QL;"
_NEW_TAIL = "const int tail_lo = n_kv - tail_n;"

_OLD_VIS = "const bool vis = live && gp < kv_end && gp <= row_limit;"
_NEW_VIS = (
    "const int tp = gp - tail_lo;\n"
    "              const bool vis = live && gp < kv_end\n"
    "                  && (tp < 0 || ((anc[m_local % QL] >> ulong(tp)) & 1ul) != 0ul);"
)


# The int8 variant's matmuls run in fp16: int8 rows convert to fp16 with two
# integer ops and one subtraction per pair (1024 + u has u in its low mantissa
# bits), where bf16 (7 mantissa bits) needs a float conversion per value.
_Q8_HEADER = """
inline half4 q8_to_half4(uint w) {
  w ^= 0x80808080u;                                                   // int8 v -> v + 128
  const half2 a = as_type<half2>((w & 0x00FF00FFu) | 0x64006400u) - half2(1152.0h);         // bytes 0, 2
  const half2 b = as_type<half2>(((w >> 8) & 0x00FF00FFu) | 0x64006400u) - half2(1152.0h);  // bytes 1, 3
  return half4(a.x, b.x, a.y, b.y);
}
"""

# (old, new) edits for the int8 variant; old is matched modulo whitespace.
_Q8_EDITS = (
    ("const device InT* k_head = keys + (size_t)kv_head * kcap * D;",
     "const device int8_t* k_head = keys + (size_t)kv_head * kcap * D;\n"
     "    const device float* ksc_head = k_scale + (size_t)kv_head * kcap;"),
    ("const device InT* v_head = values + (size_t)kv_head * kcap * D;",
     "const device int8_t* v_head = values + (size_t)kv_head * kcap * D;\n"
     "    const device float* vsc_head = v_scale + (size_t)kv_head * kcap;"),
    ("auto ct_qa = mm_qk.get_left_input_cooperative_tensor<InT, InT, float>();",
     "auto ct_qa = mm_qk.get_left_input_cooperative_tensor<half, half, float>();"),
    ("auto ct_kb = mm_qk.get_right_input_cooperative_tensor<InT, InT, float>();",
     "auto ct_kb = mm_qk.get_right_input_cooperative_tensor<half, half, float>();"),
    ("auto ct_pa = mm_pv.get_left_input_cooperative_tensor<InT, InT, float>();",
     "auto ct_pa = mm_pv.get_left_input_cooperative_tensor<half, half, float>();"),
    ("auto ct_vb = mm_pv.get_right_input_cooperative_tensor<InT, InT, float>();",
     "auto ct_vb = mm_pv.get_right_input_cooperative_tensor<half, half, float>();"),
    ("for (short j = 0; j < kElemCols; j++) ct_qa[i * kElemCols + j] = qq[j];",
     "for (short j = 0; j < kElemCols; j++) ct_qa[i * kElemCols + j] = half(float(qq[j]));"),
    ("vec<InT, 4> kk4 = vec<InT, 4>(InT(0)); if (tile_full || kr < kv_end) { "
     "kk4 = reinterpret_cast<const device vec<InT, 4>*>( k_head + (size_t)kr * D + kk * 16 + sc.x)[0]; }",
     "half4 kk4 = half4(0.0h);\n"
     "            if (tile_full || kr < kv_end) {\n"
     "              kk4 = q8_to_half4(reinterpret_cast<const device uint*>(k_head + (size_t)kr * D + kk * 16 + sc.x)[0]);\n"
     "            }"),
    # The 8 keys a lane's score columns cover, scales loaded once per tile as
    # float4s (capacities are multiples of 32; rows past kv_end are masked).
    ("// ── in-register masked online softmax ──",
     "float ksc[2][kElemCols], vsc[2][kElemCols];\n"
     "      for (short hh = 0; hh < 2; hh++) {\n"
     "        const float4 k4s = reinterpret_cast<const device float4*>(ksc_head + t0 + hh * 16 + sc.x)[0];\n"
     "        const float4 v4s = reinterpret_cast<const device float4*>(vsc_head + t0 + hh * 16 + sc.x)[0];\n"
     "        for (short j = 0; j < kElemCols; j++) {\n"
     "          ksc[hh][j] = k4s[j] * scale;\n"
     "          vsc[hh][j] = v4s[j];\n"
     "        }\n"
     "      }\n"
     "      // ── in-register masked online softmax ──"),
    ("const float s = vis ? raw * scale : -1e38f;",
     "const float s = vis ? raw * ksc[hh][j] : -1e38f;"),
    ("ct_pa[i * kElemCols + j] = InT(s_p[kk][i * kElemCols + j]);",
     "ct_pa[i * kElemCols + j] = half(s_p[kk][i * kElemCols + j] * vsc[kk][j]);"),
    ("vec<InT, 4> v4 = vec<InT, 4>(InT(0)); if (tile_full || kr < kv_end) { "
     "v4 = reinterpret_cast<const device vec<InT, 4>*>( v_head + (size_t)kr * D + g * 32 + hh * 16 + sc.x)[0]; }",
     "half4 v4 = half4(0.0h);\n"
     "              if (tile_full || kr < kv_end) {\n"
     "                v4 = q8_to_half4(reinterpret_cast<const device uint*>(v_head + (size_t)kr * D + g * 32 + hh * 16 + sc.x)[0]);\n"
     "              }"),
)


def _edit(src: str, old: str, new: str) -> str:
    pat = r"\s*".join(re.escape(t) for t in old.split())
    out, n = re.subn(pat, lambda _: new, src)
    assert n == 1, f"MTPLX's flash kernel changed; update tree_flash ({old[:40]!r} matched {n}x)"
    return out


@lru_cache(maxsize=None)
def _kernel(q8: bool = False):
    from mtplx.kernels.sdpa_nax_flash import _HEADER, _SOURCE

    src = _edit(_edit(_SOURCE, _OLD_VIS, _NEW_VIS), _OLD_TAIL, _NEW_TAIL)
    names = ["queries", "keys", "values", "offset", "kcap", "scale", "blocks", "anc", "tail_n"]
    if q8:
        for old, new in _Q8_EDITS:
            src = _edit(src, old, new)
        names += ["k_scale", "v_scale"]
    return mx.fast.metal_kernel(
        name="tree_nax_flash_partials" + ("_q8" if q8 else ""),
        input_names=names,
        output_names=["partials", "sums", "maxs"],
        header=_HEADER + (_Q8_HEADER if q8 else ""),
        source=src,
    )


def flash_blocks(capacity: int) -> int:
    """Key blocks (threadgroups per KV head) for the flash kernels. MTPLX tuned 256
    for 16-64K at 4 queries; up to 32K, 64 is faster for every query count and
    kernel (16 layers at 16K: 8 queries 6.9 -> 5.7 ms, a 16-node tree 12.2 -> 9.5;
    at 32K 3-9% less). Past 48K MTPLX's own choice."""
    from mtplx.kernels.sdpa_nax_flash import _default_blocks

    return 64 if capacity < 49152 else _default_blocks(capacity)


def ancestor_bits(anc) -> list[int]:
    """Per node, the bitmask of tail nodes it may attend to (ancestors and itself)."""
    return [sum(1 << j for j in range(len(row)) if row[j]) for row in anc]


def tree_flash(queries, cache, scale: float, bits, tail: int | None = None, stats: bool = False):
    """Attention of the n tree-node `queries` (1, Hq, n, D) over `cache`, whose last
    `tail` rows (default n) end with those nodes, with node q seeing the context
    before the tail plus the tail rows in bits[q] (an mx uint64 array of n masks;
    tail 0: every cached row). None when the contract is not met. With `stats`,
    also each block's max score and exp-sum (1, Hq, n, blocks), from which the
    softmax normaliser of the result can be rebuilt (to merge more keys in)."""
    from mtplx.kernels.sdpa_2pass import unnormalized_partials_dtype
    from mtplx.kernels.sdpa_gqa_packed import _paged_reduce_kernel
    from mtplx.nax_verify import nax_available

    keys, values = getattr(cache, "keys", None), getattr(cache, "values", None)
    bsz, hq, n, d = queries.shape
    tail = n if tail is None else tail
    if keys is None or bsz != 1 or d != 256 or not 1 <= n <= MAX_QUERIES or not (tail == 0 or n <= tail <= MAX_TAIL) or not nax_available():
        STATS["declined"] += 1
        return None
    hk, capacity = keys.shape[1], keys.shape[2]
    q8 = getattr(cache, "k_scale", None) is not None
    if hq % hk or queries.dtype not in (mx.bfloat16, mx.float16) or keys.dtype != (mx.int8 if q8 else queries.dtype):
        STATS["declined"] += 1
        return None
    gqa = hq // hk
    blocks = flash_blocks(capacity)
    nthreads = 32 * ((gqa * n + 15) // 16)
    part_dt = unnormalized_partials_dtype(queries.dtype)
    partials, sums, maxs = _kernel(q8)(
        inputs=[queries, keys, values, mx.array([cache.offset], dtype=mx.int32), capacity,
                float(scale), mx.array([blocks], dtype=mx.int32), bits, tail]
        + ([cache.k_scale, cache.v_scale] if q8 else []),
        template=[("InT", queries.dtype), ("PartT", part_dt), ("D", d), ("QL", n), ("GQA_F", gqa), ("KS", 1), ("QTG", 0)],
        grid=(hk * nthreads, 1, blocks),
        threadgroup=(nthreads, 1, 1),
        output_shapes=[(1, hq, n, blocks, d), (1, hq, n, blocks), (1, hq, n, blocks)],
        output_dtypes=[part_dt, mx.float32, mx.float32],
    )
    (out,) = _paged_reduce_kernel()(
        inputs=[partials, sums, maxs, blocks],
        template=[("InT", queries.dtype), ("V", d)],
        grid=(hq * 1024, n, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(1, hq, n, d)],
        output_dtypes=[queries.dtype],
    )
    STATS["tree_flash_q8" if q8 else "tree_flash"] += 1
    return (out, maxs, sums) if stats else out

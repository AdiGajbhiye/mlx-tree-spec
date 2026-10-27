"""Multi-row (M <= 8) 4-bit matmul on a repacked ("chip-layout") weight format.

Lithos-metal repacks weights at load time into the order its kernels consume
them. Here the target is MLX's 8x8 simdgroup-matrix fragment layout: in a
fragment each lane owns 2 elements at (fm, fn), (fm, fn + 1). For a strip of 8
output rows and a 32-wide K block, the 4 fragments (k sub-blocks t = 0..3) need
from lane l exactly 8 nibbles = one uint32. pack() rewrites MLX's affine 4-bit
weight into wp[strip][kblock][lane] (uint32), so a simdgroup reads 128 contiguous
bytes per strip and block, no lane duplicates another's load, and successive K
blocks of a strip are contiguous. Scales/biases become sp[strip][group][row].

The kernel: a threadgroup of NSG simdgroups shares each 256-wide chunk of the
M activation rows in threadgroup memory (one coalesced load), each simdgroup owns
R strips; per block it multiplies the raw 4-bit codes (exact in half) with the
activations on 8x8 half fragments, and applies scale/bias per block in float:
  sum_k (s q + b) x = s * sum_k q x + b * sum_k x.
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx

_HEADER = """
#include <metal_simdgroup_matrix>
using namespace metal;
"""

_SOURCE = """
constexpr int CK = 256;              // K chunk staged in threadgroup memory
constexpr int BPC = CK / 32;         // 32-wide blocks per chunk
threadgroup half Xt[8 * CK];         // activations, row-major [m][k], m padded to 8

const uint lane = thread_index_in_simdgroup;
const uint sgi = simdgroup_index_in_threadgroup;
const uint tid = sgi * 32 + lane;
const int K = int(K_size);
const int N = int(N_size);
const int M = int(M_size);
const int KB = K / 32;
const int G = K / GS;
const int strip0 = (int(threadgroup_position_in_grid.y) * NSG + int(sgi)) * R;
const int S = (N + 7) / 8;

const int qid = int(lane) / 4;
const int fm = (qid & 4) + ((int(lane) / 2) % 4);
const int fn = (qid & 2) * 2 + (int(lane) % 2) * 2;

simdgroup_half8x8 ones;
reinterpret_cast<thread half2&>(ones.thread_elements()) = half2(1.0h, 1.0h);

float2 acc[R];
for (int s = 0; s < R; ++s) acc[s] = float2(0.0f);

for (int c0 = 0; c0 < K; c0 += CK) {
  for (int i = int(tid); i < 8 * CK; i += NSG * 32) {
    int m = i / CK, kk = i % CK;
    Xt[i] = m < M ? half(x[m * K + c0 + kk]) : half(0.0h);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int bb = 0; bb < BPC; ++bb) {
    const int kb = c0 / 32 + bb;
    simdgroup_half8x8 B[4];
    simdgroup_half8x8 Xs = simdgroup_half8x8(0);
    for (int t = 0; t < 4; ++t) {
      int kk = bb * 32 + t * 8 + fm;
      reinterpret_cast<thread half2&>(B[t].thread_elements()) = half2(Xt[fn * CK + kk], Xt[(fn + 1) * CK + kk]);
      simdgroup_multiply_accumulate(Xs, ones, B[t], Xs);
    }
    const float2 xsum = float2(reinterpret_cast<thread half2&>(Xs.thread_elements()));
    const int g = (kb * 32) / GS;
    for (int s = 0; s < R; ++s) {
      const int strip = strip0 + s;
      if (strip >= S) break;
      uint q = wp[(strip * KB + kb) * 32 + int(lane)];
      simdgroup_half8x8 C = simdgroup_half8x8(0);
      for (int t = 0; t < 4; ++t) {
        uint byte = (q >> (8 * t)) & 0xFFu;
        simdgroup_half8x8 A;
        reinterpret_cast<thread half2&>(A.thread_elements()) = half2(half(byte & 0xFu), half(byte >> 4));
        simdgroup_multiply_accumulate(C, A, B[t], C);
      }
      float sc = float(sp[(strip * G + g) * 8 + fm]);
      float bi = float(bp[(strip * G + g) * 8 + fm]);
      acc[s] += sc * float2(reinterpret_cast<thread half2&>(C.thread_elements())) + bi * xsum;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

for (int s = 0; s < R; ++s) {
  const int row = (strip0 + s) * 8 + fm;
  if (row >= N) continue;
  if (fn < M) y[fn * N + row] = T(acc[s].x);
  if (fn + 1 < M) y[(fn + 1) * N + row] = T(acc[s].y);
}
"""


def _lane_coords():
    lanes = mx.arange(32)
    qid = lanes // 4
    fm = (qid & 4) + ((lanes // 2) % 4)
    fn = (qid & 2) * 2 + (lanes % 2) * 2
    return fm, fn


def pack(weight: mx.array, scales: mx.array, biases: mx.array):
    """MLX affine 4-bit (N, K/8 uint32) -> (wp, sp, bp) in the fragment layout."""
    n, kw = weight.shape
    k = kw * 8
    assert n % 8 == 0 and k % 32 == 0
    fm, fn = _lane_coords()
    # bytes of row r, word c: byte j holds the nibbles for k = 8c + 2j, 8c + 2j + 1
    w8 = weight.view(mx.uint8).reshape(n // 8, 8, k // 32, 4, 4)       # [S, r, b, t, j]
    w8 = w8.transpose(0, 2, 3, 1, 4).reshape(n // 8, k // 32, 4, 32)      # [S, b, t, r*4+j]
    w8 = mx.take(w8, fm * 4 + fn // 2, axis=3)                            # [S, b, t, lane]
    wp = mx.contiguous(w8.transpose(0, 1, 3, 2)).view(mx.uint32).reshape(-1)  # [S, b, lane] words
    g = scales.shape[1]
    sp = mx.contiguous(scales.reshape(n // 8, 8, g).transpose(0, 2, 1)).reshape(-1)
    bp = mx.contiguous(biases.reshape(n // 8, 8, g).transpose(0, 2, 1)).reshape(-1)
    return wp, sp, bp


@lru_cache(maxsize=None)
def _kernel():
    return mx.fast.metal_kernel(
        name="tree_spec_qmm4_packed",
        input_names=["x", "wp", "sp", "bp", "K_size", "N_size", "M_size"],
        output_names=["y"],
        source=_SOURCE,
        header=_HEADER,
    )


# ----------------------------------------------------------------- NAX variant
# MTPLX's NAX verify kernel (mtplx/nax_verify.py, ported from dflash-mlx,
# Apache-2.0): each lane dequantizes one output column's 16 K-values into a
# threadgroup tile and a Metal 4 matmul2d runs on the M5 matrix accelerators;
# the 8 simdgroups split K and their partials are summed. Two changes here:
#  - packed weights: lane l's two words for (32-row tile, 16-wide K block) are
#    adjacent, so a simdgroup reads 256 contiguous bytes per step instead of 32
#    scattered 8-byte loads (one per weight row);
#  - BM (rows of the matmul2d tile) is a parameter, to try 8 instead of 16.

_MSL_TYPE = {mx.bfloat16: "bfloat16_t", mx.float16: "half", mx.float32: "float"}

_NAX_HEADER = "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"

_NAX_SOURCE = """
using namespace metal;
using namespace mpp::tensor_ops;

constexpr int BN = 32;
constexpr int BK = 16;
constexpr int NSG = 8;
constexpr int K = KCONST;
constexpr int KB16 = K / 16;
constexpr int G = K / GS;
constexpr int K_chunk = K / NSG;

uint tid = thread_position_in_threadgroup.x;
uint sg_id = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
int tile = int(threadgroup_position_in_grid.y);
int N = int(N_size);
int n0 = tile * BN;
int k_begin = int(sg_id) * K_chunk;
int k_end = k_begin + K_chunk;

threadgroup T B_tile[NSG][BK * BN];
threadgroup float partial[NSG][BM * BN];

constexpr auto desc = matmul2d_descriptor(BM, BN, BK, false, false, false,
                                          matmul2d_descriptor::mode::multiply_accumulate);
matmul2d<desc, metal::execution_simdgroup> op;

tensor<device T, dextents<int, 2>, tensor_inline> A((device T*)x, dextents<int, 2>(K, BM), array<int, 2>({1, K}));
tensor<threadgroup T, dextents<int, 2>, tensor_inline> B(B_tile[sg_id], dextents<int, 2>(BN, BK), array<int, 2>({1, BN}));
tensor<threadgroup float, dextents<int, 2>, tensor_inline> C(partial[sg_id], dextents<int, 2>(BN, BM), array<int, 2>({1, BN}));

auto ct_c = op.template get_destination_cooperative_tensor<
    tensor<device T, extents<int, BK, BM>, tensor_inline>,
    tensor<threadgroup T, extents<int, BN, BK>, tensor_inline>, float>();
_Pragma("unroll")
for (uint16_t i = 0; i < ct_c.get_capacity(); ++i) ct_c[i] = 0.0f;

const device uint2* wl = (const device uint2*)wq2 + (size_t(tile) * KB16) * 32 + lane;
const device T* sl = sp2 + size_t(tile) * G * 32 + lane;
const device T* bl = bp2 + size_t(tile) * G * 32 + lane;
for (int k0 = k_begin; k0 < k_end; k0 += BK) {
  uint2 p = wl[(k0 / 16) * 32];
  float s0 = float(sl[(k0 / GS) * 32]);
  float b0 = float(bl[(k0 / GS) * 32]);
  float s1 = float(sl[((k0 + 8) / GS) * 32]);
  float b1 = float(bl[((k0 + 8) / GS) * 32]);
  _Pragma("unroll")
  for (int ki = 0; ki < 8; ++ki) {
    B_tile[sg_id][ki * BN + int(lane)] = T(float((p.x >> (ki * 4)) & 0xFu) * s0 + b0);
    B_tile[sg_id][(8 + ki) * BN + int(lane)] = T(float((p.y >> (ki * 4)) & 0xFu) * s1 + b1);
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);
  auto tA = A.template slice<BK, BM>(k0, 0);
  auto tB = B.template slice<BN, BK>(0, 0);
  op.run(tA, tB, ct_c);
  simdgroup_barrier(mem_flags::mem_threadgroup);
}

auto tC = C.template slice<BN, BM>(0, 0);
ct_c.store(tC);
threadgroup_barrier(mem_flags::mem_threadgroup);
for (int off = int(tid); off < BM * BN; off += NSG * 32) {
  float acc = 0.0f;
  for (int s = 0; s < NSG; ++s) acc += partial[s][off];
  int row = off / BN;
  int col = off - row * BN;
  if (row < M_size) y[row * N + n0 + col] = T(acc);
}
"""


def pack_nax(weight: mx.array, scales: mx.array, biases: mx.array):
    """MLX affine 4-bit -> (wq2, sp2, bp2): per 32-row tile and 16-wide K block,
    the 32 lanes' word pairs are adjacent; scales/biases per tile and group."""
    n, kw = weight.shape
    assert n % 32 == 0 and kw % 2 == 0
    wq2 = mx.contiguous(weight.reshape(n // 32, 32, kw // 2, 2).transpose(0, 2, 1, 3)).reshape(-1)
    g = scales.shape[1]
    sp2 = mx.contiguous(scales.reshape(n // 32, 32, g).transpose(0, 2, 1)).reshape(-1)
    bp2 = mx.contiguous(biases.reshape(n // 32, 32, g).transpose(0, 2, 1)).reshape(-1)
    return wq2, sp2, bp2


@lru_cache(maxsize=None)
def _nax_kernel(k: int, group_size: int, bm: int):
    return mx.fast.metal_kernel(
        name=f"tree_spec_nax_packed_k{k}_g{group_size}_bm{bm}",
        input_names=["x", "wq2", "sp2", "bp2", "N_size", "M_size"],
        output_names=["y"],
        source=_NAX_SOURCE,
        header=_NAX_HEADER,
    )


def nax_packed(x2: mx.array, wq2, sp2, bp2, n: int, group_size: int, *, bm: int = 8) -> mx.array:
    """y = x2 @ W.T for x2 (M <= bm, K), K % 128 == 0, N % 32 == 0, W packed by pack_nax()."""
    m, k = x2.shape
    if m < bm:
        x2 = mx.concatenate([x2, mx.zeros((bm - m, k), dtype=x2.dtype)], axis=0)
    (y,) = _nax_kernel(k, group_size, bm)(
        inputs=[mx.contiguous(x2), wq2, sp2, bp2, n, m],
        template=[("T", x2.dtype), ("KCONST", k), ("GS", group_size), ("BM", bm)],
        grid=(256, n // 32, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(m, n)],
        output_dtypes=[x2.dtype],
    )
    return y


_NAX_SPECIAL = (
    _NAX_SOURCE.replace("int N = int(N_size);", "int N = NCONST;")
    .replace("constexpr int NSG = 8;\n", "")
    .replace("dextents<int, 2>(K, BM)", "dextents<int, 2>(K, MR)")   # rows >= MR read as zero
    .replace("if (row < M_size)", "if (row < MR)")
)


@lru_cache(maxsize=None)
def nax_kernel(k: int, n: int, group_size: int, m: int, bm: int, dtype, nsg: int = 8):
    """nax_packed for exactly m rows with every parameter compiled in (no
    template or scalar inputs, ~1 us less CPU per call) and no padding: the A
    tensor is declared m rows tall and the tensor op reads the rest as zero."""
    consts = (
        f"typedef {_MSL_TYPE[dtype]} T;\n"
        + "".join(f"constexpr int {a} = {v};\n" for a, v in
                  (("KCONST", k), ("NCONST", n), ("GS", group_size), ("BM", bm), ("MR", m), ("NSG", nsg)))
    )
    kern = mx.fast.metal_kernel(
        name=f"tree_spec_nax_packed_k{k}_n{n}_g{group_size}_m{m}_bm{bm}_sg{nsg}_{_MSL_TYPE[dtype]}",
        input_names=["x", "wq2", "sp2", "bp2"],
        output_names=["y"],
        source=consts + _NAX_SPECIAL,
        header=_NAX_HEADER,
    )
    grid, tg = (32 * nsg, n // 32, 1), (32 * nsg, 1, 1)

    def run(x, wq2, sp2, bp2):
        return kern(inputs=[x, wq2, sp2, bp2], grid=grid, threadgroup=tg,
                    output_shapes=[(*x.shape[:-1], n)], output_dtypes=[dtype])[0]

    return run


# Any row count in one dispatch (prompt prefill): grid z walks BM-row chunks,
# the last one declared short so the tensor op reads its missing rows as zero.
# K steps of BK (a multiple of 16) per barrier instead of 16: at prefill sizes
# the per-step barrier and MMA setup, not the weight reads, set the pace.
_NAX_MULTI_SOURCE = """
using namespace metal;
using namespace mpp::tensor_ops;

constexpr int BN = 32;
constexpr int KB16 = K / 16;
constexpr int G = K / GS;
constexpr int K_chunk = K / NSG;

const uint tid = thread_position_in_threadgroup.x;
const uint sg_id = simdgroup_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
const int tile = int(threadgroup_position_in_grid.y);
const int m0 = int(threadgroup_position_in_grid.z) * BM;
const int mrows = min(BM, int(M_size) - m0);
const int n0 = tile * BN;
const int k_begin = int(sg_id) * K_chunk;
const int k_end = k_begin + K_chunk;

threadgroup T B_tile[NSG][BK * BN];
threadgroup float partial[NSG][BM * BN];

constexpr auto desc = matmul2d_descriptor(BM, BN, BK, false, false, false,
                                          matmul2d_descriptor::mode::multiply_accumulate);
matmul2d<desc, metal::execution_simdgroup> op;

tensor<device T, dextents<int, 2>, tensor_inline> A((device T*)x + size_t(m0) * K, dextents<int, 2>(K, mrows), array<int, 2>({1, K}));
tensor<threadgroup T, dextents<int, 2>, tensor_inline> B(B_tile[sg_id], dextents<int, 2>(BN, BK), array<int, 2>({1, BN}));
tensor<threadgroup float, dextents<int, 2>, tensor_inline> C(partial[sg_id], dextents<int, 2>(BN, BM), array<int, 2>({1, BN}));

auto ct_c = op.template get_destination_cooperative_tensor<
    tensor<device T, extents<int, BK, BM>, tensor_inline>,
    tensor<threadgroup T, extents<int, BN, BK>, tensor_inline>, float>();
_Pragma("unroll")
for (uint16_t i = 0; i < ct_c.get_capacity(); ++i) ct_c[i] = 0.0f;

const device uint2* wl = (const device uint2*)wq2 + (size_t(tile) * KB16) * 32 + lane;
const device T* sl = sp2 + size_t(tile) * G * 32 + lane;
const device T* bl = bp2 + size_t(tile) * G * 32 + lane;
for (int k0 = k_begin; k0 < k_end; k0 += BK) {
  _Pragma("unroll")
  for (int sb = 0; sb < BK / 16; ++sb) {
    const int kk = k0 + sb * 16;
    const uint2 p = wl[(kk / 16) * 32];
    const float s = float(sl[(kk / GS) * 32]);
    const float b = float(bl[(kk / GS) * 32]);
    _Pragma("unroll")
    for (int ki = 0; ki < 8; ++ki) {
      B_tile[sg_id][(sb * 16 + ki) * BN + int(lane)] = T(float((p.x >> (ki * 4)) & 0xFu) * s + b);
      B_tile[sg_id][(sb * 16 + 8 + ki) * BN + int(lane)] = T(float((p.y >> (ki * 4)) & 0xFu) * s + b);
    }
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);
  auto tA = A.template slice<BK, BM>(k0, 0);
  auto tB = B.template slice<BN, BK>(0, 0);
  op.run(tA, tB, ct_c);
  simdgroup_barrier(mem_flags::mem_threadgroup);
}

auto tC = C.template slice<BN, BM>(0, 0);
ct_c.store(tC);
threadgroup_barrier(mem_flags::mem_threadgroup);
for (int off = int(tid); off < BM * BN; off += NSG * 32) {
  float acc = 0.0f;
  for (int s2 = 0; s2 < NSG; ++s2) acc += partial[s2][off];
  const int row = off / BN;
  const int col = off - row * BN;
  if (row < mrows) y[(m0 + row) * N + n0 + col] = T(acc);
}
"""


@lru_cache(maxsize=None)
def nax_multi_kernel(k: int, n: int, group_size: int, bm: int, nsg: int, dtype, bk: int = 16):
    """y = x @ W.T for any number of rows of x (..., K), in BM-row chunks."""
    assert k % (nsg * bk) == 0 and bk % 16 == 0 and group_size % 16 == 0
    consts = (
        f"typedef {_MSL_TYPE[dtype]} T;\n"
        + "".join(f"constexpr int {a} = {v};\n" for a, v in
                  (("K", k), ("N", n), ("GS", group_size), ("BM", bm), ("NSG", nsg), ("BK", bk)))
    )
    kern = mx.fast.metal_kernel(
        name=f"tree_spec_nax_multi_k{k}_n{n}_g{group_size}_bm{bm}_sg{nsg}_bk{bk}_{_MSL_TYPE[dtype]}",
        input_names=["x", "wq2", "sp2", "bp2", "M_size"],
        output_names=["y"],
        source=consts + _NAX_MULTI_SOURCE,
        header=_NAX_HEADER,
    )

    def run(x, wq2, sp2, bp2):
        m = x.size // k
        return kern(inputs=[x, wq2, sp2, bp2, m], grid=(32 * nsg, n // 32, -(-m // bm)), threadgroup=(32 * nsg, 1, 1),
                    output_shapes=[(*x.shape[:-1], n)], output_dtypes=[dtype])[0]

    return run


def qmm4_packed(x2: mx.array, wp, sp, bp, n: int, group_size: int, *, r: int = 4, nsg: int = 8) -> mx.array:
    """y = x2 @ W.T for x2 (M <= 8, K), K % 256 == 0, with W packed by pack()."""
    m, k = x2.shape
    strips = (n + 7) // 8
    tgs = (strips + r * nsg - 1) // (r * nsg)
    (y,) = _kernel()(
        inputs=[x2, wp, sp, bp, k, n, m],
        template=[("T", x2.dtype), ("GS", group_size), ("R", r), ("NSG", nsg)],
        grid=(32, nsg * tgs, 1),
        threadgroup=(32, nsg, 1),
        output_shapes=[(m, n)],
        output_dtypes=[x2.dtype],
    )
    return y


# ------------------------------------------------------------- 1-4 row GEMV
# The NAX kernel above pads every call to an 8- or 16-row tile, so a 1-token
# forward pays for 8 rows of dequant + MMA and runs ~12% slower than MLX's qmv.
# This is a plain GEMV on the same pack_nax() layout. In a 32-row tile, the
# 16-wide K blocks are 256 contiguous bytes each (row r's uint2 at byte 8r), and
# successive blocks are contiguous too. Lane = (kq, rq): it takes RPL adjacent
# rows (RPL * 8 bytes) of block kq, so a simdgroup step reads RPL whole blocks =
# RPL * 256 contiguous bytes, and each lane's activation work (load, convert,
# pre-scale; 16 values) is shared by its RPL rows instead of redone per row.
# Nibbles are decoded MLX-style without shifts, against activations pre-scaled
# by 1, 1/16, 1/256, 1/4096:
#   sum_k (s q_k + b) x_k = s * sum_k q_k x_k + b * sum_k x_k
# The kq lanes of a row are summed with shuffles; SPLIT simdgroups split K and
# their partials are summed through threadgroup memory.

_QMV_SOURCE = """
constexpr int K = KCONST;
constexpr int KB = K / 16;
constexpr int G = K / GS;
constexpr int LPB = 32 / RPL;        // lanes per 16-wide block
constexpr int BPS = RPL;             // blocks per simdgroup step
threadgroup float part[SPLIT][MR][32];

const int lane = int(thread_index_in_simdgroup);
const int split = int(simdgroup_index_in_threadgroup);
const int tile = int(threadgroup_position_in_grid.y);
const int N = NCONST;
const int kq = lane / LPB, rq = lane % LPB;
const int b0 = split * KB / SPLIT, b1 = (split + 1) * KB / SPLIT;

float acc[RPL][MR];
for (int r = 0; r < RPL; ++r)
  for (int m = 0; m < MR; ++m) acc[r][m] = 0.0f;

const device uint4* wt = (const device uint4*)(wq2 + size_t(tile) * KB * 64) + rq * (RPL / 2);
const device T* st = sp2 + size_t(tile) * G * 32 + rq * RPL;
const device T* bt = bp2 + size_t(tile) * G * 32 + rq * RPL;
for (int kb = b0 + kq; kb < b1; kb += BPS) {
  uint4 w[RPL / 2];
  _Pragma("unroll")
  for (int i = 0; i < RPL / 2; ++i) w[i] = wt[kb * 16 + i];
  const int g = kb * 16 / GS;
  float sc[RPL], bi[RPL];
  _Pragma("unroll")
  for (int r = 0; r < RPL; ++r) { sc[r] = float(st[g * 32 + r]); bi[r] = float(bt[g * 32 + r]); }
  float xv[MR][16], xs[MR];
  _Pragma("unroll")
  for (int m = 0; m < MR; ++m) {
    const device uint4* xp = (const device uint4*)(x + m * K + kb * 16);
    const uint4 u0 = xp[0], u1 = xp[1];
    thread const T* x0 = reinterpret_cast<thread const T*>(&u0);
    thread const T* x1 = reinterpret_cast<thread const T*>(&u1);
    xs[m] = 0.0f;
    _Pragma("unroll")
    for (int i = 0; i < 8; ++i) {
      const float a = float(x0[i]), c = float(x1[i]);
      xs[m] += a + c;
      const float f = (i % 4 == 0) ? 1.0f : (i % 4 == 1) ? (1.0f / 16.0f) : (i % 4 == 2) ? (1.0f / 256.0f) : (1.0f / 4096.0f);
      xv[m][i] = a * f;
      xv[m][8 + i] = c * f;
    }
  }
  _Pragma("unroll")
  for (int r = 0; r < RPL; ++r) {
    const uint4 ww = w[r / 2];
    const uint lo = (r % 2 == 0) ? ww.x : ww.z;   // K 0..7 of row r
    const uint hi = (r % 2 == 0) ? ww.y : ww.w;   // K 8..15
    float q[16];
    _Pragma("unroll")
    for (int h = 0; h < 2; ++h) {
      const uint a16 = lo >> (16 * h), c16 = hi >> (16 * h);
      q[4 * h + 0] = float(a16 & 0x000Fu);
      q[4 * h + 1] = float(a16 & 0x00F0u);
      q[4 * h + 2] = float(a16 & 0x0F00u);
      q[4 * h + 3] = float(a16 & 0xF000u);
      q[8 + 4 * h + 0] = float(c16 & 0x000Fu);
      q[8 + 4 * h + 1] = float(c16 & 0x00F0u);
      q[8 + 4 * h + 2] = float(c16 & 0x0F00u);
      q[8 + 4 * h + 3] = float(c16 & 0xF000u);
    }
    _Pragma("unroll")
    for (int m = 0; m < MR; ++m) {
      float d = 0.0f;
      _Pragma("unroll")
      for (int i = 0; i < 16; ++i) d += q[i] * xv[m][i];
      acc[r][m] += sc[r] * d + bi[r] * xs[m];
    }
  }
}

// Sum the kq lanes of each row (lane bits above log2(LPB)).
_Pragma("unroll")
for (int r = 0; r < RPL; ++r)
  _Pragma("unroll")
  for (int m = 0; m < MR; ++m)
    for (int o = LPB; o < 32; o *= 2) acc[r][m] += simd_shuffle_xor(acc[r][m], ushort(o));
if (SPLIT == 1) {
  if (kq == 0)
    for (int m = 0; m < MR; ++m)
      for (int r = 0; r < RPL; ++r) y[m * N + tile * 32 + rq * RPL + r] = T(acc[r][m]);
} else {
  if (kq == 0)
    for (int m = 0; m < MR; ++m)
      for (int r = 0; r < RPL; ++r) part[split][m][rq * RPL + r] = acc[r][m];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (split == 0) {
    for (int m = 0; m < MR; ++m) {
      float t = 0.0f;
      for (int s2 = 0; s2 < SPLIT; ++s2) t += part[s2][m][lane];
      y[m * N + tile * 32 + lane] = T(t);
    }
  }
}
"""


@lru_cache(maxsize=None)
def _qmv_kernel(k: int, n: int, group_size: int, mr: int, split: int, rpl: int, dtype):
    """One kernel per shape with every parameter compiled in: no template or
    scalar inputs, which cost ~1 us of CPU per call (a forward makes ~400)."""
    consts = (
        f"typedef {_MSL_TYPE[dtype]} T;\n"
        + "".join(f"constexpr int {a} = {v};\n" for a, v in
                  (("KCONST", k), ("NCONST", n), ("GS", group_size), ("MR", mr), ("SPLIT", split), ("RPL", rpl)))
    )
    kern = mx.fast.metal_kernel(
        name=f"tree_spec_qmv_packed_k{k}_n{n}_g{group_size}_m{mr}_s{split}_r{rpl}_{_MSL_TYPE[dtype]}",
        input_names=["x", "wq2", "sp2", "bp2"],
        output_names=["y"],
        source=consts + _QMV_SOURCE,
    )
    grid, tg = (32 * split, n // 32, 1), (32 * split, 1, 1)

    def run(x, wq2, sp2, bp2):
        # The kernel indexes x flat, so (..., mr, K) needs no reshape either way.
        return kern(inputs=[x, wq2, sp2, bp2], grid=grid, threadgroup=tg,
                    output_shapes=[(*x.shape[:-1], n)], output_dtypes=[dtype])[0]

    return run


def qmv_packed(x2: mx.array, wq2, sp2, bp2, n: int, group_size: int, *, split: int = 8, rpl: int = 4) -> mx.array:
    """y = x2 @ W.T for x2 (M <= 4, K) with W packed by pack_nax(); group_size % 16 == 0."""
    m, k = x2.shape
    return _qmv_kernel(k, n, group_size, m, split, rpl, x2.dtype)(x2, wq2, sp2, bp2)



# ------------------------------------------------------------- prefill GEMM
# MLX's own prefill kernel (affine_qmm_t_nax in quantized_nax.h) with its weight
# loader swapped for one that reads the pack_nax() layout. A threadgroup of
# WM x WN simdgroups owns a BM x BN output tile; per BK step its threads
# dequantize the BN x BK weight block into threadgroup memory (one 16-wide K
# block of one row each, an 8-byte read: a simdgroup reads 256 contiguous bytes)
# and every simdgroup multiplies its SM x SN subtile on the NAX tiles. Unlike the
# multi-row NAX kernel above, each weight is dequantized once per BM rows, not
# per 32, and K isn't split across simdgroups. The NAX tile helpers are MLX's
# (steel/gemm/nax.h), inlined from the installed package: custom kernels are
# compiled from source without MLX's include path.

_STEEL_HEADERS = (
    "steel/defines.h",
    "steel/utils/type_traits.h",
    "steel/utils/integral_constant.h",
    "steel/gemm/nax.h",
)

_PREFILL_IMPL = """
using namespace mlx::steel;

template <typename T, int K, int N, int GS, int BM, int BN, int BK, int WM, int WN, bool aligned_m>
METAL_FUNC void qmm_prefill_packed(
    const device T* x, const device uint32_t* wq2, const device T* sp2, const device T* bp2,
    device T* y, threadgroup T* Ws, const int M, const int sgp_sm,
    const int y_row, const int y_col, const ushort sg, const ushort lane) {
  constexpr int BK_PAD = BK + 16 / sizeof(T);
  constexpr int KB16 = K / 16;
  constexpr int G = K / GS;
  constexpr int NT = WM * WN * 32;
  constexpr int ITEMS = BN * (BK / 16);       // (row, 16-wide K block) pairs per step
  static_assert(ITEMS % NT == 0, "loader items must split evenly");
  constexpr int LOADS = ITEMS / NT;
  constexpr int SM = BM / WM, SN = BN / WN, SK = 32;
  constexpr int TM = SM / 16, TN = SN / 16, TK = SK / 16;
  static_assert(GS % BK == 0 || BK % GS == 0, "group size vs BK");

  const int t = int(sg) * 32 + int(lane);
  const device uint2* wsrc[LOADS];
  const device T* ssrc[LOADS];
  const device T* bsrc[LOADS];
  threadgroup T* wdst[LOADS];
  int kbo[LOADS];
  _Pragma("unroll")
  for (int i = 0; i < LOADS; ++i) {
    const int w = t + i * NT;
    const int r32 = w % 32, rest = w / 32;
    const int toff = rest % (BN / 32);
    kbo[i] = rest / (BN / 32);
    const int tile = y_col / 32 + toff;
    wsrc[i] = (const device uint2*)wq2 + (size_t(tile) * KB16 + kbo[i]) * 32 + r32;
    ssrc[i] = sp2 + size_t(tile) * G * 32 + r32;
    bsrc[i] = bp2 + size_t(tile) * G * 32 + r32;
    wdst[i] = Ws + (toff * 32 + r32) * BK_PAD + kbo[i] * 16;
  }

  const short tm = SM * (sg / WN);
  const short tn = SN * (sg % WN);
  x += size_t(y_row + tm) * K;

  NAXTile<float, TM, TN> Dtile;
  Dtile.clear();

  for (int k = 0; k < K; k += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    _Pragma("unroll")
    for (int i = 0; i < LOADS; ++i) {
      const uint2 p = wsrc[i][(k / 16) * 32];
      const int g = (k + kbo[i] * 16) / GS;
      const float s = float(ssrc[i][g * 32]);
      const float b = float(bsrc[i][g * 32]);
      _Pragma("unroll")
      for (int ki = 0; ki < 8; ++ki) {
        wdst[i][ki] = T(float((p.x >> (ki * 4)) & 0xFu) * s + b);
        wdst[i][8 + ki] = T(float((p.y >> (ki * 4)) & 0xFu) * s + b);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    STEEL_PRAGMA_NO_UNROLL
    for (int kk1 = 0; kk1 < BK; kk1 += SK) {
      NAXTile<T, TM, TK> Atile;
      NAXTile<T, TN, TK> Btile;
      volatile int compiler_barrier;
      if (aligned_m) {
        Atile.load(x + kk1, K);
      } else {
        Atile.load_safe(x + kk1, K, short2(SK, sgp_sm));
      }
      Btile.template load<T, BK_PAD, 1>(Ws + tn * BK_PAD + kk1);
      tile_matmad_nax(Dtile, Atile, metal::bool_constant<false>{}, Btile, metal::bool_constant<true>{});
      (void)compiler_barrier;
    }
    x += BK;
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  device T* yt = y + size_t(y_row + tm) * N + y_col + tn;
  if (aligned_m) {
    Dtile.store(yt, N);
  } else if (sgp_sm > 0) {
    Dtile.store_safe(yt, N, short2(SN, sgp_sm));
  }
}
"""

_PREFILL_SOURCE = """
threadgroup T Ws[BN * (BK + 16 / sizeof(T))];
const int M = int(M_size);
const int y_row = int(threadgroup_position_in_grid.y) * BM;
const int y_col = int(threadgroup_position_in_grid.x) * BN;
const ushort sg = simdgroup_index_in_threadgroup;
const ushort lane = thread_index_in_simdgroup;
const int sgp_sm = min(BM / WM, M - (y_row + (BM / WM) * int(sg / WN)));
if (sgp_sm == BM / WM) {
  qmm_prefill_packed<T, K, N, GS, BM, BN, BK, WM, WN, true>(x, wq2, sp2, bp2, y, Ws, M, sgp_sm, y_row, y_col, sg, lane);
} else {
  qmm_prefill_packed<T, K, N, GS, BM, BN, BK, WM, WN, false>(x, wq2, sp2, bp2, y, Ws, M, sgp_sm, y_row, y_col, sg, lane);
}
"""


@lru_cache(maxsize=None)
def _steel_header() -> str:
    import os
    import re

    root = os.path.join(os.path.dirname(mx.__file__), "include", "mlx", "backend", "metal", "kernels")
    parts = []
    for rel in _STEEL_HEADERS:
        with open(os.path.join(root, rel)) as f:
            src = f.read()
        parts.append(re.sub(r'^\s*(#pragma once|#include "mlx/.*")\s*$', "", src, flags=re.MULTILINE))
    return "\n".join(parts) + _PREFILL_IMPL


PREFILL_TILE = (64, 64, 32, 2, 2)  # (BM, BN, BK, WM, WN): MLX's affine_qmm_t_nax default


@lru_cache(maxsize=None)
def prefill_kernel(k: int, n: int, group_size: int, dtype, tile: tuple = PREFILL_TILE):
    """y = x @ W.T for any number of rows of x (..., K), W packed by pack_nax()."""
    bm, bn, bk, wm, wn = tile
    assert n % bn == 0 and k % bk == 0 and bn % 32 == 0 and bk % 16 == 0
    consts = (
        f"typedef {_MSL_TYPE[dtype]} T;\n"
        + "".join(f"constexpr int {a} = {v};\n" for a, v in
                  (("K", k), ("N", n), ("GS", group_size), ("BM", bm), ("BN", bn), ("BK", bk), ("WM", wm), ("WN", wn)))
    )
    kern = mx.fast.metal_kernel(
        name=f"tree_spec_prefill_packed_k{k}_n{n}_g{group_size}_{bm}x{bn}x{bk}_{wm}x{wn}_{_MSL_TYPE[dtype]}",
        input_names=["x", "wq2", "sp2", "bp2", "M_size"],
        output_names=["y"],
        source=consts + _PREFILL_SOURCE,
        header=_steel_header(),
    )
    nt = 32 * wm * wn

    def run(x, wq2, sp2, bp2):
        m = x.size // k
        return kern(inputs=[x, wq2, sp2, bp2, m], grid=(nt * (n // bn), -(-m // bm), 1), threadgroup=(nt, 1, 1),
                    output_shapes=[(*x.shape[:-1], n)], output_dtypes=[dtype])[0]

    return run

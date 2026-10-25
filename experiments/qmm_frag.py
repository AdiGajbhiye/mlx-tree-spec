"""Multi-row (M <= 8) 4-bit matmul on 8x8 simdgroup-matrix fragments.

Each lane loads and dequantizes only the 2 weights it owns in each 8x8 fragment
(MLX steel's fragment layout), so there is no threadgroup memory and no barrier;
a simdgroup owns R strips of 8 output rows and walks K 32 at a time. Weights are
read once for all M rows. ACC selects the fragment precision: float (exact-ish,
ALU-heavy) or half (A/B/C in half, partial sums promoted to float every K block).
Candidate kernel for verify_matmul's autotuner.
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx

_HEADER = """
#include <metal_simdgroup_matrix>
using namespace metal;
"""

_SOURCE = """
const uint lane = thread_index_in_simdgroup;
const int K = int(K_size);
const int N = int(N_size);
const int M = int(M_size);
const int sg_global = int(threadgroup_position_in_grid.y) * 2 + int(simdgroup_index_in_threadgroup);
const int n_base = sg_global * (8 * R);
if (n_base >= N) return;

const int qid = int(lane) / 4;
const int fm = (qid & 4) + ((int(lane) / 2) % 4);
const int fn = (qid & 2) * 2 + (int(lane) % 2) * 2;
const int KW = K / 8;
const int KG = K / GS;
typedef simdgroup_matrix<FT, 8, 8> frag;
typedef vec<FT, 2> frag2;

float2 acc[R];
int rows[R];
bool ok[R];
for (int s = 0; s < R; ++s) {
  acc[s] = float2(0.0f);
  rows[s] = n_base + s * 8 + fm;
  ok[s] = rows[s] < N;
}
const bool m0ok = fn < M, m1ok = fn + 1 < M;

for (int k = 0; k < K; k += 32) {
  frag2 xb[4];
  for (int t = 0; t < 4; ++t) {
    int kk = k + t * 8 + fm;
    xb[t] = frag2(m0ok ? FT(x[fn * K + kk]) : FT(0), m1ok ? FT(x[(fn + 1) * K + kk]) : FT(0));
  }
  // Sum of x over this 32-wide block, for the bias term (all-ones A gives row sums).
  frag Xs = frag(0);
  frag ones = frag(FT(1));
  for (int t = 0; t < 4; ++t) {
    frag B;
    reinterpret_cast<thread frag2&>(B.thread_elements()) = xb[t];
    simdgroup_multiply_accumulate(Xs, ones, B, Xs);
  }
  const float2 xsum = float2(reinterpret_cast<thread frag2&>(Xs.thread_elements()));
  for (int s = 0; s < R; ++s) {
    uint4 p = ok[s] ? *((const device uint4*)(w + rows[s] * KW + k / 8)) : uint4(0);
    // Codes go in raw; scale and bias are applied once per block in float:
    // sum_k (sc*q+bi)*x = sc*sum_k q*x + bi*sum_k x.
    frag C = frag(0);
    for (int t = 0; t < 4; ++t) {
      uint q = p[t];
      frag A, B;
      reinterpret_cast<thread frag2&>(A.thread_elements()) =
          frag2(FT((q >> (4 * fn)) & 0xFu), FT((q >> (4 * (fn + 1))) & 0xFu));
      reinterpret_cast<thread frag2&>(B.thread_elements()) = xb[t];
      simdgroup_multiply_accumulate(C, A, B, C);
    }
    float sc = ok[s] ? float(scales[rows[s] * KG + k / GS]) : 0.0f;
    float bi = ok[s] ? float(biases[rows[s] * KG + k / GS]) : 0.0f;
    frag2 c = reinterpret_cast<thread frag2&>(C.thread_elements());
    acc[s] += sc * float2(c) + bi * xsum;
  }
}

for (int s = 0; s < R; ++s) {
  if (!ok[s]) continue;
  if (m0ok) y[fn * N + rows[s]] = T(acc[s].x);
  if (m1ok) y[(fn + 1) * N + rows[s]] = T(acc[s].y);
}
"""


@lru_cache(maxsize=None)
def _kernel():
    return mx.fast.metal_kernel(
        name="tree_spec_qmm4_frag2",
        input_names=["x", "w", "scales", "biases", "K_size", "N_size", "M_size"],
        output_names=["y"],
        source=_SOURCE,
        header=_HEADER,
    )


def qmm4_frag(x2: mx.array, w, scales, biases, group_size: int, *, half: bool = True, r: int = 4) -> mx.array:
    """y = x2 @ dequant(w).T for x2 of shape (M <= 8, K), K % 32 == 0, group_size >= 32."""
    m, k = x2.shape
    n = w.shape[0]
    sgs = (n + 8 * r - 1) // (8 * r)
    (y,) = _kernel()(
        inputs=[x2, w, scales, biases, k, n, m],
        template=[("T", x2.dtype), ("FT", mx.float16 if half else mx.float32), ("GS", group_size), ("R", r)],
        grid=(32, 2 * ((sgs + 1) // 2), 1),
        threadgroup=(32, 2, 1),
        output_shapes=[(m, n)],
        output_dtypes=[x2.dtype],
    )
    return y

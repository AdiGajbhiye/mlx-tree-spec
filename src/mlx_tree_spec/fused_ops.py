"""Fused Metal kernels for the tree pass (lithos-metal's "fewer, fatter kernels").

A tree pass makes ~1,900 GPU dispatches, and on this GPU each dependent one costs
~2 us before doing any work. Two groups of them are cheap to fuse:

gdn_prep: everything a GatedDeltaNet layer does between its input projections
  and the recurrence, per tree node: the causal conv over the node's ancestors
  (indices from Tree.conv_idx), silu, the per-head q/k RMS norms and scales,
  beta = sigmoid(b) and the decay g. One dispatch instead of ~13 (concatenate,
  gather, 3 casts, multiply, sum, silu, 2 norms, 2 scales, sigmoid, compute_g).
  Rounding follows mlx_lm's op order (bf16 after the conv, the silu and the
  norm) so outputs match the unfused path to ~1 bf16 ulp.

add_rms_norm: a residual add and the RMS norm that always follows it in a
  decoder layer, as one dispatch returning both the sum and its norm.
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx

_MSL = {mx.bfloat16: "bfloat16_t", mx.float16: "half", mx.float32: "float"}

# Grid: x = 128 threads (one per head dimension), y = head slot (Hk q heads, Hk k
# heads, Hv v heads, then one slot for g/beta), z = node.
_GDN_PREP_SOURCE = """
const int tid = int(thread_position_in_threadgroup.x);
const int slot = int(threadgroup_position_in_grid.y);
const int node = int(threadgroup_position_in_grid.z);
constexpr int QK = HK * DK;
constexpr int C = 2 * QK + HV * DV;
threadgroup float red[4];

if (slot == 2 * HK + HV) {
  // Decay and write strength of each value head.
  if (tid < HV) {
    const int i = node * HV + tid;
    const float ab = float(T(float(a[i]) + float(dt_bias[tid])));
    const float sp = ab > 20.0f ? ab : log1p(exp(ab));
    g[i] = exp(-exp(float(A_log[tid])) * float(T(sp)));
    beta[i] = T(1.0f / (1.0f + exp(-float(b[i]))));
  }
  return;
}
const bool is_v = slot >= 2 * HK;
const int ch = is_v ? 2 * QK + (slot - 2 * HK) * DV + tid : slot * DK + tid;

// Causal conv (kernel 4) over the node's window: 3 ancestors, then itself.
float acc = 0.0f;
for (int j = 0; j < 4; ++j) {
  const int r = conv_idx[node * 4 + j];
  const float xv = r < 3 ? float(conv_state[r * C + ch]) : float(qkv[(r - 3) * C + ch]);
  acc += xv * float(conv_w[ch * 4 + j]);
}
const float cv = float(T(acc));
const float s = float(T(cv / (1.0f + exp(-cv))));   // silu

if (is_v) {
  v[(node * HV + (slot - 2 * HK)) * DV + tid] = T(s);
  return;
}
// Per-head RMS norm over DK = 128 lanes (4 simdgroups).
float ss = simd_sum(s * s);
if (thread_index_in_simdgroup == 0) red[simdgroup_index_in_threadgroup] = ss;
threadgroup_barrier(mem_flags::mem_threadgroup);
const float total = red[0] + red[1] + red[2] + red[3];
const float normed = float(T(s * rsqrt(total / float(DK) + 1e-6f)));
if (slot < HK) {
  q[(node * HK + slot) * DK + tid] = T(normed * (1.0f / float(DK)));       // inv_scale ** 2
} else {
  k[(node * HK + slot - HK) * DK + tid] = T(normed * rsqrt(float(DK)));     // inv_scale
}
"""


@lru_cache(maxsize=None)
def _gdn_prep_kernel(hk: int, dk: int, hv: int, dv: int, dtype):
    assert dk == 128 and dv == 128, "one 128-thread group per head"
    consts = f"typedef {_MSL[dtype]} T;\n" + "".join(
        f"constexpr int {a} = {b};\n" for a, b in (("HK", hk), ("DK", dk), ("HV", hv), ("DV", dv))
    )
    return mx.fast.metal_kernel(
        name=f"tree_gdn_prep_{hk}_{dk}_{hv}_{dv}_{_MSL[dtype]}",
        input_names=["qkv", "conv_state", "conv_idx", "conv_w", "a", "b", "A_log", "dt_bias"],
        output_names=["q", "k", "v", "g", "beta"],
        source=consts + _GDN_PREP_SOURCE,
    )


def gdn_prep(gdn, qkv, conv_state, conv_idx, conv_w, a, b):
    """(q, k, v, g, beta) for every node, as mlx_lm's GatedDeltaNet computes them.
    qkv: (1, n, conv_dim); conv_state: (1, 3, conv_dim); conv_w: (conv_dim, 4)."""
    n = qkv.shape[1]
    hk, dk, hv, dv = gdn.num_k_heads, gdn.head_k_dim, gdn.num_v_heads, gdn.head_v_dim
    dt = qkv.dtype
    return _gdn_prep_kernel(hk, dk, hv, dv, dt)(
        inputs=[qkv, conv_state, conv_idx, conv_w, a, b, gdn.A_log, gdn.dt_bias],
        grid=(128, 2 * hk + hv + 1, n),
        threadgroup=(128, 1, 1),
        output_shapes=[(1, n, hk, dk), (1, n, hk, dk), (1, n, hv, dv), (1, n, hv), (1, n, hv)],
        output_dtypes=[dt, dt, dt, mx.float32, dt],
    )


_ADD_NORM_SOURCE = """
const int row = int(threadgroup_position_in_grid.y);
const int tid = int(thread_position_in_threadgroup.x);
constexpr int NT = 256;
threadgroup float red[NT / 32];
float vals[(D + NT - 1) / NT];
float ss = 0.0f;
for (int i = 0, c = tid; c < D; ++i, c += NT) {
  const float h = float(T(float(x[row * D + c]) + float(r[row * D + c])));
  h_out[row * D + c] = T(h);
  vals[i] = h;
  ss += h * h;
}
ss = simd_sum(ss);
if (thread_index_in_simdgroup == 0) red[simdgroup_index_in_threadgroup] = ss;
threadgroup_barrier(mem_flags::mem_threadgroup);
float total = 0.0f;
for (int i = 0; i < NT / 32; ++i) total += red[i];
const float inv = rsqrt(total / float(D) + EPS);
for (int i = 0, c = tid; c < D; ++i, c += NT) {
  n_out[row * D + c] = T(float(w[c]) * float(T(vals[i] * inv)));
}
"""


@lru_cache(maxsize=None)
def _add_norm_kernel(d: int, eps: float, dtype):
    consts = f"typedef {_MSL[dtype]} T;\nconstexpr int D = {d};\nconstexpr float EPS = {eps!r}f;\n"
    return mx.fast.metal_kernel(
        name=f"tree_add_rms_norm_{d}_{str(eps).replace('.', 'p').replace('-', 'm')}_{_MSL[dtype]}",
        input_names=["x", "r", "w"],
        output_names=["h_out", "n_out"],
        source=consts + _ADD_NORM_SOURCE,
    )


def add_rms_norm(x, r, norm):
    """(x + r, norm(x + r)) for an nn.RMSNorm `norm`, in one dispatch."""
    d = x.shape[-1]
    rows = x.size // d
    h, n = _add_norm_kernel(d, float(norm.eps), x.dtype)(
        inputs=[x, r, norm.weight],
        grid=(256, rows, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[x.shape, x.shape],
        output_dtypes=[x.dtype, x.dtype],
    )
    return h, n


# gdn_commit: the GatedDeltaNet half of committing an accepted tree path, one
# dispatch per layer instead of ~10 (gathers, concatenate, slice, sigmoid, decay,
# recurrence): replays the path's recorded q/k/v/g/beta from the pre-pass state
# into the new state, and rebuilds the conv state, which is the last 3 entries of
# the final node's conv window (Tree.conv_idx). Grid z = value heads + 1 slot
# for the conv copy.
_GDN_COMMIT_SOURCE = """
const int hv_idx = int(thread_position_in_grid.z);
constexpr int QK = HK * DK;
constexpr int C = 2 * QK + HV * DV;
if (hv_idx == HV) {
  const int t = int(thread_position_in_grid.y) * 32 + int(thread_position_in_grid.x);
  const int nt = int(threads_per_grid.y) * 32;
  for (int e = t; e < 3 * C; e += nt) {
    const int j = e / C, ch = e - j * C;
    const int src = conv_idx[last * 4 + 1 + j];
    conv_out[e] = src < 3 ? conv_state[src * C + ch] : qkv[(src - 3) * C + ch];
  }
  return;
}
const int hk_idx = hv_idx / (HV / HK);
constexpr int n_per_t = DK / 32;
const int dk_idx = int(thread_position_in_threadgroup.x);
const int dv_idx = int(thread_position_in_grid.y);
auto i_state = state_in + (hv_idx * DV + dv_idx) * DK;
auto o_state = state_out + (hv_idx * DV + dv_idx) * DK;
float state[n_per_t];
for (int i = 0; i < n_per_t; ++i) state[i] = static_cast<float>(i_state[n_per_t * dk_idx + i]);
for (int t = 0; t < M; ++t) {
  const int node = path[t];
  auto k_ = k + (node * HK + hk_idx) * DK;
  const float gv = g[node * HV + hv_idx];
  float kv_mem = 0.0f;
  for (int i = 0; i < n_per_t; ++i) {
    state[i] = state[i] * gv;
    kv_mem += state[i] * k_[n_per_t * dk_idx + i];
  }
  kv_mem = simd_sum(kv_mem);
  const float delta = (v[(node * HV + hv_idx) * DV + dv_idx] - kv_mem) * beta[node * HV + hv_idx];
  for (int i = 0; i < n_per_t; ++i) state[i] = state[i] + k_[n_per_t * dk_idx + i] * delta;
}
for (int i = 0; i < n_per_t; ++i) o_state[n_per_t * dk_idx + i] = static_cast<StT>(state[i]);
"""


@lru_cache(maxsize=None)
def _gdn_commit_kernel(hk: int, dk: int, hv: int, dv: int):
    consts = "".join(f"constexpr int {a} = {b};\n" for a, b in (("HK", hk), ("DK", dk), ("HV", hv), ("DV", dv)))
    return mx.fast.metal_kernel(
        name=f"tree_gdn_commit_{hk}_{dk}_{hv}_{dv}",
        input_names=["k", "v", "g", "beta", "state_in", "conv_state", "qkv", "conv_idx", "path", "M", "last"],
        output_names=["state_out", "conv_out"],
        source=consts + _GDN_COMMIT_SOURCE,
    )


def gdn_commit(gdn, rec, conv_idx, path_arr, nodes: list[int]):
    """New (conv_state, recurrent state) after the accepted `nodes` of a tree pass,
    from what tree_gdn recorded: (conv_state, state0, qkv, q, k, v, g, beta)."""
    conv_state, state0, qkv, _, k, v, g, beta = rec
    hk, dk, hv, dv = gdn.num_k_heads, gdn.head_k_dim, gdn.num_v_heads, gdn.head_v_dim
    state, conv = _gdn_commit_kernel(hk, dk, hv, dv)(
        inputs=[k, v, g, beta, state0, conv_state, qkv, conv_idx, path_arr, len(nodes), nodes[-1]],
        template=[("StT", state0.dtype)],
        grid=(32, dv, hv + 1),
        threadgroup=(32, 4, 1),
        output_shapes=[state0.shape, conv_state.shape],
        output_dtypes=[state0.dtype, conv_state.dtype],
    )
    return [conv, state]

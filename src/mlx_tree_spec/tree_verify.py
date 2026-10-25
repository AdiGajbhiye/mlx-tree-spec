"""Deduplicated tree verification for the hybrid Qwen3.5 target, one CPU sync per step.

tree_spec.py verifies a tree as one batch row per root-to-leaf path: every row
gets its own copy of the KV and GatedDeltaNet state (~1.4 ms per extra row) and
re-runs every shared node through the 4-bit matmuls. Here a tree is one sequence
of unique nodes (node 0 = the bonus token, each node after its parent):

  - matmuls, norms, MLPs: once per node;
  - full attention: one pass over the nodes, with a mask letting each node see
    the committed context plus its own ancestors, each node roped at its depth
    (mx.fast.rope with one offset per row, as MTPLX's mrope adapter does);
  - GatedDeltaNet: a recurrence cannot branch, so a kernel walks each
    root-to-leaf path from the one committed state (no copies; shared nodes are
    re-walked, which is cheap next to the matmuls) and writes every node's
    output once. A leaf hanging off another walk (a rank-2 sibling) is computed
    inside that walk from a register copy of the state, so it costs one node,
    not a 0.4 ms state read. The causal conv reads each node's ancestors by index.

The same pass also runs the MTP layer over every node (same tree mask), pairing
each node's hidden state with the target's prediction there. For whichever node
turns out to be the last accepted one, that is exactly the MTP step that drafts
the next root, so the next step's root candidates and their probabilities come
back with this step's single sync: the fork decision needs no sync of its own.

decode_fork drafts the tree (the head's top root candidates, each extended by a
greedy chain, drafted as a tree on the shared MTP cache in one lazy graph that
also yields the main chain's runner-up tokens for siblings), verifies it as above and commits the best path
(attention KV: trim or gather; GatedDeltaNet: replay the path's recorded
inputs). ForkPolicy picks the main chain's depth, which root alternatives to
add and how many siblings, from calibrated root probabilities and learned
rates, against the measured cost of the pass.

With FUSED (default), fused_ops kernels replace mlx_lm's op-by-op GatedDeltaNet
prep, the residual-add + RMS-norm pairs and the commit replay (~600 fewer
dispatches per step at ~2 us each), and the attention mask is additive so MLX's
fused SDPA path is taken.
"""

from __future__ import annotations

import math
import statistics
import time
from collections import Counter
from functools import lru_cache

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.cache import KVCache
from mlx_lm.models.gated_delta import compute_g, gated_delta_update

from .fused_ops import add_rms_norm, gdn_commit, gdn_prep
from . import kv8
from . import long_attention
from .long_attention import flash_tail
from .tree_flash import ancestor_bits, tree_flash
from . import tree_spec
from .tree_spec import TOP1_ACCEPT, RANK_SHARE_PRIOR, DepthPolicy, context_blend, interp_curve, prefill, repeat_cache

FUSED = True  # fused_ops kernels in the tree pass (False: mlx_lm's op-by-op path)

# ---------------------------------------------------------------- tree shape


class Tree:
    """Static structure of a node tree given each node's parent (node 0 = root)."""

    def __init__(self, parents: tuple[int, ...]):
        n = len(parents)
        assert parents[0] == -1 and all(0 <= p < i for i, p in enumerate(parents) if i)
        depth = [0] * n
        children = [[] for _ in range(n)]
        for i in range(1, n):
            depth[i] = depth[parents[i]] + 1
            children[parents[i]].append(i)

        def path(i):
            out = []
            while i != -1:
                out.append(i)
                i = parents[i]
            return out[::-1]

        self.n, self.parents, self.depth = n, parents, depth
        self.is_chain = all(p == i - 1 for i, p in enumerate(parents))
        self.paths = [path(i) for i in range(n) if not children[i]]  # every candidate row
        # GatedDeltaNet walks: a leaf whose parent another walk passes through (not
        # as its last node) rides along as that walk's sibling at that step,
        # computed from a register copy of the state: no extra state read.
        rows, sibs = [], []
        for p in self.paths:
            t = len(p) - 1
            slot = next(
                (r for r, rp in enumerate(rows) if t < len(rp) and rp[t - 1] == p[-2] and sibs[r][t] < 0),
                None,
            ) if t >= 1 else None
            if slot is None:
                rows.append(p)
                sibs.append([-1] * len(p))
            else:
                sibs[slot][t] = p[-1]
        lmax = max(len(p) for p in rows)
        owner = [-1] * n
        for r, (p, s) in enumerate(zip(rows, sibs)):
            for i in p + [x for x in s if x >= 0]:
                if owner[i] < 0:
                    owner[i] = r
        self.rows = rows
        anc = np.zeros((n, n), dtype=bool)
        for i in range(n):
            anc[i, path(i)] = True
        # Causal conv window (kernel 4) of each node: 3 ancestors then itself, as
        # indices into concat(conv_state (3 rows), node qkv (n rows)).
        conv = np.zeros((n, 4), dtype=np.int32)
        for i in range(n):
            p = path(i)
            for j in range(4):  # j = distance back from i
                conv[i, 3 - j] = 3 + p[-1 - j] if j < len(p) else 3 - (j - len(p) + 1)
        self.anc = mx.array(anc)
        self.anc_bits = mx.array(ancestor_bits(anc), dtype=mx.uint64) if n <= 64 else None
        self.tail = n  # the cache rows the masks cover: the nodes themselves
        self.depth_arr = mx.array(depth, dtype=mx.int32)
        self.conv_idx = mx.array(conv.reshape(-1))
        self.path_arr = mx.array([p + [-1] * (lmax - len(p)) for p in rows], dtype=mx.int32)
        self.sib_arr = mx.array([s + [-1] * (lmax - len(s)) for s in sibs], dtype=mx.int32)
        self.owner = mx.array(owner, dtype=mx.int32)
        self.lmax = lmax
        self._mask = None

    def mask(self, offset: int, dtype=mx.bfloat16) -> mx.array:
        """(n, offset + n) additive mask: committed context plus own ancestors.
        Additive in the activation dtype: MLX's SDPA takes its fused path for that
        (a boolean mask costs ~0.9 ms more per pass at 8 nodes)."""
        if self._mask is None or self._mask.dtype != dtype:
            self._mask = mx.where(self.anc, mx.array(0.0, dtype=dtype), mx.array(-mx.inf, dtype=dtype))
        if offset == 0:
            return self._mask
        return mx.concatenate([mx.zeros((self.n, offset), dtype=dtype), self._mask], axis=1)


@lru_cache(maxsize=256)
def tree_of(parents: tuple[int, ...]) -> Tree:
    return Tree(parents)


# ---------------------------------------------------------------- GatedDeltaNet over paths

_TREE_GDN_SOURCE = """
    auto n = thread_position_in_grid.z;          // row * Hv + head
    auto r = int(n / Hv);
    auto hv_idx = n % Hv;
    auto hk_idx = hv_idx / (Hv / Hk);
    constexpr int n_per_t = Dk / 32;
    auto dk_idx = thread_position_in_threadgroup.x;
    auto dv_idx = thread_position_in_grid.y;

    // Every row starts from the one committed state: [Hv, Dv, Dk].
    auto i_state = state_in + (hv_idx * Dv + dv_idx) * Dk;
    float state[n_per_t];
    for (int i = 0; i < n_per_t; ++i) {
      state[i] = static_cast<float>(i_state[n_per_t * dk_idx + i]);
    }
    for (int t = 0; t < L; ++t) {
      int node = paths[r * L + t];
      if (node < 0) break;
      int sib = sibs[r * L + t];
      if (sib >= 0) {
        // A leaf sibling of `node`: one step from the same state, not kept.
        auto qs = q + (sib * Hk + hk_idx) * Dk;
        auto ks = k + (sib * Hk + hk_idx) * Dk;
        float gs = static_cast<float>(g[sib * Hv + hv_idx]);
        float kvs = 0.0f;
        for (int i = 0; i < n_per_t; ++i) {
          kvs += state[i] * gs * ks[n_per_t * dk_idx + i];
        }
        kvs = simd_sum(kvs);
        auto ds = (v[(sib * Hv + hv_idx) * Dv + dv_idx] - kvs) * beta[sib * Hv + hv_idx];
        float os = 0.0f;
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          os += (state[i] * gs + ks[s_idx] * ds) * qs[s_idx];
        }
        os = simd_sum(os);
        if (thread_index_in_simdgroup == 0 && owner[sib] == r) {
          y[(sib * Hv + hv_idx) * Dv + dv_idx] = static_cast<InT>(os);
        }
      }
      auto q_ = q + (node * Hk + hk_idx) * Dk;
      auto k_ = k + (node * Hk + hk_idx) * Dk;
      float gv = static_cast<float>(g[node * Hv + hv_idx]);
      float kv_mem = 0.0f;
      for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * dk_idx + i;
        state[i] = state[i] * gv;
        kv_mem += state[i] * k_[s_idx];
      }
      kv_mem = simd_sum(kv_mem);
      auto delta = (v[(node * Hv + hv_idx) * Dv + dv_idx] - kv_mem) * beta[node * Hv + hv_idx];
      float out = 0.0f;
      for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * dk_idx + i;
        state[i] = state[i] + k_[s_idx] * delta;
        out += state[i] * q_[s_idx];
      }
      out = simd_sum(out);
      // Shared nodes are walked by several rows; the first row that has one writes it.
      if (thread_index_in_simdgroup == 0 && owner[node] == r) {
        y[(node * Hv + hv_idx) * Dv + dv_idx] = static_cast<InT>(out);
      }
    }
"""


@lru_cache(maxsize=None)
def _tree_gdn_kernel():
    return mx.fast.metal_kernel(
        name="tree_gated_delta_paths_sibs",
        input_names=["q", "k", "v", "g", "beta", "state_in", "paths", "sibs", "owner", "L"],
        output_names=["y"],
        source=_TREE_GDN_SOURCE,
    )


def tree_gated_delta(q, k, v, g, beta, state, tree: Tree):
    """Outputs of the gated delta recurrence for every node, each along its own path.
    q, k: (1, n, Hk, Dk); v: (1, n, Hv, Dv); g, beta: (1, n, Hv); state: (1, Hv, Dv, Dk)."""
    _, n, hk, dk = k.shape
    hv, dv = v.shape[2:]
    rows = len(tree.rows)
    return _tree_gdn_kernel()(
        inputs=[q, k, v, g, beta, state, tree.path_arr, tree.sib_arr, tree.owner, tree.lmax],
        template=[("InT", q.dtype), ("Dk", dk), ("Dv", dv), ("Hk", hk), ("Hv", hv)],
        grid=(32, dv, rows * hv),
        threadgroup=(32, 4, 1),
        output_shapes=[(1, n, hv, dv)],
        output_dtypes=[q.dtype],
    )[0]


def _gdn_state(gdn, cache, dtype):
    conv_state = cache[0]
    if conv_state is None:
        conv_state = mx.zeros((1, gdn.conv_kernel_size - 1, gdn.conv_dim), dtype=dtype)
    state0 = cache[1]
    if state0 is None:
        state0 = mx.zeros((1, gdn.num_v_heads, gdn.head_v_dim, gdn.head_k_dim), dtype=mx.float32)
    return conv_state, state0


def batch_gdn(gdn, x, parts):
    """The fused tree_gdn for several requests' nodes at once: the projections run
    over all rows of x (1, N, D); each request's rows [r0, r1) get their own conv
    and recurrence from its cache along its tree. parts: (r0, r1, cache, tree, record)."""
    _, n, _ = x.shape
    qkv = gdn.in_proj_qkv(x)
    z = gdn.in_proj_z(x).reshape(1, n, gdn.num_v_heads, gdn.head_v_dim)
    b = gdn.in_proj_b(x)
    a = gdn.in_proj_a(x)
    ys = []
    for r0, r1, cache, tree, record in parts:
        rows = (lambda t: t) if len(parts) == 1 else (lambda t: t[:, r0:r1])  # noqa: E731
        conv_state, state0 = _gdn_state(gdn, cache, x.dtype)
        q, k, v, g, beta = gdn_prep(
            gdn, rows(qkv), conv_state, tree.conv_idx, gdn.conv1d.weight.reshape(gdn.conv_dim, -1), rows(a), rows(b),
        )
        ys.append(tree_gated_delta(q, k, v, g, beta, state0, tree))
        record[id(gdn)] = (conv_state, state0, rows(qkv), q, k, v, g, beta)
        record["conv_idx"] = tree.conv_idx
    y = ys[0] if len(ys) == 1 else mx.concatenate(ys, axis=1)
    return gdn.out_proj(gdn.norm(y, z).reshape(1, n, -1))


def tree_gdn(gdn, x, cache, tree: Tree, record: dict):
    """mlx_lm's GatedDeltaNet over tree nodes; leaves `cache` as it was (commit
    replays the accepted path from what is recorded here)."""
    if FUSED:
        return batch_gdn(gdn, x, [(0, x.shape[1], cache, tree, record)])
    _, n, _ = x.shape
    qkv = gdn.in_proj_qkv(x)
    z = gdn.in_proj_z(x).reshape(1, n, gdn.num_v_heads, gdn.head_v_dim)
    b = gdn.in_proj_b(x)
    a = gdn.in_proj_a(x)
    conv_state, state0 = _gdn_state(gdn, cache, x.dtype)
    window = mx.take(mx.concatenate([conv_state, qkv], axis=1), tree.conv_idx, axis=1)
    window = window.reshape(1, n, gdn.conv_kernel_size, gdn.conv_dim)
    w = gdn.conv1d.weight[:, :, 0].T  # (kernel, conv_dim)
    conv_out = nn.silu((window.astype(mx.float32) * w.astype(mx.float32)).sum(axis=2).astype(x.dtype))

    q, k, v = [
        t.reshape(1, n, h, d)
        for t, h, d in zip(
            mx.split(conv_out, [gdn.key_dim, 2 * gdn.key_dim], -1),
            [gdn.num_k_heads, gdn.num_k_heads, gdn.num_v_heads],
            [gdn.head_k_dim, gdn.head_k_dim, gdn.head_v_dim],
        )
    ]
    inv_scale = k.shape[-1] ** -0.5
    q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
    k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
    beta = mx.sigmoid(b)
    g = compute_g(gdn.A_log, a, gdn.dt_bias)
    y = tree_gated_delta(q, k, v, g, beta, state0, tree)
    record[id(gdn)] = (conv_state, state0, qkv, q, k, v, a, b)
    return gdn.out_proj(gdn.norm(y, z).reshape(1, n, -1))


# ---------------------------------------------------------------- attention over nodes


def batch_attention(attn, x, parts):
    """Gated-q attention (Qwen3-Next / 3.5) over several requests' nodes: the
    q/k/v/o projections and norms run over all rows of x (1, N, D), the rope,
    cache append and attention per request. parts: (r0, r1, cache, positions,
    mask, tree, side), see tree_attention."""
    _, n, _ = x.shape
    h, hkv = attn.num_attention_heads, attn.num_key_value_heads
    queries, gate = mx.split(attn.q_proj(x).reshape(1, n, h, -1), 2, axis=-1)
    gate = gate.reshape(1, n, -1)
    keys = attn.k_norm(attn.k_proj(x).reshape(1, n, hkv, -1))
    values = attn.v_proj(x).reshape(1, n, hkv, -1)
    queries = attn.q_norm(queries)
    outs = []
    for r0, r1, cache, positions, mask, tree, side in parts:
        rows = (lambda t: t) if len(parts) == 1 else (lambda t: t[:, r0:r1])  # noqa: E731
        outs.append(_attn_core(attn, rows(queries), rows(keys), rows(values).transpose(0, 2, 1, 3), cache, positions, mask, tree, side))
    out = outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=1)
    return attn.o_proj(out * mx.sigmoid(gate))


def _attn_core(attn, queries, keys, values, cache, positions, mask, tree, side):
    """One request's attention: queries/keys (1, n, heads, d) before rope, values
    (1, Hkv, n, d); returns (1, n, Hq * d) before the gate."""
    n = queries.shape[1]
    rope = attn.rope.inner if hasattr(attn.rope, "inner") else attn.rope

    def at_depth(t):  # (1, n, heads, d) -> rope each node at its own position -> (1, heads, n, d)
        rows = mx.fast.rope(
            t.reshape(n, t.shape[2], 1, t.shape[3]), rope.dims,
            traditional=rope.traditional, base=rope.base, scale=rope.scale, offset=positions,
        )
        return rows.reshape(1, n, t.shape[2], t.shape[3]).transpose(0, 2, 1, 3)

    if side is not None:
        keys = at_depth(keys)
        prev = side.get(id(cache))
        if prev is not None:
            keys, values = mx.concatenate([prev[0], keys], axis=2), mx.concatenate([prev[1], values], axis=2)
        side[id(cache)] = (keys, values)
        out = _draft_attention(at_depth(queries), cache, keys, values, tree.tail_mask, attn.scale)
        return out.transpose(0, 2, 1, 3).reshape(1, n, -1)
    keys, values = cache.update_and_fetch(at_depth(keys), values)
    queries = at_depth(queries)
    if isinstance(cache, kv8.Q8KVCache):
        out = kv8.attention(queries, cache, attn.scale, mask, tree.anc_bits, tree.tail)
        return out.transpose(0, 2, 1, 3).reshape(1, n, -1)
    # Long contexts: MTPLX's flash kernels for a short chain (tail-causal mask),
    # the tree-mask variant for anything else up to 16 nodes.
    out = None
    if cache.offset >= long_attention.MIN_CONTEXT:
        if tree.is_chain:
            out = flash_tail(queries, cache, attn.scale)
        if out is None and tree.anc_bits is not None:
            out = tree_flash(queries, cache, attn.scale, tree.anc_bits, tree.tail)
    if out is None:
        out = mx.fast.scaled_dot_product_attention(queries, keys, values, scale=attn.scale, mask=mask)
    return out.transpose(0, 2, 1, 3).reshape(1, n, -1)


def tree_attention(attn, x, cache, positions, mask, tree, side: dict | None = None):
    """Gated-q attention over tree nodes appended to `cache`. `tree` is a Tree or a
    DraftLevel (is_chain, anc_bits, tail). With `side` (long drafts), the nodes'
    keys/values go to side[id(cache)] instead of the cache, which is only read
    (_draft_attention)."""
    return batch_attention(attn, x, [(0, x.shape[1], cache, positions, mask, tree, side)])


def _draft_attention(queries, cache, kt, vt, tail_mask, scale: float):
    """Attention of draft queries (1, Hq, n, D) over `cache`'s rows plus draft
    entries kt/vt (1, Hkv, T, D), the queries' own last, visible per the additive
    (n, T) `tail_mask`. Appending draft entries to a long cache inside a lazy
    graph copies the whole buffer per step (the previous step's still-queued
    reads keep it alive, so it cannot be updated in place: ~0.4 ms per MTP step
    at 28K); here the cache is only read, by the flash kernel, and the few
    draft entries are merged into its softmax."""
    n, hq, d = queries.shape[2], queries.shape[1], queries.shape[3]
    hkv = kt.shape[1]
    g = hq // hkv
    out_h, maxs, sums = tree_flash(queries, cache, scale, _zero_bits(n), 0, stats=True)
    m_h = mx.max(maxs, axis=-1)
    l_h = mx.sum(sums * mx.exp(maxs - m_h[..., None]), axis=-1)
    s = (queries.reshape(1, hkv, g * n, d) @ kt.swapaxes(-1, -2)).astype(mx.float32) * scale
    s = s.reshape(1, hkv, g, n, -1) + tail_mask.astype(mx.float32)
    m_t = mx.max(s, axis=-1)
    p = mx.exp(s - m_t[..., None])
    l_t = mx.sum(p, axis=-1).reshape(1, hq, n)
    o_t = (p.reshape(1, hkv, g * n, -1) @ vt.astype(mx.float32)).reshape(1, hq, n, d)
    m_t = m_t.reshape(1, hq, n)
    m = mx.maximum(m_h, m_t)
    w_h, w_t = l_h * mx.exp(m_h - m), mx.exp(m_t - m)
    out = (out_h.astype(mx.float32) * w_h[..., None] + o_t * w_t[..., None]) / (w_h + l_t * w_t)[..., None]
    return out.astype(queries.dtype)


@lru_cache(maxsize=None)
def _zero_bits(n: int) -> mx.array:
    return mx.zeros((n,), dtype=mx.uint64)


def _kv_offset(cache) -> int:
    return next(c.offset for c in cache if isinstance(c, KVCache))


def batch_forward(inner, tokens, parts):
    """Target forward over several requests' tree nodes, concatenated in `tokens`
    (1, N): (logits, post-norm hidden), both (1, N, .). parts: (r0, r1, cache,
    tree, record) per request; matmuls, norms and MLPs run once over all rows."""
    model = inner.model
    att = []
    for r0, r1, cache, tree, record in parts:
        off0 = _kv_offset(cache)
        att.append((r0, r1, cache, tree.depth_arr + off0, tree.mask(off0), tree, record))
    h = model.embed_tokens(tokens)
    layers = model.layers
    if not FUSED:
        assert len(parts) == 1, "batching needs the fused path"
        r0, r1, cache, positions, mask, tree, record = att[0]
        for layer, c in zip(layers, cache):
            if layer.is_linear:
                r = tree_gdn(layer.linear_attn, layer.input_layernorm(h), c, tree, record)
            else:
                r = tree_attention(layer.self_attn, layer.input_layernorm(h), c, positions, mask, tree)
            h = h + r
            h = h + layer.mlp(layer.post_attention_layernorm(h))
        post = model.norm(h)
        return inner.lm_head(post), post
    # Each residual add is fused with the RMS norm that follows it.
    x = layers[0].input_layernorm(h)
    for i, layer in enumerate(layers):
        if layer.is_linear:
            r = batch_gdn(layer.linear_attn, x, [(r0, r1, c[i], tree, rec) for r0, r1, c, _, _, tree, rec in att])
        else:
            r = batch_attention(layer.self_attn, x, [(r0, r1, c[i], pos, m, tree, None) for r0, r1, c, pos, m, tree, _ in att])
        h, x = add_rms_norm(h, r, layer.post_attention_layernorm)
        nxt = layers[i + 1].input_layernorm if i + 1 < len(layers) else model.norm
        h, x = add_rms_norm(h, layer.mlp(x), nxt)
    return inner.lm_head(x), x


def tree_forward(inner, tokens, cache, tree: Tree, record: dict):
    """Target forward over tree nodes: (logits, post-norm hidden), both (1, n, .)."""
    return batch_forward(inner, tokens, [(0, tokens.shape[1], cache, tree, record)])


def tree_mtp(inner, hidden, next_ids, mtp_cache, tree: Tree, variant: str, order: str):
    """_mtp_core over tree nodes: node i's entry pairs its target hidden state with
    the token the target predicts after it. Returns the MTP hidden per node."""
    off0 = mtp_cache[0].offset
    return _mtp_entries(inner, hidden, next_ids, mtp_cache, tree.depth_arr + off0, tree.mask(off0), tree, variant, order)


def _mtp_entries(inner, hidden, next_ids, mtp_cache, positions, mask, tree, variant: str, order: str, side=None):
    """_mtp_core on (1, n) entries appended to mtp_cache (or kept in `side`, see
    tree_attention), visibility from `tree`."""
    return batch_mtp_entries(inner, hidden, next_ids, [(0, hidden.shape[1], mtp_cache, positions, mask, tree, side)], variant, order)


def batch_mtp_entries(inner, hidden, next_ids, parts, variant: str, order: str):
    """_mtp_entries for several requests' entries, concatenated in hidden (1, N, D)
    and next_ids (1, N). parts: (r0, r1, mtp_cache, positions, mask, tree, side)."""
    embeds = inner.model.embed_tokens(next_ids)
    e = inner.mtp.pre_fc_norm_embedding(embeds)
    hh = inner.mtp.pre_fc_norm_hidden(hidden)
    x = inner.mtp.fc(mx.concatenate([e, hh] if order == "embedding_hidden" else [hh, e], axis=-1))
    fc_hidden = x
    for i, layer in enumerate(inner.mtp.layers):
        att = [(r0, r1, c[i], pos, m, tree, side) for r0, r1, c, pos, m, tree, side in parts]
        x = x + batch_attention(layer.self_attn, layer.input_layernorm(x), att)
        x = x + layer.mlp(layer.post_attention_layernorm(x))
    return inner._mixed_hidden(
        variant, previous=hidden, fc_hidden=fc_hidden, pre_norm=x, post_norm=inner.mtp.norm(x), input_embeds=embeds
    )


# ---------------------------------------------------------------- commit


def _keep_kv(c: KVCache, off0: int, nodes: list[int], contiguous: bool, idx) -> None:
    if not contiguous and isinstance(c, kv8.Q8KVCache):
        c.take_rows(off0, idx)
    elif not contiguous:
        k = mx.take(c.keys, idx + off0, axis=2)
        v = mx.take(c.values, idx + off0, axis=2)
        c.keys[..., off0 : off0 + len(nodes), :] = k
        c.values[..., off0 : off0 + len(nodes), :] = v
    c.offset = off0 + len(nodes)


def commit(inner, cache, mtp_cache, record, off0: int, mtp_off0: int, nodes: list[int]) -> None:
    """Keep only `nodes` (an accepted root-to-node path) of a finished tree pass."""
    contiguous = nodes == list(range(len(nodes)))
    idx = mx.array(nodes, dtype=mx.int32)
    pick = (lambda t: t[:, : len(nodes)]) if contiguous else (lambda t: mx.take(t, idx, axis=1))  # noqa: E731
    for layer, c in zip(inner.model.layers, cache):
        if isinstance(c, KVCache):
            _keep_kv(c, off0, nodes, contiguous, idx)
        elif "conv_idx" in record:  # fused pass: one replay kernel per layer
            c.cache = gdn_commit(layer.linear_attn, record[id(layer.linear_attn)], record["conv_idx"], idx, nodes)
        else:
            gdn = layer.linear_attn
            conv_state, state0, qkv, q, k, v, a, b = record[id(gdn)]
            conv_in = mx.concatenate([conv_state, pick(qkv)], axis=1)
            _, state = gated_delta_update(pick(q), pick(k), pick(v), pick(a), pick(b), gdn.A_log, gdn.dt_bias, state0)
            c.cache = [mx.contiguous(conv_in[:, -(gdn.conv_kernel_size - 1) :, :]), state]
    for c in mtp_cache:
        _keep_kv(c, mtp_off0, nodes, contiguous, idx)


# ---------------------------------------------------------------- policy


class ForkPolicy:
    """Chooses each step's tree shape BEFORE drafting: the main chain's depth D_1,
    a chain depth D_r (0 = none) for each of the next `max_branches` root
    candidates, and S rank-2 siblings at main positions 2..S+1.

    With c = P(the head's top root candidate is the target's token | its
    probability p1), s_r = P(root candidate r is | p1), rates a_j = P(main
    position j+1 accepted | position j was), and s2 = P(the head's runner-up is
    the target's token where a main token at depth >= 2 is rejected):
        E = 1 + c (1 + cont(D_1)) + sum_r s_r (1 + cont(D_r))
              + sum_{j=2}^{S+1} c a_1..a_{j-2} (1 - a_{j-1}) s2
    (an accepted sibling is worth one token: it replaces the bonus the target
    would have produced there and adds its own). cont(D) = a_1 + a_1 a_2 + ...
    up to D - 1 terms, from DepthPolicy's rates and contexts. c, s_r and s2 are
    learned from every step (the target's token after the bonus and the head's
    runner-ups are always synced), from tree_spec's measured priors. Cost: the
    measured pass time for (nodes, GDN walks), plus MTP drafting of the longest
    chain, with DepthPolicy's online scale.
    """

    BINS = (0.3, 0.5, 0.7, 0.8, 0.9, 0.95)
    PRIOR_N = 4.0
    SIB_PRIOR = RANK_SHARE_PRIOR[2][0]  # rank-2 share of depth >= 2 rejections

    def __init__(self, max_depth, max_branches, branch_depth, max_sibs, draft, curve, row_ms, long=None):
        """curve/row_ms/draft: measure_tree_curve at a short context (draft = ms per
        MTP call for one chain, and per extra chain); `long`: the same at a long
        context, as (curve, row_ms, draft, short_ctx, long_ctx), blended by the live
        context (set_context) since attention cost grows with it."""
        self.dp = DepthPolicy(max_depth, draft[0])
        self.max_branches, self.branch_depth, self.max_sibs = max_branches, branch_depth, max_sibs
        self.short, self.long = (curve, row_ms, draft), long
        self.curve, self.row_ms, self.draft = curve, row_ms, draft
        nb = len(self.BINS) + 1
        self.top_tries = [0.0] * nb
        self.top_hits = [0.0] * nb
        self.rank_hits = [[0.0] * (max_branches + 2) for _ in range(nb)]  # given a top-1 miss
        self.miss = [0.0] * nb
        self.sib_tries = self.sib_hits = 0.0
        self.shapes: list[tuple] = []
        self.node_cap: int | None = None  # most tree nodes per step (set when batching)

    def _bin(self, p: float) -> int:
        return sum(p >= b for b in self.BINS)

    def cal(self, p1: float) -> float:
        b = self._bin(p1)
        prior = next(acc for lo, acc in reversed(TOP1_ACCEPT) if p1 >= lo)
        return (self.top_hits[b] + self.PRIOR_N * prior) / (self.top_tries[b] + self.PRIOR_N)

    def share(self, p1: float, rank: int) -> float:
        b = self._bin(p1)
        prior = RANK_SHARE_PRIOR[1][rank - 2]  # depth-1 rejections: shares of ranks 2..8
        return (self.rank_hits[b][rank] + self.PRIOR_N * prior) / (self.miss[b] + self.PRIOR_N)

    def s2(self) -> float:
        return (self.sib_hits + self.PRIOR_N * self.SIB_PRIOR) / (self.sib_tries + self.PRIOR_N)

    def set_context(self, ctx: int) -> None:
        """Re-blend the pass-cost curve for the current context length."""
        if self.long is None:
            return
        (cs, rs, ds), (cl, rl, dl, s_ctx, l_ctx) = self.short, self.long
        nodes = [n for n, _ in cs]
        self.curve = [(n, context_blend(interp_curve(cs, n), interp_curve(cl, n), ctx, s_ctx, l_ctx)) for n in nodes]
        self.row_ms = context_blend(rs, rl, ctx, s_ctx, l_ctx)
        self.draft = tuple(context_blend(a, b, ctx, s_ctx, l_ctx) for a, b in zip(ds, dl))
        self.dp.draft_call_ms = self.draft[0]

    def _ms(self, tokens: int, tree: bool) -> float:
        """Pass time for `tokens` nodes; `tree`: not a plain chain, which costs the
        measured surcharge (an extra GDN walk; at long context, attention off the
        flash kernels, which only take tail-causal masks)."""
        for (n0, t0), (n1, t1) in zip(self.curve, self.curve[1:]):
            if tokens <= n1:
                return t0 + (t1 - t0) * (max(tokens, n0) - n0) / (n1 - n0) + self.row_ms * tree
        return self.curve[-1][1] + self.row_ms * tree

    def raw_ms(self, shape) -> float:
        d1, alts, s = shape
        return (
            self._ms(1 + d1 + sum(alts) + s, any(alts) or s > 0)
            + (max((d1, *alts)) - 1) * (self.draft[0] + self.draft[1] * sum(d > 0 for d in alts))
        )

    def choose(self, root_top) -> tuple:
        dp = self.dp
        p1 = math.exp(root_top[0][1])
        c = self.cal(p1)
        rates = [dp.rate(dp.ctx, j) for j in range(dp.max_depth)]
        reach = [1.0]  # reach[j] = a_1 ... a_j
        for j in range(1, dp.max_depth):
            reach.append(reach[-1] * rates[j])
        conts = [0.0] + [sum(reach[1:d]) for d in range(1, dp.max_depth + 1)]
        # Value of the sibling at main position j (j >= 2): c * reach[j-2] * (1 - a_{j-1}) * s2.
        s2 = self.s2()
        sib_gain = [0.0] * (dp.max_depth + 1)
        for j in range(2, dp.max_depth + 1):
            sib_gain[j] = sib_gain[j - 1] + c * reach[j - 2] * (1 - rates[j - 1]) * s2
        nb = min(self.max_branches, len(root_top) - 1)
        alts_p = [(1 - c) * self.share(p1, r) for r in range(2, 2 + nb)]
        max_nodes = 1 + dp.max_depth + nb * self.branch_depth + self.max_sibs
        pass_ms = [[self._ms(n, tree) for tree in (False, True)] for n in range(max_nodes + 1)]
        main_e = [1 + c * (1 + conts[d]) for d in range(dp.max_depth + 1)]
        best, best_rate = None, -1.0
        for alts in _product([range(0, self.branch_depth + 1)] * nb):
            e_alt = sum(p * (1 + conts[d]) for p, d in zip(alts_p, alts) if d)
            n_alt, d_alt = sum(alts), max(alts, default=0)
            per_draft = self.draft[0] + self.draft[1] * sum(d > 0 for d in alts)
            ms_row = pass_ms
            for d1 in range(1, dp.max_depth + 1):
                draft_ms = (max(d1, d_alt) - 1) * per_draft
                e_main = main_e[d1] + e_alt
                for s in range(0, min(self.max_sibs, d1 - 1) + 1):
                    if 1 + d1 + n_alt + s > min(self.curve[-1][0], self.node_cap or 10**9):
                        break  # past the measured curve (or the batch's share)
                    rate = (e_main + sib_gain[s + 1]) / (ms_row[1 + d1 + n_alt + s][n_alt > 0 or s > 0] + draft_ms)
                    if rate > best_rate:
                        best, best_rate = (d1, alts, s), rate
        self.shapes.append(best)
        return best

    def observe(self, root_top, target_tok, shape, n_main, sib_seen, sib_hit, step_ms) -> None:
        """root_top: this step's root candidates; target_tok: the target's token after
        the bonus; n_main: drafts accepted along the main chain; sib_seen/sib_hit:
        whether a main token at depth >= 2 was rejected, and if so whether the
        head's runner-up there was the target's token."""
        p1 = math.exp(root_top[0][1])
        b = self._bin(p1)
        self.top_tries[b] += 1
        cands = [t for t, _ in root_top]
        if cands[0] == target_tok:
            self.top_hits[b] += 1
        else:
            self.miss[b] += 1
            if target_tok in cands[1 : 1 + self.max_branches]:
                self.rank_hits[b][cands.index(target_tok) + 1] += 1
        if sib_seen:
            self.sib_tries += 1
            self.sib_hits += sib_hit
        d1 = shape[0]
        dp = self.dp
        if n_main >= 1:  # continuation rates: main positions after the first
            for j in range(dp.max_depth):
                dp.tries[dp.ctx][j] *= dp.DECAY
                dp.hits[dp.ctx][j] *= dp.DECAY
            for j in range(1, min(n_main + 1, d1)):
                dp.tries[dp.ctx][j] += 1
                dp.hits[dp.ctx][j] += j < n_main
        dp.ctx = 2 if n_main == d1 else (1 if n_main > 0 else 0)
        if step_ms is not None:  # None when batched: the step's time is shared
            dp.cost.scale = 0.8 * dp.cost.scale + 0.2 * step_ms / self.raw_ms(shape)


def _product(options):
    if not options:
        yield ()
        return
    for x in options[0]:
        for rest in _product(options[1:]):
            yield (x, *rest)


# ---------------------------------------------------------------- cost curve


def measure_tree_curve(rt, inner, drafter, tok, points=(2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24), context=None):
    """Time a whole tree pass (target + MTP over the nodes + root top-k) for chains of
    n nodes, and the extra cost of a second path at equal node count, on a
    ~200-token context or on the given `context` tokens."""
    prompt = context or tok.apply_chat_template(
        [{"role": "user", "content": "Describe a long walk through a forest. " * 20}],
        add_generation_prompt=True, enable_thinking=False,
    )
    base, mtp = prefill(rt, inner, prompt)[:2]
    mx.eval([c.state for c in base], [c.state for c in mtp])

    def ms(parents):
        tree = tree_of(parents)
        ts = []
        for i in range(6):
            c, mc = repeat_cache(base, 1, spare=256), repeat_cache(mtp, 1, spare=256)
            mx.eval([s for x in c + mc for s in (x.state or []) if s is not None])
            mx.synchronize()
            t = time.perf_counter()
            mx.eval(_pass(rt, inner, drafter, mx.array([[prompt[i % len(prompt)] for i in range(tree.n)]]), c, mc, tree, {})[:3])
            mx.synchronize()
            if i >= 2:
                ts.append((time.perf_counter() - t) * 1e3)
        return statistics.median(ts)

    chain = lambda n: (-1,) + tuple(range(n - 1))  # noqa: E731
    curve, best = [], 0.0
    for n in points:
        best = max(best, ms(chain(n)))
        curve.append((n, best))
    # Same 8 nodes as a 5-chain plus a 3-chain off the root: the surcharge for any tree.
    forked = (-1, 0, 1, 2, 3, 0, 5, 6)
    row_ms = max(0.0, ms(forked) - ms(chain(8)))

    # Drafting: per MTP call for one chain, and per extra chain drafted alongside
    # it as a tree (more rows per call; the history is still read once).
    g0 = mx.zeros((1, inner.model.norm.weight.shape[0]), dtype=mx.bfloat16)
    mc = repeat_cache(mtp, 1, spare=256)

    def draft_ms(b, depth=5):
        ts = []
        for i in range(6):
            mx.synchronize()
            t = time.perf_counter()
            toks = draft_tree(drafter, inner, mc, mx.array(prompt[:b], dtype=mx.uint32), mx.repeat(g0, b, axis=0), [depth] * b)[0]
            mx.eval(toks)
            mx.synchronize()
            if i >= 2:
                ts.append((time.perf_counter() - t) * 1e3 / (depth - 1))
        return statistics.median(ts)

    one = draft_ms(1)
    return curve, row_ms, (one, max(0.0, (draft_ms(3) - one) / 2))


# ---------------------------------------------------------------- decoding

TOP_K = 4  # root candidates kept per node


def choose(logits: mx.array, temperature: float = 0.0) -> mx.array:
    """The target's token per row: greedy, or sampled at `temperature`. A drafted
    token is accepted only if it equals this choice at its parent, so sampled
    output has exactly the target's own distribution (drafts stay greedy)."""
    if temperature <= 0:
        return mx.argmax(logits, axis=-1)
    return mx.random.categorical(logits.astype(mx.float32) * (1.0 / temperature), axis=-1)


def _pass(rt, inner, drafter, tokens, cache, mtp_cache, tree: Tree, record: dict, temperature: float = 0.0):
    """One lazy tree pass: target preds per node, MTP hidden per node, and the
    top-K next-root candidates (ids, log-probs) per node."""
    return batch_pass(rt, inner, drafter, tokens, [(0, tokens.shape[1], cache, mtp_cache, tree, record, temperature)])


def batch_pass(rt, inner, drafter, tokens, parts):
    """_pass for several requests' trees, concatenated in `tokens` (1, N); parts:
    (r0, r1, cache, mtp_cache, tree, record, temperature). Row-indexed outputs."""
    logits, post = batch_forward(inner, tokens, [(r0, r1, c, tree, rec) for r0, r1, c, _, tree, rec, _ in parts])
    temps = [t for *_, t in parts]
    if len(parts) == 1 or not any(temps):
        preds = choose(logits[0], temps[0])
    else:
        row_t = mx.concatenate([mx.full((r1 - r0,), t, dtype=mx.float32) for r0, r1, *_, t in parts])
        sampled = mx.random.categorical(logits[0].astype(mx.float32) * (1.0 / mx.maximum(row_t, 1e-6))[:, None], axis=-1)
        preds = mx.where(row_t > 0, sampled, mx.argmax(logits[0], axis=-1))
    mparts = []
    for r0, r1, _, mc, tree, _, _ in parts:
        off0 = mc[0].offset
        mparts.append((r0, r1, mc, tree.depth_arr + off0, tree.mask(off0), tree, None))
    g = batch_mtp_entries(inner, post, preds[None], mparts, drafter.variant, drafter.concat_order)
    z = drafter._logits(g[0]).astype(mx.float32)
    logp = z - mx.logsumexp(z, axis=-1, keepdims=True)
    idx = mx.argpartition(-logp, kth=TOP_K - 1, axis=-1)[:, :TOP_K]
    vals = mx.take_along_axis(logp, idx, axis=-1)
    order = mx.argsort(-vals, axis=-1)
    idx, vals = mx.take_along_axis(idx, order, -1), mx.take_along_axis(vals, order, -1)
    if drafter.ids_arr is not None:
        idx = drafter.ids_arr[idx]
    return preds, idx, vals, g


class DraftLevel:
    """One level of a branch draft: the next entry of every chain still growing,
    appended to the MTP cache after the earlier draft entries (`owners`: the chain
    of every draft entry so far, this level's n last). Each entry sees the
    committed MTP history plus its own chain's entries."""

    is_chain = False

    def __init__(self, owners: tuple[int, ...], n: int):
        vis = np.array([[o == b for o in owners] for b in owners[-n:]])
        self.n, self.tail = n, len(owners)
        self.anc_bits = mx.array(ancestor_bits(vis), dtype=mx.uint64) if self.tail <= 64 else None
        self.tail_mask = mx.where(mx.array(vis), mx.array(0.0, dtype=mx.bfloat16), mx.array(-mx.inf, dtype=mx.bfloat16))

    def mask(self, offset: int) -> mx.array:
        """(n, offset + tail) additive mask, `offset` = committed history length."""
        return mx.concatenate([mx.zeros((self.n, offset), dtype=self.tail_mask.dtype), self.tail_mask], axis=1)


@lru_cache(maxsize=1024)
def draft_level(owners: tuple[int, ...], n: int) -> DraftLevel:
    return DraftLevel(owners, n)


def draft_tree(drafter, inner, mtp_cache, roots, g0, lens: list[int], runner_ups: bool = True):
    """Greedy chains of lens[b] tokens from root tokens `roots` (B,) with MTP hidden
    g0 (B, D), drafted as one tree on the shared mtp_cache, as one lazy graph:
    level j runs step j of every chain still growing as one batch of entries
    (one read of the history for all of them, no per-chain copy of it). From
    long_attention.MIN_CONTEXT on, the entries live beside the cache (see
    _draft_attention); below it they are appended and trimmed again. Either way
    mtp_cache is left as it was. Returns the drafted tokens after the roots in
    entry order (E,), each entry's chain, the head's runner-up at each of chain
    0's entries (lens[0] - 1,; empty without `runner_ups`) and the last level's
    MTP hidden."""
    return batch_draft(drafter, inner, [(mtp_cache, roots, g0, lens, runner_ups)])[0]


def batch_draft(drafter, inner, jobs):
    """draft_tree for several requests: each level drafts the growing chains of
    every request as one batch of MTP entries (MTP layer and output layer run
    once, attention over each request's own history). jobs: (mtp_cache, roots,
    g0, lens, runner_ups); returns draft_tree's result for each."""
    st = []
    for mtp_cache, roots, g0, lens, runner_ups in jobs:
        base = mtp_cache[0].offset
        st.append(dict(
            mc=mtp_cache, lens=lens, ru=runner_ups, base=base, side={} if base >= long_attention.MIN_CONTEXT else None,
            owners=(), firsts=[], seconds=[], act=list(range(len(lens))), hidden=g0, ids=roots, r0=0,
        ))
    for j in range(1, max(max(x["lens"]) for x in st)):
        live = [x for x in st if max(x["lens"]) > j]
        parts, hs, ids, r0 = [], [], [], 0
        for x in live:
            keep = [i for i, b in enumerate(x["act"]) if x["lens"][b] > j]
            if len(keep) < len(x["act"]):
                sel = mx.array(keep, dtype=mx.int32)
                x["act"], x["hidden"], x["ids"] = [x["act"][i] for i in keep], x["hidden"][sel], x["ids"][sel]
            x["owners"] += tuple(x["act"])
            n = len(x["act"])
            level = draft_level(x["owners"], n)
            positions = mx.full((n,), x["base"] + j - 1, dtype=mx.int32)
            mask = None if x["side"] is not None else level.mask(x["base"])
            parts.append((r0, r0 + n, x["mc"], positions, mask, level, x["side"]))
            hs.append(x["hidden"])
            ids.append(x["ids"])
            x["r0"], r0 = r0, r0 + n
        one = len(live) == 1
        hidden = hs[0] if one else mx.concatenate(hs)
        g = batch_mtp_entries(
            inner, hidden[None], (ids[0] if one else mx.concatenate(ids))[None], parts, drafter.variant, drafter.concat_order,
        )[0]
        z = drafter._logits(g)
        first = mx.argmax(z, axis=-1)
        for x in live:
            a, b = x["r0"], x["r0"] + len(x["act"])
            if x["ru"] and x["act"][0] == 0:
                cols = mx.arange(z.shape[-1])
                second = mx.argmax(mx.where(cols == first[a], -mx.inf, z[a]), keepdims=True)
                x["seconds"].append(drafter.ids_arr[second] if drafter.ids_arr is not None else second)
            f = first if one else first[a:b]
            if drafter.ids_arr is not None:
                f = drafter.ids_arr[f]
            f = f.astype(mx.uint32)
            x["firsts"].append(f)
            x["hidden"], x["ids"] = (g if one else g[a:b]), f
    out = []
    for x in st:
        for c in x["mc"]:
            c.offset = x["base"]
        second = mx.concatenate(x["seconds"]).astype(mx.uint32) if x["seconds"] else mx.zeros((0,), dtype=mx.uint32)  # main chain of 1
        out.append((mx.concatenate(x["firsts"]), x["owners"], second, x["hidden"]))
    return out


class ForkState:
    """One request's tree-speculative decode, advanced a step at a time by
    batch_step (plan the shape, draft, build the tree, pass, accept and commit),
    so decode_fork and a batching server run the same per-request logic."""

    def __init__(self, inner, drafter, max_new: int, eos, state, policy: ForkPolicy, temperature: float = 0.0):
        self.inner = inner
        self.cache, self.mtp_cache, self.bonus, self.root_g, self.prefill_s = state
        self.root_top = drafter.top(self.root_g, TOP_K)[0]
        self.max_new, self.eos, self.policy, self.temperature = max_new, eos, policy, temperature
        self.out = [self.bonus]
        self.steps = self.accepted = self.rows = self.nodes = self.alt_wins = self.sib_wins = 0

    @property
    def done(self) -> bool:
        return len(self.out) >= self.max_new or self.bonus in self.eos

    def plan(self, node_cap: int | None = None):
        """Choose this step's shape; returns the batch_draft job, or None when no
        chain needs drafting past its root."""
        kv8.quantize(self.cache)
        self.policy.set_context(_kv_offset(self.cache))
        self.policy.node_cap = node_cap
        d1, alts, _ = self.shape = self.policy.choose(self.root_top)
        self.ranks = [0] + [r + 1 for r, d in enumerate(alts) if d > 0]  # root candidate index per chain
        self.lens = [d1] + [d for d in alts if d > 0]
        self.roots = mx.array([self.root_top[r][0] for r in self.ranks], dtype=mx.uint32)
        if max(self.lens) > 1:
            return (self.mtp_cache, self.roots, mx.repeat(self.root_g, len(self.ranks), axis=0), self.lens, True)
        return None

    def build(self, drafted=None) -> None:
        """The step's tree and its (lazy) tokens from plan() and batch_draft's
        result for its job. Nodes: 0 = bonus; main chain 1..d1; the other
        chains; then the siblings (runner-ups along the main chain)."""
        s = self.shape[2]
        src, owners, second = [mx.array([self.bonus], dtype=mx.uint32), self.roots], (), None
        if drafted is not None:
            tokens, owners, second, _ = drafted
            src += [tokens, second]
        parents, take, self.first_rank = [-1], [0], {}
        for b, (r, d) in enumerate(zip(self.ranks, self.lens)):
            start = len(parents)
            self.first_rank[start] = r + 1
            parents += [0] + list(range(start, start + d - 1))
            take += [1 + b] + [1 + len(self.ranks) + e for e, o in enumerate(owners) if o == b]
        self.sib0 = len(parents)
        parents += [j - 1 for j in range(2, s + 2)]  # runner-up at main position j: child of main node j - 1
        take += [1 + len(self.ranks) + len(owners) + i for i in range(s)]
        self.tree = tree_of(tuple(parents))
        self.tokens = mx.take(mx.concatenate(src), mx.array(take, dtype=mx.int32))
        self.main_seconds = second
        self.off0, self.mtp_off0, self.record = _kv_offset(self.cache), self.mtp_cache[0].offset, {}

    def finish(self, p: list[int], t: list[int], top_ids, top_lp, g, r0: int, step_ms: float | None) -> None:
        """Accept the longest matching path of the finished pass (this request's
        preds `p` and tokens `t`; its rows of the pass outputs start at r0),
        commit it and learn from the step."""
        tree, d1 = self.tree, self.shape[0]
        best_path, best_n = None, -1
        for path in tree.paths:
            n = 0
            while n + 1 < len(path) and t[path[n + 1]] == p[path[n]]:
                n += 1
            if n > best_n:
                best_path, best_n = path, n
        path = best_path[: best_n + 1]
        new_bonus = p[path[-1]]
        commit(self.inner, self.cache, self.mtp_cache, self.record, self.off0, self.mtp_off0, path)
        mx.async_eval([c.state for c in self.cache], [c.state for c in self.mtp_cache])  # replay runs while we plan
        last = r0 + path[-1]
        new_root_top = list(zip(top_ids[last].tolist(), top_lp[last].tolist()))
        self.root_g = g[0, last : last + 1]

        # Statistics: the main chain is always verified, so its outcome is known.
        n_main = 0
        while n_main < d1 and t[n_main + 1] == p[n_main]:
            n_main += 1
        sib_seen = 1 <= n_main < d1 and self.main_seconds is not None
        sib_hit = sib_seen and self.main_seconds.tolist()[n_main - 1] == p[n_main]
        if len(path) > 1:
            self.alt_wins += self.first_rank.get(path[1], 1) > 1
            self.sib_wins += path[-1] >= self.sib0
        self.policy.observe(self.root_top, p[0], self.shape, n_main, sib_seen, sib_hit, step_ms)

        self.out.extend([t[i] for i in path[1:]] + [new_bonus])
        self.root_top, self.bonus = new_root_top, new_bonus
        self.steps += 1
        self.accepted += best_n
        self.rows += len(tree.rows)
        self.nodes += tree.n - 1

    def result(self, decode_s: float) -> tuple[list[int], dict]:
        out = self.out
        for i, tk in enumerate(out):
            if tk in self.eos:
                out = out[: i + 1]
                break
        out = out[: self.max_new]
        s_ = max(self.steps, 1)
        fmt = lambda sh: f"{sh[0]}" + "".join(f"+{d}" for d in sh[1]) + (f"/s{sh[2]}" if sh[2] else "")  # noqa: E731
        return out, dict(
            steps=self.steps,
            decode_s=decode_s,
            prefill_s=self.prefill_s,
            tokens_per_step=(self.accepted + self.steps) / s_,
            mean_rows=self.rows / s_,
            mean_nodes=self.nodes / s_,
            passes_per_step=1.0,
            verify_tokens_per_step=self.nodes / s_ + 1,
            draft_s=0.0,
            verify_s=decode_s,
            alt_win_rate=self.alt_wins / s_,
            sib_win_rate=self.sib_wins / s_,
            shape_hist={fmt(k): v for k, v in Counter(self.policy.shapes).most_common(4)},
        )


def batch_caps(n: int) -> int | None:
    """Tree nodes per request when n requests share a pass: the packed matmuls cost
    about the same for 1-16 rows, then step up at 32 and 48, so the batch aims
    at those totals. None (the policy's own limit) for one request."""
    if n <= 1:
        return None
    total = 16 if n <= 2 else (32 if n <= 4 else 48)
    return max(4, total // n)


def batch_step(rt, inner, drafter, states: list[ForkState]) -> None:
    """One step of every request in `states` (none done): their drafts as one
    batch, their trees as one target + MTP pass, one CPU sync, then each
    request's accept and commit."""
    ta = time.perf_counter()
    cap = batch_caps(len(states))
    jobs = [st.plan(cap) for st in states]
    todo = [i for i, j in enumerate(jobs) if j is not None]
    drafted = dict(zip(todo, batch_draft(drafter, inner, [jobs[i] for i in todo]))) if todo else {}
    for i, st in enumerate(states):
        st.build(drafted.get(i))
    tokens = (states[0].tokens if len(states) == 1 else mx.concatenate([st.tokens for st in states]))[None]
    parts, r0 = [], 0
    for st in states:
        parts.append((r0, r0 + st.tree.n, st.cache, st.mtp_cache, st.tree, st.record, st.temperature))
        r0 += st.tree.n
    mx.async_eval(tokens)  # the GPU drafts while the pass graph is built
    preds, top_ids, top_lp, g = batch_pass(rt, inner, drafter, tokens, parts)
    mx.eval(preds, tokens, top_ids, top_lp, *[st.main_seconds for st in states if st.main_seconds is not None])  # one sync
    p, t = preds.tolist(), tokens[0].tolist()
    step_ms = (time.perf_counter() - ta) * 1e3 if len(states) == 1 else None
    for (r0, r1, *_), st in zip(parts, states):
        st.finish(p[r0:r1], t[r0:r1], top_ids, top_lp, g, r0, step_ms)


def decode_fork(
    rt, inner, drafter, prompt, max_new, eos, *, depth, branches, branch_depth, sibs, curve, row_ms, draft,
    long=None, state=None, temperature: float = 0.0,
):
    """Tree-speculative decoding of `prompt`. `state` = (cache, mtp_cache, bonus,
    root_g, prefill_s) from an outside prefill (e.g. resumed from a prompt-prefix
    snapshot); `temperature` > 0 samples (see choose)."""
    policy = ForkPolicy(depth, branches, branch_depth, sibs, draft, curve, row_ms, long)
    st = ForkState(inner, drafter, max_new, eos, state or prefill(rt, inner, prompt), policy, temperature)
    mx.eval([c.state for c in st.cache + st.mtp_cache])
    t_first = time.perf_counter()
    while not st.done:
        batch_step(rt, inner, drafter, [st])
    return st.result(time.perf_counter() - t_first)

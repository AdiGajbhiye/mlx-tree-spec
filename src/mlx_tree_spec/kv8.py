"""8-bit KV cache for the target's full-attention layers at long context.

At 32K context a 1-token decode step reads the 16 attention layers' bf16 KV
cache (2.1 GB) and little else changes with context, so the KV bytes set the
floor. Q8KVCache stores each K/V row (one token, one KV head, 256 dims) as int8
with one fp32 scale (MTPLX's kv_quant format: symmetric, max-abs / 127), i.e.
260 bytes instead of 512.

Prefill and short contexts keep the bf16 cache; quantize() swaps every
attention layer's cache for a Q8KVCache once the context reaches MIN_CONTEXT,
from where every attention call already goes through the flash kernels:
tree_flash's int8 variant reads the int8 rows directly (key scales multiply the
scores, value scales the softmax weights) for 1-16 queries with any ancestor
mask, so chain verification and tree passes use it; a single query (plain
decoding) goes to MTPLX's int8 vector kernel, which is faster there. Anything
else falls back to dequantizing the cache (slow, but correct).

Measured at 32K over 16 layers (ms per pass, bf16 -> int8): 1 query 5.9 -> 5.0,
2 queries 6.8 -> 5.5, but 4 queries 7.7 -> 9.3, 8 queries 11.0 -> 13.2 and 16
queries 20.4 -> 26.1: from 4 queries on the flash kernel is bound by its matmul
and softmax work, not by KV bytes (bf16 at 8 queries moves ~195 GB/s), and the
int8 path adds work (converting in fp16 with integer tricks or splitting keys
across simdgroups did not change that). So this saves memory (1 GB at 32K) and
a little on 1-2-query passes, and costs time on wider verify passes.
"""

from __future__ import annotations

from collections import Counter

import mlx.core as mx
from mlx_lm.models.cache import KVCache

MIN_CONTEXT = 8192  # long_attention's threshold: below it attention stays on MLX's bf16 kernels
STATS: Counter = Counter()
ENABLED = False


def quantize_rows(x: mx.array) -> tuple[mx.array, mx.array]:
    """(..., D) -> int8 rows (..., D) and fp32 scales (..., 1)."""
    xf = x.astype(mx.float32)
    s = mx.maximum(mx.max(mx.abs(xf), axis=-1, keepdims=True) / 127.0, 1e-6)
    return mx.clip(mx.round(xf / s), -127, 127).astype(mx.int8), s


# One simdgroup per row (K and V rows of every head and token): max-abs over the
# 256 values (8 per lane), then the int8 row and its scale, as quantize_rows.
_QUANT_SOURCE = """
const int row = int(threadgroup_position_in_grid.y);
const int lane = int(thread_position_in_threadgroup.x);
const bool is_v = row >= ROWS;
const device T* src = (is_v ? values : keys) + (row % ROWS) * D + lane * (D / 32);
float x[D / 32];
float m = 0.0f;
for (int i = 0; i < D / 32; ++i) {
  x[i] = float(src[i]);
  m = metal::max(m, metal::abs(x[i]));
}
const float s = metal::max(simd_max(m) / 127.0f, 1e-6f);
device int8_t* dst = (is_v ? vq : kq) + (row % ROWS) * D + lane * (D / 32);
for (int i = 0; i < D / 32; ++i) dst[i] = int8_t(metal::clamp(metal::rint(x[i] / s), -127.0f, 127.0f));
if (lane == 0) (is_v ? vs : ks)[row % ROWS] = s;
"""


def _quant_kernel(rows: int, d: int, dtype):
    key = (rows, d, dtype)
    if key not in _QUANT_KERNELS:
        msl = {mx.bfloat16: "bfloat16_t", mx.float16: "half", mx.float32: "float"}[dtype]
        _QUANT_KERNELS[key] = mx.fast.metal_kernel(
            name=f"kv8_quant_{rows}_{d}_{msl}",
            input_names=["keys", "values"],
            output_names=["kq", "ks", "vq", "vs"],
            source=f"typedef {msl} T;\nconstexpr int ROWS = {rows};\nconstexpr int D = {d};\n" + _QUANT_SOURCE,
        )
    return _QUANT_KERNELS[key]


_QUANT_KERNELS: dict = {}


def quantize_kv(keys: mx.array, values: mx.array):
    """quantize_rows of keys and values (1, H, n, D) in one dispatch: (kq, ks, vq, vs)."""
    b, h, n, d = keys.shape
    rows = b * h * n
    return _quant_kernel(rows, d, keys.dtype)(
        inputs=[keys, values],
        grid=(32, 2 * rows, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[keys.shape, (b, h, n, 1), keys.shape, (b, h, n, 1)],
        output_dtypes=[mx.int8, mx.float32, mx.int8, mx.float32],
    )


class Q8KVCache(KVCache):
    """KVCache whose keys/values are int8 rows, each with an fp32 scale in
    k_scale/v_scale (1, H, capacity, 1). offset and trim work as in KVCache."""

    def __init__(self):
        super().__init__()
        self.k_scale = self.v_scale = None

    @classmethod
    def from_cache(cls, c: KVCache, spare: int = 256) -> "Q8KVCache":
        n = cls()
        n.offset = c.offset
        room = -(-(c.offset + spare) // n.step) * n.step - c.offset  # capacity a multiple of step (the kernel reads scales in 32-row tiles)
        pad = lambda t: mx.concatenate([t, mx.zeros((*t.shape[:2], room, t.shape[3]), dtype=t.dtype)], axis=2)  # noqa: E731
        k, ks = quantize_rows(c.keys[..., : c.offset, :])
        v, vs = quantize_rows(c.values[..., : c.offset, :])
        n.keys, n.values, n.k_scale, n.v_scale = pad(k), pad(v), pad(ks), pad(vs)
        return n

    def _grow(self, need: int) -> None:
        cap = ((need + self.step - 1) // self.step) * self.step
        extra = lambda t: mx.zeros((*t.shape[:2], cap - t.shape[2], t.shape[3]), dtype=t.dtype)  # noqa: E731
        self.keys, self.values, self.k_scale, self.v_scale = (
            mx.concatenate([t[..., : self.offset, :], extra(t[..., : self.offset, :])], axis=2)
            for t in (self.keys, self.values, self.k_scale, self.v_scale)
        )

    def update_and_fetch(self, keys, values):
        prev, n = self.offset, keys.shape[2]
        if prev + n > self.keys.shape[2]:
            self._grow(prev + n + self.step)
        kq, ks, vq, vs = quantize_kv(keys, values)
        self.offset += n
        self.keys[..., prev : self.offset, :] = kq
        self.values[..., prev : self.offset, :] = vq
        self.k_scale[..., prev : self.offset, :] = ks
        self.v_scale[..., prev : self.offset, :] = vs
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]

    @property
    def state(self):
        o = self.offset
        return self.keys[..., :o, :], self.values[..., :o, :], self.k_scale[..., :o, :], self.v_scale[..., :o, :]

    def dequantized(self, dtype=mx.bfloat16) -> tuple[mx.array, mx.array]:
        k, v, ks, vs = self.state
        return (k * ks).astype(dtype), (v * vs).astype(dtype)

    def take_rows(self, start: int, idx: mx.array) -> None:
        """Move rows start + idx to start, start + 1, ... (a tree commit)."""
        for name in ("keys", "values", "k_scale", "v_scale"):
            t = getattr(self, name)
            t[..., start : start + idx.size, :] = mx.take(t, idx + start, axis=2)
            setattr(self, name, t)


def quantize(cache) -> None:
    """Swap (in place) every bf16 attention cache in `cache` that has reached
    MIN_CONTEXT for a Q8KVCache. No-op unless ENABLED."""
    if not ENABLED:
        return
    for i, c in enumerate(cache):
        if type(c) is KVCache and c.keys is not None and c.offset >= MIN_CONTEXT:
            cache[i] = Q8KVCache.from_cache(c)
            STATS["quantized"] += 1


def attention(queries, cache: Q8KVCache, scale: float, mask=None, bits=None, tail=None):
    """Attention of `queries` (1, Hq, n, D) over a Q8KVCache whose last rows are
    theirs: tail-causal (bits None, mask None or "causal") or with ancestor
    bits; anything the kernel cannot take is computed on the dequantized cache
    with `mask`."""
    from mtplx.kernels.sdpa_gqa_packed_quant import sdpa_gqa_packed_tail_quant
    from .tree_flash import tree_flash

    n = queries.shape[2]
    out = None
    if n == 1 and (tail or 1) == 1:  # one query: MTPLX's int8 vector kernel is the fastest
        out = sdpa_gqa_packed_tail_quant(
            queries=queries, k_q=cache.keys, k_scale=cache.k_scale, v_q=cache.values, v_scale=cache.v_scale,
            offset=cache.offset, scale=scale, bits=8,
        )
        STATS["vector" if out is not None else "vector_declined"] += 1
    if out is None and (bits is not None or mask is None or isinstance(mask, str)):
        if bits is None:
            bits = causal_bits(n)
        out = tree_flash(queries, cache, scale, bits, tail)
    if out is None:
        STATS["dequant"] += 1
        k, v = cache.dequantized(queries.dtype)
        if mask is None and n > 1:
            mask = "causal"
        out = mx.fast.scaled_dot_product_attention(queries, k, v, scale=scale, mask=mask)
    return out


_CAUSAL: dict[int, mx.array] = {}


def causal_bits(n: int) -> mx.array:
    if n not in _CAUSAL:
        _CAUSAL[n] = mx.array([(1 << (q + 1)) - 1 for q in range(n)], dtype=mx.uint64)
    return _CAUSAL[n]

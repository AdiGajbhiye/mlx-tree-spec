"""Long-context verify attention: MTPLX's TensorOps flash-decoding kernels.

At 32K context MLX's fused SDPA reads the whole KV cache once per query token
(16 layers: 6 ms for 1 query, 17 ms for 5) and drops to an unfused path from 6
queries on (38-43 ms), so an 8-token verify pass spends ~40 ms in attention.
MTPLX ships flash-decoding kernels on the M5 matrix units that read each KV
block once for all of a head group's queries (sdpa_nax_flash: 11.4 ms for 8
queries; the head-dim-split variant is faster up to 5). They take the cache's
full buffers plus its offset and apply a tail-causal mask, i.e. a chain.

install() routes mlx_lm's scaled_dot_product_attention (used by the stock
attention and by MTPLX's own verify path) to them for 2-10 tail-causal queries
over at least MIN_CONTEXT cached tokens, and to tree_flash's ancestor-mask
variant (with a causal mask) for 11-16; tree_verify calls them directly.
Anything else (1 query, prefill chunks) keeps MLX's kernels, except that an
8-bit cache (kv8.Q8KVCache) always goes to kv8.attention.
"""

from __future__ import annotations

import os
from collections import Counter
from functools import lru_cache

import mlx.core as mx

from . import kv8

MIN_CONTEXT = 8192  # MTPLX's own threshold for these kernels
STATS: Counter = Counter()
_ORIG = None


def flash_tail(queries, cache, scale: float):
    """Tail-causal attention of `queries` (1, Hq, L, D) over `cache`'s first
    cache.offset rows (the queries' own keys already appended), or None."""
    from mtplx.kernels.sdpa_nax_flash import sdpa_nax_flash
    from mtplx.kernels.sdpa_nax_flash_dsplit import sdpa_nax_flash_dsplit
    from .tree_flash import flash_blocks

    keys = getattr(cache, "keys", None)
    n = queries.shape[2]
    if keys is None or queries.shape[0] != 1 or cache.offset < MIN_CONTEXT:
        return None
    if 10 < n <= 16:  # past MTPLX's window: the tree-mask kernel with a causal mask
        from .tree_flash import tree_flash

        out = tree_flash(queries, cache, scale, _causal_bits(n))
        STATS["tree_flash" if out is not None else "declined"] += 1
        return out
    if not 2 <= n <= 10:
        return None
    # Block counts as tree_flash.flash_blocks (MTPLX's kernels read them from the environment per call).
    blocks = str(flash_blocks(keys.shape[2]))
    os.environ["MTPLX_NAX_FLASH_BLOCKS"] = os.environ["MTPLX_NAX_FLASH_DSPLIT_BLOCKS"] = blocks
    fn = sdpa_nax_flash_dsplit if (queries.shape[1] // keys.shape[1]) * n <= 32 else sdpa_nax_flash
    out = fn(queries=queries, keys=keys, values=cache.values, offset=cache.offset, scale=scale)
    if out is None and fn is sdpa_nax_flash_dsplit:
        out = sdpa_nax_flash(queries=queries, keys=keys, values=cache.values, offset=cache.offset, scale=scale)
    STATS["flash" if out is not None else "declined"] += 1
    return out


@lru_cache(maxsize=None)
def _causal_bits(n: int):
    return mx.array([(1 << (q + 1)) - 1 for q in range(n)], dtype=mx.uint64)


def install() -> None:
    """Route mlx_lm's SDPA helper to flash_tail where it applies."""
    global _ORIG
    from mlx_lm.models import base, qwen3_next

    if _ORIG is not None:
        return
    _ORIG = base.scaled_dot_product_attention

    def sdpa(queries, keys, values, cache, scale, mask, *args, **kwargs):
        if isinstance(cache, kv8.Q8KVCache):
            return kv8.attention(queries, cache, scale, mask)
        if cache is not None and (mask is None or isinstance(mask, str)) and not args and not kwargs:
            out = flash_tail(queries, cache, scale)
            if out is not None:
                return out
        return _ORIG(queries, keys, values, cache=cache, scale=scale, mask=mask, *args, **kwargs)

    base.scaled_dot_product_attention = sdpa
    qwen3_next.scaled_dot_product_attention = sdpa

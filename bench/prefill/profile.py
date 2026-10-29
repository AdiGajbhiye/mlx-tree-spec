"""Where one 2048-token prefill chunk's time goes, as the server runs it (packed 4-bit matmuls,
MLX attention for prefill chunks, GatedDeltaNet, then the MTP head over the chunk), at a given
cached context: full-attention layers, GatedDeltaNet layers, FFNs, MTP and the rest. The
breakdown syncs after each module, so its sum is a little above the plain chunk time, which
is timed too. Loads the model (MLX_TREE_SPEC_MODEL, default the Hugging Face one): follow CLAUDE.md's memory
checks. Usage: .venv/bin/python bench/prefill/profile.py [CTX...]   (default 0 8192 24576)"""
import os
import statistics
import sys
import time
from collections import defaultdict

import mlx.core as mx
from mlx_lm.models.cache import KVCache
from mtplx.runtime import load

from mlx_tree_spec import long_attention, packed_linear
from mlx_tree_spec import tree_spec as ts

CHUNK = 2048
times: dict[str, float] = defaultdict(float)
PROFILE = False


def arrays(caches):
    """Every array a list of caches holds (empty KV caches have none)."""
    out = []
    for c in caches:
        if getattr(c, "keys", 0) is None:
            continue
        out += [x for x in (c.state or []) if x is not None]
    return out


_stack: list[float] = []  # child time of each open timed call


def timed(cls, name):
    """Time `cls` calls exclusive of the timed calls inside them (e.g. a layer's matmuls)."""
    orig = cls.__call__

    def call(self, *a, **k):
        if not PROFILE:
            return orig(self, *a, **k)
        mx.eval([x for x in a if isinstance(x, mx.array)])  # pending inputs belong to whoever made them
        mx.synchronize()
        t = time.perf_counter()
        _stack.append(0.0)
        out = orig(self, *a, **k)
        mx.eval(out)
        mx.synchronize()
        dt = time.perf_counter() - t
        child = _stack.pop()
        times[name] += dt - child
        if _stack:
            _stack[-1] += dt
        return out
    cls.__call__ = call


def main():
    global PROFILE
    ctxs = [int(a) for a in sys.argv[1:]] or [0, 8192, 24576]
    mx.set_cache_limit(int(2e9))  # as the server
    mx.set_wired_limit(int(mx.device_info()["max_recommended_working_set_size"]))
    rt = load(ts.resolve_model(os.environ.get("MLX_TREE_SPEC_MODEL", ts.DEFAULT_MODEL)))
    inner = getattr(rt.model, "language_model", rt.model)
    _, drafter = ts.make_drafter(rt, inner, "general")
    packed_linear.install(rt.model, verbose=False)
    packed_linear.install(drafter.head, verbose=False)
    long_attention.install()
    layers = inner.model.layers
    from mlx import nn
    timed(nn.QuantizedLinear, "4-bit matmuls (all)")
    timed(type(layers[3].self_attn), "attention, excl. matmuls (16 layers)")
    timed(type(layers[0].linear_attn), "GatedDeltaNet, excl. matmuls (48 layers)")
    timed(type(layers[0].mlp), "FFN, excl. matmuls (64 layers)")
    tok = rt.tokenizer.encode((ts.DATA / "code_32k.txt").read_text())

    def mtp(hidden, ids, mtp_cache):
        return inner._mtp_core(hidden, mx.array([ids]), mtp_cache=mtp_cache, concat_order=rt.contract.concat_order,
                               mtp_hidden_variant=rt.contract.hidden_variant, emit_logits=False)[1]

    def chunk(cache, mtp_cache, prev_h, i):
        logits, hidden = rt.forward_ar(mx.array([tok[i:i + CHUNK]]), cache=cache, return_hidden=True, logits_keep=1)
        hs = mx.concatenate([prev_h, hidden[:, :-1]], axis=1) if prev_h is not None else hidden[:, :-1]
        ids = tok[i:i + CHUNK] if prev_h is not None else tok[i + 1:i + CHUNK]
        if PROFILE:
            mx.eval(hidden)
            mx.synchronize()
            t = time.perf_counter()
            _stack.append(0.0)
        g = mtp(hs, ids, mtp_cache)
        mx.eval(logits, g, [c.state for c in cache], [c.state for c in mtp_cache])
        if PROFILE:
            mx.synchronize()
            times["MTP head, excl. matmuls"] += time.perf_counter() - t - _stack.pop()
        return hidden[:, -1:]

    print(f"one {CHUNK}-token chunk, ms")
    for ctx in ctxs:
        assert ctx + CHUNK <= len(tok), f"the text has {len(tok)} tokens"
        cache, mtp_cache, prev_h = inner.make_cache(), [KVCache() for _ in rt.model.mtp.layers], None
        for i in range(0, ctx, CHUNK):
            prev_h = chunk(cache, mtp_cache, prev_h, i)
        base = (ts.repeat_cache(cache, 1), ts.repeat_cache(mtp_cache, 1), prev_h)
        mx.eval(arrays(base[0] + base[1]))

        def fresh():
            c, m = ts.repeat_cache(base[0], 1), ts.repeat_cache(base[1], 1)
            mx.eval(arrays(c + m))
            return c, m

        plain = []
        for _ in range(4):
            c, m = fresh()
            mx.synchronize()
            t = time.perf_counter()
            chunk(c, m, base[2], ctx)
            mx.synchronize()
            plain.append((time.perf_counter() - t) * 1e3)
        plain = statistics.median(plain[1:])
        times.clear()
        PROFILE = True
        runs = 3
        for _ in range(runs):
            c, m = fresh()
            mx.synchronize()
            t = time.perf_counter()
            chunk(c, m, base[2], ctx)
            mx.synchronize()
            times["total (profiled)"] += time.perf_counter() - t
        PROFILE = False
        parts = {k: v / runs * 1e3 for k, v in times.items()}
        total = parts.pop("total (profiled)")
        parts["rest (norms, embedding, logits, glue)"] = total - sum(parts.values())
        print(f"\nctx {ctx}: chunk {plain:.0f} ms plain = {CHUNK / plain * 1e3:.0f} tok/s; profiled {total:.0f} ms")
        for k, v in sorted(parts.items(), key=lambda kv: -kv[1]):
            print(f"  {k:40s} {v:7.1f}  {v / total:5.1%}")


if __name__ == "__main__":
    main()

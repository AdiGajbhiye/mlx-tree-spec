"""MTPLX's blocked GatedDeltaNet prefill kernel (MTPLX_GDN_BLOCKED_PREFILL) against mlx-lm's stock
kernel, in one process with the two alternating, so machine drift hits both alike: time of a
2048-token chunk (target forward + MTP) at a few cached contexts, and how much the chunk's
hidden states and GDN states differ. Loads the model: follow CLAUDE.md's memory checks.
Usage: MLX_TREE_SPEC_MODEL=... .venv/bin/python bench/prefill/ab_gdn.py [CTX...]"""
import os
import statistics
import sys
import time

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache
from mtplx.kernels import gdn_blocked_prefill as gbp
from mtplx.runtime import load

from mlx_tree_spec import long_attention, packed_linear
from mlx_tree_spec import tree_spec as ts

CHUNK = 2048


def main():
    ctxs = [int(a) for a in sys.argv[1:]] or [0, 8192, 24576]
    mx.set_cache_limit(int(2e9))
    mx.set_wired_limit(int(mx.device_info()["max_recommended_working_set_size"]))
    rt = load(ts.resolve_model(os.environ.get("MLX_TREE_SPEC_MODEL", ts.DEFAULT_MODEL)))
    inner = getattr(rt.model, "language_model", rt.model)
    _, drafter = ts.make_drafter(rt, inner, "general")
    packed_linear.install(rt.model, verbose=False)
    packed_linear.install(drafter.head, verbose=False)
    long_attention.install()
    tok = rt.tokenizer.encode((ts.DATA / "code_32k.txt").read_text())

    def run(cache, i):
        _, hidden = rt.forward_ar(mx.array([tok[i:i + CHUNK]]), cache=cache, return_hidden=True, logits_keep=1)
        mx.eval(hidden, [x for c in cache for x in (c.state or []) if x is not None])
        return hidden

    for ctx in ctxs:
        gbp.uninstall_gdn_blocked_prefill_patch()
        cache = inner.make_cache()
        for i in range(0, ctx, CHUNK):
            run(cache, i)
        base = ts.repeat_cache(cache, 1) if ctx else None
        times = {"stock": [], "blocked": []}
        outs = {}
        for rep in range(4):
            for name in ("stock", "blocked") if rep % 2 == 0 else ("blocked", "stock"):
                if name == "blocked":
                    gbp.install_gdn_blocked_prefill_patch()
                else:
                    gbp.uninstall_gdn_blocked_prefill_patch()
                c = ts.repeat_cache(base, 1) if base else inner.make_cache()
                mx.eval([x for y in c for x in (y.state or []) if x is not None] if base else [])
                mx.synchronize()
                t = time.perf_counter()
                h = run(c, ctx)
                mx.synchronize()
                if rep:
                    times[name].append((time.perf_counter() - t) * 1e3)
                outs[name] = (h, [y.cache[1] for y in c if isinstance(y, ArraysCache)])
        gbp.uninstall_gdn_blocked_prefill_patch()
        (hs, ss), (hb, sb) = outs["stock"], outs["blocked"]
        dh = mx.abs(hs.astype(mx.float32) - hb.astype(mx.float32)).max().item()
        ds = max(mx.abs(a - b).max().item() for a, b in zip(ss, sb))
        rel = (mx.linalg.norm((hs - hb).astype(mx.float32)) / mx.linalg.norm(hs.astype(mx.float32))).item()
        s, b = statistics.median(times["stock"]), statistics.median(times["blocked"])
        print(f"ctx {ctx:5d}: stock {s:7.1f} ms  blocked {b:7.1f} ms  ({s / b:.3f}x)  "
              f"hidden max |diff| {dh:.3g} (rel {rel:.2e}), GDN state max |diff| {ds:.3g}", flush=True)
        del cache, base, outs


if __name__ == "__main__":
    main()

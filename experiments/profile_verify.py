"""Profile the pieces of a tree_spec.py verify step on Swift 1.5.

Why does a verify step with 2+ rows cost ~115 ms when one row costs ~50-70 ms?
This times each piece in isolation on a fixed context:

  copy     repeat_cache(cache, B)            per-row copies of KV + GDN state
  fwd      Verifier.run(rows, cache)         target forward, B rows x L tokens
  commit   Verifier.commit(...)              KV trim/slice + GDN replay
  draft    MTP head on B rows x d tokens

    python experiments/profile_verify.py
"""

from __future__ import annotations

import statistics
import time

import mlx.core as mx

from mtplx.runtime import load
from mlx_tree_spec.tree_spec import DEFAULT_MODEL, KVCache, Verifier, repeat_cache, resolve_model

CONTEXT = "Explain in three short paragraphs why the sky is blue. " * 20


def timed(fn, reps=8, warmup=2):
    """Median wall time in ms of fn(); fn must return the arrays to evaluate."""
    ts = []
    for i in range(warmup + reps):
        mx.synchronize()
        t = time.perf_counter()
        mx.eval(fn())
        mx.synchronize()
        if i >= warmup:
            ts.append((time.perf_counter() - t) * 1e3)
    return statistics.median(ts)


def state_arrays(cache):
    out = []
    for c in cache:
        out.extend(c.state if isinstance(c, KVCache) else [x for x in c.cache if x is not None])
    return out


def main():
    rt = load(resolve_model(DEFAULT_MODEL))
    inner = getattr(rt.model, "language_model", rt.model)
    ver = Verifier(rt, inner)
    tok = rt.tokenizer
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": CONTEXT}], add_generation_prompt=True, enable_thinking=False
    )
    base = inner.make_cache()
    logits, hidden = rt.forward_ar(mx.array([prompt]), cache=base, return_hidden=True)
    mx.eval(logits, hidden, state_arrays(base))
    ids = mx.argmax(logits[0], axis=-1).tolist()  # plausible tokens to feed
    print(f"context {len(prompt)} tokens\n")

    def rows(b, l):
        return [[ids[(r * 7 + i) % len(ids)] for i in range(l)] for r in range(b)]

    def fwd(b, l, copy=True):
        def f():
            pc = repeat_cache(base, b) if copy else repeat_cache(base, 1)
            preds, h, rec, off0 = ver.run(rows(b, l), pc)
            return [h]
        return f

    def copy_only(b):
        return lambda: state_arrays(repeat_cache(base, b))

    copy_ms = {b: timed(copy_only(b)) for b in (1, 2, 3, 4, 6, 8, 12)}
    print("copy (repeat_cache)      " + "  ".join(f"B={b}: {t:5.1f} ms" for b, t in copy_ms.items()))

    print("\nforward minus copy, B rows x L tokens (ms):")
    print("        " + "".join(f"L={l:<7d}" for l in (1, 2, 4, 6, 12)))
    for b in (1, 2, 3, 4, 8):
        line = []
        for l in (1, 2, 4, 6, 12):
            line.append(timed(fwd(b, l)) - copy_ms[b])
        print(f"B={b:<5d} " + "".join(f"{t:<9.1f}" for t in line))

    print("\nsame total tokens (12), split into rows (forward minus copy, ms):")
    for b, l in ((1, 12), (2, 6), (3, 4), (4, 3), (6, 2), (12, 1)):
        print(f"  {b:2d} x {l:2d}: {timed(fwd(b, l)) - copy_ms[b]:6.1f}")

    # Commit: real-cache trim vs row slice, both with GDN replay of 3 tokens.
    def commit(b):
        def f():
            pc = repeat_cache(base, b)
            preds, h, rec, off0 = ver.run(rows(b, 5), pc)
            mx.eval(h, state_arrays(pc))
            tgt = repeat_cache(base, 1)
            mx.eval(state_arrays(tgt))
            t = time.perf_counter()
            ver.commit(tgt, pc if b > 1 else pc, rec, off0, b - 1, 3)
            mx.eval(state_arrays(tgt))
            f.ms.append((time.perf_counter() - t) * 1e3)
            return []
        f.ms = []
        return f

    print("\ncommit (KV slice/trim + GDN replay of 3 tokens):")
    for b in (1, 2, 4):
        f = commit(b)
        timed(f, reps=6)
        print(f"  B={b}: {statistics.median(f.ms[2:]):5.1f} ms")

    # The first KV append after a row-slice commit reallocates the whole buffer.
    def append_after_slice():
        c = repeat_cache(base, 2)
        mx.eval(state_arrays(c))
        tgt = repeat_cache(base, 1)
        for layer_c, pc in zip(tgt, c):
            if isinstance(layer_c, KVCache):
                k, v = pc.state
                layer_c.state = (k[1:2], v[1:2])
        mx.eval(state_arrays(tgt))
        return [rt.forward_ar(mx.array([[ids[0]]]), cache=tgt, return_hidden=True)[1]]

    def append_plain():
        tgt = repeat_cache(base, 1)
        mx.eval(state_arrays(tgt))
        return [rt.forward_ar(mx.array([[ids[0]]]), cache=tgt, return_hidden=True)[1]]

    print(
        f"\n1-token forward after a row-slice commit: {timed(append_after_slice):5.1f} ms"
        f"   vs plain: {timed(append_plain):5.1f} ms   (both include a batch-1 copy)"
    )

    # MTP drafting: B rows x d tokens through the MTP head (+ its cache copy).
    mtp_cache = [KVCache() for _ in rt.model.mtp.layers]
    _, g = rt.draft_mtp(hidden, mx.array([prompt[1:] + [ids[-1]]]), mtp_cache=mtp_cache, return_hidden=True)
    mx.eval(g)
    print("\nMTP draft call, B rows x d tokens (ms):")
    for b in (1, 2, 4):
        line = []
        for d in (1, 2, 4):
            hid = mx.repeat(g[:, -d:, :], b, axis=0)
            toks = mx.array(rows(b, d))
            line.append(
                timed(lambda: list(rt.draft_mtp(hid, toks, mtp_cache=repeat_cache(mtp_cache, b), return_hidden=True)))
            )
        print(f"  B={b}: " + "  ".join(f"d={d}: {t:5.1f}" for d, t in zip((1, 2, 4), line)))


if __name__ == "__main__":
    main()

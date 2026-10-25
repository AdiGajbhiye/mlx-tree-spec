"""When the target rejects a draft token, where was the target's token in the
MTP head's distribution?

Decodes with a greedy MTP chain (depth 6, so rejections are seen at every
depth) and, at each rejection, ranks the target's token in the draft
distribution at that position. That distribution was conditioned on the
correct prefix (every earlier draft token was accepted), so the rank says
directly whether a top-k branch at that position would have caught it.

    python experiments/rank_probe.py --max-new 300
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter

import mlx.core as mx

from mtplx.runtime import load
from mlx_tree_spec.tree_spec import DEFAULT_MODEL, KVCache, Verifier, mtp_root, repeat_cache, verify_step, resolve_model

PROMPTS = [
    "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring.",
    "Explain in three short paragraphs why the sky is blue.",
    "A train leaves at 9:40 and arrives at 13:05. It stops twice for 12 minutes each. How long was it moving? Show your steps.",
    "List ten interesting facts about octopuses, one sentence each.",
]
BUCKETS = ((2, 2), (3, 3), (4, 4), (5, 8), (9, 32), (33, 10**9))


def draft_chain(rt, mtp_cache, root_logp, root_g, depth):
    """Greedy MTP chain; returns its tokens and the log-prob vector at each depth."""
    dists, toks, hid = [root_logp], [int(mx.argmax(root_logp).item())], [root_g]
    for _ in range(1, depth):
        logits, g = rt.draft_mtp(
            mx.concatenate(hid, axis=0)[None], mx.array([toks]),
            mtp_cache=repeat_cache(mtp_cache, 1), return_hidden=True,
        )
        last = logits[0, -1].astype(mx.float32)
        last = last - mx.logsumexp(last)
        hid.append(g[0, -1:, :])
        dists.append(last)
        toks.append(int(mx.argmax(last).item()))
    return toks, dists


def probe(rt, inner, ver, prompt, max_new, depth, eos):
    cache = inner.make_cache()
    mtp_cache = [KVCache() for _ in rt.model.mtp.layers]
    logits, hidden = rt.forward_ar(mx.array([prompt]), cache=cache, return_hidden=True)
    bonus = int(mx.argmax(logits[0, -1]).item())
    root_logp, root_g = mtp_root(rt, mtp_cache, hidden, mx.array([prompt[1:] + [bonus]]))
    produced, rejections, accepted_p = 1, [], []
    while produced < max_new and bonus not in eos:
        toks, dists = draft_chain(rt, mtp_cache, root_logp, root_g, depth)
        acc, new_bonus, h, _ = verify_step(ver, cache, bonus, [toks], share_min=10**9)
        n = len(acc)
        accepted_p += [math.exp(dists[j][toks[j]].item()) for j in range(n)]
        if n < depth:
            d = dists[n]
            p_t = d[new_bonus].item()
            rejections.append(dict(
                depth=n + 1,
                rank=int(mx.sum(d > p_t).item()) + 1,
                p_draft=math.exp(d[toks[n]].item()),
                p_target=math.exp(p_t),
            ))
        root_logp, root_g = mtp_root(rt, mtp_cache, h, mx.array([acc + [new_bonus]]))
        bonus = new_bonus
        produced += n + 1
    return rejections, accepted_p


def bucket(r: int) -> str:
    for lo, hi in BUCKETS:
        if lo <= r <= hi:
            return str(lo) if lo == hi else (f"{lo}-{hi}" if hi < 10**9 else f">{lo - 1}")
    return "?"


def report(name: str, rej: list[dict]) -> None:
    if not rej:
        print(f"{name}: no rejections")
        return
    n = len(rej)
    c = Counter(bucket(x["rank"]) for x in rej)
    hist = "  ".join(f"r{b}: {100 * c.get(b, 0) / n:4.1f}%" for b in (bucket(lo) for lo, _ in BUCKETS))
    cov = "  ".join(f"top{k}: {100 * sum(x['rank'] <= k for x in rej) / n:4.1f}%" for k in (2, 3, 4, 8))
    med = sorted(x["rank"] for x in rej)[n // 2]
    pd = sum(x["p_draft"] for x in rej) / n
    pt = sum(x["p_target"] for x in rej) / n
    print(f"{name:<10} n={n:<4d} median rank {med:<3d} {hist}")
    print(f"{'':<10} caught by branching to {cov}   mean p(draft tok) {pd:.2f}  p(target tok) {pt:.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-new", type=int, default=300)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--json", default="rank_probe_results.json")
    args = ap.parse_args()

    rt = load(resolve_model(args.model))
    inner = getattr(rt.model, "language_model", rt.model)
    ver = Verifier(rt, inner)
    tok = rt.tokenizer
    eos = set(getattr(tok, "eos_token_ids", None) or [tok.eos_token_id])

    all_rej, all_acc, out = [], [], {}
    for text in PROMPTS:
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True, enable_thinking=False
        )
        rej, acc_p = probe(rt, inner, ver, prompt, args.max_new, args.depth, eos)
        report(text.split()[0].lower()[:10], rej)
        all_rej += rej
        all_acc += acc_p
        out[text] = rej

    print()
    report("ALL", all_rej)
    print("\nby draft depth (all prompts):")
    for d in range(1, args.depth + 1):
        report(f"depth {d}", [x for x in all_rej if x["depth"] == d])

    # How often is the head's top-1 right, by its own confidence?
    print("\nMTP top-1 confidence -> acceptance rate (all prompts)")
    edges = (0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.01)
    for lo, hi in zip(edges, edges[1:]):
        a = sum(lo <= p < hi for p in all_acc)
        r = sum(lo <= x["p_draft"] < hi for x in all_rej)
        if a + r:
            print(f"  p in [{lo:.2f}, {min(hi, 1):.2f}): accepted {100 * a / (a + r):5.1f}%  (n={a + r})")
    with open(args.json, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()

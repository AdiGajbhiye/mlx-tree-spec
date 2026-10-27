"""T1/T2 (mlx-tree-spec) vs MTPLX tables -> bench/runtimes/summary-tree.md.

Usage: compare_tree.py RUNTIMES_DIR [tree_results.jsonl]
Medians per cell, same definitions as summarize.py. Tree tokens/step comes from the
server's per-request JSON lines in logs/T*.server.log, aligned to the bench rows in order.
"""
import json
import os
import statistics
import sys
from collections import defaultdict

D = sys.argv[1]
TREE = sys.argv[2] if len(sys.argv) > 2 else os.path.join(D, "results-tree.jsonl")
FILES = [os.path.join(D, "results.jsonl"), os.path.join(D, "results-base.jsonl"), TREE]
NAMES = {
    "T1": "mlx-tree-spec · Swift 4bit · tree (defaults)",
    "C2": "MTPLX · Swift 4bit · MTP d3 + context copy",
    "C2n": "MTPLX · Swift 4bit · MTP d3 only",
    "C1": "MTPLX · Swift 4bit · no MTP",
    "T2": "mlx-tree-spec · Qwen3.8 bare-speed · tree (defaults)",
    "Q5": "MTPLX · Qwen3.8 bare-speed · MTP d3 only",
    "Q4": "MTPLX · Qwen3.8 bare-speed · no MTP",
}
GROUPS = [("Swift 1.5 4-bit text (Swift-1.5-4bit-MLX-text)", "T1", ["C2", "C2n", "C1"]),
          ("Stock Qwen3.8-27B (mtplx-bare-speed)", "T2", ["Q5", "Q4"])]
CTXS = [128, 1024, 4096, 8192, 16384, 32768]
TASKS = ["code", "prose", "math", "json", "edit", "creative"]


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def accept_len(ex):
    p = (ex or {}).get("mtplx_progress") or {}
    if p.get("verify_calls"):
        return 1 + sum(p.get("accepted_by_depth") or []) / p["verify_calls"]
    return None


cells, loads, errors, n_rows = defaultdict(list), {}, defaultdict(list), defaultdict(int)
wanted = set(NAMES)
for f in FILES:
    for line in open(f):
        if not line.strip():
            continue
        r = json.loads(line)
        L = r["label"]
        if L not in wanted:
            continue
        if r.get("scenario") == "load":
            loads[L] = r["load_s"]
        elif "error" in r:
            errors[L].append(f"{r['scenario']} {r['task']} ctx={r['ctx']} run {r['run']}: {r['error'][:160]}")
        elif r.get("scenario") in ("s1", "s2"):
            cells[(L, r["scenario"], r["task"], r["ctx"])].append(r)
            n_rows[L] += 1

# tree tokens/step: align server log lines (skip ready ping + warmup) with bench rows in order
tps = defaultdict(list)
for L in ("T1", "T2"):
    log = os.path.join(D, "logs", f"{L}.server.log")
    if not os.path.exists(log):
        continue
    recs = []
    for line in open(log):
        line = line.strip()
        if line.startswith("{") and "tokens_per_step" in line:
            try:
                recs.append(json.loads(line))
            except ValueError:
                pass
    bench = [json.loads(l) for l in open(TREE) if l.strip()]
    bench = [r for r in bench if r["label"] == L and r.get("scenario") in ("s1", "s2", "warmup")]
    srv = [r for r in recs if r["completion_tokens"] > 1]  # drop the 1-token readiness pings
    srv = srv[-(len(bench) + 1):]  # last run of this label: warmup + cells
    for b, s in zip(bench, srv[1:]):
        if "error" in b or b.get("prompt_tokens") != s["prompt_tokens"]:
            continue
        tps[(L, b["scenario"], b["task"], b["ctx"])].append(s["tokens_per_step"])
        cached = s.get("cached_tokens")
        if cached:
            errors[L].append(f"cached_tokens={cached} on {b['scenario']} {b['task']} ctx={b['ctx']}")


def get(L, sc, task, ctx, metric):
    rs = cells.get((L, sc, task, ctx), [])
    if metric == "accept_len":
        if L.startswith("T"):
            return med(tps.get((L, sc, task, ctx), []))
        return med([accept_len(r.get("server_extras")) for r in rs])
    return med([r[metric] for r in rs])


def fmt(x, nd=1):
    return "–" if x is None else f"{x:.{nd}f}"


def ratio(a, b):
    return "–" if a is None or b is None else f"{a / b:.2f}x"


out = ["# mlx-tree-spec vs MTPLX: runtime harness S1/S2", "",
       "Medians per cell (S1: 3 runs, 2 at 32768; S2: 3 runs). Greedy, thinking off, 512 max tokens, "
       "unique nonce per request, one request at a time. C1/C2/C2n from results.jsonl, Q4/Q5 from "
       "results-base.jsonl, T1/T2 from results-tree.jsonl. Speedup rows are tree / MTPLX decode tok/s "
       "(TTFT: MTPLX / tree, >1 means the tree server is faster).", ""]
for title, T, others in GROUPS:
    labels = [T] + others
    out += [f"## {title}", ""]
    for sc, keys, metric, head, nd in (
            ("s1", CTXS, "decode_tps", "S1 decode tok/s by context", 1),
            ("s1", CTXS, "ttft_s", "S1 time to first token (s) by context", 2),
            ("s1", CTXS, "prefill_tps", "S1 prefill tok/s by context", 0),
            ("s2", TASKS, "decode_tps", "S2 decode tok/s by task", 1),
            ("s2", TASKS, "accept_len", "S2 tokens per target pass (tree: tokens_per_step; MTPLX: 1 + accepted/verify)", 2)):
        task = (lambda k: "longctx") if sc == "s1" else (lambda k: k)
        ctx = (lambda k: k) if sc == "s1" else (lambda k: 0)
        out += [f"### {head}", "", "| config | " + " | ".join(map(str, keys)) + " |", "|---|" + "---:|" * len(keys)]
        for L in labels:
            out.append(f"| {L} {NAMES[L]} | " + " | ".join(fmt(get(L, sc, task(k), ctx(k), metric), nd) for k in keys) + " |")
        if metric in ("decode_tps", "ttft_s", "prefill_tps"):
            for o in others:
                if metric == "ttft_s":
                    row = [ratio(get(o, sc, task(k), ctx(k), metric), get(T, sc, task(k), ctx(k), metric)) for k in keys]
                else:
                    row = [ratio(get(T, sc, task(k), ctx(k), metric), get(o, sc, task(k), ctx(k), metric)) for k in keys]
                out.append(f"| **{T} vs {o}** | " + " | ".join(row) + " |")
        out.append("")
out += ["## Load time (s, to first successful 1-token reply)", ""]
out += [f"- {L}: {loads[L]}" for L in NAMES if L in loads]
out += ["", "## Rows per config", ""] + [f"- {L}: {n_rows[L]}" for L in NAMES if L in n_rows]
if any(errors.values()):
    out += ["", "## Errors / anomalies", ""] + [f"- {L}: {e}" for L in errors for e in errors[L]]
md = "\n".join(out) + "\n"
open(os.path.join(D, "summary-tree.md"), "w").write(md)
print(md)

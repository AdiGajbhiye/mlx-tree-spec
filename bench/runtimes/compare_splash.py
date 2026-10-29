"""T2 (mlx-tree-spec) vs Splash tables -> bench/runtimes/summary-splash.md.

Usage: compare_splash.py RUNTIMES_DIR
SP and the same-day T2 rerun come from results-splash.jsonl; the other rows are context from
results-base.jsonl (Q1/Q2 llama.cpp on the same GGUF, Q5 MTPLX) and results.jsonl (B2 mlx-dspark
with the same DFlash2 draft). Medians per cell, same definitions as summarize.py. Tree
tokens/step comes from logs/T2.server.log, Splash's from its per-request speculation counts.
"""
import json
import os
import statistics
import sys
from collections import defaultdict

D = sys.argv[1]
FILES = {"results-splash.jsonl": {"SP", "T2"}, "results-base.jsonl": {"Q1", "Q2", "Q5"},
         "results.jsonl": {"B2"}}
NAMES = {
    "T2": "mlx-tree-spec · Qwen3.8 bare-speed MLX 4bit · MTP tree (defaults)",
    "SP": "Splash 1.5.0 · Qwen3.8 UD-Q4_K_M GGUF · DFlash2 (defaults)",
    "B2": "mlx-dspark · Qwen3.8 MLX 4bit · DFlash2",
    "Q5": "MTPLX · Qwen3.8 bare-speed · MTP d3 only",
    "Q2": "llama.cpp · Qwen3.8 UD-Q4_K_M GGUF · MTP draft",
    "Q1": "llama.cpp · Qwen3.8 UD-Q4_K_M GGUF · no draft",
}
CTXS = [128, 1024, 4096, 8192, 16384, 32768]
TASKS = ["code", "prose", "math", "json", "edit", "creative"]


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def tokens_per_cycle(ex):
    s = ((ex or {}).get("metrics") or {}).get("speculation") or {}
    if s.get("cycles"):
        return 1 + s["accepted_draft_tokens"] / s["cycles"]
    return None


cells, loads, swaps, errors = defaultdict(list), {}, {}, defaultdict(list)
for f, wanted in FILES.items():
    for line in open(os.path.join(D, f)):
        if not line.strip():
            continue
        r = json.loads(line)
        L = r["label"]
        if L not in wanted:
            continue
        if r.get("scenario") == "load":
            loads[L] = r["load_s"]
        elif r.get("scenario") == "swap":
            swaps[L] = (r["before_mb"], r["after_mb"])
        elif "error" in r:
            errors[L].append(f"{r['scenario']} {r['task']} ctx={r['ctx']} run {r['run']}: {r['error'][:160]}")
        elif r.get("scenario") in ("s1", "s2"):
            cells[(L, r["scenario"], r["task"], r["ctx"])].append(r)

# tree tokens/step: align server log lines (skip ready ping + warmup) with bench rows in order
tps = defaultdict(list)
log = os.path.join(D, "logs", "T2.server.log")
if os.path.exists(log):
    srv = []
    for line in open(log):
        line = line.strip()
        if line.startswith("{") and "tokens_per_step" in line:
            try:
                srv.append(json.loads(line))
            except ValueError:
                pass
    bench = [json.loads(l) for l in open(os.path.join(D, "results-splash.jsonl")) if l.strip()]
    bench = [r for r in bench if r["label"] == "T2" and r.get("scenario") in ("s1", "s2", "warmup")]
    srv = [r for r in srv if r["completion_tokens"] > 1][-(len(bench) + 1):]
    for b, s in zip(bench, srv[1:]):
        if "error" not in b and b.get("prompt_tokens") == s["prompt_tokens"]:
            tps[("T2", b["scenario"], b["task"], b["ctx"])].append(s["tokens_per_step"])


def get(L, sc, task, ctx, metric):
    rs = cells.get((L, sc, task, ctx), [])
    if metric == "accept_len":
        if L == "T2":
            return med(tps.get((L, sc, task, ctx), []))
        return med([tokens_per_cycle(r.get("server_extras")) for r in rs])
    return med([r[metric] for r in rs])


def fmt(x, nd=1):
    return "–" if x is None else f"{x:.{nd}f}"


def ratio(a, b):
    return "–" if a is None or b is None else f"{a / b:.2f}x"


out = ["# mlx-tree-spec vs Splash: runtime harness S1/S2", "",
       "Medians per cell (S1: 3 runs, 2 at 32768; S2: 3 runs). Greedy, thinking off, 512 max tokens, "
       "unique nonce per request, one request at a time, M5 Max 36 GB. SP and T2 were measured the "
       "same day (results-splash.jsonl); T2 ran after a 10-minute cooldown, because a run started 60 s after "
       "Splash finished came out 25–38% slower from 8K (results-splash-t2-after-splash.jsonl, discarded). "
       "B2, Q1, Q2 and Q5 are earlier runs, for context. "
       "**T2 vs SP** is tree / Splash (TTFT: Splash / tree); >1 means the tree server is faster.", ""]
labels = ["T2", "SP", "B2", "Q5", "Q2", "Q1"]
for sc, keys, metric, head, nd in (
        ("s1", CTXS, "decode_tps", "S1 decode tok/s by context", 1),
        ("s1", CTXS, "ttft_s", "S1 time to first token (s) by context", 2),
        ("s1", CTXS, "prefill_tps", "S1 prefill tok/s by context", 0),
        ("s2", TASKS, "decode_tps", "S2 decode tok/s by task", 1),
        ("s2", TASKS, "accept_len", "S2 tokens per target pass (tree: tokens_per_step; Splash: 1 + accepted/cycles)", 2)):
    task = (lambda k: "longctx") if sc == "s1" else (lambda k: k)
    ctx = (lambda k: k) if sc == "s1" else (lambda k: 0)
    out += [f"## {head}", "", "| config | " + " | ".join(map(str, keys)) + " |", "|---|" + "---:|" * len(keys)]
    rows = labels if metric != "accept_len" else ["T2", "SP"]
    for L in rows:
        out.append(f"| {L} {NAMES[L]} | " + " | ".join(fmt(get(L, sc, task(k), ctx(k), metric), nd) for k in keys) + " |")
    if metric != "accept_len":
        if metric == "ttft_s":
            row = [ratio(get("SP", sc, task(k), ctx(k), metric), get("T2", sc, task(k), ctx(k), metric)) for k in keys]
        else:
            row = [ratio(get("T2", sc, task(k), ctx(k), metric), get("SP", sc, task(k), ctx(k), metric)) for k in keys]
        out.append("| **T2 vs SP** | " + " | ".join(row) + " |")
    out.append("")
out += ["## Load time (s, to first successful 1-token reply)", ""]
out += [f"- {L}: {loads[L]}" for L in labels if L in loads]
out += ["", "## Swap used (MB) before / after each same-day run", ""]
out += [f"- {L}: {a} / {b}" for L, (a, b) in swaps.items()]
if any(errors.values()):
    out += ["", "## Errors", ""] + [f"- {L}: {e}" for L in errors for e in errors[L]]
md = "\n".join(out) + "\n"
open(os.path.join(D, "summary-splash.md"), "w").write(md)
print(md)

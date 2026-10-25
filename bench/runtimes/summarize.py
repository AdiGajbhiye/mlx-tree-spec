"""Median prefill/decode tables from results.jsonl -> summary.md + summary.json."""
import json
import os
import statistics
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results.jsonl")
ORDER = ["A1", "A2", "B1", "B2", "B3", "C1", "C2", "C2n", "C3", "D1", "D2", "E1", "E2", "E2n"]
NAMES = {
    "A1": "Lithos · Qwen3.8 NVFP4 · no draft", "A2": "Lithos · Qwen3.8 NVFP4 · DSpark",
    "B1": "mlx-dspark · Qwen3.8 4bit · no draft", "B2": "mlx-dspark · Qwen3.8 4bit · DFlash2",
    "B3": "mlx-dspark · Qwen3.8 4bit · DSpark",
    "C1": "MTPLX · Swift 4bit · no MTP", "C2": "MTPLX · Swift 4bit · MTP d3 + context copy",
    "C2n": "MTPLX · Swift 4bit · MTP d3 only",
    "C3": "mlx-lm (ukisai) · Swift 4bit · no MTP",
    "D1": "llama.cpp · Swift IQ3_S · no MTP", "D2": "llama.cpp · Swift IQ3_S · MTP",
    "E1": "MTPLX · Swift AX MXFP4 · no MTP", "E2": "MTPLX · Swift AX MXFP4 · MTP d3 + context copy",
    "E2n": "MTPLX · Swift AX MXFP4 · MTP d3 only",
}


def accept_len(ex):
    """Mean tokens committed per target pass, from whatever the engine reports."""
    p = ex.get("mtplx_progress") or {}
    if p.get("verify_calls"):
        return 1 + sum(p.get("accepted_by_depth") or []) / p["verify_calls"]
    t = ex.get("timings") or {}
    if t.get("draft_n"):  # llama.cpp: accepted / drafted per round is not exposed; use rate
        return None
    x = ex.get("x_mlx_dspark") or {}
    for k in ("accept_len", "mean_accept_len", "acceptance_length"):
        if x.get(k):
            return x[k]
    return None


def accept_rate(ex):
    p = ex.get("mtplx_progress") or {}
    if p.get("drafted_by_depth") and sum(p["drafted_by_depth"]):
        return sum(p["accepted_by_depth"]) / sum(p["drafted_by_depth"])
    t = ex.get("timings") or {}
    if t.get("draft_n"):
        return t.get("draft_n_accepted", 0) / t["draft_n"]
    return None


def med(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 2) if xs else None


rows = [json.loads(l) for l in open(path) if l.strip()]
cells = defaultdict(list)
loads, errors = {}, defaultdict(list)
for r in rows:
    if r.get("scenario") == "load":
        loads[r["label"]] = r["load_s"]
    elif "error" in r:
        errors[r["label"]].append(f"{r['scenario']} {r['task']} ctx={r['ctx']}: {r['error'][:120]}")
    elif r.get("scenario") in ("s1", "s2"):
        cells[(r["label"], r["scenario"], r["task"], r["ctx"])].append(r)

summary = {}
for (label, sc, task, ctx), rs in cells.items():
    summary.setdefault(label, {}).setdefault(sc, {})[f"{task}@{ctx}"] = {
        "n": len(rs), "prompt_tokens": med([r["prompt_tokens"] for r in rs]),
        "gen_tokens": med([r["completion_tokens"] for r in rs]),
        "ttft_s": med([r["ttft_s"] for r in rs]), "prefill_tps": med([r["prefill_tps"] for r in rs]),
        "decode_tps": med([r["decode_tps"] for r in rs]),
        "accept_rate": med([accept_rate(r["server_extras"]) for r in rs]),
        "accept_len": med([accept_len(r["server_extras"]) for r in rs]),
    }
labels = [l for l in ORDER if l in summary] + [l for l in summary if l not in ORDER]
json.dump({"summary": summary, "loads": loads, "errors": errors, "names": NAMES},
          open(os.path.join(HERE, "summary.json"), "w"), indent=1)

out = []
ctxs = sorted({int(k.split("@")[1]) for l in summary for k in summary[l].get("s1", {})})
for metric, title in (("prefill_tps", "S1 prefill tok/s"), ("decode_tps", "S1 decode tok/s"),
                      ("ttft_s", "S1 time to first token (s)")):
    out += [f"## {title}", "", "| config | " + " | ".join(f"{c}" for c in ctxs) + " |",
            "|---|" + "---:|" * len(ctxs)]
    for l in labels:
        s1 = summary[l].get("s1", {})
        out.append(f"| {l} {NAMES.get(l, '')} | " + " | ".join(
            str(s1.get(f"longctx@{c}", {}).get(metric, "–")) for c in ctxs) + " |")
    out.append("")
tasks = ["code", "prose", "math", "json", "edit", "creative"]
for metric, title in (("decode_tps", "S2 decode tok/s by task"), ("accept_rate", "S2 draft acceptance rate")):
    out += [f"## {title}", "", "| config | " + " | ".join(tasks) + " |", "|---|" + "---:|" * len(tasks)]
    for l in labels:
        s2 = summary[l].get("s2", {})
        out.append(f"| {l} {NAMES.get(l, '')} | " + " | ".join(
            str(s2.get(f"{t}@0", {}).get(metric, "–")) for t in tasks) + " |")
    out.append("")
out += ["## Load time (s, to first successful 1-token reply)", ""] + [f"- {l}: {loads[l]}" for l in labels if l in loads]
if errors:
    out += ["", "## Errors", ""] + [f"- {l}: {e}" for l in errors for e in errors[l]]
open(os.path.join(HERE, "summary.md"), "w").write("\n".join(out) + "\n")
print("\n".join(out))

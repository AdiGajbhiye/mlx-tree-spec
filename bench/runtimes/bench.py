"""Engine-agnostic prefill/decode benchmark against an OpenAI-compatible server.

Prefill tok/s = prompt_tokens / TTFT (TTFT also contains the first decode step).
Decode tok/s  = (completion_tokens - 1) / (last_token_time - first_token_time).
Each request starts with a unique nonce so no engine can reuse a cached prefix.

Scenarios:
  s1  context sweep: 128..32k tokens of real code + a summarize/test task
  s2  task mix at short context: code, prose, math, json, edit, creative
"""
import argparse
import glob
import json
import os
import sysconfig
import time
import urllib.request
import uuid

from tokenizers import Tokenizer

STDLIB = sysconfig.get_paths()["stdlib"]

TASKS = {
    "longctx": "Summarize what the code above does, module by module, then write pytest "
               "unit tests for three of its functions.",
    "code": "Implement a thread-safe LRU cache in Python with get/put/delete, "
            "TTL expiry and size limits. Include docstrings and a full pytest test suite.",
    "prose": "Explain in detail how TCP congestion control works: slow start, congestion "
             "avoidance, fast retransmit, fast recovery, CUBIC and BBR. Use several paragraphs.",
    "math": "Solve step by step, showing all work: A tank holds 1200 litres. Pipe A fills it in "
            "6 hours, pipe B in 9 hours, and a drain empties it in 12 hours. The tank starts a "
            "quarter full with all three open; after 2 hours pipe B is closed. How long until the "
            "tank is full? Then verify the answer a second way.",
    "json": "Return only JSON, no prose. Produce an OpenAPI 3.1 document for a bookstore API with "
            "endpoints for books, authors, orders and reviews, including request/response schemas, "
            "pagination parameters, error responses and examples for every endpoint.",
    "edit": None,  # built from a stdlib file in build_prompt
    "creative": "Write an original short story about a lighthouse keeper who discovers that the "
                "light is signalling to something beneath the sea. Vivid, surprising, literary prose.",
}

S1_CONTEXTS = [128, 1024, 4096, 8192, 16384, 32768]
S2_TASKS = ["code", "prose", "math", "json", "edit", "creative"]


def filler_text(tok, n_tokens):
    """Non-repetitive real code (CPython stdlib) trimmed to exactly n_tokens."""
    files = sorted(glob.glob(os.path.join(STDLIB, "*.py")))
    ids = []
    for f in files:
        with open(f, encoding="utf-8", errors="ignore") as fh:
            ids += tok.encode(f"# file: {os.path.basename(f)}\n" + fh.read()).ids
        if len(ids) >= n_tokens:
            break
    return tok.decode(ids[:n_tokens])


def edit_prompt():
    with open(os.path.join(STDLIB, "textwrap.py"), encoding="utf-8") as fh:
        src = "".join(fh.readlines()[:250])
    return (f"```python\n{src}```\n\nRename the class TextWrapper to LineWrapper everywhere and add "
            "type hints to every method signature. Output the complete modified file and nothing else.")


def build_prompt(tok, task, ctx):
    if task == "edit":
        return edit_prompt()
    if task == "longctx":
        return filler_text(tok, ctx) + "\n\n" + TASKS[task]
    return TASKS[task]


def stream_request(url, api_key, body):
    req = urllib.request.Request(
        url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    t0 = time.perf_counter()
    t_first = t_last = None
    usage, extras, text = None, {}, []
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            for ch in chunk.get("choices") or []:
                d = ch.get("delta") or {}
                piece = (d.get("content") or "") + (d.get("reasoning_content") or "") + (d.get("reasoning") or "")
                if piece:
                    now = time.perf_counter()
                    t_first = t_first or now
                    t_last = now
                    text.append(piece)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for k, v in chunk.items():
                if k not in ("id", "object", "created", "model", "choices", "usage", "system_fingerprint"):
                    extras[k] = v
    return t0, t_first, t_last, time.perf_counter(), usage, extras, "".join(text)


def run_one(a, tok, scenario, task, ctx, run, prompt):
    body = {
        "model": a.model,
        "messages": [{"role": "user", "content": f"[run {uuid.uuid4()}]\n" + prompt}],
        "max_tokens": a.max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if not a.strict_openai:  # Lithos rejects any non-standard field with a 400
        body.update({"chat_template_kwargs": {"enable_thinking": False}, "cache_prompt": False})
    rec = {"label": a.label, "scenario": scenario, "task": task, "ctx": ctx, "run": run,
           "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
    try:
        t0, tf, tl, te, usage, extras, text = stream_request(a.url, a.api_key, body)
    except Exception as e:  # OOM, server crash, timeout: record and keep going
        rec["error"] = repr(e)[:500]
        return rec
    usage = usage or {}
    pt = usage.get("prompt_tokens")
    ct = usage.get("completion_tokens")
    rec["usage_from_server"] = bool(pt and ct)
    pt = pt or len(tok.encode(body["messages"][0]["content"]).ids)
    ct = ct or len(tok.encode(text).ids)
    rec.update({
        "prompt_tokens": pt, "completion_tokens": ct,
        "ttft_s": tf - t0 if tf else None,
        "prefill_tps": pt / (tf - t0) if tf else None,
        "decode_tps": (ct - 1) / (tl - tf) if tl and tl > tf and ct > 1 else None,
        "e2e_s": te - t0, "server_extras": extras, "output_head": text[:200],
    })
    return rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--api-key", default="local")
    p.add_argument("--model", default="local")
    p.add_argument("--label", required=True)
    p.add_argument("--tokenizer", required=True, help="tokenizer.json used to size prompts")
    p.add_argument("--scenarios", default="s1,s2")
    p.add_argument("--contexts", default=",".join(map(str, S1_CONTEXTS)))
    p.add_argument("--tasks", default=",".join(S2_TASKS))
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--runs-32k", type=int, default=2)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--strict-openai", action="store_true", help="send only standard OpenAI fields")
    p.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.jsonl"))
    a = p.parse_args()

    tok = Tokenizer.from_file(a.tokenizer)
    cells = []
    if "s1" in a.scenarios:
        cells += [("s1", "longctx", int(c)) for c in a.contexts.split(",")]
    if "s2" in a.scenarios:
        cells += [("s2", t, 0) for t in a.tasks.split(",")]

    warm = run_one(a, tok, "warmup", "code", 0, -1, TASKS["code"])
    print(f"{a.label} warmup: {warm.get('error') or 'ok'}", flush=True)

    for scenario, task, ctx in cells:
        prompt = build_prompt(tok, task, ctx)
        n = a.runs_32k if ctx >= 32768 else a.runs
        for run in range(n):
            rec = run_one(a, tok, scenario, task, ctx, run, prompt)
            with open(a.out, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            if "error" in rec:
                print(f"{a.label} {scenario} {task} ctx={ctx} run {run}: ERROR {rec['error']}", flush=True)
                break  # skip remaining runs of a failing cell
            print(f"{a.label} {scenario} {task} ctx={ctx} run {run}: prompt={rec['prompt_tokens']} "
                  f"gen={rec['completion_tokens']} ttft={rec['ttft_s'] or 0:.2f}s "
                  f"prefill={rec['prefill_tps'] or 0:.1f} decode={rec['decode_tps'] or 0:.1f} tok/s",
                  flush=True)
            time.sleep(3)


if __name__ == "__main__":
    main()

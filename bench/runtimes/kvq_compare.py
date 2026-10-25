"""Greedy (temperature 0) outputs of the running MTPLX server for a fixed prompt set.

Usage: kvq_compare.py run LABEL      -> kvq/LABEL.json
       kvq_compare.py diff A B ...   -> token-level agreement of each run against A
"""
import json
import os
import sys
import time
import urllib.request

from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bench

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "kvq")
tok = Tokenizer.from_file(os.environ.get("SWIFT_TOK", os.path.expanduser("~/models/swift15-mlx/Swift-1.5-4bit-MLX/tokenizer.json")))

# Facts hidden inside long code context; the model must recall them exactly.
NEEDLES = [(0.1, "Halcyon", "7419-QX"), (0.5, "Meridian", "2086-LV"), (0.9, "Tamarind", "5532-HB")]


def needle_prompt(ctx, where, name, code):
    ids = tok.encode(bench.filler_text(tok, ctx)).ids
    cut = int(len(ids) * where)
    fact = f"\n# NOTE: the deployment code for project {name} is {code}.\n"
    return (tok.decode(ids[:cut]) + fact + tok.decode(ids[cut:]) +
            f"\n\nWhat is the deployment code for project {name}? Reply with the code only.")


def prompts():
    p = [("short-code", bench.TASKS["code"], 400), ("short-math", bench.TASKS["math"], 400),
         ("short-prose", bench.TASKS["prose"], 400)]
    for ctx in (4096, 16384, 30000):
        p.append((f"summary-{ctx // 1024}k", bench.build_prompt(tok, "longctx", ctx), 400))
    for where, name, code in NEEDLES:
        p.append((f"recall-30k-{int(where * 100)}%", needle_prompt(30000, where, name, code), 20))
    return p


def run(label):
    os.makedirs(OUT, exist_ok=True)
    res = {}
    for name, prompt, max_tokens in prompts():
        body = {"model": "swift-1.5-4bit-mtp", "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens, "temperature": 0.0, "stream": False,
                "chat_template_kwargs": {"enable_thinking": False}}
        req = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": "Bearer mtplx-local"})
        t0 = time.time()
        try:
            r = json.load(urllib.request.urlopen(req, timeout=1800))
            text = r["choices"][0]["message"].get("content") or ""
            res[name] = {"text": text, "usage": r.get("usage"), "s": round(time.time() - t0, 1)}
        except urllib.error.HTTPError as e:
            res[name] = {"error": f"HTTP {e.code}: {e.read()[:200]!r}"}
        print(label, name, res[name].get("error") or repr(res[name]["text"][:70]), flush=True)
    json.dump(res, open(os.path.join(OUT, f"{label}.json"), "w"), indent=1)


def diff(base, others):
    a = json.load(open(os.path.join(OUT, f"{base}.json")))
    expected = {f"recall-30k-{int(w * 100)}%": c for w, _, c in NEEDLES}
    for o in others:
        b = json.load(open(os.path.join(OUT, f"{o}.json")))
        print(f"\n== {o} vs {base}")
        for name in a:
            if "error" in a[name] or "error" in b.get(name, {"error": 1}):
                print(f"{name:16} error"); continue
            ta, tb = tok.encode(a[name]["text"]).ids, tok.encode(b[name]["text"]).ids
            same = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), min(len(ta), len(tb)))
            exact = ta == tb
            extra = ""
            if name in expected:
                extra = f"  recall {base}={'ok' if expected[name] in a[name]['text'] else 'MISS'} {o}={'ok' if expected[name] in b[name]['text'] else 'MISS'}"
            print(f"{name:16} {'identical' if exact else f'diverges at token {same} of {len(ta)}'}{extra}")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2])
    else:
        diff(sys.argv[2], sys.argv[3:])

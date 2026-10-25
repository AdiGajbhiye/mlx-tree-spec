"""OpenAI-compatible chat server for Qwen3.8-27B on the tree-speculative decoder.

Serves POST /v1/chat/completions (non-streaming), GET /v1/models and GET /health,
enough for agent harnesses and other OpenAI-compatible clients:

  - decoding: tree-speculative decoding (tree_verify.ForkState) on the packed
    4-bit target with the MTP head drafting (fork:15,2,4,4 by default), MTPLX's
    flash kernels at long context; `temperature` samples (exact target
    distribution, see tree_verify.choose), 0 is greedy;
  - continuous batching: up to --max-batch requests decode together; each step
    drafts all of them as one batch and verifies all their trees in one target
    pass (tree_verify.batch_step), so the weights are read once for all.
    Requests join between steps (after their prefill) and leave when done;
  - prompt-prefix reuse: the GatedDeltaNet state cannot be trimmed, so the
    server keeps snapshots of every cache (up to BANK, least recently used out)
    at the start of the user turn (after the system prompt) and at the longest
    prefix the prompt shares with a recent one (goal, repo map...), and resumes
    from the longest one a new prompt starts with; reported as
    usage.prompt_tokens_details.cached_tokens;
  - thinking: chat_template_kwargs pass to the chat template; when the prompt
    opens a <think> block, the reply is split at </think>: message.content is
    the answer, message.reasoning_content the thinking, and
    usage.completion_tokens_details.reasoning_tokens counts it. --default-effort
    sets reasoning_effort for requests that send none;
  - --trace-log appends each request's prompt and reply tokens, thinking and
    answer text (one JSON line per request) for offline analysis;
  - a body with "prompt_ids" instead of "messages" decodes those tokens as is
    (no chat template, no thinking split) and returns the reply tokens as
    "gen_ids" (used by label_checkpoints.py).

HTTP requests are queued to the main thread, which loaded the model and runs
all MLX work (MLX ties streams to threads); a request whose client disconnects
(e.g. timed out) is dropped at the next step. Each request is logged as a JSON
line (--log).

  mlx-tree-spec-server --port 8090 --max-batch 4
"""

from __future__ import annotations

import argparse
import json
import queue
import socket
import threading
import time
import uuid
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.cache import KVCache

from . import long_attention
from . import packed_linear
from . import tree_spec as ts
from . import tree_verify as tv
from . import POWERED_BY
from mtplx.runtime import load

PREFILL_CHUNK = ts.PREFILL_CHUNK
MIN_SNAPSHOT = 256  # shortest shared prefix worth a snapshot
BANK = 4  # snapshots kept
RECENT = 8  # recent prompts a new one is matched against for a snapshot point


class Job:
    """One HTTP request: filled in by the engine thread, waited on by the handler."""

    def __init__(self, body: dict, stateless: bool):
        self.body, self.stateless = body, stateless
        self.done = threading.Event()
        self.response: dict | None = None
        self.error: Exception | None = None
        self.cancelled = False  # the client went away: drop it at the next step


class Engine:
    def __init__(self, model: str, config: tuple[int, int, int, int], curve_context: int, log: str | None, max_batch: int,
                 cache_limit_gb: float = 2.0, default_effort: str | None = None, trace_log: str | None = None):
        mx.set_cache_limit(int(cache_limit_gb * 1e9))
        mx.set_wired_limit(int(mx.device_info()["max_recommended_working_set_size"]))
        self.rt = load(ts.resolve_model(model))
        self.model_id = Path(model.rstrip("/")).name
        self.inner = getattr(self.rt.model, "language_model", self.rt.model)
        self.tok = self.rt.tokenizer
        _, self.drafter = ts.make_drafter(self.rt, self.inner, "general")
        packed_linear.install(self.rt.model, verbose=False)
        packed_linear.install(self.drafter.head, verbose=False)
        long_attention.install()
        self.eos = set(getattr(self.tok, "eos_token_ids", None) or [self.tok.eos_token_id])
        self.think_end = self.tok.convert_tokens_to_ids("</think>")
        self.depth, self.branches, self.branch_depth, self.sibs = config
        self.max_batch = max_batch
        self.log = open(log, "a") if log else None
        self.trace_log = open(trace_log, "a") if trace_log else None
        self.default_effort = default_effort
        self.bank: list[tuple] = []  # snapshots (tokens, cache, mtp_cache, last target hidden), most recent last
        self.user_turn = self.tok.encode("<|im_start|>user")
        self.recent: deque[list[int]] = deque(maxlen=RECENT)
        self.jobs: queue.Queue[Job] = queue.Queue()

        t = time.perf_counter()
        self.curve, self.row_ms, self.draft = tv.measure_tree_curve(self.rt, self.inner, self.drafter, self.tok)
        self.long = None
        if curve_context:
            text = (ts.DATA / "code_32k.txt").read_text()
            ids = self.tok.encode(text)[:curve_context]
            curve_l, row_l, draft_l = tv.measure_tree_curve(self.rt, self.inner, self.drafter, self.tok, context=ids)
            self.long = (curve_l, row_l, draft_l, ts.VERIFY_MS_CTX[0], len(ids))
        ts._PREFILLED.clear()
        mx.clear_cache()
        print(f"cost curves measured in {time.perf_counter() - t:.0f} s; ready", flush=True)

    # ------------------------------------------------------------- prefill

    def _mtp(self, hidden, ids, mtp_cache):
        _, g = self.inner._mtp_core(
            hidden, mx.array([ids]), mtp_cache=mtp_cache, concat_order=self.rt.contract.concat_order,
            mtp_hidden_variant=self.rt.contract.hidden_variant, emit_logits=False,
        )
        return g

    def prefill(self, prompt: list[int], temperature: float, stateless: bool):
        """Target and MTP caches over `prompt`, resumed from the longest snapshot the
        prompt extends; new snapshots at the user-turn start and at the longest
        prefix shared with a recent prompt. MTP entry i pairs target hidden i with
        token i + 1, so each piece's last entry waits for the next piece (or the
        first sampled token). Returns ForkState's state and the reused tokens."""
        t0 = time.perf_counter()
        hits = [] if stateless else [b for b in self.bank if len(b[0]) < len(prompt) and prompt[: len(b[0])] == b[0]]
        snap = max(hits, key=lambda b: len(b[0]), default=None)
        start = 0
        if snap is not None:
            self.bank.remove(snap)
            self.bank.append(snap)
            start = len(snap[0])
            cache, mtp_cache = ts.repeat_cache(snap[1], 1), ts.repeat_cache(snap[2], 1)
            prev_h = snap[3]
        else:
            cache, mtp_cache, prev_h = self.inner.make_cache(), [KVCache() for _ in self.rt.model.mtp.layers], None
        snap_at = set()
        if not stateless:
            n = max((_common(prompt, r) for r in self.recent), default=0)
            u = self._user_turn_start(prompt)
            have = {len(b[0]) for b in hits}
            snap_at = {k for k in (n, u) if MIN_SNAPSHOT <= k < len(prompt) and k > start and k not in have}
        cuts = sorted({start, len(prompt), *range(start + PREFILL_CHUNK, len(prompt), PREFILL_CHUNK)} | snap_at)
        for i, j in zip(cuts, cuts[1:]):
            logits, hidden = self.rt.forward_ar(mx.array([prompt[i:j]]), cache=cache, return_hidden=True, logits_keep=1)
            hs = hidden[:, :-1] if prev_h is None else mx.concatenate([prev_h, hidden[:, :-1]], axis=1)
            ids = prompt[i + 1 : j] if prev_h is None else prompt[i:j]
            if ids:
                self._mtp(hs, ids, mtp_cache)
            prev_h = hidden[:, -1:]
            mx.eval(logits, prev_h, [c.state for c in cache], [c.state for c in mtp_cache])
            if j in snap_at:
                snap = (prompt[:j], ts.repeat_cache(cache, 1), ts.repeat_cache(mtp_cache, 1), prev_h)
                mx.eval([c.state for c in snap[1] + snap[2]])
                self.bank = [b for b in self.bank if b[0] != snap[0]][-(BANK - 1) :] + [snap]
        bonus = int(tv.choose(logits[0, -1], temperature).item())
        root_g = self._mtp(prev_h, [bonus], mtp_cache)[:, -1, :]
        mx.eval(root_g)
        if not stateless:
            self.recent.append(prompt)
        return (cache, mtp_cache, bonus, root_g, time.perf_counter() - t0), start

    def _user_turn_start(self, prompt: list[int]) -> int:
        """Position of the last "<|im_start|>user" in `prompt` (0 if none)."""
        m = len(self.user_turn)
        for i in range(len(prompt) - m, -1, -1):
            if prompt[i : i + m] == self.user_turn:
                return i
        return 0

    # ------------------------------------------------------------- requests

    def run(self) -> None:
        """The engine loop: admit queued requests (prefill each) up to max_batch,
        step every active one together, answer the finished ones."""
        active: list[tuple[Job, tv.ForkState, dict]] = []
        while True:
            while len(active) < self.max_batch:
                try:
                    job = self.jobs.get(block=not active)
                except queue.Empty:
                    break
                if job.cancelled:
                    job.done.set()
                    continue
                try:
                    st, meta = self._begin(job)
                except Exception as e:  # noqa: BLE001
                    job.error = e
                    job.done.set()
                    continue
                if st.done:
                    self._finish(job, st, meta)
                else:
                    active.append((job, st, meta))
            for job, _, meta in [a for a in active if a[0].cancelled]:
                print(json.dumps({"cancelled": True, "prompt_tokens": len(meta["prompt"])}), flush=True)
                job.done.set()
            active = [a for a in active if not a[0].cancelled]
            if not active:
                continue
            try:
                tv.batch_step(self.rt, self.inner, self.drafter, [st for _, st, _ in active])
            except Exception as e:  # noqa: BLE001
                for job, _, _ in active:
                    job.error = e
                    job.done.set()
                active = []
                mx.clear_cache()
                continue
            for job, st, meta in [a for a in active if a[1].done]:
                self._finish(job, st, meta)
            active = [a for a in active if not a[1].done]
            if not active:
                mx.clear_cache()

    def _begin(self, job: Job):
        body = job.body
        kwargs = dict(body.get("chat_template_kwargs") or {})
        if self.default_effort and "reasoning_effort" not in kwargs:
            kwargs["reasoning_effort"] = body.get("reasoning_effort") or self.default_effort
        if "prompt_ids" in body:
            prompt, thinking = [int(t) for t in body["prompt_ids"]], False
        else:
            prompt = self.tok.apply_chat_template(body["messages"], add_generation_prompt=True, **kwargs)
            thinking = self.tok.decode(prompt[-8:]).rstrip().endswith("<think>")
        temperature = float(body.get("temperature", 0.0) or 0.0)
        max_tokens = int(body.get("max_tokens") or body.get("max_completion_tokens") or 4096)
        t0 = time.perf_counter()
        state, cached = self.prefill(prompt, temperature, job.stateless)
        policy = tv.ForkPolicy(self.depth, self.branches, self.branch_depth, self.sibs, self.draft, self.curve, self.row_ms, self.long)
        st = tv.ForkState(self.inner, self.drafter, max_tokens, self.eos, state, policy, temperature)
        meta = dict(prompt=prompt, thinking=thinking, temperature=temperature, cached=cached, t0=t0,
                    t_first=time.perf_counter(), batch=0, effort=kwargs.get("reasoning_effort"))
        return st, meta

    def _finish(self, job: Job, st: tv.ForkState, meta: dict) -> None:
        decode_s = time.perf_counter() - meta["t_first"]
        out, stats = st.result(decode_s)
        prompt, thinking = meta["prompt"], meta["thinking"]
        stopped = bool(out) and out[-1] in self.eos
        gen = out[:-1] if stopped else out
        reasoning, content, n_reason = "", self.tok.decode(gen), 0
        if thinking:
            if self.think_end in gen:
                k = gen.index(self.think_end)
                reasoning, content, n_reason = self.tok.decode(gen[:k]), self.tok.decode(gen[k + 1 :]), k + 1
            else:
                reasoning, content, n_reason = content, "", len(gen)
        content = content.lstrip()
        rec = dict(
            prompt_tokens=len(prompt), cached_tokens=meta["cached"], completion_tokens=len(gen), reasoning_tokens=n_reason,
            prefill_s=round(stats["prefill_s"], 3), decode_s=round(decode_s, 3),
            total_s=round(time.perf_counter() - meta["t0"], 3),
            decode_tok_s=round((len(out) - 1) / max(decode_s, 1e-9), 1), tokens_per_step=round(stats["tokens_per_step"], 2),
            finish="stop" if stopped else "length", temperature=meta["temperature"], thinking=thinking,
            waiting=self.jobs.qsize(),
        )
        print(json.dumps(rec), flush=True)
        if self.log:
            self.log.write(json.dumps({"ts": time.time(), **rec}) + "\n")
            self.log.flush()
        raw = "prompt_ids" in job.body
        if self.trace_log and not raw:
            self.trace_log.write(json.dumps({
                "ts": time.time(), "effort": meta["effort"], "temperature": meta["temperature"], "thinking": thinking,
                "finish": rec["finish"], "prompt_ids": prompt, "gen_ids": gen, "reasoning": reasoning, "content": content,
            }) + "\n")
            self.trace_log.flush()
        message = {"role": "assistant", "content": content}
        if thinking:
            message["reasoning_content"] = reasoning
        job.response = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": job.body.get("model", self.model_id),
            "choices": [{"index": 0, "message": message, "finish_reason": rec["finish"]}],
            "usage": {
                "prompt_tokens": len(prompt),
                "completion_tokens": len(gen),
                "total_tokens": len(prompt) + len(gen),
                "prompt_tokens_details": {"cached_tokens": meta["cached"]},
                "completion_tokens_details": {"reasoning_tokens": n_reason},
            },
        }
        if raw:
            job.response["gen_ids"] = gen
        job.done.set()


def _common(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _client_gone(sock) -> bool:
    """Whether the client closed its connection (a waiting client sends nothing more)."""
    try:
        sock.setblocking(False)
        return sock.recv(1, socket.MSG_PEEK) == b""
    except BlockingIOError:
        return False
    except OSError:
        return True
    finally:
        try:
            sock.setblocking(True)
        except OSError:
            pass


def serve(engine: Engine, host: str, port: int) -> None:
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, obj: dict) -> None:
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):  # noqa: N802
            if self.path.rstrip("/") in ("/v1/models", "/models"):
                self._send(200, {"object": "list", "data": [{"id": engine.model_id, "object": "model"}]})
            elif self.path.rstrip("/") == "/health":
                self._send(200, {"status": "ok"})
            else:
                self._send(404, {"error": {"message": f"no route {self.path}"}})

        def do_POST(self):  # noqa: N802
            if self.path.rstrip("/") not in ("/v1/chat/completions", "/chat/completions"):
                self._send(404, {"error": {"message": f"no route {self.path}"}})
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))))
                if body.get("stream"):
                    raise ValueError("streaming is not supported")
                if not isinstance(body.get("messages"), list) and not isinstance(body.get("prompt_ids"), list):
                    raise ValueError("messages (or prompt_ids) must be a list")
            except Exception as e:  # noqa: BLE001
                self._send(400, {"error": {"message": repr(e)}})
                return
            job = Job(body, self.headers.get("x-mtplx-cache-mode", "") == "stateless")
            engine.jobs.put(job)
            while not job.done.wait(1.0):
                if _client_gone(self.connection):  # e.g. its timeout fired: free the batch slot
                    job.cancelled = True
                    job.done.wait()
                    return
            if job.error is not None:
                code = 400 if isinstance(job.error, (ValueError, KeyError)) else 500
                self._send(code, {"error": {"message": repr(job.error)}})
            else:
                self._send(200, job.response)

        def log_message(self, *args):  # requests are logged by Engine._finish
            pass

    httpd = ThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    engine.run()  # MLX work stays on this (the loading) thread


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=ts.DEFAULT_MODEL, help="local MLX checkpoint with an MTP head, or a Hugging Face repo id")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--fork", default="15,2,4,4", help="depth,branches,branch_depth,sibs (tree shape limits)")
    ap.add_argument("--max-batch", type=int, default=4, help="requests decoded together (1: one at a time)")
    ap.add_argument("--curve-context", type=int, default=16384,
                    help="also measure cost curves at this many context tokens (0: short context only)")
    ap.add_argument("--cache-limit-gb", type=float, default=2.0,
                    help="cap MLX's cache of freed GPU buffers (memory the server keeps without using)")
    ap.add_argument("--log", help="append one JSON line per request (timings, token counts)")
    ap.add_argument("--trace-log", help="append each request's tokens, thinking and answer (one JSON line per request)")
    ap.add_argument("--default-effort", choices=["low", "medium", "xhigh"],
                    help="reasoning_effort for requests that send none (default: the chat template's)")
    args = ap.parse_args()
    print(POWERED_BY, flush=True)
    engine = Engine(args.model, tuple(int(x) for x in args.fork.split(",")), args.curve_context, args.log, args.max_batch,
                    args.cache_limit_gb, args.default_effort, args.trace_log)
    print(f"serving on http://{args.host}:{args.port} (max batch {args.max_batch})", flush=True)
    serve(engine, args.host, args.port)


if __name__ == "__main__":
    main()

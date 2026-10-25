"""Tree-drafted speculative decoding for Qwen3.8-27B (and fine-tunes such as Swift 1.5) on MTPLX.

MTPLX's normal decode drafts ONE chain with the MTP head (depth 3) and verifies it.
This script drafts a TREE instead: at every depth it takes the top-k tokens from the
MTP head's distribution, keeps the `width` most probable nodes (by cumulative
log-prob) to expand further, then keeps the `budget` most probable nodes overall.

Entropy gating (`gate`): a node only branches when the MTP head is unsure about it
(top-1 probability below `gate`); otherwise it gets a single child. Confident
stretches stay a chain, and the tree only widens where the draft is likely wrong.

Qwen3.8 is a hybrid model (3 Gated-DeltaNet linear-attention layers per full
attention layer). Linear-attention layers carry a recurrent state, so a single
flattened "tree attention mask" cannot verify branches. Instead every root-to-leaf
path becomes one row of a batched target forward; the longest path the target
agrees with (greedy) wins.

Prefix sharing: when all paths start with the same tokens (the "trunk"), the trunk
is verified once at batch 1 and only the diverging suffixes run batched. This costs
an extra forward pass, so it is used only when it saves at least `--share-min`
row-tokens.

Verify-token cap (`cap=N`): a target forward costs ~45 ms up to ~4 tokens and
~115 ms from ~12 tokens on (MLX's 4-bit kernels). With a cap, the tree is only a
pool of candidates: the rows sent to the target are the set of paths that
maximises the expected accepted length (sum of covered nodes' cumulative MTP
probability) within N tokens, each row costing its length + 1.

Nucleus branching (`p=X`): each node gets the fewest top-k children whose
probabilities add up to X, so with p=0.9 the tree stays a chain where the head is
confident and widens only at uncertain tokens.

Adaptive (`auto`): the tree is shaped per step by select_auto, which knows that a
branch costs a whole duplicate row on this hybrid model (extending a row costs 1
token, forking at depth j costs j + 1). It maximises expected tokens per ms using
the measured verify-time curve (rescaled online) and an acceptance model learned
online from rank_probe.py's priors: the head's top-1 at its calibrated rate, an
alternative of rank r at (1 - top-1 rate) x the share of rejections at rank r.
It stops drafting once another MTP call is unlikely to pay for itself.

Drafting (Drafter): logits only for the last position, one CPU sync per depth,
and optionally FR-Spec's pruned 64K-token output layer (`--draft-vocab frspec`);
the full output layer was ~70% of a draft call.

Single-sync chains (`lazy`, plain chains only): the chain is drafted and verified
as one GPU graph and read back once per step. `auto,lazy` picks each step's
length before drafting from learned per-position acceptance (DepthPolicy).

Fast verify (`--fast-verify`): 2-16-row matmuls go to whichever of MLX's or
MTPLX's M5 matrix-accelerator (NAX) kernels is fastest per layer shape
(verify_matmul.py), so a 12-token pass costs ~66 ms instead of ~113 ms. The cost
curve the adaptive policies use is re-measured at startup.

Commit without re-running the model: the recurrent state cannot be rewound, so
each verify pass records the per-token inputs of the linear-attention recurrence
(cheap activations, not states). After choosing the accepted path, only that
recurrence is replayed for the accepted tokens; full-attention KV caches are
simply trimmed.

Greedy only: every mode should reproduce plain greedy decoding of the target, so
the output of each mode is compared with the AR baseline.

Benchmark the decoding modes against plain greedy decoding:
    python -m mlx_tree_spec.tree_spec --max-new 200
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
import numpy as np
import mlx.nn as nn
from mlx_lm.models import qwen3_5
from mlx_lm.models.cache import ArraysCache, KVCache
from mlx_lm.models.gated_delta import gated_delta_update

from mtplx.runtime import load

from . import kv8

DEFAULT_MODEL = "Youssofal/Qwen3.8-27B-MTPLX-Bare-Speed"
DATA = Path(__file__).with_name("data")
DEFAULT_PROMPTS = [
    "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring.",
    "Explain in three short paragraphs why the sky is blue.",
]


# ------------------------------------------------- linear-attention recording

# While a verify pass runs, each GatedDeltaNet layer stores the inputs of its
# recurrence here (keyed by id(layer)): starting conv/recurrent state plus the
# per-token qkv, q, k, v, a, b. Replaying a prefix of them reproduces the state
# after any accepted token without touching the model weights again.
_RECORD: dict[int, tuple] | None = None
_ORIG_GDN_CALL = qwen3_5.GatedDeltaNet.__call__


def _gdn_call(self, inputs, mask=None, cache=None):
    """mlx_lm's GatedDeltaNet.__call__, plus recording when _RECORD is set."""
    if _RECORD is None or cache is None:
        return _ORIG_GDN_CALL(self, inputs, mask, cache)
    assert mask is None and cache.lengths is None, "recording assumes unpadded rows"
    B, S, _ = inputs.shape

    qkv = self.in_proj_qkv(inputs)
    z = self.in_proj_z(inputs).reshape(B, S, self.num_v_heads, self.head_v_dim)
    b = self.in_proj_b(inputs)
    a = self.in_proj_a(inputs)

    conv_state = cache[0]
    if conv_state is None:
        conv_state = mx.zeros((B, self.conv_kernel_size - 1, self.conv_dim), dtype=inputs.dtype)
    conv_input = mx.concatenate([conv_state, qkv], axis=1)
    cache[0] = mx.contiguous(conv_input[:, -(self.conv_kernel_size - 1) :, :])
    conv_out = nn.silu(self.conv1d(conv_input))

    q, k, v = [
        t.reshape(B, S, h, d)
        for t, h, d in zip(
            mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
            [self.num_k_heads, self.num_k_heads, self.num_v_heads],
            [self.head_k_dim, self.head_k_dim, self.head_v_dim],
        )
    ]
    state0 = cache[1]
    if state0 is None:
        state0 = mx.zeros((B, self.num_v_heads, self.head_v_dim, self.head_k_dim), dtype=mx.float32)
    inv_scale = k.shape[-1] ** -0.5
    q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
    k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

    out, state = gated_delta_update(q, k, v, a, b, self.A_log, self.dt_bias, state0, None)
    cache[1] = state
    cache.advance(S)
    _RECORD[id(self)] = (conv_state, state0, qkv, q, k, v, a, b)

    out = self.norm(out, z)
    return self.out_proj(out.reshape(B, S, -1))


def _replay_gdn(layer, rec, r: int, m: int) -> list[mx.array]:
    """Conv + recurrent state of row `r` after its first `m` recorded tokens."""
    conv_state, state0, qkv, q, k, v, a, b = rec
    conv_input = mx.concatenate([conv_state[r : r + 1], qkv[r : r + 1, :m]], axis=1)
    sl = lambda t: t[r : r + 1, :m]  # noqa: E731
    _, state = gated_delta_update(
        sl(q), sl(k), sl(v), sl(a), sl(b), layer.A_log, layer.dt_bias, state0[r : r + 1], None
    )
    return [mx.contiguous(conv_input[:, -(layer.conv_kernel_size - 1) :, :]), state]


# ---------------------------------------------------------------- cache helpers


def repeat_cache(cache, b: int, spare: int = 0):
    """Batch-`b` copies of a batch-1 cache; the original is left untouched. With
    `spare`, KV buffers get that much free room, so the next appends do not
    reallocate and copy the whole cache (~1 GB at 32K context)."""
    out = []
    for c in cache:
        if isinstance(c, kv8.Q8KVCache):
            n = kv8.Q8KVCache()
            n.offset = c.offset
            room = -(-(c.offset + (spare or n.step)) // n.step) * n.step - c.offset  # capacity a multiple of step
            n.keys, n.values, n.k_scale, n.v_scale = (
                mx.concatenate([mx.repeat(t, b, axis=0), mx.zeros((b, *t.shape[1:2], room, t.shape[3]), dtype=t.dtype)], axis=2)
                for t in c.state
            )
            out.append(n)
        elif isinstance(c, KVCache):
            n = KVCache()
            if c.keys is not None:
                k, v = c.state
                k, v = mx.repeat(k, b, axis=0), mx.repeat(v, b, axis=0)
                if spare:
                    pad = lambda t: mx.concatenate(  # noqa: E731
                        [t, mx.zeros((*t.shape[:2], spare, t.shape[3]), dtype=t.dtype)], axis=2
                    )
                    n.keys, n.values, n.offset = pad(k), pad(v), k.shape[2]
                else:
                    n.state = (k, v)
            out.append(n)
        elif isinstance(c, ArraysCache):
            n = ArraysCache(len(c.cache))
            n.cache = [None if x is None else mx.repeat(x, b, axis=0) for x in c.cache]
            out.append(n)
        else:
            raise TypeError(f"unsupported cache type {type(c).__name__}")
    return out


class Verifier:
    """Runs verify passes on the target and commits accepted tokens to its cache."""

    def __init__(self, rt, inner):
        self.rt = rt
        self.layers = inner.model.layers
        cls = type(self.layers[0].linear_attn)
        if cls.__call__ is not _ORIG_GDN_CALL:
            raise RuntimeError(f"{cls.__name__}.__call__ is not mlx_lm's; recording would be wrong")
        cls.__call__ = _gdn_call

    def run(self, rows: list[list[int]] | mx.array, cache, sync: bool = True):
        """Forward padded `rows` (or a ready token array) on `cache`; returns
        (preds, hidden, record, kv_offset_before). With sync=False, preds stays a
        lazy array so the caller can fold this pass into one CPU sync."""
        global _RECORD
        if isinstance(rows, mx.array):
            inp = rows
        else:
            lmax = max(len(r) for r in rows)
            inp = mx.array([r + [0] * (lmax - len(r)) for r in rows])
        off0 = next(c.offset for c in cache if isinstance(c, KVCache))
        _RECORD = {}
        try:
            logits, hidden = self.rt.forward_ar(inp, cache=cache, return_hidden=True)
            preds = mx.argmax(logits, axis=-1)
            if sync:
                preds = preds.tolist()
            rec = _RECORD
        finally:
            _RECORD = None
        return preds, hidden, rec, off0

    def commit(self, cache, pass_cache, rec, off0: int, r: int, m: int) -> None:
        """Make batch-1 `cache` hold row `r`'s first `m` tokens of a finished pass
        that started at KV offset `off0` (`pass_cache` may be `cache` itself)."""
        for layer, c, pc in zip(self.layers, cache, pass_cache):
            if isinstance(c, KVCache):
                if pc is c:
                    c.trim(c.offset - (off0 + m))
                else:
                    k, v = pc.state
                    c.state = (k[r : r + 1, :, : off0 + m], v[r : r + 1, :, : off0 + m])
            else:
                c.cache = _replay_gdn(layer.linear_attn, rec[id(layer.linear_attn)], r, m)


# ---------------------------------------------------------------- tree drafting


@dataclass
class Node:
    token: int
    parent: int  # -1 = root
    depth: int
    logp: float  # cumulative log-prob along the path
    cond: float = 0.0  # this token's own log-prob given its parent
    rank: int = 1  # its rank among the parent's candidates (1 = the head's pick)
    p1: float = 1.0  # probability the head gave the parent's top-1 candidate
    children: list[int] = field(default_factory=list)


def node_path(nodes: list[Node], i: int) -> list[int]:
    out = []
    while i != -1:
        out.append(i)
        i = nodes[i].parent
    return out[::-1]


TOP_N = 8  # candidates remembered per expanded node, for the rank statistics


def ranked(logp: mx.array, n: int) -> list[tuple[int, float]]:
    """The `n` most probable (token, log-prob) pairs, best first."""
    idx = mx.argpartition(-logp, kth=n - 1)[:n]
    vals = logp[idx]
    order = mx.argsort(-vals)
    return list(zip(idx[order].tolist(), vals[order].tolist()))


def pick_children(
    top: list[tuple[int, float]], k: int, gate: float | None, nucleus: float | None = None
) -> list[tuple[int, float]]:
    """The first k of `top`; just the top-1 when its probability is at least
    `gate`; with `nucleus`, the fewest of them whose probabilities add up to
    `nucleus` (1 child when the head is sure, more as it gets unsure)."""
    out = top[:k]
    if gate is not None and math.exp(out[0][1]) >= gate:
        return out[:1]
    if nucleus is not None:
        total = 0.0
        for n, (_, lp) in enumerate(out, 1):
            total += math.exp(lp)
            if total >= nucleus:
                return out[:n]
    return out


# ---------------------------------------------------------------- adaptive policy

# Target forward time (ms) by verify tokens on this M5 Max, measured with
# profile_verify.py (batch-1 rows, 233-token context) and made monotone.
# Adaptive rescales it online for thermal state and context length.
VERIFY_MS = (
    (1, 40.6), (2, 41.8), (3, 43.6), (4, 45.8), (5, 51.7), (6, 66.1), (8, 78.6),
    (10, 89.7), (12, 112.6), (24, 116.2), (48, 118.9),
)


def measure_verify_curve(rt, inner, tok, points=(1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32, 48), context=None):
    """Re-measure VERIFY_MS (and Adaptive.ROW_MS) with whatever kernels are
    installed, on a ~200-token context or on the given `context` tokens (verify
    cost grows with context: attention reads the whole KV cache every pass).
    Used with --fast-verify / --packed, which change every pass's cost."""
    global VERIFY_MS, VERIFY_MS_LONG, VERIFY_MS_CTX
    prompt = context or tok.apply_chat_template(
        [{"role": "user", "content": "Describe a long walk through a forest. " * 20}],
        add_generation_prompt=True, enable_thinking=False,
    )
    base = prefill(rt, inner, prompt)[0]
    mx.eval([c.state for c in base])

    def ms(rows: int, length: int) -> float:
        inp = mx.array([[prompt[(r + i) % len(prompt)] for i in range(length)] for r in range(rows)])
        ts = []
        for i in range(7):
            c = repeat_cache(base, rows, spare=256)
            mx.eval([s for s in (x.state for x in c) if s is not None])
            mx.synchronize()
            t = time.perf_counter()
            mx.eval(rt.forward_ar(inp, cache=c, return_hidden=True))
            mx.synchronize()
            if i >= 2:
                ts.append((time.perf_counter() - t) * 1e3)
        return sorted(ts)[len(ts) // 2]

    curve, best = [], 0.0
    for n in points:
        best = max(best, ms(1, n))  # keep it monotone
        curve.append((n, best))
    if context is not None:
        VERIFY_MS_LONG = tuple(curve)
        VERIFY_MS_CTX = (VERIFY_MS_CTX[0], len(prompt))
        return VERIFY_MS_LONG, None
    VERIFY_MS = tuple(curve)
    VERIFY_MS_CTX = (len(prompt), VERIFY_MS_CTX[1])
    Adaptive.ROW_MS = max(0.0, (ms(3, 4) - ms(1, 12)) / 2)
    return VERIFY_MS, Adaptive.ROW_MS


# The same curve measured at a long context, and the two context lengths:
# verify_ms_curve blends them by CONTEXT, the live context length the decoding
# loops keep up to date (attention reads the whole KV cache, so verify cost
# grows linearly with context).
VERIFY_MS_LONG: tuple | None = None
VERIFY_MS_CTX = (233, None)  # (short, long) context of the two curves
CONTEXT = 0


def set_context(n: int) -> None:
    global CONTEXT
    CONTEXT = n


def interp_curve(curve, n: int) -> float:
    for (n0, t0), (n1, t1) in zip(curve, curve[1:]):
        if n <= n1:
            return t0 + (t1 - t0) * (max(n, n0) - n0) / (n1 - n0)
    return curve[-1][1]


def context_blend(short: float, long: float, ctx: int, short_ctx: int, long_ctx: int) -> float:
    """Linear in context between two measurements (extrapolated past them)."""
    if long_ctx <= short_ctx:
        return short
    return max(0.5 * min(short, long), short + (long - short) * (ctx - short_ctx) / (long_ctx - short_ctx))


def verify_ms_curve(n: int) -> float:
    short = interp_curve(VERIFY_MS, n)
    if VERIFY_MS_LONG is None:
        return short
    return context_blend(short, interp_curve(VERIFY_MS_LONG, n), CONTEXT, *VERIFY_MS_CTX)


# Measured with rank_probe.py (4 prompts, 1,286 draft verdicts): how often the
# MTP head's top-1 is accepted, by the probability it gave it. The head is
# overconfident from 0.7 to 0.95. Prior for the online calibration.
TOP1_ACCEPT = (
    (0.0, 0.14), (0.3, 0.31), (0.5, 0.53), (0.7, 0.59), (0.8, 0.67), (0.9, 0.68), (0.95, 0.96),
)
# Where the target's token ranked in the head's distribution when the top-1 was
# rejected (rank_probe.py, 222 rejections): shares of ranks 2..8; the rest is
# beyond 8. Rejections at depth 1 sit closer to the top. Prior for the online
# rank model.
RANK_SHARE_PRIOR = {
    1: (0.527, 0.200, 0.091, 0.027, 0.027, 0.027, 0.027),  # depth 1
    2: (0.400, 0.130, 0.080, 0.038, 0.038, 0.038, 0.038),  # depth 2+
}


class Adaptive:
    """Policy state that adapts while decoding, used by `auto` configs.

    - acceptance model: the head's top-1 is accepted with a calibrated rate (by
      its probability; overconfident heads are corrected), and an alternative of
      rank r with (1 - top-1 rate) x the share of rejections whose target token
      had rank r. Raw MTP probabilities of alternatives (~0.1) say little; their
      rank does. Both start from rank_probe.py's measurements and keep learning.
    - cost: verify-time curve x an online scale, plus a cost per extra row (each
      branch duplicates the linear-attention state and its prefix)
    - drafting stops once no frontier node is likely enough to pay for another
      MTP call and the extra verify time
    """

    BINS = 20
    PRIOR = 8.0  # pseudo-verdicts behind each calibration bin
    RANK_PRIOR = 20.0  # pseudo-rejections behind each rank histogram
    MAX_TOKENS = 48

    ROW_MS = 5.0  # extra verify cost per additional row; measure_verify_curve updates it

    def __init__(self, row_ms: float | None = None, fixed_ms: float = 6.0):
        row_ms = self.ROW_MS if row_ms is None else row_ms
        self.hits = [0.0] * self.BINS
        self.seen = [0.0] * self.BINS
        # ranks 2..8, then "beyond 8", per depth group
        self.rank_counts = {
            g: [self.RANK_PRIOR * s for s in shares] + [self.RANK_PRIOR * (1 - sum(shares))]
            for g, shares in RANK_SHARE_PRIOR.items()
        }
        self.row_ms, self.fixed_ms = row_ms, fixed_ms
        self.scale = 1.0
        self.draft_ms = 3.0  # per MTP call
        self.ms_per_token = 15.0

    def _bin(self, p: float) -> int:
        return min(int(p * self.BINS), self.BINS - 1)

    @staticmethod
    def _prior_top1(p: float) -> float:
        return next(a for lo, a in reversed(TOP1_ACCEPT) if p >= lo)

    def cal(self, p1: float) -> float:
        """Rate at which a top-1 the head gave probability `p1` gets accepted."""
        b = self._bin(p1)
        return (self.hits[b] + self.PRIOR * self._prior_top1(p1)) / (self.seen[b] + self.PRIOR)

    def share(self, depth: int, rank: int) -> float:
        counts = self.rank_counts[1 if depth == 1 else 2]
        return counts[rank - 2] / sum(counts) if rank - 2 < len(counts) - 1 else 0.0

    def accept(self, node: Node) -> float:
        """Probability that `node` is accepted given its parent was."""
        a1 = self.cal(node.p1)
        return a1 if node.rank == 1 else (1 - a1) * self.share(node.depth, node.rank)

    def prob(self, nodes: list[Node], i: int) -> float:
        """Probability that the whole path to node `i` is accepted."""
        p = 1.0
        while i != -1:
            p *= self.accept(nodes[i])
            i = nodes[i].parent
        return p

    def _raw_ms(self, tokens: int, rows: int) -> float:
        return verify_ms_curve(tokens) + self.row_ms * max(rows - 1, 0) + self.fixed_ms

    def step_ms(self, tokens: int, rows: int) -> float:
        return self.scale * self._raw_ms(tokens, rows)

    def keep_drafting(self, prob: float, row_tokens: int) -> bool:
        """Is drafting a child worth it? It must repay the MTP call plus the extra
        verify time of growing its row to `row_tokens`, in expected tokens."""
        extra = self.step_ms(row_tokens, 1) - self.step_ms(row_tokens - 1, 1)
        return prob * self.ms_per_token >= self.draft_ms + extra

    def observe(self, nodes, tops, acc, bonus, tokens, rows, draft_ms_per_call, verify_ms, step_ms):
        """Learn from one verify step. The accepted tokens walk the tree from the
        root, and after the root and each accepted node the target's next token is
        known. For every one of them the head ranked candidates for (`tops`), record
        whether its top-1 was right and, if not, where the target's token ranked."""
        x, depth = -1, 0
        for t in acc + [bonus]:
            if x in tops:
                top = tops[x]
                rank = next((r for r, (tk, _) in enumerate(top, 1) if tk == t), len(top) + 1)
                b = self._bin(math.exp(top[0][1]))
                self.seen[b] += 1
                self.hits[b] += rank == 1
                if rank > 1:
                    counts = self.rank_counts[1 if depth == 0 else 2]
                    counts[min(rank, len(counts) + 1) - 2] += 1
            kids = [i for i, n in enumerate(nodes) if n.parent == -1] if x == -1 else nodes[x].children
            x = next((c for c in kids if nodes[c].token == t), None)
            depth += 1
            if x is None:
                break
        self.scale = 0.8 * self.scale + 0.2 * verify_ms / self._raw_ms(tokens, rows)
        if draft_ms_per_call is not None:
            self.draft_ms = 0.8 * self.draft_ms + 0.2 * draft_ms_per_call
        self.ms_per_token = 0.8 * self.ms_per_token + 0.2 * step_ms / (len(acc) + 1)


class DepthPolicy:
    """Picks each step's chain depth BEFORE drafting, for the single-sync chain.

    The per-depth decisions of `Adaptive` need the draft's probabilities, i.e. a
    CPU sync after every draft token. This instead learns, per context, the rate
    at which chain position j is accepted given position j - 1 was, and picks the
    depth D maximising expected tokens / predicted step time:

        E(D) = 1 + sum_{j<=D} prod_{i<=j} rate(ctx, i)
        T(D) = verify time of D + 1 tokens + (D - 1) MTP draft calls (+ fixed)

    Context = how the previous step went (all accepted / some / none): acceptance
    comes in streaks (boilerplate, code, lists vs. free prose). Counts decay so
    the rates follow the text as it changes. Untried depths keep an optimistic
    prior, so deeper chains get explored. Timing reuses Adaptive's verify-time
    curve, with one online scale fitted to measured whole-step time.
    """

    PRIOR_RATE = 0.8
    PRIOR_N = 3.0
    DECAY = 0.97

    def __init__(self, max_depth: int, draft_call_ms: float):
        self.max_depth = max_depth
        self.draft_call_ms = draft_call_ms
        self.cost = Adaptive()
        self.tries = {c: [0.0] * max_depth for c in range(3)}
        self.hits = {c: [0.0] * max_depth for c in range(3)}
        self.ctx = 2
        self.depths: list[int] = []

    def rate(self, ctx: int, j: int) -> float:
        return (self.hits[ctx][j] + self.PRIOR_N * self.PRIOR_RATE) / (self.tries[ctx][j] + self.PRIOR_N)

    def _raw_ms(self, d: int) -> float:
        return self.cost._raw_ms(d + 1, 1) + (d - 1) * self.draft_call_ms

    def choose(self) -> int:
        best_d, best_rate, expected, p = 1, 0.0, 1.0, 1.0
        for d in range(1, self.max_depth + 1):
            p *= self.rate(self.ctx, d - 1)
            expected += p
            r = expected / (self.cost.scale * self._raw_ms(d))
            if r > best_rate:
                best_d, best_rate = d, r
        self.depths.append(best_d)
        return best_d

    def observe(self, depth: int, n: int, step_ms: float) -> None:
        tries, hits = self.tries[self.ctx], self.hits[self.ctx]
        for j in range(self.max_depth):
            tries[j] *= self.DECAY
            hits[j] *= self.DECAY
        # Position j was tried if every earlier one was accepted.
        for j in range(min(n + 1, depth)):
            tries[j] += 1
            hits[j] += j < n
        self.ctx = 2 if n == depth else (1 if n > 0 else 0)
        self.cost.scale = 0.8 * self.cost.scale + 0.2 * step_ms / self._raw_ms(depth)


def select_auto(nodes: list[Node], keep: set[int], policy: Adaptive) -> list[list[int]]:
    """Linear-attention-aware rows maximising expected tokens per millisecond.

    Grow the row set one node at a time. Extending a row by its next node costs
    one verify token; any other node needs a NEW row that repeats the bonus token
    and the node's whole prefix (depth + 1 tokens), because branches cannot share
    recurrent state. So a fork deep in the draft costs a whole duplicate row and is
    rarely worth it, while extending a likely chain is cheap. Each time, take the
    node with the best calibrated probability per added millisecond, then return
    the point of this growth with the best (expected accepted + 1) / predicted
    step time. An empty draft (a plain decode step) competes too.
    """
    prob = {i: policy.prob(nodes, i) for i in keep}
    ends: list[int] = []  # last node of each row
    covered: set[int] = set()
    tokens, expected = 1, 0.0
    best_rate, best_ends = 1.0 / policy.step_ms(1, 1), []
    while True:
        cur_ms = policy.step_ms(tokens, max(len(ends), 1))
        best = None
        for i in keep:
            par = nodes[i].parent
            if i in covered or (par != -1 and par not in covered):
                continue
            if par in ends:
                add, new_ends = 1, [i if e == par else e for e in ends]
            elif not ends:
                add, new_ends = nodes[i].depth, [i]  # first row: bonus already counted
            else:
                add, new_ends = nodes[i].depth + 1, ends + [i]
            if tokens + add > policy.MAX_TOKENS:
                continue
            d_ms = policy.step_ms(tokens + add, len(new_ends)) - cur_ms
            score = prob[i] / max(d_ms, 0.5)
            if best is None or score > best[0]:
                best = (score, i, add, new_ends)
        if best is None:
            break
        _, i, add, ends = best
        covered.add(i)
        tokens += add
        expected += prob[i]
        rate = (expected + 1) / policy.step_ms(tokens, len(ends))
        if rate > best_rate:
            best_rate, best_ends = rate, list(ends)
    return [node_path(nodes, e) for e in best_ends]


class QHead(nn.QuantizedLinear):
    """A quantized output layer built from ready weight/scales/biases rows
    (nn.QuantizedLinear.__init__ would first allocate and quantize random
    weights). Being a QuantizedLinear, --packed repacks it like the target's."""

    def __init__(self, weight, scales, biases, group_size: int, bits: int):
        nn.Module.__init__(self)
        self.weight, self.scales, self.biases = weight, scales, biases
        self.group_size, self.bits, self.mode = group_size, bits, "affine"


def requantize_rows(weight, scales, biases, group_size: int, bits_in: int, bits_out: int, chunk: int = 16384):
    """Re-quantize an affine-quantized matrix to `bits_out`, in row chunks so the
    dequantized copy never exists all at once."""
    ws, ss, bs = [], [], []
    for s in range(0, weight.shape[0], chunk):
        w = mx.dequantize(
            weight[s : s + chunk], scales[s : s + chunk], biases[s : s + chunk],
            group_size=group_size, bits=bits_in,
        )
        q = mx.quantize(w, group_size=group_size, bits=bits_out)
        mx.eval(q)
        for out, x in zip((ws, ss, bs), q):
            out.append(x)
    return mx.concatenate(ws), mx.concatenate(ss), mx.concatenate(bs)


class Drafter:
    """The MTP head's draft path, trimmed to what drafting needs.

    rt.draft_mtp runs the full 248,320-row output layer on every position it is
    given; that layer is ~70% of a draft call (1.9 of 2.7 ms). Here logits are
    computed for the last position only, optionally over FR-Spec's pruned vocab
    (`ids`: the 64K most frequent tokens; MTPLX ships the list), optionally
    re-quantized to fewer `bits`, and the top candidates of every row come back
    in one CPU sync. With `compile`, the output layer + argmax used by greedy
    chains is mx.compile'd (the MTP layer itself is not: its KV cache grows every
    call, and a compiled graph retraces on every new shape). Verification still
    uses the target's full vocabulary, so none of this can change output, only
    which tokens get drafted.
    """

    def __init__(self, rt, inner, ids: mx.array | None = None, bits: int | None = None, compile: bool = False):
        self.rt, self.inner = rt, inner
        self.variant = rt.contract.hidden_variant
        self.concat_order = rt.contract.concat_order
        if self.variant not in ("post_norm", "pre_norm"):
            raise ValueError(f"unsupported MTP hidden variant {self.variant!r}")
        lm = inner.lm_head
        w, s, b, hb = lm.weight, lm.scales, lm.biases, lm.bits
        if ids is not None:
            w, s, b = w[ids], s[ids], b[ids]
        if bits and bits != hb:
            w, s, b = requantize_rows(w, s, b, lm.group_size, hb, bits)
            hb = bits
        self.head = QHead(w, s, b, lm.group_size, hb)
        mx.eval(self.head.parameters())
        self.ids_arr = ids
        self.ids = ids.tolist() if ids is not None else None
        self.bits = hb
        # Rough ms per MTP draft call (profiled: MTP layer ~0.9 ms, full 4-bit
        # output layer ~1.9 ms, scaling with rows and bits), for DepthPolicy.
        self.call_ms = 1.1 + 1.9 * (w.shape[0] / 248320) * (hb / 4)
        argmax = lambda g: mx.argmax(self._logits(g), axis=-1)  # noqa: E731
        self._argmax = mx.compile(argmax) if compile else argmax

    def _logits(self, g):
        return self.head(g if self.variant == "post_norm" else self.inner.mtp.norm(g))

    def next_token(self, g_last):
        """Greedy next token of each row as a lazy (B,) array: no CPU sync."""
        j = self._argmax(g_last)
        return self.ids_arr[j] if self.ids_arr is not None else j

    def root_lazy(self, mtp_cache, hidden, next_ids):
        """root(), but returning the greedy root token lazily instead of a ranking."""
        g = self.step(hidden, next_ids, mtp_cache)[:, -1, :]
        return self.next_token(g), g

    def chain_lazy(self, mtp_cache, tok, g, depth: int):
        """Greedy chain of `depth` tokens from root token `tok` (B,) and its hidden
        `g`, built as one lazy graph. Each step runs one new (hidden, token) entry
        on `mtp_cache` instead of re-running the whole path; mtp_cache is left as
        it was. One row over a long history goes through tree_verify.draft_tree
        (which keeps the history read-only; below MIN_CONTEXT MTPLX's unmasked
        path is faster). Returns tokens (B, depth) and the last hidden."""
        if tok.shape[0] == 1 and depth > 1 and mtp_cache[0].offset >= kv8.MIN_CONTEXT:
            from .tree_verify import draft_tree

            drafted, _, _, g = draft_tree(self, self.inner, mtp_cache, tok.astype(mx.uint32), g, [depth], runner_ups=False)
            return mx.concatenate([tok.astype(mx.uint32), drafted])[None], g
        toks = [tok]
        for _ in range(depth - 1):
            g = self.step(g[:, None, :], toks[-1][:, None], mtp_cache)[:, -1, :]
            toks.append(self.next_token(g))
        for c in mtp_cache:
            c.trim(depth - 1)
        return mx.stack(toks, axis=1), g

    def step(self, hidden, toks, mtp_cache):
        """Run the MTP layer on (hidden, token) pairs; returns its hidden output."""
        _, g = self.inner._mtp_core(
            hidden, toks, mtp_cache=mtp_cache, concat_order=self.concat_order,
            mtp_hidden_variant=self.variant, emit_logits=False,
        )
        return g

    def top(self, g_last, n: int) -> list[list[tuple[int, float]]]:
        """Top-n (token, log-prob) of each row's next-token distribution."""
        z = self._logits(g_last).astype(mx.float32)
        logp = z - mx.logsumexp(z, axis=-1, keepdims=True)
        idx = mx.argpartition(-logp, kth=n - 1, axis=-1)[:, :n]
        vals = mx.take_along_axis(logp, idx, axis=-1)
        order = mx.argsort(-vals, axis=-1)
        idx, vals = mx.take_along_axis(idx, order, -1), mx.take_along_axis(vals, order, -1)
        mx.eval(idx, vals)
        rows_i, rows_v = idx.tolist(), vals.tolist()
        if self.ids is not None:
            rows_i = [[self.ids[j] for j in r] for r in rows_i]
        return [list(zip(i, v)) for i, v in zip(rows_i, rows_v)]

    def root(self, mtp_cache, hidden, next_ids, n: int):
        """Append committed (hidden, next token) pairs to the MTP history; return
        the ranked candidates after the last one (the tree root) and its hidden."""
        g = self.step(hidden, next_ids, mtp_cache)[:, -1, :]
        return self.top(g, n)[0], g


def build_tree(
    drafter: Drafter, mtp_cache, root_top, root_g, *,
    depth, topk, width, budget, gate, cap=None, nucleus=None, policy=None,
):
    """Draft a tree and pick the verify rows.

    Returns (token paths, nodes used, info) where info has the node list, the rows
    as node-id paths and the MTP drafting time. Rows are: every leaf by default;
    with `cap`, the best rows within `cap` verify tokens; with `policy`, the
    adaptive linear-attention-aware choice (select_auto)."""
    score = (lambda i: policy.prob(nodes, i)) if policy else (lambda i: nodes[i].logp)
    top = root_top
    tops = {-1: top}  # expanded node (-1 = root) -> its ranked candidates
    nodes = [
        Node(t, -1, 1, lp, lp, rank=r, p1=math.exp(top[0][1]))
        for r, (t, lp) in enumerate(pick_children(top, topk, gate, nucleus), 1)
    ]
    g_of: dict[int, mx.array] = {}  # node -> MTP output hidden after its entry
    calls, draft_ms = 0, 0.0
    for d in range(1, depth):
        frontier = sorted((i for i, n in enumerate(nodes) if n.depth == d), key=lambda i: -score(i))[:width]
        if policy:
            # Only expand nodes likely enough to pay for drafting their children.
            frontier = [i for i in frontier if policy.keep_drafting(policy.prob(nodes, i), d + 2)]
        if not frontier:
            break
        # Each frontier node re-runs its whole draft path as one batch row on top of
        # the committed MTP history (paths are <= depth tokens, so this is cheap).
        t0 = time.perf_counter()
        paths = [node_path(nodes, i) for i in frontier]
        hid = mx.stack(
            [mx.concatenate([root_g] + [g_of[a] for a in p[:-1]], axis=0) for p in paths]
        )
        toks = mx.array([[nodes[a].token for a in p] for p in paths])
        g = drafter.step(hid, toks, repeat_cache(mtp_cache, len(paths)))[:, -1, :]
        row_tops = drafter.top(g, max(TOP_N, topk))
        for row, i in enumerate(frontier):
            g_of[i] = g[row : row + 1]
            top = tops[i] = row_tops[row]
            for r, (t, lp) in enumerate(pick_children(top, topk, gate, nucleus), 1):
                nodes.append(
                    Node(t, i, d + 1, nodes[i].logp + lp, lp, rank=r, p1=math.exp(top[0][1]))
                )
                nodes[i].children.append(len(nodes) - 1)
        calls += 1
        draft_ms += (time.perf_counter() - t0) * 1e3

    # Keep the `budget` most probable nodes; a child is never more probable than
    # its parent, so this set is closed under ancestors (checked anyway).
    keep: set[int] = set()
    for i in sorted(range(len(nodes)), key=lambda i: -score(i)):
        if len(keep) >= budget:
            break
        if nodes[i].parent == -1 or nodes[i].parent in keep:
            keep.add(i)
    if policy is not None:
        rows = select_auto(nodes, keep, policy)
    elif cap is not None:
        rows = select_rows(nodes, keep, cap)
    else:
        leaves = [i for i in keep if not any(c in keep for c in nodes[i].children)]
        leaves.sort(key=lambda i: -nodes[i].logp)
        rows = [node_path(nodes, i) for i in leaves]
    info = dict(
        nodes=nodes, tops=tops, id_paths=rows, draft_ms_per_call=draft_ms / calls if calls else None
    )
    used = len({a for p in rows for a in p})
    return [[nodes[a].token for a in p] for p in rows] or [[]], used, info


def select_rows(nodes: list[Node], keep: set[int], cap: int) -> list[list[int]]:
    """Pick the verify rows (root-to-node paths) that maximise the expected number
    of accepted tokens while costing at most `cap` tokens in the target forward.

    A row costs len(path) + 1 (the bonus token is repeated in every row). Under
    greedy verification the expected accepted length is the sum, over every
    distinct node the rows cover, of the probability that its whole path matches;
    the MTP head's cumulative probability stands in for that.
    """
    prob = {i: math.exp(nodes[i].logp) for i in keep}
    paths = [node_path(nodes, i) for i in keep if nodes[i].depth + 1 <= cap]
    paths.sort(key=lambda p: -prob[p[-1]])
    if cap > 8:
        # Exact search explodes for big caps: greedily add the path with the best
        # newly covered probability per token of cost.
        chosen, covered, cost = [], set(), 0
        while True:
            fits = [p for p in paths if cost + len(p) + 1 <= cap and not set(p) <= covered]
            if not fits:
                return chosen
            p = max(fits, key=lambda p: sum(prob[i] for i in p if i not in covered) / (len(p) + 1))
            chosen.append(p)
            covered |= set(p)
            cost += len(p) + 1
    best_val, best = -1.0, []

    def search(start, chosen, covered, cost):
        nonlocal best_val, best
        val = sum(prob[i] for i in covered)
        if val > best_val:
            best_val, best = val, list(chosen)
        for j in range(start, len(paths)):
            c = cost + len(paths[j]) + 1
            if c <= cap and not set(paths[j]) <= covered:
                search(j + 1, chosen + [paths[j]], covered | set(paths[j]), c)

    search(0, [], frozenset(), 0)
    return best


def common_prefix(paths: list[list[int]]) -> list[int]:
    out = []
    for toks in zip(*paths):
        if any(t != toks[0] for t in toks):
            break
        out.append(toks[0])
    return out


def matched(path: list[int], preds: list[int]) -> int:
    n = 0
    while n < len(path) and path[n] == preds[n]:
        n += 1
    return n


# ---------------------------------------------------------------- decoding loops

PREFILL_CHUNK = 2048
_PREFILLED: dict = {}


def mtp_root(rt, mtp_cache, hidden, next_ids):
    """Append committed (hidden, next token) pairs to the MTP history and return
    the MTP distribution for the token after the last one (the tree root)."""
    logits, g = rt.draft_mtp(hidden, next_ids, mtp_cache=mtp_cache, return_hidden=True)
    last = logits[0, -1, :].astype(mx.float32)
    return last - mx.logsumexp(last), g[0, -1:, :]


def prefill(rt, inner, prompt: list[int], chunk: int = PREFILL_CHUNK):
    """Prefill `prompt` on the target and the MTP history, in `chunk`-token pieces
    like MTPLX (one forward over a 32K prompt would also emit ~4 GB of logits).
    Returns fresh copies of (target cache, MTP cache), the first generated token,
    the MTP hidden after the last prompt entry (the first tree root) and the
    prefill seconds. The last prompt's result is memoised, so comparing configs
    on one long prompt prefills it once."""
    key = tuple(prompt)
    if _PREFILLED.get("key") != key:
        _PREFILLED.clear()
        cache = inner.make_cache()
        # Plain KVCaches (not MTPLX's paged ones) so repeat_cache can copy them per row.
        mtp_cache = [KVCache() for _ in rt.model.mtp.layers]
        t0 = time.perf_counter()
        for i in range(0, len(prompt), chunk):
            piece = prompt[i : i + chunk]
            logits, hidden = rt.forward_ar(mx.array([piece]), cache=cache, return_hidden=True, logits_keep=1)
            if i + chunk >= len(prompt):
                bonus = int(mx.argmax(logits[0, -1]).item())
                nxt = prompt[i + 1 :] + [bonus]
            else:
                nxt = prompt[i + 1 : i + chunk + 1]
            _, g = inner._mtp_core(
                hidden, mx.array([nxt]), mtp_cache=mtp_cache, concat_order=rt.contract.concat_order,
                mtp_hidden_variant=rt.contract.hidden_variant, emit_logits=False,
            )
            mx.eval(g, [c.state for c in cache], [c.state for c in mtp_cache])
        kv8.quantize(cache)  # long prompts: decode reads an 8-bit KV cache (with --kv8)
        _PREFILLED.update(
            key=key, cache=cache, mtp=mtp_cache, bonus=bonus, root_g=g[:, -1, :], seconds=time.perf_counter() - t0
        )
    p = _PREFILLED
    return repeat_cache(p["cache"], 1, spare=256), repeat_cache(p["mtp"], 1, spare=256), p["bonus"], p["root_g"], p["seconds"]


def decode_ar(rt, inner, prompt, max_new, eos):
    cache, _, tok, _, pf_s = prefill(rt, inner, prompt)
    mx.eval([c.state for c in cache])
    t_first = time.perf_counter()
    t0 = t_first - pf_s
    out = [tok]
    while len(out) < max_new and tok not in eos:
        kv8.quantize(cache)
        logits = rt.model(mx.array([[tok]]), cache=cache)
        tok = int(mx.argmax(logits[0, -1]).item())
        out.append(tok)
    t1 = time.perf_counter()
    return out, dict(steps=len(out) - 1, decode_s=t1 - t_first, prefill_s=t_first - t0)


def verify_step(ver: Verifier, cache, bonus: int, paths: list[list[int]], share_min: int):
    """Verify draft `paths` after `bonus`, commit the accepted tokens to `cache`.
    Returns (accepted draft tokens, new bonus token, hidden of committed tokens, stats)."""
    trunk = common_prefix(paths) if len(paths) > 1 else []
    share = len(trunk) > 0 and (len(paths) - 1) * len(trunk) >= share_min

    if len(paths) == 1 or not share:
        # One pass: every path is a row (a chain is one row on the real cache).
        pass_cache = cache if len(paths) == 1 else repeat_cache(cache, len(paths))
        rows = [[bonus] + p for p in paths]
        preds, hidden, rec, off0 = ver.run(rows, pass_cache)
        ns = [matched(p, pr) for p, pr in zip(paths, preds)]
        r = max(range(len(paths)), key=lambda i: ns[i])
        n = ns[r]
        ver.commit(cache, pass_cache, rec, off0, r, n + 1)
        return paths[r][:n], preds[r][n], hidden[r : r + 1, : n + 1], dict(
            passes=1, verify_tokens=sum(len(x) for x in rows), ns=ns
        )

    # Pass A: the shared trunk once, at batch 1, on the real cache.
    preds_a, hidden_a, rec_a, off_a = ver.run([[bonus] + trunk], cache)
    n_t = matched(trunk, preds_a[0])
    if n_t < len(trunk):
        ver.commit(cache, cache, rec_a, off_a, 0, n_t + 1)
        return trunk[:n_t], preds_a[0][n_t], hidden_a[:, : n_t + 1], dict(
            passes=1, verify_tokens=len(trunk) + 1, ns=[n_t] * len(paths)
        )

    # Pass B: the diverging suffixes, batched, starting from the end of the trunk.
    sufs = [p[len(trunk) :] for p in paths]
    pass_cache = repeat_cache(cache, len(sufs))
    preds_b, hidden_b, rec_b, off_b = ver.run(sufs, pass_cache)
    # Row r's first suffix token is predicted by the last trunk position.
    ns = [matched(s, [preds_a[0][-1]] + pr) for s, pr in zip(sufs, preds_b)]
    r = max(range(len(sufs)), key=lambda i: ns[i])
    n = ns[r]
    stats = dict(
        passes=2,
        verify_tokens=len(trunk) + 1 + sum(len(s) for s in sufs),
        ns=[len(trunk) + x for x in ns],
    )
    if n == 0:
        # Nothing past the trunk: the real cache already ends at the trunk.
        return trunk, preds_a[0][-1], hidden_a, stats
    ver.commit(cache, pass_cache, rec_b, off_b, r, n)
    hidden = mx.concatenate([hidden_a, hidden_b[r : r + 1, :n]], axis=1)
    return trunk + sufs[r][:n], preds_b[r][n - 1], hidden, stats


def decode_spec(
    rt, inner, ver, drafter, prompt, max_new, eos, *,
    depth, topk, width, budget, gate, cap, nucleus, auto, share_min,
):
    policy = Adaptive() if auto else None
    n_top = max(TOP_N, topk)
    cache, mtp_cache, bonus, root_g, pf_s = prefill(rt, inner, prompt)
    root_top = drafter.top(root_g, n_top)[0]
    mx.eval([c.state for c in cache + mtp_cache])
    t_first = time.perf_counter()
    t0 = t_first - pf_s

    out = [bonus]
    steps = accepted_total = rows_total = nodes_total = passes_total = vtok_total = 0
    draft_s = verify_s = 0.0
    while len(out) < max_new and bonus not in eos:
        kv8.quantize(cache)
        set_context(next(c.offset for c in cache if isinstance(c, KVCache)))
        ta = time.perf_counter()
        paths, n_nodes, info = build_tree(
            drafter, mtp_cache, root_top, root_g,
            depth=depth, topk=topk, width=width, budget=budget, gate=gate, cap=cap,
            nucleus=nucleus, policy=policy,
        )
        tb = time.perf_counter()
        acc, new_bonus, h, st = verify_step(ver, cache, bonus, paths, share_min)
        root_top, root_g = drafter.root(mtp_cache, h, mx.array([acc + [new_bonus]]), n_top)
        mx.eval(root_g, *[c.state for c in cache])
        tc = time.perf_counter()
        if policy is not None:
            policy.observe(
                info["nodes"], info["tops"], acc, new_bonus, st["verify_tokens"], len(paths),
                info["draft_ms_per_call"], (tc - tb) * 1e3, (tc - ta) * 1e3,
            )

        out.extend(acc + [new_bonus])
        bonus = new_bonus
        steps += 1
        accepted_total += len(acc)
        rows_total += len(paths)
        nodes_total += n_nodes
        passes_total += st["passes"]
        vtok_total += st["verify_tokens"]
        draft_s += tb - ta
        verify_s += tc - tb
    t1 = time.perf_counter()

    # Trim anything generated past EOS / max_new so outputs compare like-for-like.
    for i, t in enumerate(out):
        if t in eos:
            out = out[: i + 1]
            break
    out = out[:max_new]
    s = max(steps, 1)
    return out, dict(
        steps=steps,
        decode_s=t1 - t_first,
        prefill_s=t_first - t0,
        tokens_per_step=(accepted_total + steps) / s,
        mean_rows=rows_total / s,
        mean_nodes=nodes_total / s,
        passes_per_step=passes_total / s,
        verify_tokens_per_step=vtok_total / s,
        draft_s=draft_s,
        verify_s=verify_s,
    )


def decode_chain(rt, inner, ver, drafter, prompt, max_new, eos, *, depth, overlap=False, auto=False):
    """A greedy chain with ONE CPU sync per step.

    With `auto`, each step's chain length (1..depth) is picked by DepthPolicy
    before drafting; otherwise it is always `depth`.

    The chain is drafted as a lazy graph (no per-token syncs), started on the GPU
    with async_eval while Python builds the verify graph, and verified straight
    from the GPU token array; the CPU reads drafts and predictions once.

    With `overlap`, during verification a second GPU stream drafts the NEXT step's
    chain, assuming every draft token is accepted and the bonus token is the MTP
    head's own guess (with its hidden state standing in for the target's). When
    both hold, that chain is used and drafting drops out of the step; otherwise
    it is thrown away. Outputs are unaffected either way: the target verifies.
    """
    spec_stream = mx.new_stream(mx.gpu) if overlap else None
    cache, mtp_cache, bonus, root_g, pf_s = prefill(rt, inner, prompt)
    root_tok = drafter.next_token(root_g)
    mx.eval(root_tok, [c.state for c in cache + mtp_cache])
    t_first = time.perf_counter()
    t0 = t_first - pf_s

    out = [bonus]
    steps = accepted_total = hits = depth_total = 0
    policy = DepthPolicy(depth, drafter.call_ms) if auto else None
    spec = None  # (chain tokens, last hidden, its private MTP cache) drafted last step
    while len(out) < max_new and bonus not in eos:
        kv8.quantize(cache)
        set_context(next(c.offset for c in cache if isinstance(c, KVCache)))
        ta = time.perf_counter()
        dd = policy.choose() if policy else depth
        if spec is not None:
            draft, g_last, dcache = spec
        else:
            draft, g_last = drafter.chain_lazy(mtp_cache, root_tok, root_g, dd)
            dcache = mtp_cache
        mx.async_eval(draft)
        inp = mx.concatenate([mx.array([[bonus]], dtype=mx.uint32), draft.astype(mx.uint32)], axis=1)
        preds, hidden, rec, off0 = ver.run(inp, cache, sync=False)
        mx.async_eval(preds)
        guess = None
        if overlap:
            with mx.stream(spec_stream):
                sc = dcache if dcache is not mtp_cache else repeat_cache(mtp_cache, 1)
                g_end = drafter.step(g_last[:, None, :], draft[:, -1:], sc)[:, -1, :]
                guess = drafter.next_token(g_end)
                g_root = drafter.step(g_end[:, None, :], guess[:, None], sc)[:, -1, :]
                s_toks, s_g = drafter.chain_lazy(sc, drafter.next_token(g_root), g_root, depth)
                mx.async_eval(guess, s_toks, s_g)
        mx.eval(preds, draft)
        p, d = preds.tolist()[0], draft.tolist()[0]
        n = matched(d, p)
        new_bonus = p[n]
        ver.commit(cache, cache, rec, off0, 0, n + 1)
        # MTP history gets the committed tokens with the target's hidden states.
        root_tok, root_g = drafter.root_lazy(mtp_cache, hidden[:, : n + 1], mx.array([d[:n] + [new_bonus]]))
        spec = None
        if guess is not None and n == depth and new_bonus == guess.item():
            spec, hits = (s_toks, s_g, sc), hits + 1
        out.extend(d[:n] + [new_bonus])
        bonus = new_bonus
        steps += 1
        accepted_total += n
        depth_total += dd
        if policy is not None:
            # This step's commit work runs lazily inside the next step, so each
            # measured step carries the previous one's; consistent across steps.
            policy.observe(dd, n, (time.perf_counter() - ta) * 1e3)
    t1 = time.perf_counter()

    for i, t in enumerate(out):
        if t in eos:
            out = out[: i + 1]
            break
    out = out[:max_new]
    s = max(steps, 1)
    return out, dict(
        steps=steps,
        decode_s=t1 - t_first,
        prefill_s=t_first - t0,
        tokens_per_step=(accepted_total + steps) / s,
        mean_rows=1.0,
        mean_nodes=depth_total / s,
        passes_per_step=1.0,
        verify_tokens_per_step=depth_total / s + 1,
        depth_hist=dict(sorted(Counter(policy.depths).items())) if policy else None,
        draft_s=0.0,
        verify_s=t1 - t_first,
        spec_hit_rate=hits / s if overlap else None,
    )


def resolve_model(model: str) -> str:
    """A local checkpoint directory, or a Hugging Face repo id (downloaded on first use)."""
    path = Path(model).expanduser()
    if path.is_dir():
        return str(path)
    from huggingface_hub import snapshot_download

    return snapshot_download(model)


def gpu_problem(macos_version: str, architecture: str | None, device_name: str) -> str | None:
    """Why this machine can't run the packed verify kernels (None if it can).
    They use the GPU tensor units (NAX) of M5-class GPUs on macOS 26.2+; there is
    no fallback for M1-M4 yet."""
    from mtplx.nax_detect import nax_available_for_platform

    if nax_available_for_platform(macos_version, architecture):
        return None
    return (
        "mlx-tree-spec needs an Apple M5-family GPU and macOS 26.2 or later: its verify kernels use the M5 GPU's "
        f"neural accelerators and have no fallback for M1-M4 yet. This machine: {device_name} "
        f"({architecture or 'unknown GPU'}), macOS {macos_version or 'unknown'}."
    )


def require_m5_gpu() -> None:
    """Exit with a readable message on a GPU the kernels don't support, before any download."""
    import platform
    import sys

    from mtplx.nax_detect import gpu_architecture

    problem = gpu_problem(platform.mac_ver()[0], gpu_architecture(), mx.device_info().get("device_name", "unknown"))
    if problem:
        sys.exit(problem)


def load_draft_vocab(spec: str) -> mx.array | None:
    """Token ids for a pruned draft output layer, sorted; None for the full vocab."""
    if spec == "full":
        return None
    if spec == "general":
        spec = str(DATA / "general_vocab.npy")
    if spec == "frspec":
        import mtplx

        spec = str(Path(mtplx.__file__).with_name("data") / "qwen38_code_ranked_64k.npy")
    if spec.endswith(".json"):
        with open(spec) as f:
            ids = np.array(json.load(f)["ids"])
    else:
        ids = np.load(spec)
    return mx.array(np.unique(ids.astype(np.int64)).astype(np.uint32))


def first_divergence(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def parse_cfg(s: str) -> dict:
    # "name:depth,topk,width,budget[,gate][,cap=N][,p=X][,auto]"
    # e.g. "gated:5,4,4,16,0.8", "opt:3,3,2,12,cap=4", "adapt:6,3,3,16,p=0.9,auto"
    # or "fork:depth,branches,branch_depth[,sibs]" (tree_verify.decode_fork)
    name, nums = s.split(":")
    parts = nums.split(",")
    if name == "fork":
        d, nb, bd, ns = (int(x) for x in parts + ["0"] * (4 - len(parts)))
        return dict(name=name, fork=True, depth=d, branches=nb, branch_depth=bd, sibs=ns)
    d, k, w, b = (int(x) for x in parts[:4])
    cfg = dict(
        name=name, depth=d, topk=k, width=w, budget=b, gate=None, cap=None, nucleus=None,
        auto=False, lazy=False, overlap=False,
    )
    for p in parts[4:]:
        if p.startswith("cap="):
            cfg["cap"] = int(p[4:])
        elif p.startswith("p="):
            cfg["nucleus"] = float(p[2:])
        elif p in ("auto", "lazy", "overlap"):
            cfg[p] = True
        else:
            cfg["gate"] = float(p.removeprefix("gate="))
    if cfg["lazy"] or cfg["overlap"]:
        plain = k == 1 and w == 1 and cfg["cap"] is None and cfg["gate"] is None and cfg["nucleus"] is None
        if not plain or (cfg["auto"] and cfg["overlap"]):
            raise ValueError(
                f"{s}: lazy/overlap need a plain chain (topk 1, width 1, no cap/gate/p); "
                "auto works with lazy, not overlap"
            )
    return cfg


def cfg_label(c: dict) -> str:
    if c.get("fork"):
        return f"fork d{c['depth']} +{c['branches']}x{c['branch_depth']} s{c['sibs']}"
    g = f" g{c['gate']:g}" if c["gate"] is not None else ""
    cap = f" cap{c['cap']}" if c["cap"] is not None else ""
    nuc = f" p{c['nucleus']:g}" if c["nucleus"] is not None else ""
    flags = "".join(f" {f}" for f in ("auto", "lazy", "overlap") if c[f])
    return f"{c['name']} d{c['depth']} k{c['topk']} w{c['width']} b{c['budget']}{g}{cap}{nuc}{flags}"


def make_drafter(rt, inner, spec: str) -> tuple[str, Drafter]:
    """'<vocab>[:bits][+compile]', e.g. 'full', 'frspec:3', 'general_vocab.npy:3+compile'.
    Returns the label suffix and the Drafter."""
    compile_ = spec.endswith("+compile")
    vocab, _, bits = spec.removesuffix("+compile").partition(":")
    d = Drafter(rt, inner, load_draft_vocab(vocab), bits=int(bits) if bits else None, compile=compile_)
    tags = []
    if d.ids is not None:
        tags.append(f"{Path(vocab).stem if vocab != 'frspec' else 'code'} {len(d.ids) // 1024}k")
    if bits:
        tags.append(f"q{bits}")
    if compile_:
        tags.append("compiled")
    return (f" [{', '.join(tags)}]" if tags else ""), d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--prompt", action="append", help="repeatable; defaults to two built-in prompts")
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--thinking", action="store_true", help="enable the chat template's thinking mode")
    ap.add_argument(
        "--config",
        action="append",
        type=parse_cfg,
        help="name:depth,topk,width,budget[,gate] or fork:depth,branches,branch_depth[,sibs] (repeatable)",
    )
    ap.add_argument(
        "--share-min",
        type=int,
        default=16,
        help="verify a shared trunk separately only if that saves at least this many row-tokens",
    )
    ap.add_argument(
        "--fast-verify",
        nargs="?",
        const="verify_matmul_table.json",
        help="route 2-16-row 4-bit matmuls to the fastest kernel (MLX or MTPLX's NAX "
        "verify kernels; see verify_matmul.py), using/creating this tuning table, "
        "and re-measure the verify cost curve. Not bit-exact vs MLX's kernels",
    )
    ap.add_argument(
        "--packed",
        action="store_true",
        help="repack the target's 4-bit linears in place for the packed kernels "
        "(packed_linear.py): any 1-16-row matmul costs about a 1-row one, 1 row as "
        "much as MLX's. Applies to the AR baseline and prefill too. Not bit-exact vs MLX's kernels",
    )
    ap.add_argument(
        "--draft-vocab",
        action="append",
        help="draft output layer '<vocab>[:bits][+compile]': vocab is 'full' (default), 'frspec' "
        "(MTPLX's 64K code vocab), 'general' (frspec plus prose tokens, what the server uses) "
        "or a .npy / .json token-id list; bits re-quantizes it (e.g. 3); "
        "+compile compiles output layer + argmax. Repeat to compare (each config runs once per drafter)",
    )
    ap.add_argument(
        "--flash-attn", action="store_true",
        help="route 2-10-query causal attention over long contexts (>= 8192 cached tokens) to MTPLX's "
        "flash-decoding kernels (long_attention.py), for every decoder alike",
    )
    ap.add_argument("--prompt-file", action="append", help="read a prompt from this text file (repeatable)")
    ap.add_argument(
        "--curve-context", action="store_true",
        help="also measure the cost curves at the longest prompt's context; the policies blend the short and "
        "long curves by the live context length (attention cost grows with it)",
    )
    ap.add_argument(
        "--cache-limit-gb", type=float, default=4.0,
        help="cap MLX's cache of freed GPU buffers (it otherwise grows across runs until the machine swaps)",
    )
    ap.add_argument(
        "--wired-gb", type=float, default=None,
        help="wire (pin) up to this much of MLX's GPU memory so macOS cannot page the model out "
        "(default: the GPU's recommended working set; 0 disables)",
    )
    ap.add_argument(
        "--kv8", action="store_true",
        help=f"8-bit KV cache for the attention layers from {kv8.MIN_CONTEXT} cached tokens on (kv8.py); "
        "not bit-exact vs bf16 KV",
    )
    ap.add_argument("--json", help="write per-prompt results to this file")
    args = ap.parse_args()
    configs = args.config or [
        parse_cfg(s)
        for s in ("chain:3,1,1,3", "tree:4,2,2,8", "gated:5,4,4,12,0.8", "gated:6,4,4,16,0.6")
    ]

    mx.set_cache_limit(int(args.cache_limit_gb * 1e9))
    # Wired buffers stay resident (Metal residency sets): weights and KV cache
    # cannot be swapped out under memory pressure; other apps get paged instead.
    wired = args.wired_gb * 1e9 if args.wired_gb is not None else mx.device_info()["max_recommended_working_set_size"]
    if wired:
        mx.set_wired_limit(int(wired))
    from . import POWERED_BY

    print(POWERED_BY)
    require_m5_gpu()
    rt = load(resolve_model(args.model))
    inner = getattr(rt.model, "language_model", rt.model)
    ver = Verifier(rt, inner)
    drafters = dict(make_drafter(rt, inner, v) for v in (args.draft_vocab or ["full"]))
    tok = rt.tokenizer
    if args.fast_verify and args.packed:
        raise SystemExit("--fast-verify and --packed are alternatives")
    if args.fast_verify:
        from . import verify_matmul

        verify_matmul.install(rt.model, cache_file=args.fast_verify, verbose=False)
    if args.packed:
        from . import packed_linear

        rep = packed_linear.install(rt.model, verbose=False)
        for d in drafters.values():
            packed_linear.install(d.head, verbose=False)
        print(
            f"packed {rep['packed']} linears; active memory "
            f"{rep['memory_before_gb']:.1f} -> {rep['memory_after_gb']:.1f} GB"
        )
    if args.flash_attn or args.kv8:
        from . import long_attention

        long_attention.install()
    kv8.ENABLED = args.kv8
    # (text, name, max_new): a prompt file named "...@N.txt" generates up to N tokens.
    texts = [(t, t[:70], args.max_new) for t in args.prompt or []]
    for path in args.prompt_file or []:
        m = re.search(r"@(\d+)$", Path(path).stem)
        texts.append((Path(path).read_text(), Path(path).name, int(m.group(1)) if m else args.max_new))
    texts = texts or [(t, t[:70], args.max_new) for t in DEFAULT_PROMPTS]
    chat = lambda text: tok.apply_chat_template(  # noqa: E731
        [{"role": "user", "content": text}], add_generation_prompt=True, enable_thinking=args.thinking,
    )
    # Cost curves at ~200 tokens and, with --curve-context, at the longest prompt;
    # the policies blend the two by the live context length.
    long_ctx = max((chat(t) for t, _, _ in texts), key=len) if args.curve_context else None
    show = lambda curve: " ".join(f"{n}:{t:.0f}" for n, t in curve)  # noqa: E731
    if args.fast_verify or args.packed:
        curve, row_ms = measure_verify_curve(rt, inner, tok)
        print(f"verify cost: tokens->ms {show(curve)}  extra row {row_ms:.1f} ms")
        if long_ctx:
            print(f"verify cost at {len(long_ctx)} tokens: {show(measure_verify_curve(rt, inner, tok, context=long_ctx)[0])}")
    tree_long = None
    if any(c.get("fork") for c in configs):
        from . import tree_verify

        drafter0 = next(iter(drafters.values()))
        tree_curve, tree_row_ms, tree_draft = tree_verify.measure_tree_curve(rt, inner, drafter0, tok)
        print(
            f"tree pass cost: nodes->ms {show(tree_curve)}  extra path {tree_row_ms:.1f} ms  "
            f"draft {tree_draft[0]:.2f} ms/call +{tree_draft[1]:.2f} per extra chain"
        )
        if long_ctx:
            curve_l, row_l, draft_l = tree_verify.measure_tree_curve(rt, inner, drafter0, tok, context=long_ctx)
            print(
                f"tree pass cost at {len(long_ctx)} tokens: {show(curve_l)}  extra path {row_l:.1f} ms  "
                f"draft {draft_l[0]:.2f} ms/call +{draft_l[1]:.2f} per extra chain"
            )
            tree_long = (curve_l, row_l, draft_l, VERIFY_MS_CTX[0], len(long_ctx))
    eos = set(getattr(tok, "eos_token_ids", None) or [tok.eos_token_id])

    results = []
    for text, name, max_new in texts:
        prompt = chat(text)
        print(f"\n=== {name!r} ({len(prompt)} prompt tokens, up to {max_new} new)")
        mx.clear_cache()
        ref, st = decode_ar(rt, inner, prompt, max_new, eos)
        ar_tps = (len(ref) - 1) / st["decode_s"]
        print(f"{'ar':<26} {len(ref):4d} tok  {ar_tps:6.1f} tok/s  (prefill {st['prefill_s']:.1f} s)")
        row = {"prompt": text, "ar": {"tokens": len(ref), "tok_s": ar_tps}}
        for cfg, (suffix, drafter) in ((c, d) for c in configs for d in drafters.items()):
            label = cfg_label(cfg) + suffix
            mx.clear_cache()  # every run starts from the same memory state
            if cfg.get("fork"):
                out, st = tree_verify.decode_fork(
                    rt, inner, drafter, prompt, max_new, eos, depth=cfg["depth"], branches=cfg["branches"],
                    branch_depth=cfg["branch_depth"], sibs=cfg["sibs"], curve=tree_curve, row_ms=tree_row_ms,
                    draft=tree_draft, long=tree_long,
                )
            elif cfg["lazy"] or cfg["overlap"]:
                out, st = decode_chain(
                    rt, inner, ver, drafter, prompt, max_new, eos,
                    depth=cfg["depth"], overlap=cfg["overlap"], auto=cfg["auto"],
                )
            else:
                out, st = decode_spec(
                    rt, inner, ver, drafter, prompt, max_new, eos,
                    share_min=args.share_min,
                    **{k: cfg[k] for k in ("depth", "topk", "width", "budget", "gate", "cap", "nucleus", "auto")},
                )
            tps = (len(out) - 1) / st["decode_s"]
            div = first_divergence(ref, out)
            print(
                f"{label:<26} {len(out):4d} tok  {tps:6.1f} tok/s  "
                f"tok/step {st['tokens_per_step']:.2f}  rows {st['mean_rows']:.1f}  "
                f"nodes {st['mean_nodes']:.1f}  passes {st['passes_per_step']:.2f}  "
                f"vtok {st['verify_tokens_per_step']:.1f}  "
                f"[draft {st['draft_s']:.1f}s verify {st['verify_s']:.1f}s]  "
                + (f"spec hits {st['spec_hit_rate']:.0%}  " if st.get("spec_hit_rate") is not None else "")
                + (f"depths {st['depth_hist']}  " if st.get("depth_hist") else "")
                + (
                    f"alt wins {st['alt_win_rate']:.0%} sib wins {st['sib_win_rate']:.0%} shapes {st['shape_hist']}  "
                    if st.get("shape_hist") else ""
                )
                + ("== ar" if div is None else f"DIVERGES at token {div}")
            )
            row[label] = {"tokens": len(out), "tok_s": tps, "diverges_at": div, **st}
        print("--- ar output ---\n" + tok.decode(ref))
        results.append(row)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()

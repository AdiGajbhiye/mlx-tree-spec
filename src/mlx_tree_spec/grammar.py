"""Grammar-constrained decoding for tree-speculative decoding (llguidance).

A request's grammar (a JSON schema, a Lark grammar or a regex) constrains its reply;
with thinking on, only the part after </think>. Output stays target-exact: at every
position the emitted token is the target's own choice among the tokens the grammar
allows there (its argmax over them, or a sample from its distribution restricted to
them). That costs little in a tree pass:

- If the target's unconstrained choice at a node is allowed, it is also the
  constrained one (the argmax over a set that contains the overall argmax), so the
  usual draft == target comparison stands. One llguidance check per node on the
  walked paths, no mask.
- Only where that choice is not allowed is the mask computed (~0.06 ms for the
  248K-token vocabulary) and the choice redone on the node's logits. For sampling,
  drawing from the full distribution and redrawing from the masked one only on a
  disallowed draw gives exactly the masked distribution.

Drafts the grammar forbids are still verified (and then rejected) rather than pruned
before the pass, and forced spans are not fast-forwarded: both would need the drafted
tokens on the CPU before the pass, an extra sync per step.

Request fields: `response_format` as in OpenAI's API ({"type": "json_schema",
"json_schema": {"schema": ...}} or {"type": "json_object"}), or `grammar`:
{"json_schema": ...} | {"lark": "..."} | {"regex": "..."}.
"""

from __future__ import annotations

import json

import mlx.core as mx
import numpy as np


class GrammarError(ValueError):
    pass


def grammar_spec(body: dict) -> str | None:
    """The llguidance grammar a request body asks for, or None."""
    from llguidance import LLMatcher

    g = body.get("grammar")
    rf = body.get("response_format") or {}
    if g is not None:
        if not isinstance(g, dict) or len(g) != 1:
            raise GrammarError('"grammar" must be one of {"json_schema": ...}, {"lark": "..."}, {"regex": "..."}')
        (kind, value), = g.items()
        if kind == "json_schema":
            return LLMatcher.grammar_from_json_schema(value if isinstance(value, (str, dict)) else json.dumps(value))
        if kind == "lark":
            return LLMatcher.grammar_from_lark(value)
        if kind == "regex":
            return LLMatcher.grammar_from_regex(value)
        raise GrammarError(f"unknown grammar kind {kind!r}")
    if rf.get("type") == "json_schema":
        schema = (rf.get("json_schema") or {}).get("schema")
        if schema is None:
            raise GrammarError('response_format json_schema needs "json_schema": {"schema": ...}')
        return LLMatcher.grammar_from_json_schema(schema)
    if rf.get("type") == "json_object":
        return LLMatcher.grammar_from_json_schema({"type": "object"})
    if rf.get("type") not in (None, "text"):
        raise GrammarError(f"unsupported response_format type {rf.get('type')!r}")
    return None


class Grammars:
    """The llguidance tokenizer for one model (built once, ~1 s)."""

    def __init__(self, hf_tokenizer, eos_id: int):
        import llguidance.hf

        self.tok = llguidance.hf.from_tokenizer(hf_tokenizer, eos_token=eos_id)
        self.eos_id = eos_id

    def constraint(self, spec: str, eos, think_end: int | None, active: bool) -> Constraint:
        from llguidance import LLMatcher

        m = LLMatcher(self.tok, spec, log_level=0)
        if m.is_error():
            raise GrammarError(m.get_error())
        return Constraint(m, self, set(eos), think_end, active)


class Constraint:
    """One request's grammar state over its output so far. `push` and `pop` move it
    along a candidate path (tokens before and including </think> pass freely while
    thinking), `allowed` checks a token and `pick` redoes a choice under the mask."""

    def __init__(self, matcher, grammars: Grammars, eos: set[int], think_end: int | None, active: bool):
        self.m, self.g, self.eos, self.think_end, self.active = matcher, grammars, eos, think_end, active
        self.stack: list[str] = []  # per pushed token: "c" consumed, "a" activated (</think>), "-" passed
        self.masks = self.repicks = 0

    def allowed(self, tok: int) -> bool:
        if not self.active:
            return True
        if self.m.is_stopped():
            return tok in self.eos
        return self.m.validate_tokens([tok]) == 1

    def push(self, tok: int) -> None:
        if not self.active:
            self.active = tok == self.think_end
            self.stack.append("a" if self.active else "-")
        elif self.m.is_stopped():
            self.stack.append("-")
        else:
            self.m.consume_token(tok)
            self.stack.append("c")

    def pop(self, n: int) -> None:
        consumed = 0
        for _ in range(n):
            kind = self.stack.pop()
            if kind == "c":
                consumed += 1
            elif kind == "a":
                self.active = False
        if consumed:
            self.m.rollback(consumed)

    def commit(self) -> None:
        """Make the pushed tokens permanent (nothing to pop back to)."""
        self.stack.clear()

    def choose(self, pred: int, logits: mx.array, temperature: float) -> int:
        """The target's constrained choice from its unconstrained one `pred` and the
        node's logits (V,)."""
        if self.allowed(pred):
            return pred
        self.repicks += 1
        n = logits.shape[-1]
        if self.m.is_stopped():
            bits = np.zeros(n, dtype=bool)
            bits[self.g.eos_id] = True
        else:
            self.masks += 1
            raw = np.frombuffer(self.m.compute_bitmask(), dtype=np.uint8)
            bits = np.unpackbits(raw, bitorder="little")[:n].astype(bool)
            if len(bits) < n:  # logits padded past the tokenizer's vocabulary
                bits = np.concatenate([bits, np.zeros(n - len(bits), dtype=bool)])
        z = mx.where(mx.array(bits), logits.astype(mx.float32), -mx.inf)
        if temperature > 0:
            return int(mx.random.categorical(z * (1.0 / temperature)).item())
        return int(mx.argmax(z).item())

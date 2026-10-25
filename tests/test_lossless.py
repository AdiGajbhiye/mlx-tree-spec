"""Tree decoding emits only the target's own greedy choices.

Token-for-token equality with plain decoding is not the guarantee: verify passes
run 8-16 rows through other kernels than one-token decoding, so bf16 rounding
flips near-ties and the texts drift apart. What must hold is that every token the
tree decoder emits is the target's argmax given its prefix, up to such near-ties.
This re-scores the whole output in one teacher-forced pass and checks exactly that.

Loads the 27B model: set MLX_TREE_SPEC_MODEL to a local checkpoint or repo id.
"""

import os

import mlx.core as mx
import pytest

MODEL = os.environ.get("MLX_TREE_SPEC_MODEL")
pytestmark = [pytest.mark.model, pytest.mark.skipif(not MODEL, reason="set MLX_TREE_SPEC_MODEL to run")]

PROMPTS = [
    "Write a Python function that merges two sorted lists, with a docstring and doctests.",
    "Explain how a hash map handles collisions, in two short paragraphs.",
]
MAX_NEW = 160
NEAR_TIE = 0.5  # logits; bf16 spacing is 0.125 at magnitudes 16-32


@pytest.fixture(scope="module")
def setup():
    from mtplx.runtime import load

    from mlx_tree_spec import tree_spec as ts
    from mlx_tree_spec import tree_verify as tv

    rt = load(ts.resolve_model(MODEL))
    inner = getattr(rt.model, "language_model", rt.model)
    tok = rt.tokenizer
    _, drafter = ts.make_drafter(rt, inner, "general")
    curve, row_ms, draft = tv.measure_tree_curve(rt, inner, drafter, tok)
    eos = set(getattr(tok, "eos_token_ids", None) or [tok.eos_token_id])
    return rt, inner, tok, drafter, (curve, row_ms, draft), eos


def _greedy_mismatches(rt, inner, prompt, out):
    """Positions where `out` is not the teacher-forced argmax, with the logit margin."""
    seq = prompt + out[:-1]
    res = rt.forward_ar(mx.array([seq]), cache=inner.make_cache())
    logits = (res[0] if isinstance(res, tuple) else res)[0, len(prompt) - 1 :].astype(mx.float32)
    top = mx.argmax(logits, axis=-1)
    chosen = mx.take_along_axis(logits, mx.array(out)[:, None], axis=-1)[:, 0]
    margin = mx.max(logits, axis=-1) - chosen
    mx.eval(top, margin)
    return [(i, margin[i].item()) for i in range(len(out)) if top[i].item() != out[i]]


@pytest.mark.parametrize("text", PROMPTS)
def test_tree_decoding_emits_target_choices(setup, text):
    from mlx_tree_spec import tree_verify as tv

    rt, inner, tok, drafter, (curve, row_ms, draft), eos = setup
    prompt = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, enable_thinking=False)
    out, st = tv.decode_fork(
        rt, inner, drafter, prompt, MAX_NEW, eos, depth=15, branches=2, branch_depth=4, sibs=4,
        curve=curve, row_ms=row_ms, draft=draft,
    )
    assert st["tokens_per_step"] > 1.5, "speculation accepted almost nothing"
    bad = [(i, m) for i, m in _greedy_mismatches(rt, inner, prompt, out) if m > NEAR_TIE]
    assert not bad, f"tokens that are not the target's choice (position, logit margin): {bad[:10]}"

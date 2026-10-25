"""Build a general-text draft vocabulary for Drafter (tree_spec.py --draft-vocab).

MTPLX's FR-Spec list holds the 64K most frequent Qwen3.8 tokens on a CODE corpus;
on prose it misses tokens the target uses, which costs 2-8% acceptance. This
generates text with the target itself (greedy, same as the benchmark) across
varied non-code prompts, then writes the union of the code list and every token
the target produced. None of the prompts overlap tree_spec.py's benchmark
prompts. Every 5th prompt is held out to measure coverage on unseen text.

    python experiments/build_draft_vocab.py --out general_vocab.npy
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter

import mlx.core as mx
import numpy as np

from mtplx.runtime import load
from mlx_tree_spec.tree_spec import DEFAULT_MODEL, Drafter, Verifier, decode_spec, load_draft_vocab, resolve_model

PROMPTS = [
    # explanation / science
    "How do vaccines train the immune system? Explain for a curious teenager.",
    "Why do we have seasons on Earth?",
    "Explain how a refrigerator keeps food cold.",
    "What causes inflation, and how do central banks respond to it?",
    "Describe how photosynthesis works in plain language.",
    "Why is the ocean salty but rivers are not?",
    "Explain the difference between weather and climate.",
    "How does GPS figure out where you are?",
    # history / culture
    "Summarise the causes of the First World War.",
    "Tell me about the history of the printing press and its impact.",
    "Who was Ada Lovelace and why is she remembered?",
    "Describe daily life in ancient Rome for an ordinary family.",
    "What was the Silk Road and what was traded along it?",
    # writing / creative
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Write a poem about autumn rain in a city.",
    "Write a friendly email to a colleague asking to reschedule our Thursday meeting.",
    "Draft a polite complaint letter to an airline about lost luggage.",
    "Write a product description for a handmade ceramic coffee mug.",
    "Write a two-paragraph opening for a mystery novel set in Mumbai.",
    "Write a toast for my sister's wedding, warm and a little funny.",
    # advice / how-to
    "Give me practical tips for sleeping better.",
    "How should I prepare for a job interview at a startup?",
    "What are good habits for managing personal finances in your twenties?",
    "How do I start running if I've never exercised before?",
    "Explain how to make a basic sourdough starter.",
    "Plan a three-day trip to Kyoto for someone who loves food and temples.",
    "How can a small team run effective weekly meetings?",
    # lists / structured
    "List the pros and cons of working from home.",
    "Give me a markdown table comparing tea, coffee and yerba mate.",
    "List twelve common English idioms and what they mean.",
    "Outline a beginner's study plan for learning Spanish in six months.",
    "Give a checklist for moving to a new apartment.",
    # reasoning / maths in prose
    "If I save 500 rupees a week at 6% annual interest, roughly how much will I have in 3 years? Explain.",
    "A recipe serves 4 and needs 300 g of flour. How much for 10 people? Show the steps.",
    "Is it better to pay off a loan early or invest the money? Walk through the trade-offs.",
    "Three friends split a bill of 2,450 with a 10% tip. How much does each pay?",
    # dialogue / summarising / opinion
    "Write a short dialogue between a customer and a barista who got the order wrong.",
    "Summarise the plot of Romeo and Juliet in a few sentences.",
    "What are the arguments for and against a four-day work week?",
    "Explain the rules of cricket to an American.",
    "Describe the feeling of standing at the top of a mountain at sunrise.",
    # other languages and formats
    "Translate into French and Spanish: 'The museum opens at nine, but the café is already busy.'",
    "Write a haiku sequence about the sea.",
    "Answer in Hindi: what is the capital of Australia and why was it chosen?",
    "Write a JSON object describing a fictional bookstore with name, address, opening hours and three books.",
    "Write a limerick about a forgetful robot.",
    "Explain what a blockchain is, then give an analogy a child would understand.",
    "Write meeting notes for a project kickoff with action items.",
    "Describe the taste of a mango to someone who has never had one.",
    "Give a short motivational speech for a team before a product launch.",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-new", type=int, default=300)
    ap.add_argument("--out", default="general_vocab.npy")
    ap.add_argument("--report", default="general_vocab_report.json")
    args = ap.parse_args()

    rt = load(resolve_model(args.model))
    inner = getattr(rt.model, "language_model", rt.model)
    ver = Verifier(rt, inner)
    drafter = Drafter(rt, inner)  # full vocab: the generated text is the target's own greedy output
    tok = rt.tokenizer
    eos = set(getattr(tok, "eos_token_ids", None) or [tok.eos_token_id])

    train, held = Counter(), Counter()
    t0 = time.perf_counter()
    for n, text in enumerate(PROMPTS):
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True, enable_thinking=False
        )
        out, _ = decode_spec(
            rt, inner, ver, drafter, prompt, args.max_new, eos,
            depth=3, topk=1, width=1, budget=3, gate=None, cap=None, nucleus=None, auto=False,
            share_min=10**9,
        )
        (held if n % 5 == 4 else train).update(out)
        print(f"[{n + 1:2d}/{len(PROMPTS)}] {len(out):3d} tok  {time.perf_counter() - t0:5.0f}s  {text[:60]}")

    code = set(load_draft_vocab("frspec").tolist())
    merged = sorted(code | set(train))
    np.save(args.out, np.array(merged, dtype=np.uint32))

    def coverage(vocab: set[int], counts: Counter) -> float:
        return sum(c for t, c in counts.items() if t in vocab) / max(sum(counts.values()), 1)

    report = dict(
        prompts=len(PROMPTS),
        train_tokens=sum(train.values()),
        held_tokens=sum(held.values()),
        distinct_train=len(train),
        code_size=len(code),
        merged_size=len(merged),
        added=len(merged) - len(code),
        held_coverage_code=coverage(code, held),
        held_coverage_merged=coverage(set(merged), held),
        train_coverage_code=coverage(code, train),
    )
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

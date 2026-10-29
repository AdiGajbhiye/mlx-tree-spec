"""Save real FFN inputs (2048 tokens of code_32k.txt) and dequantized 4-bit FFN weights of layers
3, 31 and 62 to real_layers.npz, for split_accuracy.py. Loads the model: run it with this repo's
venv, under CLAUDE.md's memory checks. Usage: .venv/bin/python bench/ane/capture_layers.py MODEL_DIR"""
import sys

import mlx.core as mx
import numpy as np
from mtplx.runtime import load

from mlx_tree_spec import tree_spec as ts

LAYERS = (3, 31, 62)
r = load(sys.argv[1])
inner = getattr(r.model, "language_model", r.model)
which = {id(inner.model.layers[i].mlp): i for i in LAYERS}
saved = {}
cls = type(inner.model.layers[0].mlp)
orig = cls.__call__


def hook(self, x):
    if id(self) in which:
        saved[which[id(self)]] = x
    return orig(self, x)


cls.__call__ = hook
tok = r.tokenizer.encode((ts.DATA / "code_32k.txt").read_text())[:2048]
mx.eval(r.model(mx.array([tok])), list(saved.values()))
out = {}
for i in LAYERS:
    out[f"x{i}"] = np.array(saved[i].reshape(-1, saved[i].shape[-1]).astype(mx.float32))
    for n in ("gate_proj", "up_proj", "down_proj"):
        q = getattr(inner.model.layers[i].mlp, n)
        out[f"{n}{i}"] = np.array(mx.dequantize(q.weight, q.scales, q.biases, group_size=64, bits=4).astype(mx.float32))
np.savez("real_layers.npz", **out)

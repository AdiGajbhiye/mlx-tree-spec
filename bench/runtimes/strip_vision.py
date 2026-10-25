"""Make a text-only copy of an MLX checkpoint by dropping top-level `visual.*` (vision encoder) tensors.

Weights are copied bit-for-bit (no re-quantization); other files are symlinked.
"""
import json
import os
import sys

import mlx.core as mx

src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
index = json.load(open(os.path.join(src, "model.safetensors.index.json")))
dropped = 0
for shard in sorted(set(index["weight_map"].values())):
    tensors, meta = mx.load(os.path.join(src, shard), return_metadata=True)
    keep = {k: v for k, v in tensors.items() if not k.startswith("visual.")}
    dropped += len(tensors) - len(keep)
    mx.save_safetensors(os.path.join(dst, shard), keep, metadata=meta)
    print(shard, len(keep), "kept", flush=True)
index["weight_map"] = {k: v for k, v in index["weight_map"].items() if not k.startswith("visual.")}
json.dump(index, open(os.path.join(dst, "model.safetensors.index.json"), "w"), indent=2)
for f in os.listdir(src):
    if not f.endswith(".safetensors") and f != "model.safetensors.index.json" and not os.path.exists(os.path.join(dst, f)):
        os.symlink(os.path.join(src, f), os.path.join(dst, f))
print("dropped", dropped, "vision tensors")

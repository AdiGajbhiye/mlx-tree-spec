"""tree_flash's key-block count at long context, through the real wrapper (bf16 and kv8 caches):
ms over 16 layers for 32 / 48 / 64 blocks. Usage: python bench/attn/blocks.py"""
import sys
import mlx.core as mx
from mlx_tree_spec import tree_flash as tf
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from flash_bench import caches, med  # noqa: E402

orig = tf.flash_blocks
print("ctx    kv   n  |   32     48     64  (ms; best marked *)")
for ctx in (16384, 24576, 32768):
    for q8 in (False, True):
        cs = caches(ctx - 100, q8)
        for n in (4, 8, 12, 16):
            q = mx.random.normal((1, 24, n, 256)).astype(mx.bfloat16)
            bits = mx.array(tf.ancestor_bits([[j <= i for j in range(n)] for i in range(n)]), dtype=mx.uint64)
            ts = []
            for b in (32, 48, 64):
                tf.flash_blocks = lambda cap, b=b: b
                ts.append(med(lambda: [tf.tree_flash(q, c, 256 ** -0.5, bits) for c in cs], n=11))
            best = min(ts)
            print(f"{ctx:6d} {'q8' if q8 else 'bf16':>4} {n:3d} | " + " ".join(f"{t:5.2f}{'*' if t == best else ' '}" for t in ts), flush=True)
        del cs
tf.flash_blocks = orig

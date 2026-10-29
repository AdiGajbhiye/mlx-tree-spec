"""Where does the ANE's error come from? The same exact-weight models on CPU_ONLY / CPU_AND_GPU /
CPU_AND_NE, 14.7% share; and the ANE with fp32 compute precision requested."""
import sys
sys.argv = ["x"]
exec(open("exact_split.py").read().split("silu = lambda")[0])
import numpy as np
silu = lambda v: v / (1 + np.exp(-v))
rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
x = (np.random.default_rng(0).standard_normal((T, H)) * 0.5).astype(np.float32); x[:, 7] *= 40
xin = {"x": x.T.reshape(1, H, 1, T).astype(np.float16)}
ch = 2560
wg, wu, wd = q4((ch, H), 1), q4((ch, H), 2), q4((H, ch), 3)
ref = (silu(x @ wg["deq"].T) * (x @ wu["deq"].T)) @ wd["deq"].T
x16 = xin["x"].reshape(H, T).T.astype(np.float32)
ref16 = (silu(x16 @ wg["deq"].T) * (x16 @ wu["deq"].T)) @ wd["deq"].T
print(f"fp16 rounding of the input alone: {rel(ref16, ref):.2e}")
for fmt in ("fp16", "u4 exact"):
    m = ane(wg, wu, wd, fmt)
    for cu in (ct.ComputeUnit.CPU_ONLY, ct.ComputeUnit.CPU_AND_GPU, ct.ComputeUnit.CPU_AND_NE):
        mm = ct.models.MLModel(m.get_spec(), weights_dir=m.weights_dir, compute_units=cu)
        t = med(lambda: mm.predict(xin), n=3, warm=1)
        y = mm.predict(xin)["y"].reshape(H, T).T.astype(np.float32)
        print(f"{fmt:>9} {cu.name:11s} {t:7.1f} ms  rel err {rel(y, ref):.2e}", flush=True)

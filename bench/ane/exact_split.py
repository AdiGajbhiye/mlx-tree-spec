"""Exact 4-bit weights on the ANE: MLX's affine group-64 format (w = s*q + b) written as Core ML's
blockwise s*(q - offset), offset = -b/s in fp16. Speed and error of each weight format for the
ANE's share of one FFN layer (T=2048 rows), vs the GPU (MLX quantized_matmul, bf16) and an fp32
reference built from the same 4-bit weights. Also the process footprint after loading each model.
Usage: python exact_split.py"""
import os, statistics, subprocess, time
import coremltools as ct, mlx.core as mx, numpy as np
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types
from coremltools.optimize.coreml import OpLinearQuantizerConfig, OptimizationConfig, linear_quantize_weights

H, T = 5120, 2048


def med(f, n=7, warm=2):
    ts = []
    for i in range(n + warm):
        t = time.perf_counter(); f()
        if i >= warm: ts.append((time.perf_counter() - t) * 1e3)
    return statistics.median(ts)


def footprint_mb():
    out = subprocess.run(["footprint", str(os.getpid())], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if "phys_footprint:" in line:
            n, u = line.split()[1:3]
            return float(n) * {"KB": 1 / 1024, "MB": 1, "GB": 1024}[u]


def q4(shape, seed):
    """Random bf16 weights quantized the way the model is (affine, group 64, 4 bits): q, s, b as numpy."""
    w = mx.random.normal(shape, key=mx.random.key(seed)).astype(mx.bfloat16) * 0.02
    wq, s, b = mx.quantize(w, group_size=64, bits=4)
    deq = mx.dequantize(wq, s, b, group_size=64, bits=4)
    srep, brep = (mx.repeat(a, 64, axis=1) for a in (s, b))
    q = mx.round((deq.astype(mx.float32) - brep.astype(mx.float32)) / srep.astype(mx.float32))
    mx.eval(q, deq)
    return dict(packed=(wq, s, b), q=np.array(q).astype(np.uint8), s=np.array(s.astype(mx.float32)),
                b=np.array(b.astype(mx.float32)), deq=np.array(deq.astype(mx.float32)))


def weight(w, fmt):
    """The conv weight (Cout, Cin, 1, 1) for one 4-bit matrix in the given ANE format."""
    co, ci = w["q"].shape
    if fmt == "fp16":
        return w["deq"].reshape(co, ci, 1, 1).astype(np.float16)
    data = w["q"].reshape(co, ci, 1, 1)
    data = data.astype(types.np_uint4_dtype) if fmt == "u4 exact" else data.astype(np.int8)
    scale = w["s"].reshape(co, ci // 64, 1, 1).astype(np.float16)
    offset = (-w["b"] / w["s"]).reshape(co, ci // 64, 1, 1).astype(np.float16)
    return mb.constexpr_blockwise_shift_scale(data=data, scale=scale, offset=offset)


def ane(wg, wu, wd, fmt):
    ch = wg["q"].shape[0]
    base = "fp16" if fmt == "i8 per-channel" else fmt

    @mb.program(input_specs=[mb.TensorSpec((1, H, 1, T), types.fp16)], opset_version=ct.target.macOS15)
    def prog(x):
        g = mb.conv(x=x, weight=weight(wg, base))
        u = mb.conv(x=x, weight=weight(wu, base))
        return mb.conv(x=mb.mul(x=mb.silu(x=g), y=u), weight=weight(wd, base), name="y")

    m = ct.convert(prog, convert_to="mlprogram", minimum_deployment_target=ct.target.macOS15,
                   compute_precision=ct.precision.FLOAT16, compute_units=ct.ComputeUnit.CPU_AND_NE)
    if fmt == "i8 per-channel":
        m = linear_quantize_weights(m, OptimizationConfig(global_config=OpLinearQuantizerConfig(
            mode="linear_symmetric", dtype="int8", granularity="per_channel")))
    return m


silu = lambda v: v / (1 + np.exp(-v))
rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
x = (np.random.default_rng(0).standard_normal((T, H)) * 0.5).astype(np.float32)
x[:, 7] *= 40  # an outlier channel, as real activations have
xin = {"x": x.T.reshape(1, H, 1, T).astype(np.float16)}
xm = mx.array(x).astype(mx.bfloat16)
print(f"{'share':>6} {'ANE weights':>16} | {'ANE ms':>6} {'+MB':>6} | rel err vs fp32: {'GPU bf16':>8} {'ANE':>8}")
for ch in (2560, 3584):
    wg, wu, wd = q4((ch, H), 1), q4((ch, H), 2), q4((H, ch), 3)
    ref = (silu(x @ wg["deq"].T) * (x @ wu["deq"].T)) @ wd["deq"].T
    qmm = lambda a, w: mx.quantized_matmul(a, *w["packed"], transpose=True, group_size=64, bits=4)
    g, u = qmm(xm, wg), qmm(xm, wu)
    gpu = np.array(qmm(mx.sigmoid(g) * g * u, wd).astype(mx.float32))
    for fmt in ("fp16", "i8 per-channel", "i8 exact", "u4 exact"):
        before = footprint_mb()
        try:
            m = ane(wg, wu, wd, fmt)
            t = med(lambda: m.predict(xin))
            y = m.predict(xin)["y"].reshape(H, T).T.astype(np.float32)
            grew = footprint_mb() - before
            print(f"{ch / 17408:6.1%} {fmt:>16} | {t:6.1f} {grew:6.0f} | {'':16} {rel(gpu, ref):8.2e} {rel(y, ref):8.2e}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"{ch / 17408:6.1%} {fmt:>16} | failed: {str(e)[:120]}", flush=True)
        m = None

"""Accuracy of the split on real weights and activations (layers 3, 31, 62; 2048 tokens of code):
error vs an fp32 reference of the 4-bit FFN, for the GPU alone (bf16) and for GPU lead (bf16) +
ANE tail (int8 per-channel weights, fp16 activations)."""
import coremltools as ct, mlx.core as mx, numpy as np
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types
from coremltools.optimize.coreml import OpLinearQuantizerConfig, OptimizationConfig, linear_quantize_weights
d = np.load("real_layers.npz")  # from capture_layers.py
T, H = 2048, 5120
silu = lambda v: v / (1 + np.exp(-v))
def ane(wg, wu, wd):
    ch = wg.shape[0]
    @mb.program(input_specs=[mb.TensorSpec((1, H, 1, T), types.fp16)], opset_version=ct.target.macOS15)
    def prog(x):
        g = mb.conv(x=x, weight=wg.reshape(ch, H, 1, 1).astype(np.float16))
        u = mb.conv(x=x, weight=wu.reshape(ch, H, 1, 1).astype(np.float16))
        return mb.conv(x=mb.mul(x=mb.silu(x=g), y=u), weight=wd.reshape(H, ch, 1, 1).astype(np.float16), name="y")
    m = ct.convert(prog, convert_to="mlprogram", minimum_deployment_target=ct.target.macOS15,
                   compute_precision=ct.precision.FLOAT16, compute_units=ct.ComputeUnit.CPU_AND_NE)
    return linear_quantize_weights(m, OptimizationConfig(global_config=OpLinearQuantizerConfig(
        mode="linear_symmetric", dtype="int8", granularity="per_channel")))
def gpu(x, wg, wu, wd):
    x, wg, wu, wd = (mx.array(a).astype(mx.bfloat16) for a in (x, wg, wu, wd))
    g, u = x @ wg.T, x @ wu.T
    return np.array((mx.sigmoid(g) * g * u @ wd.T).astype(mx.float32))
rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
for i in (3, 31, 62):
    x, wg, wu, wd = d[f"x{i}"], d[f"gate_proj{i}"], d[f"up_proj{i}"], d[f"down_proj{i}"]
    ref = (silu(x @ wg.T) * (x @ wu.T)) @ wd.T
    alone = gpu(x, wg, wu, wd)
    for ch in (2560, 3584):
        k = wg.shape[0] - ch
        m = ane(wg[k:], wu[k:], wd[:, k:])
        ya = m.predict({"x": x.T.reshape(1, H, 1, T).astype(np.float16)})["y"].reshape(H, T).T.astype(np.float32)
        split = gpu(x, wg[:k], wu[:k], wd[:, :k]) + ya
        tail_ref = (silu(x @ wg[k:].T) * (x @ wu[k:].T)) @ wd[:, k:].T
        print(f"layer {i:2d} ane {ch / wg.shape[0]:.1%}: rel err GPU alone {rel(alone, ref):.2e}  split {rel(split, ref):.2e}  "
              f"(ANE tail alone {rel(ya, tail_ref):.2e}, finite {np.isfinite(ya).all()})  "
              f"max abs GPU {np.abs(alone - ref).max():.3f} split {np.abs(split - ref).max():.3f}", flush=True)

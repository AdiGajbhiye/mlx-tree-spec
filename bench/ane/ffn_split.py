"""Can the Neural Engine (through public Core ML) take a share of one Qwen3.8-27B FFN layer's
prefill work while the GPU (MLX, 4-bit) does the rest, and finish sooner than the GPU alone?

Random weights at the real shapes (hidden 5120, intermediate 17408, SwiGLU). Times, per chunk of
T rows: GPU full FFN; GPU on the leading (1-share) channels; ANE on the trailing share channels
(fp16 activations; int8 per-channel or uint4 per-block weights); both at once (GPU async + ANE in a thread), including the
copy of the input out of MLX and the partial result back. Usage: python ffn_split.py
"""
import statistics
import sys
import threading
import time

import coremltools as ct
import mlx.core as mx
import numpy as np
from coremltools.converters.mil import Builder as mb

H, I = 5120, 17408
UNITS, WMODES = (5, 7, 9), ("i8", "u4b")


def med(f, n=7, warm=2):
    ts = []
    for i in range(n + warm):
        t = time.perf_counter()
        f()
        if i >= warm:
            ts.append((time.perf_counter() - t) * 1e3)
    return statistics.median(ts)


def gpu_ffn(ch):
    """4-bit affine (group 64) SwiGLU over `ch` intermediate channels, like the packed model."""
    k = mx.random.split(mx.random.key(ch), 3)
    qs = [mx.quantize(mx.random.normal(s, key=kk).astype(mx.bfloat16) * 0.02, group_size=64, bits=4)
          for s, kk in (((ch, H), k[0]), ((ch, H), k[1]), ((H, ch), k[2]))]
    mx.eval(qs)

    def f(x):
        g = mx.quantized_matmul(x, *qs[0], transpose=True, group_size=64, bits=4)
        u = mx.quantized_matmul(x, *qs[1], transpose=True, group_size=64, bits=4)
        return mx.quantized_matmul(mx.sigmoid(g) * g * u, *qs[2], transpose=True, group_size=64, bits=4)
    return f


def ane_ffn(ch, T, wmode):
    rng = np.random.default_rng(ch)
    wg, wu = (rng.standard_normal((ch, H), dtype=np.float32) * 0.02 for _ in range(2))
    wd = rng.standard_normal((H, ch), dtype=np.float32) * 0.02

    # 1x1 convs on (1, C, 1, T), the layout the ANE prefers
    @mb.program(input_specs=[mb.TensorSpec(shape=(1, H, 1, T), dtype=ct.converters.mil.mil.types.fp16)],
                opset_version=ct.target.macOS15)
    def prog(x):
        g = mb.conv(x=x, weight=wg.reshape(ch, H, 1, 1).astype(np.float16))
        u = mb.conv(x=x, weight=wu.reshape(ch, H, 1, 1).astype(np.float16))
        h = mb.mul(x=mb.silu(x=g), y=u)
        return mb.conv(x=h, weight=wd.reshape(H, ch, 1, 1).astype(np.float16), name="y")

    m = ct.convert(prog, convert_to="mlprogram", minimum_deployment_target=ct.target.macOS15,
                   compute_precision=ct.precision.FLOAT16, compute_units=ct.ComputeUnit.CPU_AND_NE)
    if wmode != "f16":
        from coremltools.optimize.coreml import OpLinearQuantizerConfig, OptimizationConfig, linear_quantize_weights
        cfg = (OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8", granularity="per_channel") if wmode == "i8"
               else OpLinearQuantizerConfig(mode="linear", dtype="uint4", granularity="per_block", block_size=64))
        m = linear_quantize_weights(m, OptimizationConfig(global_config=cfg))
    return m


def main():
    chunks = [int(a) for a in sys.argv[1:]] or [512, 1024, 2048]
    full = gpu_ffn(I)
    print(f"{'T':>5} {'share':>6} {'w':>4} | {'gpu full':>8} {'gpu lead':>8} {'ane tail':>8} {'ane cpu-only':>12} "
          f"{'both':>7} | speedup")
    for T in chunks:
        x = (mx.random.normal((T, H)) * 0.5).astype(mx.bfloat16)
        mx.eval(x)
        t_full = med(lambda: mx.eval(full(x)))
        for units in UNITS:  # 512-channel units on the ANE (of 34)
            ch = units * 512
            lead = gpu_ffn(I - ch)
            t_lead = med(lambda: mx.eval(lead(x)))
            for wmode in WMODES:
                m = ane_ffn(ch, T, wmode)
                xin = {"x": np.zeros((1, H, 1, T), np.float16)}
                t_ane = med(lambda: m.predict(xin))
                t_cpu = None
                if units == 5 and wmode == "f16":
                    mc = ct.models.MLModel(m.get_spec(), weights_dir=m.weights_dir, compute_units=ct.ComputeUnit.CPU_ONLY)
                    t_cpu = med(lambda: mc.predict(xin), n=3, warm=1)

                def both():
                    xh = np.asarray(x.astype(mx.float16)).T.reshape(1, H, 1, T)  # out of MLX (syncs x)
                    out = {}
                    th = threading.Thread(target=lambda: out.update(m.predict({"x": xh})))
                    yg = lead(x)
                    mx.async_eval(yg)
                    th.start()
                    th.join()
                    ya = mx.array(out["y"].reshape(H, T).T)
                    mx.eval(yg + ya.astype(mx.bfloat16))
                t_both = med(both)
                cpu = f"{t_cpu:12.1f}" if t_cpu else " " * 12
                print(f"{T:5d} {ch / I:6.1%} {wmode:>4} | {t_full:8.1f} {t_lead:8.1f} {t_ane:8.1f} "
                      f"{cpu} {t_both:7.1f} | {t_full / t_both:.2f}x", flush=True)


if __name__ == "__main__":
    main()

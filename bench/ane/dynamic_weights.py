"""Weights as Core ML inputs (fed per layer) instead of constants: does the ANE still run it?"""
import statistics, time
import coremltools as ct, numpy as np
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types
H, T, ch = 5120, 2048, 3584
def med(f, n=5, warm=2):
    ts = []
    for i in range(n + warm):
        t = time.perf_counter(); f()
        if i >= warm: ts.append((time.perf_counter() - t) * 1e3)
    return statistics.median(ts)
for wdt, name in ((types.fp16, "fp16 weights in"),):
    @mb.program(input_specs=[mb.TensorSpec((1, H, 1, T), types.fp16), mb.TensorSpec((ch, H, 1, 1), wdt),
                             mb.TensorSpec((ch, H, 1, 1), wdt), mb.TensorSpec((H, ch, 1, 1), wdt)],
                opset_version=ct.target.macOS15)
    def prog(x, wg, wu, wd):
        g = mb.conv(x=x, weight=wg); u = mb.conv(x=x, weight=wu)
        return mb.conv(x=mb.mul(x=mb.silu(x=g), y=u), weight=wd, name="y")
    for cu in (ct.ComputeUnit.CPU_AND_NE, ct.ComputeUnit.CPU_AND_GPU):
        m = ct.convert(prog, convert_to="mlprogram", minimum_deployment_target=ct.target.macOS15,
                       compute_precision=ct.precision.FLOAT16, compute_units=cu)
        npd = np.float16
        inp = {"x": np.zeros((1, H, 1, T), np.float16), "wg": np.ones((ch, H, 1, 1), npd), "wu": np.ones((ch, H, 1, 1), npd),
               "wd": np.ones((H, ch, 1, 1), npd)}
        print(f"{name:36s} {cu.name:11s} {med(lambda: m.predict(inp)):7.1f} ms", flush=True)

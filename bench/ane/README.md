# Neural Engine FFN split: prototype (2026-10-30)

Can part of each layer's FFN run on the Neural Engine, through public Core ML, while the GPU
does the rest during prefill? Results and the go/no-go are in `docs/devlog.md`
("Neural Engine FFN split for prefill").

These scripts need `coremltools`, which this package doesn't depend on. Use a separate env:

```bash
uv venv -p 3.12 /tmp/ane-venv && uv pip install -p /tmp/ane-venv/bin/python "mlx==0.32.2" coremltools numpy
```

- `ffn_split.py [ROWS...]`: one FFN layer at the real shapes with random weights. Times the GPU
  alone, the GPU on the leading channels, the ANE on the rest (int8 or 4-bit block weights), and
  both at once. The both-at-once time includes copying data out of MLX and back.
- `dynamic_weights.py`: the same with weights passed as model inputs instead of constants.
- `capture_layers.py MODEL_DIR` (this repo's venv; loads the model): saves real FFN inputs and
  weights of three layers to `real_layers.npz` (3.3 GB).
- `split_accuracy.py`: error of GPU + ANE against an fp32 reference, next to the GPU alone's.

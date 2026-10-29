import numpy as np
import pytest

from mlx_tree_spec import POWERED_BY
from mlx_tree_spec.tree_flash import ancestor_bits
from mlx_tree_spec.tree_spec import DATA, first_divergence, load_draft_vocab, parse_cfg, resolve_model


def test_parse_fork_config():
    cfg = parse_cfg("fork:15,2,4,4")
    assert cfg["fork"] and (cfg["depth"], cfg["branches"], cfg["branch_depth"], cfg["sibs"]) == (15, 2, 4, 4)
    assert parse_cfg("fork:15")["branches"] == 0


def test_parse_chain_config():
    cfg = parse_cfg("adapt:6,3,3,16,p=0.9,auto")
    assert (cfg["depth"], cfg["topk"], cfg["width"], cfg["budget"]) == (6, 3, 3, 16)
    assert cfg["nucleus"] == 0.9 and cfg["auto"]
    with pytest.raises(ValueError):
        parse_cfg("bad:3,3,2,12,lazy")  # lazy needs a plain chain


def test_first_divergence():
    assert first_divergence([1, 2, 3], [1, 2, 3]) is None
    assert first_divergence([1, 2, 3], [1, 5, 3]) == 1
    assert first_divergence([1, 2], [1, 2, 3]) == 2


def test_ancestor_bits():
    # root 0, child 1 of 0, child 2 of 0 (sibling of 1)
    anc = [[1, 0, 0], [1, 1, 0], [1, 0, 1]]
    assert ancestor_bits(anc) == [0b001, 0b011, 0b101]


def test_general_vocab_is_packaged():
    ids = np.array(load_draft_vocab("general"))
    assert (DATA / "code_32k.txt").exists()
    assert len(ids) > 65536 and np.all(np.diff(ids) > 0)


def test_resolve_local_dir(tmp_path):
    assert resolve_model(str(tmp_path)) == str(tmp_path)


def test_attribution():
    assert "MTPLX" in POWERED_BY and "github.com/youssofal/mtplx" in POWERED_BY


class _ByteTok:
    """Byte-level stand-in tokenizer: ids < 256 are bytes, 1000 is </think>."""

    def decode(self, ids):
        return b"".join(b"</think>" if i == 1000 else bytes([i]) for i in ids).decode("utf-8", "replace")


def _stream(ids, sizes, thinking):
    from mlx_tree_spec.server import DeltaStream

    ds, out, i = DeltaStream(_ByteTok(), 1000, thinking), {"reasoning_content": "", "content": ""}, 0
    for n in sizes:
        for d in ds.feed(ids[i : i + n]):
            for k, v in d.items():
                assert v
                out[k] += v
        i += n
    for d in ds.feed(ids[i:], final=True):
        for k, v in d.items():
            out[k] += v
    return out


def test_delta_stream_matches_whole_decode():
    reasoning, content = "Think: naïve → café 🙂\n", "\n\n  Answer — 文字 🙂 done."
    ids = list(reasoning.encode()) + [1000] + list(content.encode())
    for sizes in ([1] * len(ids), [3, 1, 4, 1, 5, 9, 2, 6] * 20, [len(ids)]):
        assert _stream(ids, sizes, thinking=True) == {"reasoning_content": reasoning, "content": content.lstrip()}
    plain = list(content.encode())
    assert _stream(plain, [2] * 40, thinking=False) == {"reasoning_content": "", "content": content.lstrip()}


def test_gpu_problem():
    from mlx_tree_spec.tree_spec import gpu_problem

    assert gpu_problem("26.2", "applegpu_g17s", "Apple M5 Max") is None
    assert "M1-M4" in gpu_problem("26.2", "applegpu_g16s", "Apple M4 Max")
    assert "macOS 15.5" in gpu_problem("15.5", "applegpu_g17s", "Apple M5")
    assert gpu_problem("26.2", None, "unknown")


@pytest.mark.parametrize("m", [64, 100, 2048])
@pytest.mark.parametrize("k, n, bk", [(512, 128, 64), (1024, 192, 128)])
def test_prefill_kernel(m, k, n, bk):
    """Bit-identical to mx.quantized_matmul where MLX runs its NAX GEMM (here 2048
    rows of the larger shape); where it picks another kernel, no further from a
    float32 reference than MLX's own result."""
    import mlx.core as mx

    from mlx_tree_spec.qmm_packed import pack_nax, prefill_kernel

    mx.random.seed(0)
    wq, s, b = mx.quantize((mx.random.normal((n, k)) * 0.02).astype(mx.bfloat16), group_size=64, bits=4)
    x = (mx.random.normal((1, m, k)) * 0.5).astype(mx.bfloat16)
    ref = mx.quantized_matmul(x, wq, s, b, transpose=True, group_size=64, bits=4)
    y = prefill_kernel(k, n, 64, mx.bfloat16, (64, 64, bk, 2, 2))(x, *pack_nax(wq, s, b))
    assert y.shape == ref.shape
    if (k, m) == (1024, 2048):
        assert mx.array_equal(y, ref).item()
    exact = x.astype(mx.float32) @ mx.dequantize(wq, s, b, group_size=64, bits=4).astype(mx.float32).T
    err = lambda a: mx.abs(a.astype(mx.float32) - exact).max().item()
    assert err(y) <= err(ref) * 1.01


def test_curve_cache_roundtrip(tmp_path):
    from mlx_tree_spec.server import curves_key, load_curves, save_curves

    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "model.safetensors").write_bytes(b"x" * 10)
    key = curves_key(str(model), 16384)
    assert key == curves_key(str(model), 16384) != curves_key(str(model), 8192)
    path = tmp_path / "curves.json"
    assert load_curves(path, key) is None
    curve, draft = ((2, 30.0), (4, 33.5)), (1.2, 0.4)
    long = (((2, 41.0), (4, 46.0)), 3.0, (1.9, 0.6), 233, 16384)
    save_curves(path, key, "m", curve, 2.5, draft, long)
    save_curves(path, "other", "m", curve, 2.5, draft, None)
    assert load_curves(path, key) == (curve, 2.5, draft, long)
    assert load_curves(path, "other") == (curve, 2.5, draft, None)
    (model / "model.safetensors").write_bytes(b"x" * 11)  # new weights: a new key
    assert curves_key(str(model), 16384) != key


def _fake_snapshot(n):
    import mlx.core as mx
    from mlx_lm.models.cache import ArraysCache, KVCache

    kv, gdn, mtp = KVCache(), ArraysCache(2), KVCache()
    kv.state = (mx.random.normal((1, 2, n, 8)).astype(mx.bfloat16), mx.random.normal((1, 2, n, 8)).astype(mx.bfloat16))
    gdn.cache = [mx.random.normal((1, 3, 4)), None]
    mtp.state = (mx.ones((1, 1, n, 8)), mx.zeros((1, 1, n, 8)))
    return list(range(100, 100 + n)), [kv, gdn, KVCache()], [mtp], mx.ones((1, 1, 16))


def _fresh():
    from mlx_lm.models.cache import ArraysCache, KVCache

    return [KVCache(), ArraysCache(2), KVCache()], [KVCache()]


def test_snapshot_store(tmp_path):
    import mlx.core as mx

    from mlx_tree_spec.snapshot_store import SnapshotStore

    store = SnapshotStore(tmp_path, "setup", budget_bytes=10**9)
    snap = _fake_snapshot(12)
    store.queue(snap)
    store.queue(snap)  # queued once
    assert len(store.pending) == 1
    store.flush()
    assert not store.pending and len(store.entries) == 1
    store.queue(snap)  # already on disk
    assert not store.pending

    store = SnapshotStore(tmp_path, "setup", budget_bytes=10**9)  # a restart finds it
    prompt = snap[0] + [7, 8, 9]
    assert store.find(snap[0]) is None  # never the whole prompt
    assert store.find(prompt, longer_than=12) is None
    n, f = store.find(prompt)
    assert n == 12
    tokens, cache, mtp_cache, prev_h = store.load(f, prompt, *_fresh())
    assert tokens == snap[0] and cache[0].offset == 12 and cache[2].keys is None and cache[1].cache[1] is None
    for a, b in ((cache[0].state[0], snap[1][0].state[0]), (cache[1].cache[0], snap[1][1].cache[0]),
                 (mtp_cache[0].state[1], snap[2][0].state[1]), (prev_h, snap[3])):
        assert a.dtype == b.dtype and mx.array_equal(a, b)
    assert store.load(f, [0] + prompt, *_fresh()) is None  # not a prefix: dropped
    assert store.find(prompt) is None and not f.exists()


def test_snapshot_store_budget_and_layout(tmp_path):
    from mlx_lm.models.cache import KVCache

    from mlx_tree_spec.snapshot_store import SnapshotStore

    store = SnapshotStore(tmp_path, "a", budget_bytes=10**9)
    for n in (8, 16):
        store.queue(_fake_snapshot(n))
        store.flush()
    size = max(f.stat().st_size for f in tmp_path.glob("a/*"))
    other = SnapshotStore(tmp_path, "b", budget_bytes=size + 1)  # the budget spans setups
    other.queue(_fake_snapshot(4))
    other.flush()
    assert len(list(tmp_path.glob("*/*.safetensors"))) == 1 and len(other.entries) == 1
    store = SnapshotStore(tmp_path, "b", budget_bytes=10**9)
    n, f = store.find(list(range(100, 110)))
    assert store.load(f, list(range(100, 110)), [KVCache()], [KVCache()]) is None  # wrong layout: dropped
    assert not f.exists()

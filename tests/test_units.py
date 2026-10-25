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

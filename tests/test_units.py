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

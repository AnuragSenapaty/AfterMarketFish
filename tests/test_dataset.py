# tests/test_dataset.py
from pathlib import Path
import numpy as np
import torch

from aftermarketfish.dataset import MoveDataset


def test_move_dataset_loads_npz_and_shapes(tmp_path: Path):
    X = np.zeros((4, 12, 8, 8), dtype=np.uint8)
    y = np.array(["e2e4", "e2e4", "g1f3", "g1f3"], dtype=object)

    npz_path = tmp_path / "tiny.npz"
    np.savez_compressed(npz_path, X=X, y=y)

    ds = MoveDataset(str(npz_path))
    assert len(ds) == 4
    assert isinstance(ds.vocab, dict)
    assert set(ds.vocab.keys()) == {"e2e4", "g1f3"}

    x0, y0 = ds[0]
    assert isinstance(x0, torch.Tensor)
    assert x0.shape == (12 * 8 * 8,)
    assert y0.dtype == torch.int64
    assert 0 <= int(y0.item()) < len(ds.vocab)

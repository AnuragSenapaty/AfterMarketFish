"""Dataset helpers for training on preprocessed NPZ files."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class MoveDataset(Dataset):
    """Dataset wrapping magnus_dataset.npz-style files.

    NPZ is expected to contain:
    - X: (N, 12, 8, 8) bitboard planes
    - y: (N,) array of UCI move strings
    """

    def __init__(self, npz_path: str, vocab: Optional[Dict[str, int]] = None) -> None:
        super().__init__()
        data = np.load(npz_path, allow_pickle=True)
        X = data["X"]
        y_uci = data["y"]

        if vocab is None:
            uniq = sorted(set(map(str, y_uci.tolist())))
            self.vocab: Dict[str, int] = {m: i for i, m in enumerate(uniq)}
        else:
            self.vocab = vocab

        y_ids = []
        kept_idx = []
        for i, m in enumerate(y_uci):
            m = str(m)
            if m in self.vocab:
                y_ids.append(self.vocab[m])
                kept_idx.append(i)

        if len(kept_idx) != len(y_uci):
            X = X[kept_idx]

        y = np.array(y_ids, dtype=np.int64)

        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y)
        self.N = self.X.shape[0]

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int):
        x = self.X[idx].view(-1)  # flatten 12×8×8 → vector
        y = self.y[idx]
        return x, y

"""Feed-forward network model and checkpoint loader."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict, Any

import torch
import torch.nn as nn


class FFNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        h1: int = 1024,
        h2: int = 512,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


def load_checkpoint(
    ckpt_path: str | Path,
    map_location: str | torch.device = "cpu",
) -> Tuple[FFNN, Dict[str, Any]]:
    """Load a model checkpoint saved by the training script.

    Returns (model, meta_dict) where meta_dict contains hyperparameters.
    """
    ckpt_path = Path(ckpt_path)
    blob = torch.load(ckpt_path, map_location=map_location)

    input_dim = int(blob["input_dim"])
    num_classes = int(blob["num_classes"])
    h1 = int(blob.get("h1", 1024))
    h2 = int(blob.get("h2", 512))
    dropout = float(blob.get("dropout", 0.2))

    model = FFNN(
        input_dim=input_dim,
        num_classes=num_classes,
        h1=h1,
        h2=h2,
        dropout=dropout,
    )
    model.load_state_dict(blob["model_state_dict"])
    model.eval()

    meta: Dict[str, Any] = {
        "input_dim": input_dim,
        "num_classes": num_classes,
        "h1": h1,
        "h2": h2,
        "dropout": dropout,
    }
    return model, meta

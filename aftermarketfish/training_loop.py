"""Training utilities for the FFNN model."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from .paths import resolve_repo_path
from .dataset import MoveDataset
from .model_ffnn import FFNN


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    tot_loss, tot_acc, tot_n = 0.0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)

        opt.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()

        bs = y.size(0)
        tot_loss += loss.item() * bs
        tot_acc += accuracy(logits.detach(), y) * bs
        tot_n += bs

    return tot_loss / tot_n, tot_acc / tot_n


@torch.no_grad()
def eval_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    tot_loss, tot_acc, tot_n = 0.0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)

        bs = y.size(0)
        tot_loss += loss.item() * bs
        tot_acc += accuracy(logits, y) * bs
        tot_n += bs

    return tot_loss / tot_n, tot_acc / tot_n


def train_ffnn(
    npz_path: str,
    out_dir: str,
    epochs: int = 10,
    batch_size: int = 512,
    lr: float = 3e-4,
    dropout: float = 0.2,
    hidden1: int = 1024,
    hidden2: int = 512,
    val_split: float = 0.1,
    seed: int = 42,
) -> Dict[str, Any]:
    """High-level training entrypoint used by the CLI wrapper."""
    npz_path = resolve_repo_path(npz_path)
    out_dir = resolve_repo_path(out_dir)

    set_seed(seed)

    ds_full = MoveDataset(npz_path, vocab=None)
    num_classes = len(ds_full.vocab)
    input_dim = 12 * 8 * 8

    n_total = len(ds_full)
    n_val = max(1, int(val_split * n_total))
    n_train = n_total - n_val
    ds_train, ds_val = random_split(ds_full, [n_train, n_val])

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FFNN(
        input_dim=input_dim,
        num_classes=num_classes,
        h1=hidden1,
        h2=hidden2,
        dropout=dropout,
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    vocab_json = out_path / "vocab.json"
    model_name = out_path / "ffnn.pt"
    meta_json = out_path / "training_meta.json"

    best_stats: Dict[str, Any] = {}

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, dl_train, opt, loss_fn, device)
        va_loss, va_acc = eval_one_epoch(model, dl_val, loss_fn, device)
        print(f"[E{epoch:02d}] train {tr_loss:.4f}/{tr_acc:.4f} | val {va_loss:.4f}/{va_acc:.4f}")

        if va_loss < best_val:
            best_val = va_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": input_dim,
                    "num_classes": num_classes,
                    "h1": hidden1,
                    "h2": hidden2,
                    "dropout": dropout,
                },
                model_name,
            )

            with open(vocab_json, "w", encoding="utf-8") as f:
                json.dump(ds_full.vocab, f, indent=2, ensure_ascii=False)

            meta: Dict[str, Any] = {
                "npz": npz_path,
                "out_dir": out_dir,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "dropout": dropout,
                "hidden1": hidden1,
                "hidden2": hidden2,
                "val_split": val_split,
                "seed": seed,
                "device": str(device),
                "best_val_loss": va_loss,
                "best_val_acc": va_acc,
            }
            with open(meta_json, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            best_stats = meta
            print(f"[SAVE] {model_name} | {vocab_json} | {meta_json}")

    print("[DONE] Training complete.")
    return best_stats

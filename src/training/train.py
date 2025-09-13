# src/train.py
import os
import json
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# -------------------
# Data utilities
# -------------------

class MoveDataset(Dataset):
    def __init__(self, npz_path: str, vocab: dict | None = None):
        data = np.load(npz_path, allow_pickle=False)
        X = data["X"]  # (N, 12, 8, 8), uint8
        y_uci = data["y"]  # (N,), UCI strings

        # Build / use vocab
        if vocab is None:
            uniq = sorted(set(str(m) for m in y_uci.tolist()))
            self.vocab = {m: i for i, m in enumerate(uniq)}
        else:
            self.vocab = vocab

        # Map UCI -> class id (ignore unseen moves)
        y = []
        dropped = 0
        for m in y_uci:
            m = str(m)
            if m in self.vocab:
                y.append(self.vocab[m])
            else:
                dropped += 1
        if dropped:
            print(f"[WARN] {dropped} samples had moves not in vocab and were dropped.")

        # If we dropped any, we must align X as well
        if dropped:
            # rebuild filtered X in the same pass
            X_filtered, y_filtered = [], []
            vi = {m: i for i, m in enumerate(y_uci)}  # index by original order
            k = 0
            for m in y_uci:
                m = str(m)
                if m in self.vocab:
                    X_filtered.append(X[k])
                    y_filtered.append(self.vocab[m])
                k += 1
            X = np.stack(X_filtered)
            y = np.array(y_filtered, dtype=np.int64)
        else:
            y = np.array(y, dtype=np.int64)

        # Save tensors
        self.X = torch.from_numpy(X.astype(np.float32))  # (N, 12, 8, 8)
        self.y = torch.from_numpy(y)                     # (N,)

        # Normalize to 0/1 floats (already 0/1, but make explicit)
        # self.X = self.X  # bitmaps are already 0/1

        self.N = self.X.shape[0]

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        # Return flattened features for FFNN
        x = self.X[idx].view(-1)   # (12*8*8,)
        y = self.y[idx]
        return x, y

# -------------------
# Model
# -------------------

class FFNN(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, h1=1024, h2=512, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(h2, num_classes),
        )

    def forward(self, x):
        return self.net(x)

# -------------------
# Train / Eval
# -------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def accuracy(logits, targets):
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).float().mean().item()

def train_one_epoch(model, loader, opt, loss_fn, device):
    model.train()
    total_loss, total_acc, total_n = 0.0, 0.0, 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
        bs = y.size(0)
        total_loss += loss.item() * bs
        total_acc  += accuracy(logits.detach(), y) * bs
        total_n    += bs
    return total_loss / total_n, total_acc / total_n

@torch.no_grad()
def eval_one_epoch(model, loader, loss_fn, device):
    model.eval()
    total_loss, total_acc, total_n = 0.0, 0.0, 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        bs = y.size(0)
        total_loss += loss.item() * bs
        total_acc  += accuracy(logits, y) * bs
        total_n    += bs
    return total_loss / total_n, total_acc / total_n

# -------------------
# Main
# -------------------

def main():
    parser = argparse.ArgumentParser(description="Train FFNN on bitboard → move classification")
    parser.add_argument("--npz", type=str, default="data/Magnus/Processed/magnus_dataset.npz",
                        help="Path to compressed dataset .npz (with X and y arrays)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--hidden1", type=int, default=1024)
    parser.add_argument("--hidden2", type=int, default=512)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="models",
                        help="Directory to save model + vocab + meta")
    parser.add_argument("--vocab_json", type=str, default="vocab.json")
    parser.add_argument("--model_name", type=str, default="ffnn.pt")
    parser.add_argument("--meta_json", type=str, default="training_meta.json")
    args = parser.parse_args()

    set_seed(args.seed)

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    npz_path = os.path.join(project_root, "data", "Magnus", "Processed", "magnus_dataset.npz")
    print("Resolved npz_path:", npz_path)
    args.npz = npz_path
    # Load full dataset to build vocab
    print(f"[INFO] Loading dataset: {args.npz}")
    ds_full = MoveDataset(args.npz, vocab=None)
    num_classes = len(ds_full.vocab)
    input_dim = 12 * 8 * 8
    print(f"[INFO] Samples: {len(ds_full)} | Classes: {num_classes}")

    if len(ds_full) == 0:
        raise RuntimeError("Empty dataset after loading. Check your .npz contents.")

    # Train / val split
    val_size = int(len(ds_full) * args.val_split)
    train_size = len(ds_full) - val_size
    ds_train, ds_val = random_split(ds_full, [train_size, val_size],
                                    generator=torch.Generator().manual_seed(args.seed))
    print(f"[INFO] Split → train: {train_size}, val: {val_size}")

    # Dataloaders
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=0)
    dl_val   = DataLoader(ds_val,   batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=0)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    model = FFNN(input_dim=input_dim, num_classes=num_classes,
                 h1=args.hidden1, h2=args.hidden2, dropout=args.dropout).to(device)

    # Loss / Optim
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, dl_train, opt, loss_fn, device)
        va_loss, va_acc = eval_one_epoch(model, dl_val, loss_fn, device)
        print(f"[E{epoch:02d}] train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {va_loss:.4f} acc {va_acc:.4f}")
        if va_loss < best_val:
            best_val = va_loss
            # Save checkpoint
            out_dir = Path(args.out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = out_dir / args.model_name
            torch.save({
                "model_state_dict": model.state_dict(),
                "input_dim": input_dim,
                "num_classes": num_classes,
                "h1": args.hidden1,
                "h2": args.hidden2,
                "dropout": args.dropout,
            }, ckpt_path)
            # Save vocab
            vocab_path = out_dir / args.vocab_json
            with open(vocab_path, "w", encoding="utf-8") as f:
                json.dump(ds_full.vocab, f, indent=2, ensure_ascii=False)
            # Save meta
            meta = {
                "npz": os.path.abspath(args.npz),
                "samples_total": len(ds_full),
                "train_size": train_size,
                "val_size": val_size,
                "classes": num_classes,
                "epochs_seen": epoch,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "seed": args.seed,
            }
            meta_path = out_dir / args.meta_json
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            print(f"[SAVE] best ckpt → {ckpt_path}")
            print(f"[SAVE] vocab     → {vocab_path}")
            print(f"[SAVE] meta      → {meta_path}")

    print("[DONE] Training complete.")

if __name__ == "__main__":
    main()

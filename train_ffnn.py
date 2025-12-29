"""CLI wrapper for training the FFNN model."""

from __future__ import annotations

import argparse

from aftermarketfish.training_loop import train_ffnn


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FFNN on bitboard → move classification")
    parser.add_argument("--npz", type=str, default="data/Magnus/Processed/magnus_dataset.npz")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--hidden1", type=int, default=1024)
    parser.add_argument("--hidden2", type=int, default=512)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="models")

    args = parser.parse_args()

    train_ffnn(
        npz_path=args.npz,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        dropout=args.dropout,
        hidden1=args.hidden1,
        hidden2=args.hidden2,
        val_split=args.val_split,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

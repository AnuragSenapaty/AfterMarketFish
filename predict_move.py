"""CLI wrapper for predicting a move from a FEN position."""

from __future__ import annotations

import argparse

import chess

from aftermarketfish.inference import predict_move_from_board


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict a move from a FEN using a trained FFNN model.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint (ffnn.pt)")
    parser.add_argument("--vocab", type=str, required=True, help="Path to vocab.json")
    parser.add_argument("--fen", type=str, required=True, help="FEN string of the position")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run on (e.g. 'cpu', 'cuda', or 'cuda:0')",
    )
    args = parser.parse_args()

    board = chess.Board(args.fen)
    move_uci, prob = predict_move_from_board(
        board,
        ckpt_path=args.ckpt,
        vocab_path=args.vocab,
        device=args.device,
    )
    print(f"Predicted move: {move_uci} (p={prob:.3f})")


if __name__ == "__main__":
    main()
